#!/usr/bin/env python3
"""Measure native Aumann-Shapley/Fisher attribution from sealed MCG K4 candidates.

The fast Qwen control published exact K3/K4 MCG reconstructions before the
native causal-attribution producer was wired into the full-model run.  This
bridge consumes that immutable candidate inventory directly from Hugging Face,
uses uniform K4 as the explicit provisional path endpoint, and preserves the
source, teacher, candidate, code, and numeric identities in sealed receipts.
It does not re-encode weights or substitute a fake-quant proxy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np

from quant_pipeline.campaign.qwen_attribution import (
    load_teacher_logits,
    measure_native_causal_attribution,
    verify_attribution_inputs,
    write_attribution_inputs,
)
from quant_pipeline.core.artifacts import (
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.evaluation.kld_window import verify_kld_window
from quant_pipeline.normalization.artifact_v31 import tensor_sha256
from quant_pipeline.scoring.attribution import split_layer_damage


MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
DATASET_REPO = "brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility"
DATASET_REVISION = "e4d8a67ddb1f0b4c7605c5efcdc3c54e87e22b9f"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
REVISION = re.compile(r"[0-9a-f]{40}")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    expected = document.get(field)
    observed = _hash_json({key: value for key, value in document.items() if key != field})
    if expected != observed:
        raise ValueError(f"{label} seal mismatch")


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if REVISION.fullmatch(value) is None:
        raise ValueError("pipeline Git revision is not immutable")
    return value


def _lfs_sha(item: Any) -> str | None:
    value = getattr(item, "lfs", None)
    if value is None:
        return None
    return value.get("sha256") if isinstance(value, dict) else getattr(value, "sha256", None)


def _load_receipt(repo: str, revision: str, layer: int) -> tuple[dict[str, Any], Path]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=f"candidates/layer-{layer:03d}/encode-receipt.json",
    )
    receipt = json.loads(Path(path).read_text())
    _verify_seal(receipt, "receipt_sha256", f"layer {layer} encode receipt")
    if (
        receipt.get("schema") != "quant-pipeline.qwen-fast-k34-encode.v2"
        or receipt.get("layer") != layer
        or receipt.get("experts") != list(range(128))
        or receipt.get("candidate_count") != 128 * 3 * 2
    ):
        raise ValueError(f"layer {layer} candidate receipt is incomplete")
    return receipt, Path(path)


def _inventory(repo: str, revision: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from huggingface_hub import HfApi

    if REVISION.fullmatch(revision) is None:
        raise ValueError("candidate dataset revision must be immutable")
    api = HfApi()
    info = api.repo_info(repo_id=repo, repo_type="dataset", revision=revision)
    if str(info.sha) != revision:
        raise ValueError("candidate dataset revision did not resolve exactly")
    with ThreadPoolExecutor(max_workers=12) as pool:
        loaded = list(pool.map(lambda layer: _load_receipt(repo, revision, layer), range(48)))
    rows = []
    for layer, (receipt, receipt_path) in enumerate(loaded):
        candidate_path = f"candidates/layer-{layer:03d}/{receipt['candidate_tensor_file']}"
        remote = api.get_paths_info(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            paths=[candidate_path],
            expand=True,
        )
        if len(remote) != 1:
            raise ValueError(f"layer {layer} candidate payload is absent from the Hub")
        item = remote[0]
        if int(item.size) != int(receipt["candidate_tensor_bytes"]):
            raise ValueError(f"layer {layer} candidate payload byte count drifted")
        if _lfs_sha(item) != receipt["candidate_tensor_sha256"]:
            raise ValueError(f"layer {layer} candidate Hub SHA-256 drifted")
        rows.append({
            "layer": layer,
            "receipt_path": f"candidates/layer-{layer:03d}/encode-receipt.json",
            "receipt_file_sha256": sha256_file(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "candidate_path": candidate_path,
            "candidate_bytes": int(item.size),
            "candidate_sha256": receipt["candidate_tensor_sha256"],
            "uniform_k4_score_rows_sha256": _hash_json([
                row for row in receipt["scores"] if int(row["bits"]) == 4
            ]),
        })
    body = {
        "schema": "quant-pipeline.qwen-hf-mcg-candidate-inventory.v1",
        "repo_id": repo,
        "repo_type": "dataset",
        "revision": revision,
        "provisional_bit_triplet": [4, 4, 4],
        "layers": rows,
    }
    body["inventory_sha256"] = _hash_json(body)
    return body, [value[0] for value in loaded]


def _candidate_path(
    *,
    row: dict[str, Any],
    local_root: Path | None,
    cache_root: Path,
    repo: str,
    revision: str,
) -> tuple[Path, bool]:
    if local_root is not None:
        local = local_root / f"layer-{int(row['layer']):03d}" / "k34-candidates.safetensors"
        if local.is_file() and sha256_file(local) == row["candidate_sha256"]:
            return local, False
    from huggingface_hub import hf_hub_download

    value = Path(hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=row["candidate_path"],
        local_dir=cache_root,
    ))
    if value.stat().st_size != row["candidate_bytes"] or sha256_file(value) != row["candidate_sha256"]:
        raise ValueError(f"downloaded layer {row['layer']} candidate payload drifted")
    return value, True


def _load_uniform_k4(
    *,
    inventory: dict[str, Any],
    receipts: list[dict[str, Any]],
    local_root: Path | None,
    cache_root: Path,
) -> dict[int, dict[str, Any]]:
    import torch
    from safetensors import safe_open

    decoded: dict[int, dict[str, Any]] = {}
    for row, receipt in zip(inventory["layers"], receipts, strict=True):
        started = time.monotonic()
        path, temporary = _candidate_path(
            row=row,
            local_root=local_root,
            cache_root=cache_root,
            repo=inventory["repo_id"],
            revision=inventory["revision"],
        )
        scores = {
            (int(value["expert"]), str(value["projection"])): value
            for value in receipt["scores"]
            if int(value["bits"]) == 4
        }
        if len(scores) != 128 * 3:
            raise ValueError(f"layer {row['layer']} lacks the complete uniform-K4 score inventory")
        values: dict[str, list[Any]] = {projection: [] for projection in PROJECTIONS}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for projection in PROJECTIONS:
                for expert in range(128):
                    key = f"K4.E{expert:03d}.{projection}.reconstruction_hf"
                    tensor = handle.get_tensor(key).contiguous()
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f"uniform-K4 tensor {key} is not BF16")
                    if tensor_sha256(tensor) != scores[(expert, projection)]["stored_bf16_reconstruction_sha256"]:
                        raise ValueError(f"uniform-K4 tensor {key} identity mismatch")
                    values[projection].append(tensor)
        decoded[int(row["layer"])] = {
            projection: torch.stack(values[projection]).contiguous()
            for projection in PROJECTIONS
        }
        if temporary:
            path.unlink()
        print(json.dumps({
            "stage": "uniform-k4-load",
            "layer": row["layer"],
            "decoded_bytes": sum(value.numel() * value.element_size() for value in decoded[int(row["layer"])].values()),
            "elapsed_seconds": time.monotonic() - started,
        }, sort_keys=True), flush=True)
    return decoded


def _attribution_document(arrays: dict[str, np.ndarray], inventory_sha256: str) -> dict[str, Any]:
    layers = []
    for index, layer in enumerate(arrays["layer_indices"]):
        split = split_layer_damage(
            float(arrays["measured_layer_damage"][index]),
            arrays["projected_expert_residuals"][index],
            projected_routing_residual=arrays["projected_routing_residuals"][index],
        )
        layers.append({
            "layer_index": int(layer),
            "aumann_shapley": float(arrays["measured_layer_damage"][index]),
            **split,
        })
    measured = float(arrays["measured_end_to_end_delta"][0])
    layer_total = float(np.sum(arrays["measured_layer_damage"]))
    body = {
        "schema": "quant-pipeline.qwen-hf-mcg-attribution.v2",
        "candidate_inventory_sha256": inventory_sha256,
        "path_nodes": int(len(arrays["path_nodes"])),
        "fisher_rank": int(arrays["projected_expert_residuals"].shape[-1]),
        "source_kld": float(arrays["source_kld"][0]),
        "candidate_kld": float(arrays["candidate_kld"][0]),
        "measured_end_to_end_delta": measured,
        "sum_measured_layer_damage": layer_total,
        "unresolved_path_quadrature_and_nonlinear_remainder": measured - layer_total,
        "closed_end_to_end_delta": measured,
        "sum_closed_damage": float(sum(row["closed_total"] for row in layers)),
        "remainder_policy": "explicit-unresolved-nonlinear-remainder",
        "layers": layers,
    }
    body["attribution_sha256"] = _hash_json(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--local-encode-root", type=Path)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--kld-window", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--path-nodes", type=int, default=5)
    parser.add_argument("--fisher-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if REVISION.fullmatch(args.model_revision) is None or REVISION.fullmatch(args.dataset_revision) is None:
        parser.error("model and dataset revisions must be immutable 40-hex commits")
    plan = {
        "schema": "quant-pipeline.qwen-hf-mcg-native-attribution-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "model_revision": args.model_revision,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "local_encode_root": str(args.local_encode_root.resolve()) if args.local_encode_root else None,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "kld_window": str(args.kld_window.resolve()),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "output": str(args.output.resolve()),
        "device_map": args.device_map,
        "attention_backend": args.attention_backend,
        "path_nodes": args.path_nodes,
        "fisher_rank": args.fisher_rank,
        "seed": args.seed,
        "pipeline_git_revision": _git_revision(),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    inventory, receipts = _inventory(args.dataset_repo, args.dataset_revision)
    write_json(output / "candidate-inventory.json", inventory)
    cache = args.candidate_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    decoded = _load_uniform_k4(
        inventory=inventory,
        receipts=receipts,
        local_root=args.local_encode_root.resolve() if args.local_encode_root else None,
        cache_root=cache,
    )
    window_root = args.kld_window.resolve()
    window = json.loads((window_root / "kld-window.json").read_text())
    verify_kld_window(window, window_root)

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.source_model.resolve(),
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    arrays = measure_native_causal_attribution(
        model=model,
        token_ids=window["token_ids"],
        decoded_by_layer=decoded,
        teacher_logits=load_teacher_logits(args.teacher.resolve()),
        path_nodes=args.path_nodes,
        fisher_rank=args.fisher_rank,
        seed=args.seed,
    )
    del model, decoded
    torch.cuda.empty_cache()
    attribution_inputs = write_attribution_inputs(
        output / "attribution-inputs.npz",
        arrays,
        provenance={
            "implementation": "native-qwen-hf-uniform-k4-mcg-blend-fisher-v2",
            "model_revision": args.model_revision,
            "kld_window_seal_sha256": window["seal_sha256"],
            "teacher_reference_sha256": sha256_file(args.teacher),
            "candidate_inventory_sha256": inventory["inventory_sha256"],
            "candidate_dataset_repo": args.dataset_repo,
            "candidate_dataset_revision": args.dataset_revision,
            "provisional_bit_triplet": [4, 4, 4],
            "path_nodes": args.path_nodes,
            "fisher_rank": args.fisher_rank,
            "seed": args.seed,
            "test_only": False,
        },
    )
    verify_attribution_inputs(attribution_inputs)
    attribution = _attribution_document(arrays, inventory["inventory_sha256"])
    write_json(output / "attribution.json", attribution)
    print(json.dumps({
        "ok": True,
        "attribution_sha256": attribution["attribution_sha256"],
        "candidate_kld": attribution["candidate_kld"],
        "sum_measured_layer_damage": attribution["sum_measured_layer_damage"],
        "path_remainder": attribution["unresolved_path_quadrature_and_nonlinear_remainder"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
