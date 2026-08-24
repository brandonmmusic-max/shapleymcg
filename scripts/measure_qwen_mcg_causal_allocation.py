#!/usr/bin/env python3
"""Install a sealed causal MCG allocation and measure prefix/final KLD.

Candidate layers are streamed from the immutable Hub inventory, verified by
whole-file and selected-tensor SHA-256, installed cumulatively, and re-anchored
every configured layer interval.  Only the final student logits are retained;
every prefix retains its exact token-KLD vector and allocation-prefix seal.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
from quant_pipeline.core.artifacts import (
    atomic_write,
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.evaluation.kld_window import verify_kld_window
from quant_pipeline.normalization.artifact_v31 import tensor_sha256
from quant_pipeline.scoring.kld import summarize


MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_teacher_logits(path: Path) -> np.ndarray:
    """Load the canonical LM logits without depending on campaign internals."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        if "logits" not in handle.keys():
            raise ValueError("teacher safetensors must contain a logits tensor")
        return np.asarray(handle.get_tensor("logits"))


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    observed = _hash_json({key: value for key, value in document.items() if key != field})
    if document.get(field) != observed:
        raise ValueError(f"{label} seal mismatch")


def _token_kld(teacher: np.ndarray, student: np.ndarray, chunk: int = 16) -> np.ndarray:
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student logit geometry mismatch")
    result = np.empty(teacher.shape[0], dtype=np.float64)
    for start in range(0, len(result), chunk):
        stop = min(start + chunk, len(result))
        target = np.asarray(teacher[start:stop], dtype=np.float64)
        observed = np.asarray(student[start:stop], dtype=np.float64)
        if not np.isfinite(target).all() or not np.isfinite(observed).all():
            raise ValueError("teacher/student logits contain non-finite values")
        target -= np.max(target, axis=-1, keepdims=True)
        observed -= np.max(observed, axis=-1, keepdims=True)
        target -= np.logaddexp.reduce(target, axis=-1, keepdims=True)
        observed -= np.logaddexp.reduce(observed, axis=-1, keepdims=True)
        result[start:stop] = np.sum(np.exp(target) * (target - observed), axis=-1)
    return result


def _save_npy(path: Path, value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_write(path, buffer.getvalue())
    return sha256_file(path)


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


def _capture(model: Any, token_ids: list[int]) -> Any:
    import torch

    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
    return logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()


def _install_layer(
    *,
    model: Any,
    layer: int,
    candidate_path: Path,
    candidate_file_sha256: str,
    choices: dict[tuple[int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    blocks = qwen_moe_layers(model)
    gate_up = []
    down = []
    selected = []
    with safe_open(candidate_path, framework="pt", device="cpu") as handle:
        for expert in range(128):
            projections = {}
            values = {}
            for projection in PROJECTIONS:
                choice = choices[(layer, expert, projection)]
                bits = int(choice["bits"])
                key = f"K{bits}.E{expert:03d}.{projection}.reconstruction_hf"
                tensor = handle.get_tensor(key).contiguous()
                if tensor_sha256(tensor) != choice["stored_bf16_reconstruction_sha256"]:
                    raise ValueError(f"selected tensor identity mismatch: {key}")
                values[projection] = tensor
                projections[projection] = {
                    "bits": bits,
                    "tensor": key,
                    "sha256": choice["stored_bf16_reconstruction_sha256"],
                }
            gate_up.append(torch.cat((values["gate_proj"], values["up_proj"]), dim=0))
            down.append(values["down_proj"])
            selected.append({"expert": expert, "projections": projections})
    experts = blocks[layer].experts
    gate_up_tensor = torch.stack(gate_up).to(
        device=experts.gate_up_proj.device,
        dtype=experts.gate_up_proj.dtype,
    )
    down_tensor = torch.stack(down).to(
        device=experts.down_proj.device,
        dtype=experts.down_proj.dtype,
    )
    with torch.no_grad():
        experts.gate_up_proj.copy_(gate_up_tensor)
        experts.down_proj.copy_(down_tensor)
    body = {
        "schema": "quant-pipeline.qwen-mcg-installed-layer.v1",
        "layer": layer,
        "candidate_file_sha256": candidate_file_sha256,
        "selected": selected,
    }
    body["installed_layer_sha256"] = _hash_json(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--local-encode-root", type=Path)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--kld-window", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--reanchor-every-layers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.reanchor_every_layers < 1 or 48 % args.reanchor_every_layers:
        parser.error("--reanchor-every-layers must be a positive divisor of 48")
    plan = {
        "schema": "quant-pipeline.qwen-mcg-causal-kld-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "model_revision": args.model_revision,
        "allocation": str(args.allocation.resolve()),
        "allocation_file_sha256": sha256_file(args.allocation),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "local_encode_root": str(args.local_encode_root.resolve()) if args.local_encode_root else None,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "kld_window": str(args.kld_window.resolve()),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "output": str(args.output.resolve()),
        "device_map": args.device_map,
        "attention_backend": args.attention_backend,
        "reanchor_every_layers": args.reanchor_every_layers,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    allocation = json.loads(args.allocation.read_text())
    inventory = json.loads(args.candidate_inventory.read_text())
    _verify_seal(allocation, "allocation_sha256", "research allocation")
    _verify_seal(inventory, "inventory_sha256", "candidate inventory")
    if allocation.get("candidate_inventory_sha256") != inventory["inventory_sha256"]:
        raise ValueError("research allocation belongs to a different candidate inventory")
    if (
        allocation.get("average_weight_bits") != 3.5
        or allocation.get("k3_count") != 9216
        or allocation.get("k4_count") != 9216
        or len(allocation.get("choices", ())) != 18432
    ):
        raise ValueError("research allocation is not exact 3.5 routed-expert BPW")
    choices = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): row
        for row in allocation["choices"]
    }
    if len(choices) != 18432:
        raise ValueError("research allocation matrix inventory is incomplete")
    window_root = args.kld_window.resolve()
    window = json.loads((window_root / "kld-window.json").read_text())
    verify_kld_window(window, window_root)
    teacher = load_teacher_logits(args.teacher.resolve())

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.source_model.resolve(),
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = args.candidate_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    installed = []
    reanchors = []
    final_student = None
    started = time.monotonic()
    for row in inventory["layers"]:
        layer = int(row["layer"])
        layer_started = time.monotonic()
        candidate_path, temporary = _candidate_path(
            row=row,
            local_root=args.local_encode_root.resolve() if args.local_encode_root else None,
            cache_root=cache,
            repo=inventory["repo_id"],
            revision=inventory["revision"],
        )
        installed.append(_install_layer(
            model=model,
            layer=layer,
            candidate_path=candidate_path,
            candidate_file_sha256=row["candidate_sha256"],
            choices=choices,
        ))
        if temporary:
            candidate_path.unlink()
        print(json.dumps({
            "stage": "install",
            "layer": layer,
            "installed_layer_sha256": installed[-1]["installed_layer_sha256"],
            "elapsed_seconds": time.monotonic() - layer_started,
        }, sort_keys=True), flush=True)
        if (layer + 1) % args.reanchor_every_layers == 0:
            student = _capture(model, window["token_ids"])
            values = _token_kld(teacher, student.numpy())
            prefix = output / f"reanchor-layer-{layer:03d}.token-kld.npy"
            token_sha = _save_npy(prefix, values)
            prefix_choices = [choice for choice in allocation["choices"] if int(choice["layer"]) <= layer]
            reanchor = {
                "installed_through_layer": layer,
                "installed_layer_count": layer + 1,
                "allocation_prefix_sha256": _hash_json(prefix_choices),
                "installed_prefix_sha256": _hash_json(installed),
                "token_kld_file": prefix.name,
                "token_kld_sha256": token_sha,
                "summary": summarize(values),
                "top1_agreement": float(np.mean(np.argmax(teacher, axis=-1) == np.argmax(student.numpy(), axis=-1))),
            }
            reanchor["reanchor_sha256"] = _hash_json(reanchor)
            reanchors.append(reanchor)
            print(json.dumps({"stage": "reanchor", **reanchor}, sort_keys=True), flush=True)
            if layer == 47:
                final_student = student
            else:
                del student
    if final_student is None:
        raise RuntimeError("final causal reanchor did not execute")
    student_path = output / "student-logits.safetensors"
    save_file({"logits": final_student}, student_path, metadata={
        "role": "kld-student",
        "allocation_sha256": allocation["allocation_sha256"],
        "token_sha256": str(window["token_sha256"]),
    })
    final_values = np.load(output / reanchors[-1]["token_kld_file"], allow_pickle=False)
    report = {
        "schema": "quant-pipeline.qwen-mcg-causal-kld.v1",
        "model_revision": args.model_revision,
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "allocation_sha256": allocation["allocation_sha256"],
        "teacher_sha256": sha256_file(args.teacher),
        "student_sha256": sha256_file(student_path),
        "student_shape": list(final_student.shape),
        "installed_layers": installed,
        "installed_prefix_sha256": _hash_json(installed),
        "reanchors": reanchors,
        "final_token_kld_sha256": reanchors[-1]["token_kld_sha256"],
        "summary": summarize(final_values),
        "top1_agreement": reanchors[-1]["top1_agreement"],
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = _hash_json(report)
    write_json(output / "kld-report.json", report)
    print(json.dumps({
        "ok": True,
        "report_sha256": report["report_sha256"],
        "mean_kld": report["summary"]["mean"],
        "top1_agreement": report["top1_agreement"],
        "reanchor_count": len(reanchors),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
