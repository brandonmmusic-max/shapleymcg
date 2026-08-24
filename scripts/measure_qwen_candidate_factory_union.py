#!/usr/bin/env python3
"""Select between TurboDerp and MCG candidate factories at fixed 3.5 BPW.

The full causal K3/K4 allocation is frozen.  A disjoint 2,048-token WikiText
row selects whole-layer factory swaps by direct end-to-end KLD, and the other
nine rows provide the untouched validation endpoint.  Candidate factory is
free only within an already selected bit rate; no swap can change the K3/K4
count, non-expert body, teacher, or evaluator.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from measure_qwen_turboderp_exact_3p5 import (
    _install_turbo_selected_k3,
    _load_allocation,
    _validate_turbo_scope,
)
from measure_qwen_turboderp_hybrid_k4 import _install_turboderp_full
from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
from quant_pipeline.core.artifacts import (
    atomic_write,
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.normalization.artifact_v31 import tensor_sha256
from quant_pipeline.scoring.kld import summarize


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


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


def _save_npy(path: Path, value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_write(path, buffer.getvalue())
    return sha256_file(path)


def _save_logits(path: Path, value: np.ndarray, metadata: dict[str, str]) -> str:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite student logits: {path}")
    save_file({"logits": np.ascontiguousarray(value, dtype=np.float32)}, path, metadata=metadata)
    return sha256_file(path)


def _load_npy(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"saved token artifact hash mismatch: {path}")
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)


def _resume_layer_rows(
    log_path: Path,
    output: Path,
    baseline_token: np.ndarray,
    inventory: dict[str, Any],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for raw in log_path.read_text().splitlines():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if value.get("stage") != "single-layer-swap":
            continue
        value = dict(value)
        value.pop("stage", None)
        rows[int(value["layer"])] = value
    if sorted(rows) != list(range(48)):
        raise ValueError("resume log does not contain 48 completed single-layer swaps")
    result = []
    for layer in range(48):
        row = rows[layer]
        inventory_row = inventory["layers"][layer]
        if row["candidate_file_sha256"] != inventory_row["candidate_sha256"]:
            raise ValueError(f"resume layer {layer} candidate identity differs from inventory")
        token = _load_npy(
            output / "single-layer-token-kld" / f"layer-{layer:03d}.npy",
            row["token_kld_sha256"],
        )
        delta = _load_npy(
            output / "single-layer-token-delta" / f"layer-{layer:03d}.npy",
            row["token_delta_sha256"],
        )
        if not np.array_equal(token - baseline_token, delta):
            raise ValueError(f"resume layer {layer} token delta does not reproduce")
        if float(token.mean()) != float(row["mcg_mean_kld"]):
            raise ValueError(f"resume layer {layer} mean KLD does not reproduce")
        if float(delta.mean()) != float(row["delta_mean_kld_vs_turbo"]):
            raise ValueError(f"resume layer {layer} mean delta does not reproduce")
        interval = _paired_block_interval(delta, seed=seed + layer)
        if interval != [float(value) for value in row["paired_block_95_interval"]]:
            raise ValueError(f"resume layer {layer} bootstrap interval does not reproduce")
        result.append(row)
    return result


def _teacher(path: Path) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise ValueError(f"teacher file has an unexpected tensor inventory: {path}")
        value = np.asarray(handle.get_tensor(keys[0]), dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"teacher logits are not rank two: {path}")
    return np.ascontiguousarray(value)


def _capture(model: Any, token_ids: np.ndarray) -> np.ndarray:
    import torch

    device = model.get_input_embeddings().weight.device
    ids = torch.from_numpy(token_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
    return logits.float().cpu().reshape(-1, logits.shape[-1]).numpy()


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


def _paired_block_interval(
    delta: np.ndarray,
    *,
    seed: int,
    block_tokens: int = 64,
    draws: int = 20_000,
) -> list[float]:
    if delta.ndim != 1 or len(delta) % block_tokens:
        raise ValueError("paired block bootstrap requires a divisible token vector")
    blocks = delta.reshape(-1, block_tokens).mean(axis=1)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(blocks), size=(draws, len(blocks)))
    means = blocks[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _snapshot_layer(model: Any, layer: int) -> dict[str, Any]:
    experts = qwen_moe_layers(model)[layer].experts
    return {
        "gate_up": experts.gate_up_proj.detach().cpu().clone(),
        "down": experts.down_proj.detach().cpu().clone(),
    }


def _restore_layer(model: Any, layer: int, snapshot: dict[str, Any]) -> None:
    import torch

    experts = qwen_moe_layers(model)[layer].experts
    with torch.no_grad():
        experts.gate_up_proj.copy_(snapshot["gate_up"].to(
            device=experts.gate_up_proj.device,
            dtype=experts.gate_up_proj.dtype,
        ))
        experts.down_proj.copy_(snapshot["down"].to(
            device=experts.down_proj.device,
            dtype=experts.down_proj.dtype,
        ))


def _install_mcg_layer(
    model: Any,
    layer: int,
    path: Path,
    choices: dict[tuple[int, int, str], int],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    score_rows = {
        (int(row["expert"]), str(row["projection"]), int(row["bits"])): row
        for row in receipt["scores"]
    }
    experts = qwen_moe_layers(model)[layer].experts
    gate_up = []
    down = []
    selected = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for expert in range(128):
            values = {}
            projections = {}
            for projection in PROJECTIONS:
                bits = choices[(layer, expert, projection)]
                key = f"K{bits}.E{expert:03d}.{projection}.reconstruction_hf"
                value = handle.get_tensor(key).contiguous()
                score = score_rows[(expert, projection, bits)]
                if tensor_sha256(value) != score["stored_bf16_reconstruction_sha256"]:
                    raise ValueError(f"MCG selected tensor identity mismatch: {key}")
                values[projection] = value
                projections[projection] = {
                    "bits": bits,
                    "tensor": key,
                    "sha256": score["stored_bf16_reconstruction_sha256"],
                }
            gate_up.append(torch.cat((values["gate_proj"], values["up_proj"]), dim=0))
            down.append(values["down_proj"])
            selected.append({"expert": expert, "projections": projections})
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.stack(gate_up).to(
            device=experts.gate_up_proj.device,
            dtype=experts.gate_up_proj.dtype,
        ))
        experts.down_proj.copy_(torch.stack(down).to(
            device=experts.down_proj.device,
            dtype=experts.down_proj.dtype,
        ))
    body = {
        "schema": "quant-pipeline.qwen-mcg-factory-layer-install.v1",
        "layer": layer,
        "candidate_file_sha256": receipt["candidate_tensor_sha256"],
        "selected": selected,
    }
    body["installed_layer_sha256"] = _hash_json(body)
    return body


def _load_receipt(inventory: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=inventory["repo_id"],
        repo_type="dataset",
        revision=inventory["revision"],
        filename=row["receipt_path"],
    ))
    receipt = json.loads(path.read_text())
    _verify_seal(receipt, "receipt_sha256", f"layer {row['layer']} candidate receipt")
    if (
        receipt["receipt_sha256"] != row["receipt_sha256"]
        or receipt["candidate_tensor_sha256"] != row["candidate_sha256"]
        or int(receipt["candidate_tensor_bytes"]) != int(row["candidate_bytes"])
    ):
        raise ValueError(f"layer {row['layer']} candidate receipt differs from inventory")
    return receipt


def _evaluate_rows(
    model: Any,
    token_ids: np.ndarray,
    teacher_paths: list[Path],
    rows: list[int],
    *,
    logit_root: Path,
    arm: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    records = []
    for row in rows:
        started = time.monotonic()
        teacher = _teacher(teacher_paths[row])
        student = _capture(model, token_ids[row])
        student_sha = _save_logits(
            logit_root / f"row-{row:02d}.safetensors",
            student,
            {"arm": arm, "row": str(row)},
        )
        token = _token_kld(teacher, student)
        values.append(token)
        records.append({
            "row": row,
            "teacher_sha256": sha256_file(teacher_paths[row]),
            "student_logits_sha256": student_sha,
            "mean_kld": float(token.mean()),
            "top1_agreement": float(np.mean(
                np.argmax(teacher, axis=-1) == np.argmax(student, axis=-1)
            )),
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"stage": "evaluate", **records[-1]}, sort_keys=True), flush=True)
    return np.stack(values), records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--turboderp-k3-model", type=Path, required=True)
    parser.add_argument("--turboderp-k3-revision", required=True)
    parser.add_argument("--turboderp-k4-model", type=Path, required=True)
    parser.add_argument("--exllamav3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--selection-row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--resume-log",
        type=Path,
        help="resume after all 48 layer swaps using the append-only producer log",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.selection_row not in range(10):
        parser.error("--selection-row must be in [0, 9]")
    plan = {
        "schema": "quant-pipeline.qwen-candidate-factory-union-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "allocation": str(args.allocation.resolve()),
        "allocation_file_sha256": sha256_file(args.allocation),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "candidate_cache": str(args.candidate_cache.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "turboderp_k3_model": str(args.turboderp_k3_model.resolve()),
        "turboderp_k3_revision": args.turboderp_k3_revision,
        "turboderp_k4_model": str(args.turboderp_k4_model.resolve()),
        "exllamav3_root": str(args.exllamav3_root.resolve()),
        "output": str(args.output.resolve()),
        "attention_backend": args.attention_backend,
        "selection_row": args.selection_row,
        "validation_rows": [row for row in range(10) if row != args.selection_row],
        "fixed_rate": "exact 9216 K3 plus 9216 K4 routed-expert matrices",
        "factory_granularity": "whole routed-expert layer",
        "seed": args.seed,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    if args.resume_log is None:
        output = prepare_empty_destination(args.output.resolve())
        write_json(output / "plan.json", plan | {"dry_run": False})
    else:
        output = args.output.resolve()
        if not output.is_dir():
            raise ValueError("resume output does not exist")
        existing_plan = json.loads((output / "plan.json").read_text())
        for key in (
            "allocation_file_sha256",
            "candidate_inventory_file_sha256",
            "source_model",
            "panel_root",
            "turboderp_k3_model",
            "turboderp_k3_revision",
            "turboderp_k4_model",
            "exllamav3_root",
            "attention_backend",
            "selection_row",
            "validation_rows",
            "fixed_rate",
            "factory_granularity",
            "seed",
        ):
            if existing_plan.get(key) != plan.get(key):
                raise ValueError(f"resume plan differs at {key}")
        plan = existing_plan
    allocation, choices = _load_allocation(args.allocation.resolve())
    inventory = json.loads(args.candidate_inventory.read_text())
    _verify_seal(inventory, "inventory_sha256", "candidate inventory")
    if allocation.get("candidate_inventory_sha256") != inventory["inventory_sha256"]:
        raise ValueError("allocation and MCG candidate inventory differ")
    if [int(row["layer"]) for row in inventory["layers"]] != list(range(48)):
        raise ValueError("candidate inventory does not contain 48 ordered layers")
    k3_scope = _validate_turbo_scope(args.turboderp_k3_model.resolve(), 3.0)
    k4_scope = _validate_turbo_scope(args.turboderp_k4_model.resolve(), 4.0)

    panel = json.loads((args.panel_root / "panel.json").read_text())
    _verify_seal(panel, "panel_sha256", "evaluation panel")
    with np.load(args.panel_root / panel["token_file"]) as handle:
        token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
    if list(token_ids.shape) != [10, 2048]:
        raise ValueError("evaluation panel must be 10 x 2,048 tokens")
    teacher_paths = sorted((args.panel_root / "teacher-logits").glob("row-*.safetensors"))
    if len(teacher_paths) != 10:
        raise ValueError("evaluation panel lacks ten teacher-logit rows")

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.source_model.resolve(),
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fixed_scope = _install_turboderp_full(
        model, args.turboderp_k4_model.resolve(), args.exllamav3_root.resolve()
    )
    rate = _install_turbo_selected_k3(
        model,
        args.turboderp_k3_model.resolve(),
        choices,
        args.exllamav3_root.resolve(),
    )
    selection_teacher = _teacher(teacher_paths[args.selection_row])
    baseline_student = _capture(model, token_ids[args.selection_row])
    baseline_token = _token_kld(selection_teacher, baseline_student)
    baseline_mean = float(baseline_token.mean())
    baseline_student_path = (
        output / "selection-baseline-student-logits" / f"row-{args.selection_row:02d}.safetensors"
    )
    if args.resume_log is None:
        baseline_student_sha = _save_logits(
            baseline_student_path,
            baseline_student,
            {"arm": "selection-baseline", "row": str(args.selection_row)},
        )
        _save_npy(output / "selection-baseline.token-kld.npy", baseline_token)
    else:
        saved_baseline_student = _teacher(baseline_student_path)
        if not np.array_equal(saved_baseline_student, baseline_student):
            raise ValueError("resumed Turbo baseline logits differ from the original capture")
        saved_baseline_token = np.asarray(
            np.load(output / "selection-baseline.token-kld.npy", allow_pickle=False),
            dtype=np.float64,
        )
        if not np.array_equal(saved_baseline_token, baseline_token):
            raise ValueError("resumed Turbo baseline token KLD differs from the original capture")
        baseline_student_sha = sha256_file(baseline_student_path)
    print(json.dumps({
        "stage": "selection-baseline", "mean_kld": baseline_mean,
    }, sort_keys=True), flush=True)

    cache = args.candidate_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    layer_rows = []
    retained_paths: dict[int, Path] = {}
    receipts: dict[int, dict[str, Any]] = {}
    inventory_rows = [] if args.resume_log is not None else inventory["layers"]
    for inventory_row in inventory_rows:
        layer = int(inventory_row["layer"])
        started = time.monotonic()
        snapshot = _snapshot_layer(model, layer)
        receipt = _load_receipt(inventory, inventory_row)
        path, temporary = _candidate_path(
            row=inventory_row,
            local_root=None,
            cache_root=cache,
            repo=inventory["repo_id"],
            revision=inventory["revision"],
        )
        install = _install_mcg_layer(model, layer, path, choices, receipt)
        student = _capture(model, token_ids[args.selection_row])
        token = _token_kld(selection_teacher, student)
        delta = token - baseline_token
        interval = _paired_block_interval(delta, seed=args.seed + layer)
        row = {
            "layer": layer,
            "mcg_mean_kld": float(token.mean()),
            "delta_mean_kld_vs_turbo": float(delta.mean()),
            "relative_delta_vs_turbo": float(delta.mean() / baseline_mean),
            "paired_block_95_interval": interval,
            "single_swap_improves_mean": bool(delta.mean() < 0.0),
            "single_swap_interval_below_zero": bool(interval[1] < 0.0),
            "candidate_file_sha256": inventory_row["candidate_sha256"],
            "installed_layer_sha256": install["installed_layer_sha256"],
            "token_kld_sha256": _save_npy(
                output / "single-layer-token-kld" / f"layer-{layer:03d}.npy", token
            ),
            "token_delta_sha256": _save_npy(
                output / "single-layer-token-delta" / f"layer-{layer:03d}.npy", delta
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
        layer_rows.append(row)
        _restore_layer(model, layer, snapshot)
        del snapshot, student, token, delta
        if temporary:
            path.unlink()
        print(json.dumps({"stage": "single-layer-swap", **row}, sort_keys=True), flush=True)

    if args.resume_log is not None:
        layer_rows = _resume_layer_rows(
            args.resume_log.resolve(),
            output,
            baseline_token,
            inventory,
            seed=args.seed,
        )
        print(json.dumps({
            "stage": "resume-layer-swaps-verified",
            "layer_count": len(layer_rows),
            "baseline_mean_kld": baseline_mean,
        }, sort_keys=True), flush=True)

    selected_layers = []
    greedy_rows = []
    current_token = baseline_token
    current_mean = baseline_mean
    retained_root = output / "retained-candidates"
    retained_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(layer_rows, key=lambda row: (row["delta_mean_kld_vs_turbo"], row["layer"]))
    for candidate in ranked:
        if not candidate["single_swap_improves_mean"]:
            continue
        layer = int(candidate["layer"])
        inventory_row = inventory["layers"][layer]
        snapshot = _snapshot_layer(model, layer)
        receipt = receipts.get(layer) or _load_receipt(inventory, inventory_row)
        receipts[layer] = receipt
        retained = retained_root / f"layer-{layer:03d}.safetensors"
        if retained.is_file():
            if sha256_file(retained) != inventory_row["candidate_sha256"]:
                raise ValueError(f"retained layer {layer} candidate identity mismatch")
            path, temporary = retained, False
        else:
            path, temporary = _candidate_path(
                row=inventory_row,
                local_root=None,
                cache_root=cache,
                repo=inventory["repo_id"],
                revision=inventory["revision"],
            )
        install = _install_mcg_layer(model, layer, path, choices, receipt)
        student = _capture(model, token_ids[args.selection_row])
        token = _token_kld(selection_teacher, student)
        mean = float(token.mean())
        accepted = mean < current_mean
        greedy = {
            "layer": layer,
            "mean_before": current_mean,
            "mean_after": mean,
            "delta": mean - current_mean,
            "accepted": accepted,
            "installed_layer_sha256": install["installed_layer_sha256"],
        }
        greedy_rows.append(greedy)
        if accepted:
            selected_layers.append(layer)
            current_token = token
            current_mean = mean
            if temporary:
                retained = retained_root / f"layer-{layer:03d}.safetensors"
                path.replace(retained)
                retained_paths[layer] = retained
        else:
            _restore_layer(model, layer, snapshot)
            if temporary:
                path.unlink()
        print(json.dumps({"stage": "greedy-factory-selection", **greedy}, sort_keys=True), flush=True)
        progress = {
            "schema": "quant-pipeline.qwen-candidate-factory-greedy-progress.v1",
            "baseline_mean_kld": baseline_mean,
            "current_mean_kld": current_mean,
            "selected_layers": selected_layers,
            "greedy_path": greedy_rows,
        }
        progress["progress_sha256"] = _hash_json(progress)
        write_json(output / "greedy-progress.json", progress)

    selection_union_sha = _save_npy(output / "selection-union.token-kld.npy", current_token)
    selection_union_student = _capture(model, token_ids[args.selection_row])
    selection_union_check = _token_kld(selection_teacher, selection_union_student)
    if not np.array_equal(selection_union_check, current_token):
        raise ValueError("selection-union recapture differs from the accepted greedy state")
    selection_union_student_sha = _save_logits(
        output / "selection-union-student-logits" / f"row-{args.selection_row:02d}.safetensors",
        selection_union_student,
        {"arm": "selection-union", "row": str(args.selection_row)},
    )
    validation_rows = plan["validation_rows"]
    union_validation, union_validation_records = _evaluate_rows(
        model,
        token_ids,
        teacher_paths,
        validation_rows,
        logit_root=output / "validation-union-student-logits",
        arm="validation-union",
    )
    if selected_layers:
        # Rebuild the exact Turbo baseline rather than trusting retained CPU
        # snapshots from an exploratory sequence.
        _install_turboderp_full(
            model, args.turboderp_k4_model.resolve(), args.exllamav3_root.resolve()
        )
        _install_turbo_selected_k3(
            model,
            args.turboderp_k3_model.resolve(),
            choices,
            args.exllamav3_root.resolve(),
        )
        baseline_validation, baseline_validation_records = _evaluate_rows(
            model,
            token_ids,
            teacher_paths,
            validation_rows,
            logit_root=output / "validation-baseline-student-logits",
            arm="validation-baseline",
        )
    else:
        baseline_validation = union_validation.copy()
        baseline_validation_records = [dict(row) for row in union_validation_records]

    validation_delta = baseline_validation - union_validation
    validation_interval = _paired_block_interval(
        validation_delta.reshape(-1),
        seed=args.seed + 10_000,
        block_tokens=64,
    )
    factory_choices = []
    selected_set = set(selected_layers)
    for row in allocation["choices"]:
        factory_choices.append({
            "layer": int(row["layer"]),
            "expert": int(row["expert"]),
            "projection": str(row["projection"]),
            "bits": int(row["bits"]),
            "factory": "mcg" if int(row["layer"]) in selected_set else "turboderp-v0.0.1",
        })
    factory_allocation = {
        "schema": "quant-pipeline.qwen-candidate-factory-union-allocation.v1",
        "bit_allocation_sha256": allocation["allocation_sha256"],
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "selection_row": args.selection_row,
        "selection_teacher_sha256": sha256_file(teacher_paths[args.selection_row]),
        "factory_granularity": "whole routed-expert layer",
        "selected_mcg_layers": sorted(selected_layers),
        "turboderp_layer_count": 48 - len(selected_layers),
        "mcg_layer_count": len(selected_layers),
        "k3_count": rate["k3_matrices"],
        "k4_count": rate["k4_matrices"],
        "average_weight_bits": 3.5,
        "choices": factory_choices,
    }
    factory_allocation["allocation_sha256"] = _hash_json(factory_allocation)
    write_json(output / "factory-allocation.json", factory_allocation)
    report = {
        "schema": "quant-pipeline.qwen-candidate-factory-union.v1",
        "factory_allocation_sha256": factory_allocation["allocation_sha256"],
        "bit_allocation_sha256": allocation["allocation_sha256"],
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "panel_sha256": panel["panel_sha256"],
        "turboderp_k3_scope": k3_scope,
        "turboderp_k4_scope": k4_scope,
        "fixed_nonexpert_scope": fixed_scope,
        "selection": {
            "row": args.selection_row,
            "baseline_mean_kld": baseline_mean,
            "baseline_student_logits_sha256": baseline_student_sha,
            "union_mean_kld": current_mean,
            "absolute_reduction": baseline_mean - current_mean,
            "relative_reduction": (baseline_mean - current_mean) / baseline_mean,
            "union_token_kld_sha256": selection_union_sha,
            "union_student_logits_sha256": selection_union_student_sha,
            "single_layer_swaps": layer_rows,
            "greedy_path": greedy_rows,
        },
        "untouched_validation": {
            "rows": validation_rows,
            "baseline_summary": summarize(baseline_validation.reshape(-1)),
            "union_summary": summarize(union_validation.reshape(-1)),
            "absolute_reduction": float(validation_delta.mean()),
            "relative_reduction": float(validation_delta.mean() / baseline_validation.mean()),
            "paired_block_95_interval_for_baseline_minus_union": validation_interval,
            "rows_union_better": int(np.count_nonzero(
                union_validation.mean(axis=1) < baseline_validation.mean(axis=1)
            )),
            "row_count": len(validation_rows),
            "baseline_records": baseline_validation_records,
            "union_records": union_validation_records,
            "baseline_token_kld_sha256": _save_npy(
                output / "validation-baseline.token-kld.npy", baseline_validation
            ),
            "union_token_kld_sha256": _save_npy(
                output / "validation-union.token-kld.npy", union_validation
            ),
        },
        "interpretation": {
            "claim": "candidate-factory union at frozen full-causal bit allocation",
            "allocator_superiority_retested": False,
            "factory_selection_is_exploratory": True,
            "primary_endpoint_uses_rows_unseen_by_factory_selection": True,
        },
    }
    report["report_sha256"] = _hash_json(report)
    write_json(output / "report.json", report)
    print(json.dumps({
        "ok": True,
        "selected_mcg_layers": selected_layers,
        "selection_mean_kld": current_mean,
        "validation_baseline_mean_kld": report["untouched_validation"]["baseline_summary"]["mean"],
        "validation_union_mean_kld": report["untouched_validation"]["union_summary"]["mean"],
        "validation_relative_reduction": report["untouched_validation"]["relative_reduction"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
