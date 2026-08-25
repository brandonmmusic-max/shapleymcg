#!/usr/bin/env python3
"""Select between two MCG candidate factories at one frozen Qwen allocation.

Both factories must expose complete, immutable 48-layer candidate inventories
in the same runtime representation. One WikiText row selects whole-layer
factory swaps by direct end-to-end KLD; the other nine rows are untouched until
the assembled union is fixed. Bit choices, BF16 parent, non-expert weights,
teacher logits, token panel, attention backend, and KLD arithmetic are shared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from measure_qwen_candidate_factory_union import (
    _capture,
    _evaluate_rows,
    _paired_block_interval,
    _restore_layer,
    _save_logits,
    _save_npy,
    _snapshot_layer,
    _teacher,
    _token_kld,
    _verify_seal,
)
from measure_qwen_mcg_causal_allocation import _candidate_path, _install_layer
from quant_pipeline.core.artifacts import (
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.scoring.kld import summarize


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _load_inventory(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _verify_seal(value, "inventory_sha256", f"{label} candidate inventory")
    layers = value.get("layers")
    if not isinstance(layers, list) or [int(row["layer"]) for row in layers] != list(range(48)):
        raise ValueError(f"{label} candidate inventory must contain 48 ordered layers")
    return value


def _load_allocation(
    path: Path,
    inventory: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    value = json.loads(path.read_text())
    _verify_seal(value, "allocation_sha256", f"{label} allocation")
    rows = value.get("choices")
    if (
        value.get("candidate_inventory_sha256") != inventory["inventory_sha256"]
        or value.get("average_weight_bits") != 3.5
        or value.get("k3_count") != 9216
        or value.get("k4_count") != 9216
        or not isinstance(rows, list)
        or len(rows) != 18432
    ):
        raise ValueError(f"{label} allocation is not a bound exact-3.5 candidate allocation")
    choices = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): dict(row)
        for row in rows
    }
    if len(choices) != 18432:
        raise ValueError(f"{label} allocation matrix inventory is incomplete")
    return value, choices


def _candidate_for_layer(
    *,
    inventory: dict[str, Any],
    layer: int,
    local_root: Path | None,
    cache_root: Path,
) -> tuple[Path, bool]:
    row = inventory["layers"][layer]
    if int(row["layer"]) != layer:
        raise ValueError("candidate inventory layer order drifted")
    return _candidate_path(
        row=row,
        local_root=local_root,
        cache_root=cache_root,
        repo=inventory["repo_id"],
        revision=inventory["revision"],
    )


def _install_factory_layer(
    *,
    model: Any,
    inventory: dict[str, Any],
    choices: dict[tuple[int, int, str], dict[str, Any]],
    layer: int,
    local_root: Path | None,
    cache_root: Path,
) -> tuple[dict[str, Any], Path, bool]:
    path, temporary = _candidate_for_layer(
        inventory=inventory,
        layer=layer,
        local_root=local_root,
        cache_root=cache_root,
    )
    row = inventory["layers"][layer]
    installed = _install_layer(
        model=model,
        layer=layer,
        candidate_path=path,
        candidate_file_sha256=row["candidate_sha256"],
        choices=choices,
    )
    return installed, path, temporary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--baseline-allocation", type=Path, required=True)
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--baseline-local-root", type=Path)
    parser.add_argument("--baseline-label", default="native-mcg")
    parser.add_argument("--challenger-allocation", type=Path, required=True)
    parser.add_argument("--challenger-inventory", type=Path, required=True)
    parser.add_argument("--challenger-local-root", type=Path)
    parser.add_argument("--challenger-label", default="model-guided-mcg")
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--selection-row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.selection_row not in range(10):
        parser.error("--selection-row must be in [0, 9]")
    if not args.baseline_label or not args.challenger_label or args.baseline_label == args.challenger_label:
        parser.error("factory labels must be distinct and non-empty")
    plan = {
        "schema": "quant-pipeline.qwen-mcg-factory-union-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "model_revision": args.model_revision,
        "baseline_allocation": str(args.baseline_allocation.resolve()),
        "baseline_allocation_file_sha256": sha256_file(args.baseline_allocation),
        "baseline_inventory": str(args.baseline_inventory.resolve()),
        "baseline_inventory_file_sha256": sha256_file(args.baseline_inventory),
        "baseline_local_root": str(args.baseline_local_root.resolve()) if args.baseline_local_root else None,
        "baseline_label": args.baseline_label,
        "challenger_allocation": str(args.challenger_allocation.resolve()),
        "challenger_allocation_file_sha256": sha256_file(args.challenger_allocation),
        "challenger_inventory": str(args.challenger_inventory.resolve()),
        "challenger_inventory_file_sha256": sha256_file(args.challenger_inventory),
        "challenger_local_root": str(args.challenger_local_root.resolve()) if args.challenger_local_root else None,
        "challenger_label": args.challenger_label,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "teacher_receipt": str(args.teacher_receipt.resolve()),
        "teacher_receipt_file_sha256": sha256_file(args.teacher_receipt),
        "output": str(args.output.resolve()),
        "attention_backend": args.attention_backend,
        "selection_row": args.selection_row,
        "validation_rows": [row for row in range(10) if row != args.selection_row],
        "factory_granularity": "whole-routed-expert-layer",
        "fixed_rate": "exact-9216-K3-plus-9216-K4-routed-expert-matrices",
        "seed": args.seed,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    baseline_inventory = _load_inventory(args.baseline_inventory.resolve(), "baseline")
    challenger_inventory = _load_inventory(args.challenger_inventory.resolve(), "challenger")
    baseline_allocation, baseline_choices = _load_allocation(
        args.baseline_allocation.resolve(), baseline_inventory, "baseline"
    )
    challenger_allocation, challenger_choices = _load_allocation(
        args.challenger_allocation.resolve(), challenger_inventory, "challenger"
    )
    baseline_bits = {identity: int(row["bits"]) for identity, row in baseline_choices.items()}
    challenger_bits = {identity: int(row["bits"]) for identity, row in challenger_choices.items()}
    if baseline_bits != challenger_bits:
        raise ValueError("factory comparison changed one or more frozen K3/K4 choices")

    panel_root = args.panel_root.resolve()
    panel = json.loads((panel_root / "panel.json").read_text())
    _verify_seal(panel, "panel_sha256", "evaluation panel")
    token_path = panel_root / str(panel["token_file"])
    if sha256_file(token_path) != panel["token_file_sha256"]:
        raise ValueError("evaluation panel token file drifted")
    with np.load(token_path) as handle:
        token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
    if list(token_ids.shape) != [10, 2048]:
        raise ValueError("evaluation panel must be 10 x 2,048 tokens")
    teacher_receipt = json.loads(args.teacher_receipt.read_text())
    _verify_seal(teacher_receipt, "receipt_sha256", "teacher receipt")
    if (
        teacher_receipt.get("model_revision") != args.model_revision
        or teacher_receipt.get("panel_sha256") != panel["panel_sha256"]
        or teacher_receipt.get("attention_backend") != args.attention_backend
    ):
        raise ValueError("teacher receipt differs from model, panel, or attention backend")
    teacher_paths = [panel_root / str(row["path"]) for row in teacher_receipt["teacher_files"]]
    if len(teacher_paths) != 10:
        raise ValueError("teacher receipt must seal ten rows")
    for path, row in zip(teacher_paths, teacher_receipt["teacher_files"], strict=True):
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"teacher row drifted: {path.name}")

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.source_model.resolve(),
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = args.candidate_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    baseline_local = args.baseline_local_root.resolve() if args.baseline_local_root else None
    challenger_local = args.challenger_local_root.resolve() if args.challenger_local_root else None

    installed_baseline = []
    for layer in range(48):
        installed, path, temporary = _install_factory_layer(
            model=model,
            inventory=baseline_inventory,
            choices=baseline_choices,
            layer=layer,
            local_root=baseline_local,
            cache_root=cache,
        )
        installed_baseline.append(installed["installed_layer_sha256"])
        if temporary:
            path.unlink()
        print(json.dumps({"stage": "install-baseline", "layer": layer}, sort_keys=True), flush=True)

    selection = args.selection_row
    selection_teacher = _teacher(teacher_paths[selection])
    baseline_student = _capture(model, token_ids[selection])
    baseline_token = _token_kld(selection_teacher, baseline_student)
    baseline_mean = float(baseline_token.mean())
    baseline_student_sha = _save_logits(
        output / "selection-baseline-student-logits" / f"row-{selection:02d}.safetensors",
        baseline_student,
        {"arm": args.baseline_label, "row": str(selection)},
    )
    baseline_token_sha = _save_npy(output / "selection-baseline.token-kld.npy", baseline_token)
    print(json.dumps({"stage": "selection-baseline", "mean_kld": baseline_mean}, sort_keys=True), flush=True)

    layer_rows = []
    for layer in range(48):
        started = time.monotonic()
        snapshot = _snapshot_layer(model, layer)
        installed, _, _ = _install_factory_layer(
            model=model,
            inventory=challenger_inventory,
            choices=challenger_choices,
            layer=layer,
            local_root=challenger_local,
            cache_root=cache,
        )
        student = _capture(model, token_ids[selection])
        token = _token_kld(selection_teacher, student)
        delta = token - baseline_token
        interval = _paired_block_interval(delta, seed=args.seed + layer)
        row = {
            "layer": layer,
            "challenger_mean_kld": float(token.mean()),
            "delta_mean_kld_vs_baseline": float(delta.mean()),
            "relative_delta_vs_baseline": float(delta.mean() / baseline_mean),
            "paired_block_95_interval": interval,
            "single_swap_improves_mean": bool(delta.mean() < 0.0),
            "single_swap_interval_below_zero": bool(interval[1] < 0.0),
            "baseline_candidate_sha256": baseline_inventory["layers"][layer]["candidate_sha256"],
            "challenger_candidate_sha256": challenger_inventory["layers"][layer]["candidate_sha256"],
            "installed_challenger_layer_sha256": installed["installed_layer_sha256"],
            "token_kld_sha256": _save_npy(output / "single-layer-token-kld" / f"layer-{layer:03d}.npy", token),
            "token_delta_sha256": _save_npy(output / "single-layer-token-delta" / f"layer-{layer:03d}.npy", delta),
            "elapsed_seconds": time.monotonic() - started,
        }
        layer_rows.append(row)
        _restore_layer(model, layer, snapshot)
        print(json.dumps({"stage": "single-layer-swap", **row}, sort_keys=True), flush=True)

    selected_layers: list[int] = []
    baseline_snapshots: dict[int, dict[str, Any]] = {}
    greedy_rows = []
    current_token = baseline_token
    current_mean = baseline_mean
    for candidate in sorted(layer_rows, key=lambda row: (row["delta_mean_kld_vs_baseline"], row["layer"])):
        if not candidate["single_swap_improves_mean"]:
            continue
        layer = int(candidate["layer"])
        snapshot = _snapshot_layer(model, layer)
        installed, _, _ = _install_factory_layer(
            model=model,
            inventory=challenger_inventory,
            choices=challenger_choices,
            layer=layer,
            local_root=challenger_local,
            cache_root=cache,
        )
        student = _capture(model, token_ids[selection])
        token = _token_kld(selection_teacher, student)
        mean = float(token.mean())
        accepted = mean < current_mean
        greedy = {
            "layer": layer,
            "mean_before": current_mean,
            "mean_after": mean,
            "delta": mean - current_mean,
            "accepted": accepted,
            "installed_challenger_layer_sha256": installed["installed_layer_sha256"],
        }
        greedy_rows.append(greedy)
        if accepted:
            selected_layers.append(layer)
            baseline_snapshots[layer] = snapshot
            current_token = token
            current_mean = mean
        else:
            _restore_layer(model, layer, snapshot)
        print(json.dumps({"stage": "greedy-factory-selection", **greedy}, sort_keys=True), flush=True)

    selection_union_sha = _save_npy(output / "selection-union.token-kld.npy", current_token)
    selection_union_student = _capture(model, token_ids[selection])
    if not np.array_equal(_token_kld(selection_teacher, selection_union_student), current_token):
        raise ValueError("selection union recapture differs from accepted greedy state")
    selection_union_student_sha = _save_logits(
        output / "selection-union-student-logits" / f"row-{selection:02d}.safetensors",
        selection_union_student,
        {"arm": "factory-union", "row": str(selection)},
    )
    validation_rows = plan["validation_rows"]
    union_validation, union_records = _evaluate_rows(
        model,
        token_ids,
        teacher_paths,
        validation_rows,
        logit_root=output / "validation-union-student-logits",
        arm="factory-union",
    )
    for layer in selected_layers:
        _restore_layer(model, layer, baseline_snapshots[layer])
    baseline_validation, baseline_records = _evaluate_rows(
        model,
        token_ids,
        teacher_paths,
        validation_rows,
        logit_root=output / "validation-baseline-student-logits",
        arm=args.baseline_label,
    )
    validation_delta = baseline_validation - union_validation
    validation_interval = _paired_block_interval(
        validation_delta.reshape(-1),
        seed=args.seed + 10_000,
        block_tokens=64,
    )

    selected_set = set(selected_layers)
    factory_allocation = {
        "schema": "quant-pipeline.qwen-mcg-factory-union-allocation.v1",
        "baseline_allocation_sha256": baseline_allocation["allocation_sha256"],
        "challenger_allocation_sha256": challenger_allocation["allocation_sha256"],
        "baseline_candidate_inventory_sha256": baseline_inventory["inventory_sha256"],
        "challenger_candidate_inventory_sha256": challenger_inventory["inventory_sha256"],
        "selection_row": selection,
        "factory_granularity": "whole-routed-expert-layer",
        "baseline_label": args.baseline_label,
        "challenger_label": args.challenger_label,
        "selected_challenger_layers": sorted(selected_layers),
        "baseline_layer_count": 48 - len(selected_layers),
        "challenger_layer_count": len(selected_layers),
        "k3_count": 9216,
        "k4_count": 9216,
        "average_weight_bits": 3.5,
        "choices": [
            {
                "layer": identity[0],
                "expert": identity[1],
                "projection": identity[2],
                "bits": baseline_bits[identity],
                "factory": args.challenger_label if identity[0] in selected_set else args.baseline_label,
            }
            for identity in sorted(baseline_bits)
        ],
    }
    factory_allocation["allocation_sha256"] = _hash_json(factory_allocation)
    write_json(output / "factory-allocation.json", factory_allocation)
    report = {
        "schema": "quant-pipeline.qwen-mcg-factory-union.v1",
        "factory_allocation_sha256": factory_allocation["allocation_sha256"],
        "panel_sha256": panel["panel_sha256"],
        "teacher_receipt_sha256": teacher_receipt["receipt_sha256"],
        "model_revision": args.model_revision,
        "attention_backend": args.attention_backend,
        "fixed_rate": {"k3_count": 9216, "k4_count": 9216, "average_weight_bits": 3.5},
        "selection": {
            "row": selection,
            "baseline_mean_kld": baseline_mean,
            "baseline_token_kld_sha256": baseline_token_sha,
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
            "rows_union_better": int(np.count_nonzero(union_validation.mean(axis=1) < baseline_validation.mean(axis=1))),
            "row_count": len(validation_rows),
            "baseline_records": baseline_records,
            "union_records": union_records,
            "baseline_token_kld_sha256": _save_npy(output / "validation-baseline.token-kld.npy", baseline_validation),
            "union_token_kld_sha256": _save_npy(output / "validation-union.token-kld.npy", union_validation),
        },
        "claim_boundary": {
            "candidate_family_was_selected_only_on_the_selection_row": True,
            "validation_rows_were_untouched_until_the_union_was_fixed": True,
            "bit_rate_choices_changed": False,
            "joint_factory_plus_rate_allocation_claimed": False,
        },
    }
    report["report_sha256"] = _hash_json(report)
    write_json(output / "report.json", report)
    print(json.dumps({
        "ok": True,
        "selected_challenger_layers": selected_layers,
        "selection_mean_kld": current_mean,
        "validation_baseline_mean_kld": report["untouched_validation"]["baseline_summary"]["mean"],
        "validation_union_mean_kld": report["untouched_validation"]["union_summary"]["mean"],
        "validation_relative_reduction": report["untouched_validation"]["relative_reduction"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
