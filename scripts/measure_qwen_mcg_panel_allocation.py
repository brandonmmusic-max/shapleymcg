#!/usr/bin/env python3
"""Measure a sealed exact-3.5 MCG allocation on a sealed 10x2048 panel.

The BF16 panel teacher must already have been captured with the same attention
backend requested here. Candidate tensors are verified and installed exactly as
in ``measure_qwen_mcg_causal_allocation.py``; the only difference is that this
runner captures and scores every row in the broader WikiText panel after the
complete allocation has been installed.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from measure_qwen_mcg_causal_allocation import (
    MODEL_REVISION,
    _candidate_path,
    _hash_json,
    _install_layer,
    _token_kld,
    _verify_seal,
)
from quant_pipeline.core.artifacts import (
    atomic_write,
    prepare_empty_destination,
    sha256_file,
    write_json,
)
from quant_pipeline.scoring.kld import summarize


ROWS = 10
ROW_LENGTH = 2048


def _save_npy(path: Path, value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_write(path, buffer.getvalue())
    return sha256_file(path)


def _load_panel(panel_root: Path) -> tuple[dict[str, Any], np.ndarray]:
    panel = json.loads((panel_root / "panel.json").read_text())
    _verify_seal(panel, "panel_sha256", "WikiText comparison panel")
    token_path = panel_root / str(panel["token_file"])
    if sha256_file(token_path) != panel["token_file_sha256"]:
        raise ValueError("WikiText comparison panel token file drifted")
    with np.load(token_path) as handle:
        token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
    if list(token_ids.shape) != [ROWS, ROW_LENGTH]:
        raise ValueError("comparison panel is not 10x2048")
    return panel, token_ids


def _capture_row(model: Any, token_ids: np.ndarray) -> Any:
    import torch

    device = model.get_input_embeddings().weight.device
    ids = torch.from_numpy(token_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
    return logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--local-encode-root", type=Path)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-mcg-panel-allocation-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "model_revision": args.model_revision,
        "allocation": str(args.allocation.resolve()),
        "allocation_file_sha256": sha256_file(args.allocation),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "local_encode_root": str(args.local_encode_root.resolve()) if args.local_encode_root else None,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "teacher_root": str(args.teacher_root.resolve()),
        "teacher_receipt": str(args.teacher_receipt.resolve()),
        "teacher_receipt_file_sha256": sha256_file(args.teacher_receipt),
        "output": str(args.output.resolve()),
        "device_map": args.device_map,
        "attention_backend": args.attention_backend,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

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
    panel_root = args.panel_root.resolve()
    panel, token_ids = _load_panel(panel_root)
    teacher_paths = sorted(args.teacher_root.resolve().glob("row-*.safetensors"))
    if len(teacher_paths) != ROWS:
        raise ValueError("panel teacher root must contain ten row-*.safetensors files")
    teacher_receipt = json.loads(args.teacher_receipt.read_text())
    _verify_seal(teacher_receipt, "receipt_sha256", "panel teacher receipt")
    if teacher_receipt.get("model_revision") != args.model_revision:
        raise ValueError("panel teacher model revision mismatch")
    if teacher_receipt.get("panel_sha256") != panel["panel_sha256"]:
        raise ValueError("panel teacher belongs to a different token panel")
    if teacher_receipt.get("attention_backend") != args.attention_backend:
        raise ValueError("teacher and student attention backends differ")
    receipt_teacher_files = teacher_receipt.get("teacher_files", [])
    if len(receipt_teacher_files) != ROWS:
        raise ValueError("panel teacher receipt does not seal ten rows")
    for path, row in zip(teacher_paths, receipt_teacher_files, strict=True):
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"panel teacher row drifted: {path.name}")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
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
    started = time.monotonic()
    for row in inventory["layers"]:
        layer = int(row["layer"])
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
        }, sort_keys=True), flush=True)

    student_root = output / "student-logits"
    student_root.mkdir()
    per_row = []
    all_values = []
    total_top1 = 0
    teacher_files = []
    student_files = []
    for index, (ids, teacher_path) in enumerate(zip(token_ids, teacher_paths, strict=True)):
        student = _capture_row(model, ids)
        with safe_open(teacher_path, framework="np") as handle:
            teacher = np.asarray(handle.get_tensor("logits"), dtype=np.float32)
        values = _token_kld(teacher, student.numpy())
        top1 = int(np.count_nonzero(np.argmax(teacher, axis=-1) == np.argmax(student.numpy(), axis=-1)))
        path = student_root / f"row-{index:02d}.safetensors"
        save_file({"logits": student}, path, metadata={
            "role": "kld-student",
            "row": str(index),
            "allocation_sha256": allocation["allocation_sha256"],
            "panel_sha256": panel["panel_sha256"],
            "attention_backend": args.attention_backend,
        })
        teacher_files.append({"path": str(teacher_path), "sha256": sha256_file(teacher_path)})
        student_files.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
        per_row.append({
            "row": index,
            "summary": summarize(values),
            "top1_agreement": top1 / ROW_LENGTH,
            "teacher_sha256": teacher_files[-1]["sha256"],
            "student_sha256": student_files[-1]["sha256"],
        })
        all_values.append(values)
        total_top1 += top1
        print(json.dumps({"stage": "capture", **per_row[-1]}, sort_keys=True), flush=True)

    combined = np.concatenate(all_values)
    token_kld_sha256 = _save_npy(output / "token-kld.npy", combined)
    report = {
        "schema": "quant-pipeline.qwen-mcg-panel-allocation.v1",
        "model_revision": args.model_revision,
        "attention_backend": args.attention_backend,
        "panel_sha256": panel["panel_sha256"],
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "allocation_sha256": allocation["allocation_sha256"],
        "installed_prefix_sha256": _hash_json(installed),
        "installed_layers": installed,
        "teacher_files": teacher_files,
        "student_files": student_files,
        "token_kld_sha256": token_kld_sha256,
        "summary": summarize(combined),
        "mean_of_row_means": float(np.mean([row["summary"]["mean"] for row in per_row])),
        "top1_agreement": total_top1 / (ROWS * ROW_LENGTH),
        "per_row": per_row,
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = _hash_json(report)
    write_json(output / "kld-report.json", report)
    print(json.dumps({
        "ok": True,
        "report_sha256": report["report_sha256"],
        "mean_kld": report["summary"]["mean"],
        "top1_agreement": report["top1_agreement"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
