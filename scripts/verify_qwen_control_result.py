#!/usr/bin/env python3
"""Independently verify selected Qwen reconstructions and exact token KLD."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.normalization.absolute_v31 import tensor_sha256
from quant_pipeline.scoring.kld import summarize


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict, field: str, label: str) -> None:
    expected = document.get(field)
    body = {key: value for key, value in document.items() if key != field}
    if expected != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def _independent_kld(teacher: np.ndarray, student: np.ndarray, chunk: int = 8) -> np.ndarray:
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student logit geometry mismatch")
    result = np.empty(teacher.shape[0], dtype=np.float64)
    for start in range(0, teacher.shape[0], chunk):
        stop = min(start + chunk, teacher.shape[0])
        t = np.asarray(teacher[start:stop], dtype=np.float64)
        s = np.asarray(student[start:stop], dtype=np.float64)
        if not np.isfinite(t).all() or not np.isfinite(s).all():
            raise ValueError("non-finite teacher/student logits")
        t_max = np.max(t, axis=-1, keepdims=True)
        s_max = np.max(s, axis=-1, keepdims=True)
        t_shift = t - t_max
        s_shift = s - s_max
        t_log_z = np.logaddexp.reduce(t_shift, axis=-1, keepdims=True)
        s_log_z = np.logaddexp.reduce(s_shift, axis=-1, keepdims=True)
        t_log_p = t_shift - t_log_z
        s_log_p = s_shift - s_log_z
        result[start:stop] = np.sum(np.exp(t_log_p) * (t_log_p - s_log_p), axis=-1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--kld-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--kld-window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-control-independent-verification-plan.v1",
        "encode_root": str(args.encode_root.resolve()),
        "kld_root": str(args.kld_root.resolve()),
        "teacher": str(args.teacher.resolve()),
        "kld_window": str(args.kld_window.resolve()),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch
    from safetensors import safe_open

    started = time.monotonic()
    encode_root = args.encode_root.resolve()
    kld_root = args.kld_root.resolve()
    allocation_path = kld_root / "allocation.json"
    report_path = kld_root / "kld-report.json"
    student_path = kld_root / "student-logits.safetensors"
    values_path = kld_root / "token-kld.npy"
    allocation = json.loads(allocation_path.read_text())
    report = json.loads(report_path.read_text())
    _verify_seal(allocation, "allocation_sha256", "allocation")
    _verify_seal(report, "report_sha256", "KLD report")
    choices = list(allocation["choices"])
    if (
        len(choices) != 48 * 128 * 3
        or allocation["k3_count"] != len(choices) // 2
        or allocation["k4_count"] != len(choices) // 2
        or allocation["average_weight_bits"] != 3.5
    ):
        raise ValueError("selected allocation is not the complete exact-half-K4 control")

    by_layer: dict[int, list[dict]] = {}
    for choice in choices:
        by_layer.setdefault(int(choice["layer"]), []).append(choice)
    checked = 0
    for layer in range(48):
        candidate_path = encode_root / f"layer-{layer:03d}" / "k34-candidates.safetensors"
        receipt = json.loads(
            (encode_root / f"layer-{layer:03d}" / "encode-receipt.json").read_text()
        )
        _verify_seal(receipt, "receipt_sha256", f"encode layer {layer}")
        if sha256_file(candidate_path) != receipt["candidate_tensor_sha256"]:
            raise ValueError(f"candidate file hash mismatch at layer {layer}")
        with safe_open(candidate_path, framework="pt", device="cpu") as handle:
            for choice in by_layer.get(layer, []):
                key = (
                    f"K{int(choice['bits'])}.E{int(choice['expert']):03d}."
                    f"{choice['projection']}.reconstruction_hf"
                )
                tensor = handle.get_tensor(key).contiguous()
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"selected reconstruction dtype mismatch: {key}")
                if tensor_sha256(tensor) != choice["stored_bf16_reconstruction_sha256"]:
                    raise ValueError(f"selected reconstruction hash mismatch: {key}")
                checked += 1
        print(json.dumps({"stage": "selected-tensors", "layer": layer, "checked": checked}), flush=True)
    if checked != len(choices):
        raise ValueError("selected reconstruction inventory is incomplete")

    window = json.loads((args.kld_window.resolve() / "kld-window.json").read_text())
    with safe_open(args.teacher.resolve(), framework="np") as teacher_handle:
        teacher_metadata = teacher_handle.metadata() or {}
        teacher = teacher_handle.get_tensor("logits")
    with safe_open(student_path, framework="np") as student_handle:
        student_metadata = student_handle.metadata() or {}
        student = student_handle.get_tensor("logits")
    token_sha256 = str(window["token_sha256"])
    if (
        teacher_metadata.get("token_sha256") != token_sha256
        or student_metadata.get("token_sha256") != token_sha256
    ):
        raise ValueError("teacher/student token identity mismatch")
    independent = _independent_kld(teacher, student)
    stored = np.load(values_path, allow_pickle=False)
    max_abs_delta = float(np.max(np.abs(independent - stored)))
    if not np.allclose(independent, stored, rtol=1e-11, atol=1e-12):
        raise ValueError(f"independent token KLD mismatch; max absolute delta {max_abs_delta}")
    independent_summary = summarize(independent)
    for key, value in independent_summary.items():
        if isinstance(value, int):
            if report["summary"][key] != value:
                raise ValueError(f"KLD summary mismatch: {key}")
        elif not np.isclose(report["summary"][key], value, rtol=1e-11, atol=1e-12):
            raise ValueError(f"KLD summary mismatch: {key}")
    buffer = io.BytesIO()
    np.save(buffer, independent, allow_pickle=False)
    independent_path = args.output.resolve().with_suffix(".token-kld.npy")
    atomic_write(independent_path, buffer.getvalue())
    result = {
        "schema": "quant-pipeline.qwen-control-independent-verification.v1",
        "allocation_sha256": allocation["allocation_sha256"],
        "kld_report_sha256": report["report_sha256"],
        "teacher_sha256": sha256_file(args.teacher.resolve()),
        "student_sha256": sha256_file(student_path),
        "token_sha256": token_sha256,
        "selected_reconstruction_count": checked,
        "selected_reconstruction_verification": "all selected BF16 tensor payload SHA256",
        "candidate_layer_count": len(by_layer),
        "independent_token_kld_sha256": sha256_file(independent_path),
        "stored_token_kld_sha256": sha256_file(values_path),
        "max_abs_token_kld_delta": max_abs_delta,
        "summary": independent_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    result["verification_sha256"] = _hash_json(result)
    write_json(args.output.resolve(), result)
    print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
