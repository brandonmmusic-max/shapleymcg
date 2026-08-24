#!/usr/bin/env python3
"""Independently replay and seal a causal MCG KLD report from saved logits."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from quant_pipeline.core.artifacts import (
    atomic_write,
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    observed = _hash_json({key: value for key, value in document.items() if key != field})
    if document.get(field) != observed:
        raise ValueError(f"{label} seal mismatch")


def _logits(path: Path) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        if "logits" not in handle.keys():
            raise ValueError(f"{path} lacks logits")
        return np.asarray(handle.get_tensor("logits"), dtype=np.float64)


def _independent_token_kld(teacher: np.ndarray, student: np.ndarray, chunk: int = 8) -> np.ndarray:
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student logit geometry mismatch")
    result = np.empty(teacher.shape[0], dtype=np.float64)
    for start in range(0, len(result), chunk):
        stop = min(start + chunk, len(result))
        target = teacher[start:stop]
        observed = student[start:stop]
        target_shift = target - np.max(target, axis=1, keepdims=True)
        observed_shift = observed - np.max(observed, axis=1, keepdims=True)
        target_partition = np.sum(np.exp(target_shift), axis=1, keepdims=True)
        observed_partition = np.sum(np.exp(observed_shift), axis=1, keepdims=True)
        target_probability = np.exp(target_shift) / target_partition
        target_log_probability = target_shift - np.log(target_partition)
        observed_log_probability = observed_shift - np.log(observed_partition)
        result[start:stop] = np.sum(
            target_probability * (target_log_probability - observed_log_probability),
            axis=1,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-mcg-causal-kld-verification-plan.v1",
        "report": str(args.report.resolve()),
        "report_file_sha256": sha256_file(args.report),
        "teacher": str(args.teacher.resolve()),
        "teacher_file_sha256": sha256_file(args.teacher),
        "student": str(args.student.resolve()),
        "student_file_sha256": sha256_file(args.student),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    report = json.loads(args.report.read_text())
    _verify_seal(report, "report_sha256", "causal KLD report")
    if report["teacher_sha256"] != plan["teacher_file_sha256"]:
        raise ValueError("report teacher identity mismatch")
    if report["student_sha256"] != plan["student_file_sha256"]:
        raise ValueError("report student identity mismatch")
    if _hash_json(report["installed_layers"]) != report["installed_prefix_sha256"]:
        raise ValueError("installed-layer prefix seal mismatch")
    for row in report["installed_layers"]:
        _verify_seal(row, "installed_layer_sha256", f"installed layer {row['layer']}")
    root = args.report.resolve().parent
    for row in report["reanchors"]:
        _verify_seal(row, "reanchor_sha256", f"reanchor layer {row['installed_through_layer']}")
        token_path = root / row["token_kld_file"]
        if sha256_file(token_path) != row["token_kld_sha256"]:
            raise ValueError("reanchor token-KLD identity mismatch")
    teacher = _logits(args.teacher.resolve())
    student = _logits(args.student.resolve())
    values = _independent_token_kld(teacher, student)
    producer_values = np.load(root / report["reanchors"][-1]["token_kld_file"], allow_pickle=False)
    max_difference = float(np.max(np.abs(values - producer_values)))
    if max_difference > 1e-12:
        raise RuntimeError("independent token-KLD replay differs from producer")
    mean_kld = float(np.mean(values))
    if abs(mean_kld - float(report["summary"]["mean"])) > 1e-14:
        raise RuntimeError("independent mean KLD differs from report")
    top1 = float(np.mean(np.argmax(teacher, axis=1) == np.argmax(student, axis=1)))
    if abs(top1 - float(report["top1_agreement"])) > 1e-15:
        raise RuntimeError("independent top-1 agreement differs from report")
    output = prepare_empty_destination(args.output.resolve())
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    token_path = output / "independent.token-kld.npy"
    atomic_write(token_path, buffer.getvalue())
    verification = {
        "schema": "quant-pipeline.qwen-mcg-causal-kld-independent-verification.v1",
        "report_sha256": report["report_sha256"],
        "report_file_sha256": plan["report_file_sha256"],
        "teacher_sha256": plan["teacher_file_sha256"],
        "student_sha256": plan["student_file_sha256"],
        "independent_token_kld_sha256": sha256_file(token_path),
        "producer_token_kld_sha256": report["final_token_kld_sha256"],
        "max_token_kld_difference": max_difference,
        "mean_kld": mean_kld,
        "top1_agreement": top1,
        "installed_layer_count": len(report["installed_layers"]),
        "reanchor_count": len(report["reanchors"]),
    }
    verification["verification_sha256"] = _hash_json(verification)
    write_json(output / "plan.json", plan | {"dry_run": False})
    write_json(output / "verification.json", verification)
    print(json.dumps({"ok": True, **verification}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
