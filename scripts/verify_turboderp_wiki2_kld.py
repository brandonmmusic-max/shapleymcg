#!/usr/bin/env python3
"""Independently verify a sealed WikiText KLD result with torch.kl_div.

This deliberately does not import the pipeline scorer.  It recomputes
KL(teacher || student) from the stored logits using PyTorch's reference
functional implementation and checks the stored per-token values and top-1
agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: dict) -> str:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_logits(path: Path) -> np.ndarray:
    with safe_open(path, framework="np") as handle:
        return np.asarray(handle.get_tensor("logits"), dtype=np.float32)


def token_kld_float64(teacher: np.ndarray, student: np.ndarray, chunk: int = 16) -> np.ndarray:
    result = np.empty(teacher.shape[0], dtype=np.float64)
    for start in range(0, len(result), chunk):
        stop = min(start + chunk, len(result))
        target = np.asarray(teacher[start:stop], dtype=np.float64)
        observed = np.asarray(student[start:stop], dtype=np.float64)
        target -= np.max(target, axis=-1, keepdims=True)
        observed -= np.max(observed, axis=-1, keepdims=True)
        target -= np.logaddexp.reduce(target, axis=-1, keepdims=True)
        observed -= np.logaddexp.reduce(observed, axis=-1, keepdims=True)
        result[start:stop] = np.sum(np.exp(target) * (target - observed), axis=-1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    root = args.result.resolve()
    report = json.loads((root / "kld-report.json").read_text())
    stored = np.load(root / "token-kld.npy").astype(np.float64, copy=False).reshape(-1)
    teacher_root = args.teacher_root.resolve() if args.teacher_root else root / "teacher-logits"
    teacher_paths = sorted(teacher_root.glob("row-*.safetensors"))
    student_paths = sorted((root / "student-logits").glob("row-*.safetensors"))
    if len(teacher_paths) != len(student_paths) or not teacher_paths:
        raise ValueError("teacher/student capture count mismatch")
    if "report_sha256" in report and report["report_sha256"] != hash_json(
        {key: value for key, value in report.items() if key != "report_sha256"}
    ):
        raise ValueError("KLD report seal mismatch")
    if "teacher_files" in report:
        expected = [row["sha256"] for row in report["teacher_files"]]
        observed = [sha256_file(path) for path in teacher_paths]
        if expected != observed:
            raise ValueError("teacher file inventory drifted")
    if "student_files" in report:
        expected = [row["sha256"] for row in report["student_files"]]
        observed = [sha256_file(path) for path in student_paths]
        if expected != observed:
            raise ValueError("student file inventory drifted")

    values: list[np.ndarray] = []
    torch_values: list[np.ndarray] = []
    agreements = 0
    count = 0
    for teacher_path, student_path in zip(teacher_paths, student_paths, strict=True):
        teacher = load_logits(teacher_path)
        student = load_logits(student_path)
        if teacher.shape != student.shape or teacher.ndim != 2:
            raise ValueError(f"logit geometry mismatch: {teacher_path.name}")
        per_token = token_kld_float64(teacher, student)
        values.append(per_token)
        # Retain the float32 PyTorch/ExLlama-style value as a secondary
        # numerical cross-check, but do not compare its per-token rounding
        # directly against the stored float64 producer vector.
        teacher_torch = torch.from_numpy(teacher)
        student_torch = torch.from_numpy(student)
        torch_per_token = F.kl_div(
            F.log_softmax(student_torch, dim=-1),
            F.softmax(teacher_torch, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        torch_values.append(torch_per_token.numpy())
        agreements += int(np.count_nonzero(teacher.argmax(-1) == student.argmax(-1)))
        count += int(teacher.shape[0])

    independent = np.concatenate(values).astype(np.float64, copy=False)
    torch_reference = np.concatenate(torch_values).astype(np.float32, copy=False)
    if independent.shape != stored.shape:
        raise ValueError(f"token array shape mismatch: {independent.shape} != {stored.shape}")
    delta = np.abs(independent - stored)
    observed_mean = float(independent.mean())
    reported_mean = float(report["summary"]["mean"])
    observed_top1 = agreements / count
    reported_top1 = float(report["top1_agreement"])
    result = {
        "schema": "quant-pipeline.turboderp-wiki2-kld-independent-verification.v1",
        "ok": bool(
            float(delta.max(initial=0.0)) <= args.atol
            and abs(observed_mean - reported_mean) <= args.atol
            and abs(observed_top1 - reported_top1) <= 1e-12
        ),
        "direction": "KL(bf16 teacher || quantized student)",
        "implementation": "independent NumPy float64 log-softmax and KL reduction",
        "secondary_float32_implementation": (
            "torch.nn.functional.kl_div(log_softmax(student), softmax(teacher))"
        ),
        "positions": count,
        "stored_token_kld_sha256": sha256_file(root / "token-kld.npy"),
        "independent_mean": observed_mean,
        "reported_mean": reported_mean,
        "secondary_float32_mean": float(torch_reference.astype(np.float64).mean()),
        "secondary_float32_mean_delta": float(
            abs(torch_reference.astype(np.float64).mean() - reported_mean)
        ),
        "mean_absolute_delta": float(delta.mean()),
        "max_absolute_delta": float(delta.max(initial=0.0)),
        "independent_top1_agreement": observed_top1,
        "reported_top1_agreement": reported_top1,
        "atol": args.atol,
        "attention_backend": report.get("attention_backend"),
        "panel_sha256": report.get("panel_sha256"),
        "report_sha256": report.get("report_sha256"),
        "report_file_sha256": sha256_file(root / "kld-report.json"),
        "teacher_files": [sha256_file(path) for path in teacher_paths],
        "student_files": [sha256_file(path) for path in student_paths],
    }
    result["verification_sha256"] = hash_json(result)
    destination = root / "independent-verification.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
