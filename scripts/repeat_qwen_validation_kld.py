#!/usr/bin/env python3
"""Repeat the sealed Qwen validation-model KLD capture five times."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.scoring.kld import summarize


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _token_kld(teacher: np.ndarray, student: np.ndarray, chunk: int = 8) -> np.ndarray:
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student logit geometry mismatch")
    result = np.empty(teacher.shape[0], dtype=np.float64)
    for start in range(0, teacher.shape[0], chunk):
        stop = min(start + chunk, teacher.shape[0])
        t = np.asarray(teacher[start:stop], dtype=np.float64)
        s = np.asarray(student[start:stop], dtype=np.float64)
        if not np.isfinite(t).all() or not np.isfinite(s).all():
            raise ValueError("non-finite logits")
        t -= np.max(t, axis=-1, keepdims=True)
        s -= np.max(s, axis=-1, keepdims=True)
        t -= np.logaddexp.reduce(t, axis=-1, keepdims=True)
        s -= np.logaddexp.reduce(s, axis=-1, keepdims=True)
        result[start:stop] = np.sum(np.exp(t) * (t - s), axis=-1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kld-window", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    plan = {
        "schema": "quant-pipeline.qwen-validation-kld-repeat-plan.v1",
        "model": str(args.model.resolve()),
        "kld_window": str(args.kld_window.resolve()),
        "teacher": str(args.teacher.resolve()),
        "expected_student": str(args.expected_student.resolve()),
        "output": str(args.output.resolve()),
        "runs": args.runs,
        "attention_backend": args.attention_backend,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    from quant_pipeline.normalization.absolute_v31 import tensor_sha256

    started = time.monotonic()
    window = json.loads((args.kld_window.resolve() / "kld-window.json").read_text())
    with safe_open(args.teacher.resolve(), framework="np") as handle:
        teacher_metadata = handle.metadata() or {}
        teacher = handle.get_tensor("logits")
    with safe_open(args.expected_student.resolve(), framework="pt", device="cpu") as handle:
        expected_metadata = handle.metadata() or {}
        expected = handle.get_tensor("logits").contiguous()
    if (
        teacher_metadata.get("token_sha256") != window["token_sha256"]
        or expected_metadata.get("token_sha256") != window["token_sha256"]
    ):
        raise ValueError("teacher/student token identity mismatch")
    expected_raw_sha256 = tensor_sha256(expected)

    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(),
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ids = torch.tensor(
        [window["token_ids"]],
        dtype=torch.long,
        device=model.get_input_embeddings().weight.device,
    )
    rows = []
    token_values = []
    for run in range(args.runs):
        run_started = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
        student = logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()
        exact = bool(torch.equal(student, expected))
        max_abs_delta = float((student - expected).abs().max().item())
        values = _token_kld(teacher, student.numpy())
        row = {
            "run": run,
            "student_raw_sha256": tensor_sha256(student),
            "exact_expected_student": exact,
            "max_abs_expected_student_delta": max_abs_delta,
            "summary": summarize(values),
            "elapsed_seconds": time.monotonic() - run_started,
        }
        rows.append(row)
        token_values.append(values)
        print(json.dumps({"stage": "run", **row}, sort_keys=True), flush=True)
        del logits, student
    del model
    torch.cuda.empty_cache()

    stacked = np.stack(token_values)
    token_path = args.output.resolve().with_suffix(".token-kld.npy")
    buffer = io.BytesIO()
    np.save(buffer, stacked, allow_pickle=False)
    atomic_write(token_path, buffer.getvalue())
    means = np.asarray([row["summary"]["mean"] for row in rows], dtype=np.float64)
    result = {
        "schema": "quant-pipeline.qwen-validation-kld-repeat.v1",
        "model_manifest_sha256": json.loads(
            (args.model.resolve() / "model-manifest.json").read_text()
        )["manifest_sha256"],
        "teacher_sha256": sha256_file(args.teacher.resolve()),
        "expected_student_sha256": sha256_file(args.expected_student.resolve()),
        "expected_student_raw_sha256": expected_raw_sha256,
        "token_sha256": str(window["token_sha256"]),
        "runs": rows,
        "run_count": args.runs,
        "exact_expected_student_count": sum(bool(row["exact_expected_student"]) for row in rows),
        "mean_of_run_means": float(np.mean(means)),
        "sample_std_of_run_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
        "token_kld_matrix_sha256": sha256_file(token_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    if result["exact_expected_student_count"] != args.runs:
        raise ValueError("one or more repeated captures differ from the measured student logits")
    result["report_sha256"] = _hash_json(result)
    write_json(args.output.resolve(), result)
    print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
