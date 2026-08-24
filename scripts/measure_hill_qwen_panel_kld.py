#!/usr/bin/env python3
"""Measure BF16-to-student KL on the sealed Hill Qwen panel reconstruction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.scoring.kld import summarize


PAPER_AS_ADD_KLD_4P2 = 0.0353
PAPER_MODELOPT_KLD_4P2 = 0.0429
EXPERT_ELEMENTS = 28_991_029_248
ATTENTION_ELEMENTS = 905_969_664
PAPER_ALLOCATION_ELEMENTS = EXPERT_ELEMENTS + ATTENTION_ELEMENTS
NONEXPERT_ALL_ELEMENTS = 1_541_093_376
TOTAL_MODEL_ELEMENTS = 30_532_122_624
EXPERT_PAYLOAD_BYTES = 12_787_384_320


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


def _score_pair(payload: tuple[int, str, str]) -> tuple[int, np.ndarray]:
    from safetensors import safe_open

    index, teacher_name, student_name = payload
    with safe_open(teacher_name, framework="np") as handle:
        teacher_logits = handle.get_tensor("logits")
    with safe_open(student_name, framework="np") as handle:
        student_logits = handle.get_tensor("logits")
    return index, _token_kld(teacher_logits, student_logits)


def _load_model(path: Path, attention_backend: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _capture(model, token_ids: np.ndarray, root: Path, role: str) -> list[Path]:
    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=False)
    device = model.get_input_embeddings().weight.device
    paths = []
    for index, values in enumerate(token_ids):
        ids = torch.from_numpy(values.astype(np.int64, copy=False)).unsqueeze(0).to(device)
        started = time.monotonic()
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
        logits = output.float().cpu().reshape(-1, output.shape[-1]).contiguous()
        path = root / f"sequence-{index:02d}.safetensors"
        save_file(
            {"logits": logits},
            path,
            metadata={"role": role, "sequence": str(index)},
        )
        paths.append(path)
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "role": role,
                    "sequence": index,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del ids, output, logits
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--reuse-captures", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.hill-qwen-panel-kld-plan.v1",
        "teacher_model": str(args.teacher_model.resolve()),
        "student_model": str(args.student_model.resolve()),
        "panel": str(args.panel.resolve()),
        "output": str(args.output.resolve()),
        "attention_backend": args.attention_backend,
        "reuse_captures": args.reuse_captures,
        "workers": args.workers,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    started = time.monotonic()
    panel_root = args.panel.resolve()
    panel = json.loads((panel_root / "panel.json").read_text())
    if panel.get("panel_sha256") != _hash_json(
        {key: value for key, value in panel.items() if key != "panel_sha256"}
    ):
        raise ValueError("panel seal mismatch")
    token_path = panel_root / panel["token_file"]
    if sha256_file(token_path) != panel["token_file_sha256"]:
        raise ValueError("panel token file drifted")
    with np.load(token_path) as handle:
        token_ids = np.asarray(handle["evaluation"], dtype=np.int32)
    if list(token_ids.shape) != [16, 2048]:
        raise ValueError("paper evaluation panel must be 16x2048")

    output = args.output.resolve()
    if args.reuse_captures:
        teacher_paths = sorted((output / "teacher-logits").glob("sequence-*.safetensors"))
        student_paths = sorted((output / "student-logits").glob("sequence-*.safetensors"))
        if len(teacher_paths) != 16 or len(student_paths) != 16:
            raise ValueError("--reuse-captures requires exactly 16 teacher and 16 student files")
    else:
        import torch

        output.mkdir(parents=True, exist_ok=False)
        teacher = _load_model(args.teacher_model.resolve(), args.attention_backend)
        teacher_paths = _capture(teacher, token_ids, output / "teacher-logits", "bf16-teacher")
        del teacher
        torch.cuda.empty_cache()

        student = _load_model(args.student_model.resolve(), args.attention_backend)
        student_paths = _capture(student, token_ids, output / "student-logits", "quantized-student")
        del student
        torch.cuda.empty_cache()

    payloads = [
        (index, str(teacher_path), str(student_path))
        for index, (teacher_path, student_path) in enumerate(
            zip(teacher_paths, student_paths, strict=True)
        )
    ]
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.workers == 1:
        scored = [_score_pair(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, 16)) as pool:
            scored = list(pool.map(_score_pair, payloads))
    scored.sort(key=lambda row: row[0])

    per_sequence = []
    token_rows = []
    for index, values in scored:
        teacher_path = teacher_paths[index]
        student_path = student_paths[index]
        token_rows.append(values)
        per_sequence.append(
            {
                "sequence": index,
                "source": "bfcl" if index < 8 else "ruler",
                "teacher_sha256": sha256_file(teacher_path),
                "student_sha256": sha256_file(student_path),
                "summary": summarize(values),
            }
        )

    matrix = np.stack(token_rows)
    token_kld_path = output / "token-kld.npy"
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    atomic_write(token_kld_path, buffer.getvalue())
    sequence_means = np.asarray([row["summary"]["mean"] for row in per_sequence])
    overall = summarize(matrix.reshape(-1))
    paper_scope_logical_bpw = (
        EXPERT_ELEMENTS * 3.5 + ATTENTION_ELEMENTS * 16
    ) / PAPER_ALLOCATION_ELEMENTS
    paper_scope_payload_bpw = (
        EXPERT_PAYLOAD_BYTES * 8 + ATTENTION_ELEMENTS * 16
    ) / PAPER_ALLOCATION_ELEMENTS
    whole_model_payload_bpw = (
        EXPERT_PAYLOAD_BYTES * 8 + NONEXPERT_ALL_ELEMENTS * 16
    ) / TOTAL_MODEL_ELEMENTS
    result = {
        "schema": "quant-pipeline.hill-qwen-panel-kld.v1",
        "panel_sha256": panel["panel_sha256"],
        "status": panel["status"],
        "metric": "mean tokenwise KL(bf16 || quantized) across 16 equal-length evaluation sequences",
        "per_sequence": per_sequence,
        "summary": overall,
        "mean_of_sequence_means": float(np.mean(sequence_means)),
        "sample_std_of_sequence_means": float(np.std(sequence_means, ddof=1)),
        "standard_error_of_sequence_means": float(np.std(sequence_means, ddof=1) / np.sqrt(16)),
        "token_kld_sha256": sha256_file(token_kld_path),
        "rate": {
            "routed_expert_logical_bpw": 3.5,
            "routed_expert_payload_bpw": EXPERT_PAYLOAD_BYTES * 8 / EXPERT_ELEMENTS,
            "paper_allocation_scope": "48 layers x (qkv, o, routed gate, routed up, routed down) = 240 linear units; embeddings, router, and norms excluded",
            "paper_allocation_scope_elements": PAPER_ALLOCATION_ELEMENTS,
            "paper_scope_logical_effective_bpw": paper_scope_logical_bpw,
            "paper_scope_payload_effective_bpw": paper_scope_payload_bpw,
            "attention_assigned_bits": 16,
            "whole_model_payload_effective_bpw_including_excluded_bf16_parameters": whole_model_payload_bpw,
        },
        "paper_comparison": {
            "closest_same_corpus_budget": "Qwen3-30B Table 2 effective 4.2 bits",
            "paper_as_add_kld": PAPER_AS_ADD_KLD_4P2,
            "paper_modelopt_kld": PAPER_MODELOPT_KLD_4P2,
            "lower_than_paper_as_add": bool(overall["mean"] < PAPER_AS_ADD_KLD_4P2),
            "lower_than_paper_modelopt": bool(overall["mean"] < PAPER_MODELOPT_KLD_4P2),
            "relative_change_vs_as_add": float(overall["mean"] / PAPER_AS_ADD_KLD_4P2 - 1),
            "relative_change_vs_modelopt": float(overall["mean"] / PAPER_MODELOPT_KLD_4P2 - 1),
            "format_caveat": "ours is mixed K3/K4 expert-weight-only with BF16 activations; paper is W4A4 NVFP4 over {16,8,4} units",
            "model_caveat": "ours quantizes Qwen/Qwen3-30B-A3B-Base; the paper names Qwen3-30B-A3B and reports a 73.3 AIME bf16 anchor, indicating the post-trained hybrid checkpoint",
            "panel_caveat": "paper source categories and dimensions are matched, but its exact row IDs and token IDs were not published",
            "strict_head_to_head_valid": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result["report_sha256"] = _hash_json(result)
    write_json(output / "kld-report.json", result)
    print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
