#!/usr/bin/env python3
"""Measure a model pair with ExLlamaV3 model_diff's WikiText-2 protocol.

The published EXL3 cards describe this as "wiki2 20k tokens".  The upstream
evaluator forms consecutive, non-overlapping 2,048-token rows from the raw
WikiText-2 test split.  Ten rows therefore contain 20,480 input/logit positions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.core.artifacts import (
    atomic_write,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.scoring.kld import summarize


ROWS = 10
ROW_LENGTH = 2048
EXPERT_ELEMENTS = 28_991_029_248
ATTENTION_ELEMENTS = 905_969_664
ROUTER_ELEMENTS = 12_582_912
EXPERT_PAYLOAD_BYTES = 12_787_384_320
EXL3_BODY_LINEAR_ELEMENTS = EXPERT_ELEMENTS + ATTENTION_ELEMENTS + ROUTER_ELEMENTS
TURBODERP_3BPW_KLD = 0.0688
TURBODERP_4BPW_KLD = 0.0215


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _score_pair(payload: tuple[int, str, str]) -> tuple[int, np.ndarray, int]:
    """Return float32 EXL3-equivalent KL values and exact top-1 matches."""
    import torch
    from safetensors import safe_open

    index, teacher_name, student_name = payload
    with safe_open(teacher_name, framework="pt", device="cpu") as handle:
        teacher = handle.get_tensor("logits").float()
    with safe_open(student_name, framework="pt", device="cpu") as handle:
        student = handle.get_tensor("logits").float()
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student logit geometry mismatch")
    # exllamav3.util.measures.compute_kl_div(student, teacher, vocab_size)
    # computes KL(softmax(teacher) || softmax(student)) in float32.
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    student_logp = torch.log_softmax(student, dim=-1)
    values = torch.sum(torch.exp(teacher_logp) * (teacher_logp - student_logp), dim=-1)
    top1 = int(torch.eq(teacher.argmax(-1), student.argmax(-1)).sum().item())
    return index, values.double().numpy(), top1


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
            output = model(input_ids=ids, use_cache=False, return_dict=True).logits
        logits = output.float().cpu().reshape(-1, output.shape[-1]).contiguous()
        path = root / f"row-{index:02d}.safetensors"
        save_file(
            {"logits": logits},
            path,
            metadata={"role": role, "row": str(index), "positions": str(len(values))},
        )
        paths.append(path)
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "role": role,
                    "row": index,
                    "sha256": sha256_file(path),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del ids, output, logits
    return paths


def _prepare_panel(model: Path, output: Path) -> tuple[dict, np.ndarray]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    flat = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
    required = ROWS * ROW_LENGTH
    if flat.size < required:
        raise ValueError(f"WikiText-2 test token stream has only {flat.size} tokens")
    token_ids = flat[:required].reshape(ROWS, ROW_LENGTH)
    token_buffer = io.BytesIO()
    np.savez(token_buffer, input_ids=token_ids)
    token_path = output / "input-token-ids.npz"
    atomic_write(token_path, token_buffer.getvalue())
    panel = {
        "schema": "quant-pipeline.turboderp-wiki2-panel.v1",
        "dataset": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "text_construction": "double-newline join of the text column",
        "tokenizer": str(model.resolve()),
        "tokenizer_add_special_tokens": False,
        "rows": ROWS,
        "row_length": ROW_LENGTH,
        "stride": ROW_LENGTH,
        "input_positions": required,
        "scored_logit_positions": required,
        "source_stream_tokens": int(flat.size),
        "token_file": token_path.name,
        "token_file_sha256": sha256_file(token_path),
        "token_ids_sha256": sha256_bytes(token_ids.tobytes(order="C")),
        "upstream_protocol": "turboderp-org/exllamav3 eval/model_diff.py",
        "status": "exact-upstream-construction-with-transformers-tokenizer-backend",
    }
    panel["panel_sha256"] = _hash_json(panel)
    write_json(output / "panel.json", panel)
    return panel, token_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-captures", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.turboderp-wiki2-kld-plan.v1",
        "teacher_model": str(args.teacher_model.resolve()),
        "student_model": str(args.student_model.resolve()),
        "output": str(args.output.resolve()),
        "rows": ROWS,
        "row_length": ROW_LENGTH,
        "workers": args.workers,
        "attention_backend": args.attention_backend,
        "reuse_captures": args.reuse_captures,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    import torch

    started = time.monotonic()
    output = args.output.resolve()
    if args.reuse_captures:
        panel = json.loads((output / "panel.json").read_text())
        with np.load(output / panel["token_file"]) as handle:
            token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
        teacher_paths = sorted((output / "teacher-logits").glob("row-*.safetensors"))
        student_paths = sorted((output / "student-logits").glob("row-*.safetensors"))
        if len(teacher_paths) != ROWS or len(student_paths) != ROWS:
            raise ValueError("--reuse-captures requires ten teacher and ten student rows")
    else:
        output.mkdir(parents=True, exist_ok=False)
        panel, token_ids = _prepare_panel(args.teacher_model.resolve(), output)
        teacher = _load_model(args.teacher_model.resolve(), args.attention_backend)
        teacher_paths = _capture(teacher, token_ids, output / "teacher-logits", "bf16-reference")
        del teacher
        torch.cuda.empty_cache()
        student = _load_model(args.student_model.resolve(), args.attention_backend)
        student_paths = _capture(student, token_ids, output / "student-logits", "quantized-student")
        del student
        torch.cuda.empty_cache()

    payloads = [
        (index, str(teacher), str(student))
        for index, (teacher, student) in enumerate(zip(teacher_paths, student_paths, strict=True))
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, ROWS)) as pool:
        scored = list(pool.map(_score_pair, payloads))
    scored.sort(key=lambda row: row[0])

    per_row = []
    values_by_row = []
    top1_total = 0
    for index, values, top1 in scored:
        values_by_row.append(values)
        top1_total += top1
        per_row.append(
            {
                "row": index,
                "summary": summarize(values),
                "top1_agreement": top1 / ROW_LENGTH,
                "teacher_sha256": sha256_file(teacher_paths[index]),
                "student_sha256": sha256_file(student_paths[index]),
            }
        )
    matrix = np.stack(values_by_row)
    token_path = output / "token-kld.npy"
    token_buffer = io.BytesIO()
    np.save(token_buffer, matrix, allow_pickle=False)
    atomic_write(token_path, token_buffer.getvalue())
    row_means = np.asarray([row["summary"]["mean"] for row in per_row])
    result = {
        "schema": "quant-pipeline.turboderp-wiki2-kld.v1",
        "metric": "mean tokenwise KL(bf16 reference || quantized student)",
        "numeric_protocol": "float32 softmax/log-softmax matching exllamav3 compute_kl_div",
        "panel_sha256": panel["panel_sha256"],
        "summary": summarize(matrix.reshape(-1)),
        "mean_of_row_means": float(row_means.mean()),
        "sample_std_of_row_means": float(row_means.std(ddof=1)),
        "standard_error_of_row_means": float(row_means.std(ddof=1) / np.sqrt(ROWS)),
        "top1_agreement": top1_total / (ROWS * ROW_LENGTH),
        "per_row": per_row,
        "token_kld_sha256": sha256_file(token_path),
        "rate": {
            "routed_expert_logical_bpw": 3.5,
            "routed_expert_payload_bpw": EXPERT_PAYLOAD_BYTES * 8 / EXPERT_ELEMENTS,
            "exl3_body_linear_scope": "all body Linear modules except lm_head; includes routed experts, attention projections, and routers",
            "exl3_body_linear_elements": EXL3_BODY_LINEAR_ELEMENTS,
            "exl3_body_linear_logical_bpw": (
                EXPERT_ELEMENTS * 3.5
                + (ATTENTION_ELEMENTS + ROUTER_ELEMENTS) * 16
            ) / EXL3_BODY_LINEAR_ELEMENTS,
            "exl3_body_linear_payload_bpw": (
                EXPERT_PAYLOAD_BYTES * 8
                + (ATTENTION_ELEMENTS + ROUTER_ELEMENTS) * 16
            ) / EXL3_BODY_LINEAR_ELEMENTS,
        },
        "turboderp_comparison": {
            "published_model": "turboderp/Qwen3-30B-A3B-exl3",
            "published_parent": "Qwen/Qwen3-30B-A3B (post-trained, not Base)",
            "published_3bpw_kld": TURBODERP_3BPW_KLD,
            "published_4bpw_kld": TURBODERP_4BPW_KLD,
            "same_dataset_construction": True,
            "same_tokenizer_family": True,
            "strict_head_to_head_valid": False,
            "caveat": "same evaluator corpus but a different parent checkpoint and different quantization scope/format",
        },
        "reference_dtype": "bfloat16 model execution; TurboDerp card labels its parent FP16",
        "kv_cache": "disabled (use_cache=False)",
        "elapsed_seconds": time.monotonic() - started,
    }
    result["report_sha256"] = _hash_json(result)
    write_json(output / "kld-report.json", result)
    print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
