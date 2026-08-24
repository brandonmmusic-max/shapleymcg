#!/usr/bin/env python3
"""Measure same-parent uniform expert-K3/K4 controls on the WikiText panel."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
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


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _score_pair(payload: tuple[int, str, str]) -> tuple[int, np.ndarray, int]:
    import torch
    from safetensors import safe_open

    index, teacher_name, student_name = payload
    with safe_open(teacher_name, framework="pt", device="cpu") as handle:
        teacher = handle.get_tensor("logits").float()
    with safe_open(student_name, framework="pt", device="cpu") as handle:
        student = handle.get_tensor("logits").float()
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    student_logp = torch.log_softmax(student, dim=-1)
    values = torch.sum(torch.exp(teacher_logp) * (teacher_logp - student_logp), dim=-1)
    top1 = int(torch.eq(teacher.argmax(-1), student.argmax(-1)).sum().item())
    return index, values.double().numpy(), top1


def _verify_candidate_inventory(root: Path) -> list[dict]:
    receipts = []
    for layer in range(48):
        layer_root = root / f"layer-{layer:03d}"
        receipt_path = layer_root / "encode-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        seal = receipt.get("receipt_sha256")
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if seal != _hash_json(body):
            raise ValueError(f"layer {layer} encode receipt seal mismatch")
        if receipt.get("layer") != layer or receipt.get("experts") != list(range(128)):
            raise ValueError(f"layer {layer} candidate inventory is incomplete")
        tensor_path = layer_root / str(receipt["candidate_tensor_file"])
        if tensor_path.stat().st_size != int(receipt["candidate_tensor_bytes"]):
            raise ValueError(f"layer {layer} candidate file size drifted")
        receipts.append(
            {
                "layer": layer,
                "receipt_sha256": seal,
                "candidate_tensor_sha256": receipt["candidate_tensor_sha256"],
                "candidate_tensor_bytes": receipt["candidate_tensor_bytes"],
            }
        )
    return receipts


def _install_uniform(model, encode_root: Path, bit: int) -> None:
    import torch
    from safetensors import safe_open

    blocks = qwen_moe_layers(model)
    for layer in range(48):
        path = encode_root / f"layer-{layer:03d}" / "k34-candidates.safetensors"
        gate_up = []
        down = []
        with safe_open(path, framework="pt", device="cpu") as handle:
            for expert in range(128):
                gate = handle.get_tensor(
                    f"K{bit}.E{expert:03d}.gate_proj.reconstruction_hf"
                )
                up = handle.get_tensor(
                    f"K{bit}.E{expert:03d}.up_proj.reconstruction_hf"
                )
                gate_up.append(torch.cat((gate, up), dim=0))
                down.append(
                    handle.get_tensor(
                        f"K{bit}.E{expert:03d}.down_proj.reconstruction_hf"
                    )
                )
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
        del gate_up, down, gate_up_tensor, down_tensor
        print(json.dumps({"stage": "install", "bit": bit, "layer": layer}), flush=True)


def _capture(model, token_ids: np.ndarray, output: Path, bit: int) -> list[Path]:
    import torch
    from safetensors.torch import save_file

    root = output / f"uniform-k{bit}" / "student-logits"
    root.mkdir(parents=True, exist_ok=False)
    device = model.get_input_embeddings().weight.device
    paths = []
    for index, values in enumerate(token_ids):
        ids = torch.from_numpy(values.astype(np.int64, copy=False)).unsqueeze(0).to(device)
        started = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        logits = logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()
        path = root / f"row-{index:02d}.safetensors"
        save_file(
            {"logits": logits},
            path,
            metadata={"role": f"uniform-expert-k{bit}-student", "row": str(index)},
        )
        paths.append(path)
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "bit": bit,
                    "row": index,
                    "sha256": sha256_file(path),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del ids, logits
    return paths


def _score(
    teacher_paths: list[Path],
    student_paths: list[Path],
    output: Path,
    bit: int,
    workers: int,
) -> dict:
    payloads = [
        (index, str(teacher), str(student))
        for index, (teacher, student) in enumerate(
            zip(teacher_paths, student_paths, strict=True)
        )
    ]
    # The model is already initialized on CUDA at this point.  Forking after
    # CUDA/PyTorch initialization can deadlock worker threads, so require fresh
    # interpreter processes for the CPU-only scorer.
    with ProcessPoolExecutor(
        max_workers=min(workers, ROWS),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        scored = list(pool.map(_score_pair, payloads))
    scored.sort(key=lambda row: row[0])
    values_by_row = []
    per_row = []
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
    token_path = output / f"uniform-k{bit}" / "token-kld.npy"
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    atomic_write(token_path, buffer.getvalue())
    row_means = np.asarray([row["summary"]["mean"] for row in per_row])
    result = {
        "schema": "quant-pipeline.qwen-uniform-expert-control.v1",
        "expert_logical_bpw": float(bit),
        "expert_elements": EXPERT_ELEMENTS,
        "attention_precision": "source BF16",
        "router_precision": "source BF16",
        "lm_head_precision": "source BF16",
        "kv_cache": "disabled (use_cache=False)",
        "metric": "float32 mean tokenwise KL(Base BF16 || reconstructed student)",
        "summary": summarize(matrix.reshape(-1)),
        "mean_of_row_means": float(row_means.mean()),
        "sample_std_of_row_means": float(row_means.std(ddof=1)),
        "top1_agreement": top1_total / (ROWS * ROW_LENGTH),
        "per_row": per_row,
        "token_kld_sha256": sha256_file(token_path),
    }
    result["report_sha256"] = _hash_json(result)
    write_json(output / f"uniform-k{bit}" / "kld-report.json", result)
    print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-uniform-expert-controls-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "encode_root": str(args.encode_root.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "output": str(args.output.resolve()),
        "expert_bits": [3, 4],
        "attention_backend": args.attention_backend,
        "workers": args.workers,
        "resume": args.resume,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    import torch
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    write_json(output / "plan.json", plan | {"dry_run": False})
    receipts = _verify_candidate_inventory(args.encode_root.resolve())
    panel_root = args.panel_root.resolve()
    panel = json.loads((panel_root / "panel.json").read_text())
    if panel.get("panel_sha256") != _hash_json(
        {key: value for key, value in panel.items() if key != "panel_sha256"}
    ):
        raise ValueError("WikiText comparison panel seal mismatch")
    with np.load(panel_root / panel["token_file"]) as handle:
        token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
    if list(token_ids.shape) != [ROWS, ROW_LENGTH]:
        raise ValueError("comparison panel is not 10x2048")
    teacher_paths = sorted((panel_root / "teacher-logits").glob("row-*.safetensors"))
    if len(teacher_paths) != ROWS:
        raise ValueError("comparison panel lacks ten teacher-logit rows")

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
    results = {}
    for bit in (3, 4):
        _install_uniform(model, args.encode_root.resolve(), bit)
        capture_root = output / f"uniform-k{bit}" / "student-logits"
        existing = sorted(capture_root.glob("row-*.safetensors"))
        if args.resume and len(existing) == ROWS:
            student_paths = existing
            print(
                json.dumps(
                    {"stage": "capture-reuse", "bit": bit, "rows": ROWS},
                    sort_keys=True,
                ),
                flush=True,
            )
        elif existing:
            raise ValueError(
                f"partial K{bit} captures found ({len(existing)}); refusing ambiguous resume"
            )
        else:
            student_paths = _capture(model, token_ids, output, bit)
        results[f"k{bit}"] = _score(
            teacher_paths, student_paths, output, bit, args.workers
        )
    del model
    torch.cuda.empty_cache()
    summary = {
        "schema": "quant-pipeline.qwen-uniform-expert-controls.v1",
        "panel_sha256": panel["panel_sha256"],
        "candidate_receipts": receipts,
        "controls": {
            key: {
                "expert_logical_bpw": value["expert_logical_bpw"],
                "mean_kld": value["summary"]["mean"],
                "top1_agreement": value["top1_agreement"],
                "report_sha256": value["report_sha256"],
            }
            for key, value in results.items()
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    summary["summary_sha256"] = _hash_json(summary)
    write_json(output / "summary.json", summary)
    print(json.dumps({"ok": True, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
