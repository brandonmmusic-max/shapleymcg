#!/usr/bin/env python3
"""Measure seeded score-blind 3.5-bpw expert allocations on a sealed panel.

Each seed assigns exactly 64 of 128 experts to K4 independently within every
layer and projection; the other 64 use K3.  This fixes the rate at exactly 3.5
logical bits per expert-weight element while deliberately using no calibration,
Hessian, routed-damage, or end-to-end score to choose K4 locations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from measure_qwen_uniform_expert_controls import (
    EXPERT_ELEMENTS,
    ROW_LENGTH,
    ROWS,
    _hash_json,
    _score_pair,
    _verify_candidate_inventory,
)
from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
from quant_pipeline.core.artifacts import atomic_write, sha256_file, write_json
from quant_pipeline.scoring.kld import summarize


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _allocation(seed: int) -> tuple[np.ndarray, dict]:
    bits = np.full((48, len(PROJECTIONS), 128), 3, dtype=np.uint8)
    choices = []
    for layer in range(48):
        for projection_index, projection in enumerate(PROJECTIONS):
            # SeedSequence makes every layer/projection stream explicit and
            # independently reproducible rather than dependent on loop order.
            rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([seed, layer, projection_index]))
            )
            selected = np.sort(rng.permutation(128)[:64])
            bits[layer, projection_index, selected] = 4
            choices.append(
                {
                    "layer": layer,
                    "projection": projection,
                    "k4_experts": selected.tolist(),
                }
            )
    k4_count = int(np.count_nonzero(bits == 4))
    k3_count = int(np.count_nonzero(bits == 3))
    if k3_count != 9216 or k4_count != 9216:
        raise AssertionError(f"unbalanced naive allocation: K3={k3_count}, K4={k4_count}")
    document = {
        "schema": "quant-pipeline.qwen-score-blind-3p5-allocation.v1",
        "seed": seed,
        "rng": "numpy.PCG64(SeedSequence([seed, layer, projection_index]))",
        "stratification": "exactly 64 K4 and 64 K3 per layer and projection",
        "score_information_used": False,
        "expert_logical_bpw": 3.5,
        "k3_count": k3_count,
        "k4_count": k4_count,
        "choices": choices,
    }
    document["allocation_sha256"] = _hash_json(document)
    return bits, document


def _install(model, encode_root: Path, bits: np.ndarray, seed: int) -> None:
    import torch
    from safetensors import safe_open

    blocks = qwen_moe_layers(model)
    for layer in range(48):
        path = encode_root / f"layer-{layer:03d}" / "k34-candidates.safetensors"
        gate_up = []
        down = []
        with safe_open(path, framework="pt", device="cpu") as handle:
            for expert in range(128):
                gate_bit = int(bits[layer, 0, expert])
                up_bit = int(bits[layer, 1, expert])
                down_bit = int(bits[layer, 2, expert])
                gate = handle.get_tensor(
                    f"K{gate_bit}.E{expert:03d}.gate_proj.reconstruction_hf"
                )
                up = handle.get_tensor(
                    f"K{up_bit}.E{expert:03d}.up_proj.reconstruction_hf"
                )
                gate_up.append(torch.cat((gate, up), dim=0))
                down.append(
                    handle.get_tensor(
                        f"K{down_bit}.E{expert:03d}.down_proj.reconstruction_hf"
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
        print(
            json.dumps({"stage": "install", "seed": seed, "layer": layer}, sort_keys=True),
            flush=True,
        )


def _capture(
    model,
    token_ids: np.ndarray,
    seed_root: Path,
    seed: int,
) -> list[Path]:
    import torch
    from safetensors.torch import save_file

    root = seed_root / "student-logits"
    root.mkdir(parents=True, exist_ok=False)
    device = model.get_input_embeddings().weight.device
    paths = []
    for row, values in enumerate(token_ids):
        ids = torch.from_numpy(values.astype(np.int64, copy=False)).unsqueeze(0).to(device)
        started = time.monotonic()
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=False, return_dict=True).logits
        logits = output.float().cpu().reshape(-1, output.shape[-1]).contiguous()
        path = root / f"row-{row:02d}.safetensors"
        save_file(
            {"logits": logits},
            path,
            metadata={"role": "score-blind-3p5-student", "seed": str(seed), "row": str(row)},
        )
        paths.append(path)
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "seed": seed,
                    "row": row,
                    "elapsed_seconds": time.monotonic() - started,
                    "sha256": sha256_file(path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del ids, output, logits
    return paths


def _score(
    teacher_paths: list[Path],
    student_paths: list[Path],
    seed_root: Path,
    seed: int,
    allocation: dict,
    workers: int,
) -> dict:
    payloads = [
        (index, str(teacher), str(student))
        for index, (teacher, student) in enumerate(
            zip(teacher_paths, student_paths, strict=True)
        )
    ]
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
    token_path = seed_root / "token-kld.npy"
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    atomic_write(token_path, buffer.getvalue())
    row_means = np.asarray([row["summary"]["mean"] for row in per_row])
    result = {
        "schema": "quant-pipeline.qwen-score-blind-3p5-control.v1",
        "seed": seed,
        "allocation_sha256": allocation["allocation_sha256"],
        "expert_logical_bpw": 3.5,
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
    write_json(seed_root / "kld-report.json", result)
    print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    plan = {
        "schema": "quant-pipeline.qwen-score-blind-3p5-controls-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "encode_root": str(args.encode_root.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "output": str(args.output.resolve()),
        "seeds": args.seeds,
        "workers": args.workers,
        "attention_backend": args.attention_backend,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan | {"dry_run": False})
    candidate_receipts = _verify_candidate_inventory(args.encode_root.resolve())

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
        raise ValueError("comparison panel lacks ten BF16 teacher rows")
    mixed = json.loads((panel_root / "kld-report.json").read_text())

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

    results = []
    for seed in args.seeds:
        seed_root = output / f"seed-{seed:03d}"
        seed_root.mkdir()
        bits, allocation = _allocation(seed)
        write_json(seed_root / "allocation.json", allocation)
        _install(model, args.encode_root.resolve(), bits, seed)
        student_paths = _capture(model, token_ids, seed_root, seed)
        results.append(
            _score(
                teacher_paths,
                student_paths,
                seed_root,
                seed,
                allocation,
                args.workers,
            )
        )

    del model
    torch.cuda.empty_cache()
    naive_kld = np.asarray([row["summary"]["mean"] for row in results], dtype=np.float64)
    naive_top1 = np.asarray([row["top1_agreement"] for row in results], dtype=np.float64)
    selected_kld = float(mixed["summary"]["mean"])
    selected_top1 = float(mixed["top1_agreement"])
    summary = {
        "schema": "quant-pipeline.qwen-score-blind-3p5-controls.v1",
        "panel_sha256": panel["panel_sha256"],
        "candidate_receipts": candidate_receipts,
        "seeds": args.seeds,
        "controls": [
            {
                "seed": row["seed"],
                "mean_kld": row["summary"]["mean"],
                "top1_agreement": row["top1_agreement"],
                "allocation_sha256": row["allocation_sha256"],
                "report_sha256": row["report_sha256"],
            }
            for row in results
        ],
        "naive_mean_kld": float(naive_kld.mean()),
        "naive_sample_std_kld": float(naive_kld.std(ddof=1)),
        "naive_min_kld": float(naive_kld.min()),
        "naive_max_kld": float(naive_kld.max()),
        "naive_mean_top1_agreement": float(naive_top1.mean()),
        "naive_sample_std_top1_agreement": float(naive_top1.std(ddof=1)),
        "selected_mean_kld": selected_kld,
        "selected_top1_agreement": selected_top1,
        "selected_kld_reduction_vs_naive_mean": float(
            (naive_kld.mean() - selected_kld) / naive_kld.mean()
        ),
        "selected_top1_gain_vs_naive_mean": float(selected_top1 - naive_top1.mean()),
        "naive_seeds_beating_selected_kld": int(np.count_nonzero(naive_kld < selected_kld)),
        "elapsed_seconds": time.monotonic() - started,
    }
    summary["summary_sha256"] = _hash_json(summary)
    write_json(output / "summary.json", summary)
    print(json.dumps({"ok": True, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
