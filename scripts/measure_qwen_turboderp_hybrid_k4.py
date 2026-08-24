#!/usr/bin/env python3
"""Measure post-trained Qwen K4 arms with a common BF16 Transformers replay.

The three causal arms are captured in one source-model load:

1. corrected-R10 K4 experts with source-BF16 attention/head;
2. TurboDerp K4 body plus K6 head, reconstructed into the same HF model; and
3. TurboDerp dense K4/K6 components with only its experts replaced by ours.

The third arm is the matched-component attribution: parent, evaluator, dense
reconstructions, router, head, and token panel are fixed; expert reconstruction
is the only changed component.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

from measure_qwen_uniform_expert_controls import (
    ROW_LENGTH,
    ROWS,
    _hash_json,
    _score_pair,
    _verify_candidate_inventory,
)
from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
from quant_pipeline.core.artifacts import atomic_write, sha256_file, write_json
from quant_pipeline.scoring.kld import summarize


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def _install_ours_k4(model, encode_root: Path) -> None:
    import torch
    from safetensors import safe_open

    blocks = qwen_moe_layers(model)
    for layer in range(48):
        path = encode_root / f"layer-{layer:03d}" / "k34-candidates.safetensors"
        experts = blocks[layer].experts
        with safe_open(path, framework="pt", device="cpu") as handle:
            for expert in range(128):
                gate = handle.get_tensor(f"K4.E{expert:03d}.gate_proj.reconstruction_hf")
                up = handle.get_tensor(f"K4.E{expert:03d}.up_proj.reconstruction_hf")
                down = handle.get_tensor(f"K4.E{expert:03d}.down_proj.reconstruction_hf")
                with torch.no_grad():
                    experts.gate_up_proj[expert, : gate.shape[0]].copy_(
                        gate.to(experts.gate_up_proj.device)
                    )
                    experts.gate_up_proj[expert, gate.shape[0] :].copy_(
                        up.to(experts.gate_up_proj.device)
                    )
                    experts.down_proj[expert].copy_(down.to(experts.down_proj.device))
        print(json.dumps({"stage": "install-ours-k4", "layer": layer}), flush=True)


def _copy_exl3_linear(source, destination) -> None:
    import torch

    # EXL3 exposes its reconstructed matrix as [in_features, out_features];
    # Transformers stores linear weights as [out_features, in_features].
    reconstructed = source.inner.get_weight_tensor().T.contiguous()
    if tuple(reconstructed.shape) != tuple(destination.shape):
        raise ValueError(
            f"EXL3/HF shape mismatch: {tuple(reconstructed.shape)} != {tuple(destination.shape)}"
        )
    with torch.no_grad():
        destination.copy_(reconstructed.to(device=destination.device, dtype=destination.dtype))
    del reconstructed


def _install_turboderp_full(model, turbo_model: Path, exllamav3_root: Path) -> dict:
    import torch

    sys.path.insert(0, str(exllamav3_root))
    try:
        from exllamav3 import Config, Model
    finally:
        # Imported modules remain in sys.modules; avoid affecting later imports.
        sys.path.pop(0)

    config = Config.from_directory(str(turbo_model))
    quant = json.loads((turbo_model / "quantization_config.json").read_text())
    if (
        quant.get("bits") != 4.0
        or quant.get("head_bits") != 6
        or quant.get("calibration") != {"rows": 100, "cols": 2048}
    ):
        raise ValueError("TurboDerp checkpoint is not the pinned K4/K6 reference scope")
    qmodel = Model.from_config(config)
    if len(qmodel.modules) != 51:
        raise ValueError(f"unexpected EXL3 module inventory: {len(qmodel.modules)}")

    for layer in range(48):
        qblock = qmodel.modules[layer + 1]
        config.stc.begin_deferred_load()
        qblock.load(torch.device("cuda:0"))
        config.stc.end_deferred_load()
        hf = model.model.layers[layer]
        for projection in ATTENTION:
            source = qblock.find_module(f"model.layers.{layer}.self_attn.{projection}")
            _copy_exl3_linear(source, getattr(hf.self_attn, projection).weight)
        for expert in range(128):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate = qblock.find_module(prefix + ".gate_proj")
            up = qblock.find_module(prefix + ".up_proj")
            down = qblock.find_module(prefix + ".down_proj")
            target = hf.mlp.experts
            gate_rows = target.gate_up_proj.shape[1] // 2
            _copy_exl3_linear(gate, target.gate_up_proj[expert, :gate_rows])
            _copy_exl3_linear(up, target.gate_up_proj[expert, gate_rows:])
            _copy_exl3_linear(down, target.down_proj[expert])
        qblock.unload()
        torch.cuda.empty_cache()
        print(json.dumps({"stage": "install-turboderp-full", "layer": layer}), flush=True)

    qhead = qmodel.modules[-1]
    config.stc.begin_deferred_load()
    qhead.load(torch.device("cuda:0"))
    config.stc.end_deferred_load()
    _copy_exl3_linear(qhead, model.lm_head.weight)
    qhead.unload()
    torch.cuda.empty_cache()
    return {
        "repo": "turboderp/Qwen3-30B-A3B-exl3",
        "revision": "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef",
        "quantization_config_sha256": sha256_file(
            turbo_model / "quantization_config.json"
        ),
        "bits": 4.0,
        "head_bits": 6,
        "attention_projections": 48 * 4,
        "expert_matrices": 48 * 128 * 3,
        "router": "source BF16",
        "replay_dtype": "BF16",
    }


def _capture(model, token_ids: np.ndarray, root: Path, arm: str) -> list[Path]:
    import torch
    from safetensors.torch import save_file

    destination = root / arm / "student-logits"
    destination.mkdir(parents=True, exist_ok=False)
    device = model.get_input_embeddings().weight.device
    paths = []
    for row, values in enumerate(token_ids):
        ids = torch.from_numpy(values.astype(np.int64, copy=False)).unsqueeze(0).to(device)
        started = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        value = logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()
        path = destination / f"row-{row:02d}.safetensors"
        save_file(
            {"logits": value},
            path,
            metadata={"role": "posttrained-k4-student", "arm": arm, "row": str(row)},
        )
        paths.append(path)
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "arm": arm,
                    "row": row,
                    "elapsed_seconds": time.monotonic() - started,
                    "sha256": sha256_file(path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del ids, logits, value
    return paths


def _score(
    teacher_paths: list[Path],
    student_paths: list[Path],
    root: Path,
    arm: str,
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
    matrix = np.stack([row[1] for row in scored])
    top1 = sum(row[2] for row in scored) / (ROWS * ROW_LENGTH)
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    token_path = root / arm / "token-kld.npy"
    atomic_write(token_path, buffer.getvalue())
    report = {
        "schema": "quant-pipeline.qwen-posttrained-k4-arm.v1",
        "arm": arm,
        "metric": "float32 mean tokenwise KL(posttrained BF16 || reconstructed student)",
        "summary": summarize(matrix.reshape(-1)),
        "top1_agreement": top1,
        "token_kld_sha256": sha256_file(token_path),
        "teacher_files": [sha256_file(path) for path in teacher_paths],
        "student_files": [sha256_file(path) for path in student_paths],
    }
    report["report_sha256"] = _hash_json(report)
    write_json(root / arm / "kld-report.json", report)
    print(json.dumps({"ok": True, **report}, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--turboderp-model", type=Path, required=True)
    parser.add_argument("--exllamav3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-posttrained-turboderp-hybrid-k4-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "source_revision": args.source_revision,
        "encode_root": str(args.encode_root.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "turboderp_model": str(args.turboderp_model.resolve()),
        "turboderp_revision": "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef",
        "exllamav3_root": str(args.exllamav3_root.resolve()),
        "output": str(args.output.resolve()),
        "workers": args.workers,
        "attention_backend": args.attention_backend,
        "arms": ["ours-expert-k4", "turboderp-full-k4", "hybrid-ours-experts"],
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch
    from transformers import AutoModelForCausalLM

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan | {"dry_run": False})
    candidate_receipts = _verify_candidate_inventory(args.encode_root.resolve())
    panel = json.loads((args.panel_root / "panel.json").read_text())
    if panel.get("panel_sha256") != _hash_json(
        {key: value for key, value in panel.items() if key != "panel_sha256"}
    ):
        raise ValueError("post-trained WikiText panel seal mismatch")
    with np.load(args.panel_root / panel["token_file"]) as handle:
        token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
    if list(token_ids.shape) != [ROWS, ROW_LENGTH]:
        raise ValueError("post-trained panel is not 10x2048")
    teacher_paths = sorted((args.panel_root / "teacher-logits").glob("row-*.safetensors"))
    if len(teacher_paths) != ROWS:
        raise ValueError("post-trained panel lacks ten BF16 teacher rows")

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

    _install_ours_k4(model, args.encode_root.resolve())
    ours = _score(
        teacher_paths,
        _capture(model, token_ids, output, "ours-expert-k4"),
        output,
        "ours-expert-k4",
        args.workers,
    )
    turbo_scope = _install_turboderp_full(
        model,
        args.turboderp_model.resolve(),
        args.exllamav3_root.resolve(),
    )
    turbo = _score(
        teacher_paths,
        _capture(model, token_ids, output, "turboderp-full-k4"),
        output,
        "turboderp-full-k4",
        args.workers,
    )
    _install_ours_k4(model, args.encode_root.resolve())
    hybrid = _score(
        teacher_paths,
        _capture(model, token_ids, output, "hybrid-ours-experts"),
        output,
        "hybrid-ours-experts",
        args.workers,
    )
    del model
    torch.cuda.empty_cache()

    summary = {
        "schema": "quant-pipeline.qwen-posttrained-turboderp-hybrid-k4.v1",
        "source_revision": args.source_revision,
        "panel_sha256": panel["panel_sha256"],
        "candidate_receipts": candidate_receipts,
        "turboderp_scope": turbo_scope,
        "arms": {
            row["arm"]: {
                "mean_kld": row["summary"]["mean"],
                "top1_agreement": row["top1_agreement"],
                "report_sha256": row["report_sha256"],
            }
            for row in (ours, turbo, hybrid)
        },
        "matched_hybrid_kld_reduction_vs_turboderp": (
            turbo["summary"]["mean"] - hybrid["summary"]["mean"]
        )
        / turbo["summary"]["mean"],
        "matched_hybrid_top1_gain_vs_turboderp": (
            hybrid["top1_agreement"] - turbo["top1_agreement"]
        ),
        "published_turboderp_context": {
            "mean_kld": 0.0215,
            "top1_agreement": 0.9433,
            "provenance": "upstream model card; not substituted for local remeasurement",
        },
    }
    summary["summary_sha256"] = _hash_json(summary)
    write_json(output / "summary.json", summary)
    print(json.dumps({"ok": True, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
