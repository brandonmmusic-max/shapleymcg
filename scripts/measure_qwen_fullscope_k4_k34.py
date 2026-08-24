#!/usr/bin/env python3
"""Measure full K4-body ShapleyMCG arms against matched TurboDerp controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from measure_qwen_turboderp_exact_3p5 import _load_allocation
from measure_qwen_turboderp_hybrid_k4 import (
    ROW_LENGTH,
    ROWS,
    _capture,
    _copy_exl3_linear,
    _hash_json,
    _import_exllamav3_reader,
    _install_ours,
    _score,
)
from quant_pipeline.core.artifacts import sha256_file, write_json


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def _load_report(path: Path, teacher_hashes: list[str]) -> dict:
    report = json.loads(path.read_text())
    expected = report.get("report_sha256")
    if expected != _hash_json({key: value for key, value in report.items() if key != "report_sha256"}):
        raise ValueError(f"comparison report seal mismatch: {path}")
    if report.get("teacher_files") != teacher_hashes:
        raise ValueError(f"comparison report teacher lineage mismatch: {path}")
    return report


def _verify_attention_inventory(root: Path) -> list[dict]:
    result = []
    for layer in range(48):
        directory = root / f"layer-{layer:03d}"
        receipt = json.loads((directory / "encode-receipt.json").read_text())
        expected = receipt.get("receipt_sha256")
        if expected != _hash_json({key: value for key, value in receipt.items() if key != "receipt_sha256"}):
            raise ValueError(f"attention layer {layer} receipt seal mismatch")
        if receipt.get("layer") != layer or receipt.get("bits") != 4:
            raise ValueError(f"attention layer {layer} receipt scope mismatch")
        tensor_path = directory / receipt["tensor_file"]
        if sha256_file(tensor_path) != receipt["tensor_file_sha256"]:
            raise ValueError(f"attention layer {layer} tensor hash mismatch")
        result.append(receipt)
    return result


def _install_attention(model, root: Path) -> None:
    import torch
    from safetensors import safe_open

    for layer in range(48):
        path = root / f"layer-{layer:03d}" / "attention-k4.safetensors"
        target = model.model.layers[layer].self_attn
        with safe_open(path, framework="pt", device="cpu") as handle:
            with torch.no_grad():
                for projection in ATTENTION:
                    destination = getattr(target, projection).weight
                    reconstructed = handle.get_tensor(f"K4.{projection}.reconstruction_hf")
                    if tuple(reconstructed.shape) != tuple(destination.shape):
                        raise ValueError(
                            f"layer {layer} {projection} shape mismatch: "
                            f"{tuple(reconstructed.shape)} != {tuple(destination.shape)}"
                        )
                    destination.copy_(
                        reconstructed.to(device=destination.device, dtype=destination.dtype)
                    )
        print(json.dumps({"stage": "install-ours-attention-k4", "layer": layer}), flush=True)


def _install_turbo_head(model, turbo_model: Path, exllamav3_root: Path) -> dict:
    import torch

    Config, Model = _import_exllamav3_reader(exllamav3_root)
    config = Config.from_directory(str(turbo_model))
    quant = json.loads((turbo_model / "quantization_config.json").read_text())
    if quant.get("bits") != 4.0 or quant.get("head_bits") != 6:
        raise ValueError("TurboDerp checkpoint is not the pinned K4/K6 scope")
    qmodel = Model.from_config(config)
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
        "bits": 6,
        "quantization_config_sha256": sha256_file(turbo_model / "quantization_config.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--expert-encode-root", type=Path, required=True)
    parser.add_argument("--attention-encode-root", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--turboderp-k4-model", type=Path, required=True)
    parser.add_argument("--exllamav3-root", type=Path, required=True)
    parser.add_argument("--turbo-k34-report", type=Path, required=True)
    parser.add_argument("--turbo-k4-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-fullscope-k4-k34-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "expert_encode_root": str(args.expert_encode_root.resolve()),
        "attention_encode_root": str(args.attention_encode_root.resolve()),
        "allocation": str(args.allocation.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "turboderp_k4_model": str(args.turboderp_k4_model.resolve()),
        "exllamav3_root": str(args.exllamav3_root.resolve()),
        "turbo_k34_report": str(args.turbo_k34_report.resolve()),
        "turbo_k4_report": str(args.turbo_k4_report.resolve()),
        "output": str(args.output.resolve()),
        "fixed_scope": "TurboDerp K6 head; source BF16 embedding/router/norm",
        "ours_scope": "ShapleyMCG K4 q/k/v/o plus ShapleyMCG expert K3/K4 or K4",
        "arms": ["ours-full-k34", "ours-full-k4"],
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
    attention_receipts = _verify_attention_inventory(args.attention_encode_root.resolve())
    allocation, choices = _load_allocation(args.allocation.resolve())
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
    teacher_hashes = [sha256_file(path) for path in teacher_paths]
    turbo_k34 = _load_report(args.turbo_k34_report.resolve(), teacher_hashes)
    turbo_k4 = _load_report(args.turbo_k4_report.resolve(), teacher_hashes)

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
    head_scope = _install_turbo_head(
        model, args.turboderp_k4_model.resolve(), args.exllamav3_root.resolve()
    )
    _install_attention(model, args.attention_encode_root.resolve())
    _install_ours(
        model,
        args.expert_encode_root.resolve(),
        choices,
        "install-ours-full-selected-k34",
    )
    ours_k34 = _score(
        teacher_paths,
        _capture(model, token_ids, output, "ours-full-k34"),
        output,
        "ours-full-k34",
        args.workers,
        "quant-pipeline.qwen-fullscope-k4-k34-arm.v1",
    )
    _install_ours(
        model,
        args.expert_encode_root.resolve(),
        None,
        "install-ours-full-k4",
    )
    ours_k4 = _score(
        teacher_paths,
        _capture(model, token_ids, output, "ours-full-k4"),
        output,
        "ours-full-k4",
        args.workers,
        "quant-pipeline.qwen-fullscope-k4-k34-arm.v1",
    )
    del model
    torch.cuda.empty_cache()

    comparisons = {}
    for label, turbo, ours in (
        ("exact_3p5", turbo_k34, ours_k34),
        ("uniform_k4", turbo_k4, ours_k4),
    ):
        turbo_mean = float(turbo["summary"]["mean"])
        ours_mean = float(ours["summary"]["mean"])
        comparisons[label] = {
            "turboderp_mean_kld": turbo_mean,
            "ours_mean_kld": ours_mean,
            "ours_kld_reduction_vs_turboderp": (turbo_mean - ours_mean) / turbo_mean,
            "turboderp_top1_agreement": float(turbo["top1_agreement"]),
            "ours_top1_agreement": float(ours["top1_agreement"]),
            "ours_top1_gain": float(ours["top1_agreement"] - turbo["top1_agreement"]),
            "turboderp_report_sha256": turbo["report_sha256"],
            "ours_report_sha256": ours["report_sha256"],
        }
    summary = {
        "schema": "quant-pipeline.qwen-fullscope-k4-k34.v1",
        "panel_sha256": panel["panel_sha256"],
        "allocation_sha256": allocation["allocation_sha256"],
        "attention_encode_receipts": [row["receipt_sha256"] for row in attention_receipts],
        "fixed_k6_head": head_scope,
        "fixed_unquantized_scope": ["model.embed_tokens", "routers", "norms"],
        "attention_scope": {"bits": 4, "matrices": 192},
        "expert_scopes": {
            "exact_3p5": {"k3_matrices": 9216, "k4_matrices": 9216},
            "uniform_k4": {"k4_matrices": 18432},
        },
        "comparisons": comparisons,
    }
    summary["summary_sha256"] = _hash_json(summary)
    write_json(output / "summary.json", summary)
    print(json.dumps({"ok": True, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
