#!/usr/bin/env python3
"""Measure an exact TurboDerp-pool-vs-R10-pool 3.5 expert-BPW control.

Both arms use TurboDerp K4 attention/dense weights and its K6 head.  Both use
the same sealed half-K3/half-K4 matrix allocation.  One expert arm selects
reconstructions from TurboDerp's published K3/K4 checkpoints; the other uses
separately calibrated and encoded R10/MCG candidates.  This matches parent,
rate, allocation, non-expert weights, panel, and evaluator, but it does not
isolate a codebook because the candidate-production pipelines also differ in
calibration state, Hessians, rotations, scaling, and numeric encoder policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from measure_qwen_turboderp_hybrid_k4 import (
    ROW_LENGTH,
    ROWS,
    _capture,
    _copy_exl3_linear,
    _hash_json,
    _import_exllamav3_reader,
    _install_ours,
    _install_turboderp_full,
    _score,
)
from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
from quant_pipeline.core.artifacts import sha256_file, write_json


TURBO_K4_REVISION = "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _load_allocation(path: Path) -> tuple[dict, dict[tuple[int, int, str], int]]:
    allocation = json.loads(path.read_text())
    expected = allocation.get("allocation_sha256")
    body = {key: value for key, value in allocation.items() if key != "allocation_sha256"}
    if expected != _hash_json(body):
        raise ValueError("selected allocation seal mismatch")
    choices = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): int(row["bits"])
        for row in allocation.get("choices", [])
    }
    if (
        len(choices) != 48 * 128 * 3
        or sum(bit == 3 for bit in choices.values()) != 48 * 128 * 3 // 2
        or sum(bit == 4 for bit in choices.values()) != 48 * 128 * 3 // 2
    ):
        raise ValueError("allocation is not exact half-K3/half-K4")
    return allocation, choices


def _validate_turbo_scope(model_root: Path, bits: float) -> dict:
    path = model_root / "quantization_config.json"
    quant = json.loads(path.read_text())
    if (
        quant.get("bits") != bits
        or quant.get("head_bits") != 6
        or quant.get("calibration") != {"rows": 100, "cols": 2048}
    ):
        raise ValueError(f"TurboDerp checkpoint is not pinned K{bits:g}/K6 scope")
    return {
        "bits": bits,
        "head_bits": 6,
        "calibration": quant["calibration"],
        "quantization_config_sha256": sha256_file(path),
    }


def _install_turbo_selected_k3(
    model,
    turbo_k3: Path,
    choices: dict[tuple[int, int, str], int],
    exllamav3_root: Path,
) -> dict:
    import torch

    Config, Model = _import_exllamav3_reader(exllamav3_root)
    config = Config.from_directory(str(turbo_k3))
    qmodel = Model.from_config(config)
    if len(qmodel.modules) != 51:
        raise ValueError(f"unexpected EXL3 K3 module inventory: {len(qmodel.modules)}")
    blocks = qwen_moe_layers(model)
    installed = 0
    for layer in range(48):
        qblock = qmodel.modules[layer + 1]
        config.stc.begin_deferred_load()
        qblock.load(torch.device("cuda:0"))
        config.stc.end_deferred_load()
        target = blocks[layer].experts
        gate_rows = target.gate_up_proj.shape[1] // 2
        for expert in range(128):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            for projection in PROJECTIONS:
                if choices[(layer, expert, projection)] != 3:
                    continue
                source = qblock.find_module(prefix + "." + projection)
                if projection == "gate_proj":
                    destination = target.gate_up_proj[expert, :gate_rows]
                elif projection == "up_proj":
                    destination = target.gate_up_proj[expert, gate_rows:]
                else:
                    destination = target.down_proj[expert]
                _copy_exl3_linear(source, destination)
                installed += 1
        qblock.unload()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {"stage": "install-turboderp-selected-k3", "layer": layer},
                sort_keys=True,
            ),
            flush=True,
        )
    if installed != 48 * 128 * 3 // 2:
        raise ValueError(f"installed {installed} K3 matrices, expected 9216")
    return {"k3_matrices": installed, "k4_matrices": 48 * 128 * 3 - installed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--encode-root", type=Path)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--turboderp-k3-model", type=Path, required=True)
    parser.add_argument("--turboderp-k3-revision", required=True)
    parser.add_argument("--turboderp-k4-model", type=Path, required=True)
    parser.add_argument("--exllamav3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument(
        "--turboderp-pool-only",
        action="store_true",
        help="score only the selected allocation reconstructed from the published TurboDerp K3/K4 pool",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-turboderp-exact-3p5-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "encode_root": str(args.encode_root.resolve()) if args.encode_root else None,
        "allocation": str(args.allocation.resolve()),
        "panel_root": str(args.panel_root.resolve()),
        "turboderp_k3_model": str(args.turboderp_k3_model.resolve()),
        "turboderp_k3_revision": args.turboderp_k3_revision,
        "turboderp_k4_model": str(args.turboderp_k4_model.resolve()),
        "turboderp_k4_revision": TURBO_K4_REVISION,
        "exllamav3_root": str(args.exllamav3_root.resolve()),
        "output": str(args.output.resolve()),
        "expert_rate": "exact half K3 / half K4 = 3.5 logical BPW",
        "fixed_nonexpert_scope": "TurboDerp K4 body and K6 head",
        "arms": (
            ["turboderp-selected-k34"]
            if args.turboderp_pool_only
            else ["turboderp-selected-k34", "hybrid-ours-selected-k34"]
        ),
        "comparison_kind": "matched allocation across independently produced reconstruction pools",
        "codec_only_ablation": False,
        "native_turboderp_3p5_published": False,
        "confounded_candidate_production_factors": [
            "calibration corpus and routed subsets",
            "progressive versus source-BF16 Hessian state",
            "rotation and scaling policy",
            "procedural codebook selection",
            "numeric encoder implementation",
        ],
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if not args.turboderp_pool_only and args.encode_root is None:
        raise ValueError("--encode-root is required unless --turboderp-pool-only is selected")

    import torch
    from transformers import AutoModelForCausalLM

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan | {"dry_run": False})
    allocation, choices = _load_allocation(args.allocation.resolve())
    k3_scope = _validate_turbo_scope(args.turboderp_k3_model.resolve(), 3.0)
    k4_scope = _validate_turbo_scope(args.turboderp_k4_model.resolve(), 4.0)
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

    fixed_scope = _install_turboderp_full(
        model,
        args.turboderp_k4_model.resolve(),
        args.exllamav3_root.resolve(),
    )
    rate = _install_turbo_selected_k3(
        model,
        args.turboderp_k3_model.resolve(),
        choices,
        args.exllamav3_root.resolve(),
    )
    turbo = _score(
        teacher_paths,
        _capture(model, token_ids, output, "turboderp-selected-k34"),
        output,
        "turboderp-selected-k34",
        args.workers,
        "quant-pipeline.qwen-turboderp-exact-3p5-arm.v1",
    )
    ours = None
    if not args.turboderp_pool_only:
        _install_ours(
            model,
            args.encode_root.resolve(),
            choices,
            "install-ours-selected-k34-matched",
        )
        ours = _score(
            teacher_paths,
            _capture(model, token_ids, output, "hybrid-ours-selected-k34"),
            output,
            "hybrid-ours-selected-k34",
            args.workers,
            "quant-pipeline.qwen-turboderp-exact-3p5-arm.v1",
        )
    del model
    torch.cuda.empty_cache()
    summary = {
        "schema": "quant-pipeline.qwen-turboderp-exact-3p5.v1",
        "panel_sha256": panel["panel_sha256"],
        "allocation_sha256": allocation["allocation_sha256"],
        "expert_rate": rate,
        "turboderp_k3_revision": args.turboderp_k3_revision,
        "turboderp_k4_revision": TURBO_K4_REVISION,
        "turboderp_k3_scope": k3_scope,
        "turboderp_k4_scope": k4_scope,
        "fixed_nonexpert_scope": fixed_scope,
        "arms": {
            row["arm"]: {
                "mean_kld": row["summary"]["mean"],
                "top1_agreement": row["top1_agreement"],
                "report_sha256": row["report_sha256"],
            }
            for row in (turbo,) + ((ours,) if ours is not None else ())
        },
    }
    if ours is not None:
        summary["ours_kld_reduction_vs_turboderp_at_exact_3p5"] = (
            turbo["summary"]["mean"] - ours["summary"]["mean"]
        ) / turbo["summary"]["mean"]
        summary["ours_top1_gain_vs_turboderp_at_exact_3p5"] = (
            ours["top1_agreement"] - turbo["top1_agreement"]
        )
    summary["summary_sha256"] = _hash_json(summary)
    write_json(output / "summary.json", summary)
    print(json.dumps({"ok": True, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
