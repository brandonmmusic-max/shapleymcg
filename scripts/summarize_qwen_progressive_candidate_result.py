#!/usr/bin/env python3
"""Synthesize sealed progressive-candidate results without re-scoring logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _load(path: Path, seal: str, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get(seal) != _hash_json({key: row for key, row in value.items() if key != seal}):
        raise ValueError(f"{label} seal mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-panel-report", type=Path, required=True)
    parser.add_argument("--fast-progressive-report", type=Path, required=True)
    parser.add_argument("--progressive-panel-report", type=Path, required=True)
    parser.add_argument("--factory-union-report", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-progressive-candidate-summary-plan.v1",
        "native_panel_report": str(args.native_panel_report.resolve()),
        "native_panel_report_file_sha256": sha256_file(args.native_panel_report),
        "fast_progressive_report": str(args.fast_progressive_report.resolve()),
        "fast_progressive_report_file_sha256": sha256_file(args.fast_progressive_report),
        "progressive_panel_report": str(args.progressive_panel_report.resolve()),
        "progressive_panel_report_file_sha256": sha256_file(args.progressive_panel_report),
        "factory_union_report": str(args.factory_union_report.resolve()),
        "factory_union_report_file_sha256": sha256_file(args.factory_union_report),
        "lineage": str(args.lineage.resolve()),
        "lineage_file_sha256": sha256_file(args.lineage),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    native = _load(args.native_panel_report, "report_sha256", "native panel")
    fast = _load(args.fast_progressive_report, "report_sha256", "fast progressive")
    progressive = _load(args.progressive_panel_report, "report_sha256", "progressive panel")
    union = _load(args.factory_union_report, "report_sha256", "factory union")
    lineage = _load(args.lineage, "lineage_sha256", "GLM lineage")
    native_mean = float(native["summary"]["mean"])
    progressive_mean = float(progressive["summary"]["mean"])
    endpoint = union["untouched_validation"]
    union_baseline = float(endpoint["baseline_summary"]["mean"])
    union_mean = float(endpoint["union_summary"]["mean"])
    interval = [float(value) for value in endpoint["paired_block_95_interval_for_baseline_minus_union"]]
    if len(interval) != 2 or interval[0] > interval[1]:
        raise ValueError("factory-union paired interval is invalid")
    body = {
        "schema": "quant-pipeline.qwen-progressive-candidate-result.v1",
        "model_revision": progressive["model_revision"],
        "attention_backend": progressive["attention_backend"],
        "rate": {
            "scope": "routed-expert-matrix-weights",
            "logical_bpw": 3.5,
            "k3_count": 9216,
            "k4_count": 9216,
        },
        "glm_lineage": {
            "lineage_sha256": lineage["lineage_sha256"],
            "original_glm_repo": lineage["original_glm_model"]["repo"],
            "original_glm_revision": lineage["original_glm_model"]["revision"],
            "calibration_sha256": lineage["original_glm_model"]["verified_files"][0]["sha256"],
            "qwen_teacher_sha256": lineage["progressive_fast_kld"]["teacher_sha256"],
        },
        "fast_progressive_diagnostic": {
            "positions": int(fast["summary"]["count"]),
            "mean_kld": float(fast["summary"]["mean"]),
            "allocation_sha256": fast["allocation_sha256"],
            "report_sha256": fast["report_sha256"],
            "interpretation": "fast diagonal-allocation candidate-pipeline diagnostic",
        },
        "frozen_causal_rate_panel": {
            "positions": int(progressive["summary"]["count"]),
            "native_source_state_mean_kld": native_mean,
            "progressive_state_mean_kld": progressive_mean,
            "absolute_native_minus_progressive": native_mean - progressive_mean,
            "relative_native_minus_progressive": (native_mean - progressive_mean) / native_mean,
            "progressive_is_lower": progressive_mean < native_mean,
            "native_report_sha256": native["report_sha256"],
            "progressive_report_sha256": progressive["report_sha256"],
            "allocation_sha256": progressive["allocation_sha256"],
        },
        "factory_union_untouched_validation": {
            "selection_row": int(union["selection"]["row"]),
            "validation_rows": list(endpoint["rows"]),
            "positions": int(endpoint["union_summary"]["count"]),
            "native_baseline_mean_kld": union_baseline,
            "selected_union_mean_kld": union_mean,
            "absolute_native_minus_union": union_baseline - union_mean,
            "relative_native_minus_union": (union_baseline - union_mean) / union_baseline,
            "union_is_lower": union_mean < union_baseline,
            "paired_block_95_interval_for_native_minus_union": interval,
            "interval_excludes_zero_in_favor_of_union": interval[0] > 0.0,
            "rows_union_better": int(endpoint["rows_union_better"]),
            "row_count": int(endpoint["row_count"]),
            "report_sha256": union["report_sha256"],
            "factory_allocation_sha256": union["factory_allocation_sha256"],
        },
        "claim_boundary": {
            "frozen_rate_candidate_quality_ablation": True,
            "candidate_family_selected_inside_process": True,
            "selection_and_validation_rows_disjoint": True,
            "joint_matrix_level_factory_plus_rate_claim": False,
            "packed_runtime_throughput_claim": False,
        },
        "source_files": plan,
    }
    body["summary_sha256"] = _hash_json(body)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, body)
    print(json.dumps({
        "ok": True,
        "summary_sha256": body["summary_sha256"],
        "fast_progressive_mean_kld": body["fast_progressive_diagnostic"]["mean_kld"],
        "progressive_panel_mean_kld": progressive_mean,
        "factory_union_validation_mean_kld": union_mean,
        "factory_union_interval_excludes_zero": interval[0] > 0.0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
