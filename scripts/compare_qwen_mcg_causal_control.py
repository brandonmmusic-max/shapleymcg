#!/usr/bin/env python3
"""Seal the matched causal-vs-historical exact-3.5 KLD comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify(document: dict[str, Any], field: str, label: str) -> None:
    if document.get(field) != _hash_json({key: value for key, value in document.items() if key != field}):
        raise ValueError(f"{label} seal mismatch")


def _choice_map(document: dict[str, Any]) -> dict[tuple[int, int, str], int]:
    result = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): int(row["bits"])
        for row in document.get("choices", ())
    }
    if len(result) != 18432:
        raise ValueError("allocation does not contain 18,432 unique matrix choices")
    if sum(bit == 3 for bit in result.values()) != 9216 or sum(bit == 4 for bit in result.values()) != 9216:
        raise ValueError("allocation is not exact 3.5 routed-expert BPW")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-report", type=Path, required=True)
    parser.add_argument("--causal-plan", type=Path, required=True)
    parser.add_argument("--causal-allocation", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--control-allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    causal_report = json.loads(args.causal_report.read_text())
    control_report = json.loads(args.control_report.read_text())
    causal_plan = json.loads(args.causal_plan.read_text())
    control_plan = json.loads(args.control_plan.read_text())
    causal_allocation = json.loads(args.causal_allocation.read_text())
    control_allocation = json.loads(args.control_allocation.read_text())
    _verify(causal_report, "report_sha256", "causal report")
    _verify(control_report, "report_sha256", "control report")
    _verify(causal_allocation, "allocation_sha256", "causal allocation")
    _verify(control_allocation, "allocation_sha256", "control allocation")
    causal_choices = _choice_map(causal_allocation)
    control_choices = _choice_map(control_allocation)

    matched = {
        "model_revision": (causal_report["model_revision"], control_report["model_revision"]),
        "candidate_inventory_sha256": (
            causal_report["candidate_inventory_sha256"],
            control_report["candidate_inventory_sha256"],
        ),
        "teacher_sha256": (causal_report["teacher_sha256"], control_report["teacher_sha256"]),
        "attention_backend": (causal_plan["attention_backend"], control_plan["attention_backend"]),
        "device_map": (causal_plan["device_map"], control_plan["device_map"]),
        "kld_window": (causal_plan["kld_window"], control_plan["kld_window"]),
        "reanchor_every_layers": (
            causal_plan["reanchor_every_layers"],
            control_plan["reanchor_every_layers"],
        ),
    }
    drift = {key: values for key, values in matched.items() if values[0] != values[1]}
    if drift:
        raise ValueError(f"causal/control protocol drift: {drift}")
    causal_mean = float(causal_report["summary"]["mean"])
    control_mean = float(control_report["summary"]["mean"])
    if control_mean <= 0:
        raise ValueError("control KLD must be positive")
    body = {
        "schema": "quant-pipeline.qwen-mcg-causal-control-comparison.v1",
        "protocol": {key: values[0] for key, values in matched.items()},
        "rate": {
            "scope": "routed-expert matrices",
            "logical_bpw": 3.5,
            "k3_matrix_count": 9216,
            "k4_matrix_count": 9216,
            "matrix_count": 18432,
        },
        "causal": {
            "allocation_sha256": causal_allocation["allocation_sha256"],
            "report_sha256": causal_report["report_sha256"],
            "mean_kld": causal_mean,
            "top1_agreement": float(causal_report["top1_agreement"]),
        },
        "historical_control": {
            "allocation_sha256": control_allocation["allocation_sha256"],
            "report_sha256": control_report["report_sha256"],
            "mean_kld": control_mean,
            "top1_agreement": float(control_report["top1_agreement"]),
        },
        "effect": {
            "absolute_kld_reduction": control_mean - causal_mean,
            "relative_kld_reduction": 1.0 - causal_mean / control_mean,
            "top1_agreement_delta": float(causal_report["top1_agreement"])
            - float(control_report["top1_agreement"]),
            "changed_matrix_choices": sum(
                causal_choices[key] != control_choices[key] for key in causal_choices
            ),
        },
        "files": {
            "causal_report_sha256": sha256_file(args.causal_report),
            "causal_plan_sha256": sha256_file(args.causal_plan),
            "causal_allocation_file_sha256": sha256_file(args.causal_allocation),
            "control_report_sha256": sha256_file(args.control_report),
            "control_plan_sha256": sha256_file(args.control_plan),
            "control_allocation_file_sha256": sha256_file(args.control_allocation),
        },
    }
    body["comparison_sha256"] = _hash_json(body)
    write_json(args.output.resolve(), body)
    print(json.dumps({"ok": True, **body["effect"], "comparison_sha256": body["comparison_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
