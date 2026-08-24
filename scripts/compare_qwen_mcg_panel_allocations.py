#!/usr/bin/env python3
"""Seal a matched 20,480-position causal-vs-historical allocation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify(document: dict[str, Any], field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def _choice_map(document: dict[str, Any]) -> dict[tuple[int, int, str], int]:
    result = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): int(
            row["bits"]
        )
        for row in document.get("choices", ())
    }
    if len(result) != 18432:
        raise ValueError("allocation does not contain 18,432 unique matrix choices")
    if sum(bit == 3 for bit in result.values()) != 9216 or sum(
        bit == 4 for bit in result.values()
    ) != 9216:
        raise ValueError("allocation is not exact 3.5 routed-expert BPW")
    return result


def _load_result(root: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads((root / "kld-report.json").read_text())
    verification = json.loads((root / "independent-verification.json").read_text())
    _verify(report, "report_sha256", f"{label} report")
    _verify(verification, "verification_sha256", f"{label} verification")
    if (
        not verification.get("ok")
        or verification.get("report_sha256") != report["report_sha256"]
        or verification.get("panel_sha256") != report["panel_sha256"]
        or verification.get("attention_backend") != report["attention_backend"]
        or verification.get("positions") != report.get("summary", {}).get("count")
        or float(verification.get("max_absolute_delta", float("inf"))) > 1e-10
    ):
        raise ValueError(f"{label} result is not independently verified")
    return report, verification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--causal-allocation", type=Path, required=True)
    parser.add_argument("--control-allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    causal_root = args.causal_result.resolve()
    control_root = args.control_result.resolve()
    causal_report, causal_verification = _load_result(causal_root, "causal")
    control_report, control_verification = _load_result(control_root, "control")
    causal_allocation = json.loads(args.causal_allocation.read_text())
    control_allocation = json.loads(args.control_allocation.read_text())
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
        "panel_sha256": (causal_report["panel_sha256"], control_report["panel_sha256"]),
        "attention_backend": (
            causal_report["attention_backend"],
            control_report["attention_backend"],
        ),
        "teacher_files": (causal_report["teacher_files"], control_report["teacher_files"]),
        "positions": (
            causal_report["summary"]["count"],
            control_report["summary"]["count"],
        ),
    }
    drift = {key: values for key, values in matched.items() if values[0] != values[1]}
    if drift:
        raise ValueError(f"causal/control panel drift: {sorted(drift)}")
    if matched["attention_backend"][0] != "sdpa" or matched["positions"][0] != 20480:
        raise ValueError("comparison is not the required 20,480-position SDPA panel")
    if causal_report["allocation_sha256"] != causal_allocation["allocation_sha256"]:
        raise ValueError("causal report/allocation identity mismatch")
    if control_report["allocation_sha256"] != control_allocation["allocation_sha256"]:
        raise ValueError("control report/allocation identity mismatch")

    causal_mean = float(causal_report["summary"]["mean"])
    control_mean = float(control_report["summary"]["mean"])
    if control_mean <= 0:
        raise ValueError("control KLD must be positive")
    body = {
        "schema": "quant-pipeline.qwen-mcg-panel-causal-control-comparison.v1",
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
            "verification_sha256": causal_verification["verification_sha256"],
            "mean_kld": causal_mean,
            "top1_agreement": float(causal_report["top1_agreement"]),
        },
        "historical_control": {
            "allocation_sha256": control_allocation["allocation_sha256"],
            "report_sha256": control_report["report_sha256"],
            "verification_sha256": control_verification["verification_sha256"],
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
            "causal_report_file_sha256": sha256_file(causal_root / "kld-report.json"),
            "causal_verification_file_sha256": sha256_file(
                causal_root / "independent-verification.json"
            ),
            "causal_allocation_file_sha256": sha256_file(args.causal_allocation),
            "control_report_file_sha256": sha256_file(control_root / "kld-report.json"),
            "control_verification_file_sha256": sha256_file(
                control_root / "independent-verification.json"
            ),
            "control_allocation_file_sha256": sha256_file(args.control_allocation),
        },
    }
    body["comparison_sha256"] = _hash_json(body)
    write_json(args.output.resolve(), body)
    print(
        json.dumps(
            {"ok": True, **body["effect"], "comparison_sha256": body["comparison_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
