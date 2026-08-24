#!/usr/bin/env python3
"""Refresh a sealed validation model card and metadata without rewriting weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assemble_qwen_validation_model import (
    _load_inputs,
    _load_publication_evidence,
    _model_card,
    _seal_manifest,
    _write_sha256s,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kld-root", type=Path, required=True)
    parser.add_argument("--causal-comparison", type=Path, required=True)
    parser.add_argument("--panel-report", type=Path, required=True)
    parser.add_argument("--panel-control-report", type=Path, required=True)
    parser.add_argument("--panel-verification", type=Path, required=True)
    parser.add_argument("--panel-control-verification", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    allocation, report, verification, _ = _load_inputs(args.kld_root.resolve())
    (
        comparison,
        panel_report,
        panel_control_report,
        panel_verification,
        panel_control_verification,
    ) = _load_publication_evidence(
        args.causal_comparison,
        args.panel_report,
        args.panel_control_report,
        args.panel_verification,
        args.panel_control_verification,
        allocation,
        report,
    )
    provenance = json.loads((model / "provenance.json").read_text())
    logit_verification = json.loads((model / "model-logit-verification.json").read_text())
    if provenance["allocation_sha256"] != allocation["allocation_sha256"]:
        raise ValueError("model/allocation identity mismatch")
    if provenance["kld_report_sha256"] != report["report_sha256"]:
        raise ValueError("model/KLD identity mismatch")
    if (
        provenance["model_logit_verification_sha256"]
        != logit_verification["verification_sha256"]
    ):
        raise ValueError("model/logit-verification identity mismatch")
    expected_evidence = {
        "causal_comparison_sha256": comparison["comparison_sha256"],
        "panel_report_sha256": panel_report["report_sha256"],
        "panel_control_report_sha256": panel_control_report["report_sha256"],
        "panel_verification_sha256": panel_verification["verification_sha256"],
        "panel_control_verification_sha256": panel_control_verification[
            "verification_sha256"
        ],
    }
    for key, expected in expected_evidence.items():
        if provenance.get(key) != expected:
            raise ValueError(f"model/{key} identity mismatch")

    (model / "README.md").write_text(
        _model_card(
            report,
            allocation,
            verification,
            comparison,
            panel_report,
            panel_control_report,
            panel_verification,
            panel_control_verification,
        )
    )
    _write_sha256s(model)
    manifest = _seal_manifest(model, provenance)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "readme_bytes": (model / "README.md").stat().st_size,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
