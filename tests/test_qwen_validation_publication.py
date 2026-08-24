from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assemble_qwen_validation_model",
    ROOT / "scripts" / "assemble_qwen_validation_model.py",
)
assert SPEC is not None and SPEC.loader is not None
ASSEMBLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLER)


def _sealed(body: dict, field: str) -> dict:
    result = dict(body)
    result[field] = sha256_bytes(canonical_json(body))
    return result


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _evidence(tmp_path: Path) -> tuple[dict, dict, list[Path]]:
    causal_allocation = "causal-allocation"
    control_allocation = "control-allocation"
    panel = "sealed-panel"
    teachers = ["teacher-0", "teacher-1"]
    comparison = _sealed(
        {
            "causal": {
                "allocation_sha256": causal_allocation,
                "mean_kld": 0.055,
                "report_sha256": "single-report",
                "top1_agreement": 0.912,
            },
            "historical_control": {
                "allocation_sha256": control_allocation,
                "mean_kld": 0.062,
                "top1_agreement": 0.901,
            },
            "effect": {
                "relative_kld_reduction": 1.0 - 0.055 / 0.062,
                "top1_agreement_delta": 0.011,
                "changed_matrix_choices": 9392,
            },
            "protocol": {"attention_backend": "sdpa"},
            "rate": {
                "k3_matrix_count": 9216,
                "k4_matrix_count": 9216,
                "logical_bpw": 3.5,
            },
        },
        "comparison_sha256",
    )

    def report(allocation: str, mean: float, top1: float) -> dict:
        return _sealed(
            {
                "allocation_sha256": allocation,
                "attention_backend": "sdpa",
                "panel_sha256": panel,
                "teacher_files": teachers,
                "summary": {"count": 20480, "mean": mean},
                "top1_agreement": top1,
            },
            "report_sha256",
        )

    causal_report = report(causal_allocation, 0.045, 0.911)
    control_report = report(control_allocation, 0.052, 0.903)

    def verification(kld_report: dict) -> dict:
        return _sealed(
            {
                "attention_backend": "sdpa",
                "max_absolute_delta": 0.0,
                "ok": True,
                "panel_sha256": panel,
                "positions": 20480,
                "report_sha256": kld_report["report_sha256"],
            },
            "verification_sha256",
        )

    paths = [
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "causal-report.json", causal_report),
        _write(tmp_path / "control-report.json", control_report),
        _write(tmp_path / "causal-verification.json", verification(causal_report)),
        _write(tmp_path / "control-verification.json", verification(control_report)),
    ]
    report = {
        "report_sha256": "single-report",
        "summary": {"mean": 0.055, "p50": 0.01, "p95": 0.2, "max": 2.0},
        "top1_agreement": 0.912,
    }
    return {"allocation_sha256": causal_allocation}, report, paths


def test_publication_evidence_requires_matched_sealed_sdpa_panel(tmp_path: Path) -> None:
    allocation, report, paths = _evidence(tmp_path)
    loaded = ASSEMBLER._load_publication_evidence(*paths, allocation, report)
    assert loaded[1]["summary"]["count"] == 20480
    assert loaded[3]["ok"] is True

    control_report = json.loads(paths[2].read_text())
    control_report["attention_backend"] = "eager"
    body = {key: value for key, value in control_report.items() if key != "report_sha256"}
    _write(paths[2], _sealed(body, "report_sha256"))
    with pytest.raises(ValueError, match="matched SDPA panel"):
        ASSEMBLER._load_publication_evidence(*paths, allocation, report)


def test_model_card_reports_causal_and_panel_effects(tmp_path: Path) -> None:
    allocation, report, paths = _evidence(tmp_path)
    comparison, causal, control, causal_verify, control_verify = (
        ASSEMBLER._load_publication_evidence(*paths, allocation, report)
    )
    card = ASSEMBLER._model_card(
        report,
        {
            "allocation_sha256": "causal-allocation",
            "k3_count": 9216,
            "k4_count": 9216,
        },
        {
            "verification_sha256": "single-verification",
            "max_token_kld_difference": 9e-13,
        },
        comparison,
        causal,
        control,
        causal_verify,
        control_verify,
    )
    assert "11.2903%" in card
    assert "13.4615%" in card
    assert "Transformers SDPA" in card
    assert "exact expanded BF16 validation model" in card
