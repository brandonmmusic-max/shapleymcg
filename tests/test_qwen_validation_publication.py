from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

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

    causal_verification = verification(causal_report)
    control_verification = verification(control_report)
    panel_comparison = _sealed(
        {
            "causal": {
                "allocation_sha256": causal_allocation,
                "mean_kld": 0.045,
                "report_sha256": causal_report["report_sha256"],
                "top1_agreement": 0.911,
                "verification_sha256": causal_verification["verification_sha256"],
            },
            "historical_control": {
                "allocation_sha256": control_allocation,
                "mean_kld": 0.052,
                "report_sha256": control_report["report_sha256"],
                "top1_agreement": 0.903,
                "verification_sha256": control_verification["verification_sha256"],
            },
            "effect": {
                "relative_kld_reduction": 1.0 - 0.045 / 0.052,
                "top1_agreement_delta": 0.008,
            },
            "protocol": {"attention_backend": "sdpa", "positions": 20480},
            "rate": {
                "k3_matrix_count": 9216,
                "k4_matrix_count": 9216,
                "logical_bpw": 3.5,
            },
        },
        "comparison_sha256",
    )
    paths = [
        _write(tmp_path / "comparison.json", comparison),
        _write(tmp_path / "panel-comparison.json", panel_comparison),
        _write(tmp_path / "causal-report.json", causal_report),
        _write(tmp_path / "control-report.json", control_report),
        _write(tmp_path / "causal-verification.json", causal_verification),
        _write(tmp_path / "control-verification.json", control_verification),
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
    assert loaded[2]["summary"]["count"] == 20480
    assert loaded[4]["ok"] is True

    control_report = json.loads(paths[3].read_text())
    control_report["attention_backend"] = "eager"
    body = {key: value for key, value in control_report.items() if key != "report_sha256"}
    _write(paths[3], _sealed(body, "report_sha256"))
    with pytest.raises(ValueError, match="matched SDPA panel"):
        ASSEMBLER._load_publication_evidence(*paths, allocation, report)


def test_model_card_reports_causal_and_panel_effects(tmp_path: Path) -> None:
    allocation, report, paths = _evidence(tmp_path)
    comparison, panel_comparison, causal, control, causal_verify, control_verify = (
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
        panel_comparison,
        causal,
        control,
        causal_verify,
        control_verify,
    )
    assert "11.2903%" in card
    assert "13.4615%" in card
    assert "Transformers SDPA" in card
    assert "exact expanded BF16 validation model" in card


def test_candidate_resolver_downloads_missing_hash_bound_layers(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {18: b"layer-18", 19: b"layer-19"}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_download(*, filename: str, local_dir: Path, **_: object) -> str:
        layer = int(filename.split("/")[1].split("-")[1])
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[layer])
        return str(path)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    paths, downloaded = ASSEMBLER._resolve_candidate_files(
        layers=[18, 19],
        encode_root=tmp_path / "local",
        expected_sha256={layer: sha256_bytes(data) for layer, data in payloads.items()},
        hf_repo="owner/repo",
        hf_revision="immutable-revision",
        download_root=tmp_path / "downloads",
        workers=2,
    )
    assert [paths[layer].read_bytes() for layer in (18, 19)] == [
        payloads[18],
        payloads[19],
    ]
    assert set(downloaded) == set(paths.values())

    for path in downloaded:
        path.unlink()
    assert not any(path.exists() for path in downloaded)
