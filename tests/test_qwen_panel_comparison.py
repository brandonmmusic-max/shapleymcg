from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_qwen_mcg_panel_allocations",
    ROOT / "scripts" / "compare_qwen_mcg_panel_allocations.py",
)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def _sealed(body: dict, field: str) -> dict:
    value = dict(body)
    value[field] = sha256_bytes(canonical_json(body))
    return value


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _allocation(reverse: bool) -> dict:
    choices = []
    for index in range(18432):
        bit = 3 if index < 9216 else 4
        if reverse:
            bit = 7 - bit
        choices.append(
            {
                "layer": index // (128 * 3),
                "expert": (index // 3) % 128,
                "projection": ("gate_proj", "up_proj", "down_proj")[index % 3],
                "bits": bit,
            }
        )
    return _sealed(
        {
            "average_weight_bits": 3.5,
            "choices": choices,
            "k3_count": 9216,
            "k4_count": 9216,
        },
        "allocation_sha256",
    )


def _result(root: Path, allocation_sha: str, mean: float, top1: float) -> None:
    report = _sealed(
        {
            "allocation_sha256": allocation_sha,
            "attention_backend": "sdpa",
            "candidate_inventory_sha256": "inventory",
            "model_revision": "revision",
            "panel_sha256": "panel",
            "summary": {"count": 20480, "mean": mean},
            "teacher_files": [{"sha256": "teacher"}],
            "top1_agreement": top1,
        },
        "report_sha256",
    )
    verification = _sealed(
        {
            "attention_backend": "sdpa",
            "max_absolute_delta": 0.0,
            "ok": True,
            "panel_sha256": "panel",
            "positions": 20480,
            "report_sha256": report["report_sha256"],
        },
        "verification_sha256",
    )
    _write(root / "kld-report.json", report)
    _write(root / "independent-verification.json", verification)


def test_seals_matched_panel_comparison(tmp_path: Path, monkeypatch) -> None:
    causal_allocation = _allocation(False)
    control_allocation = _allocation(True)
    causal_allocation_path = _write(tmp_path / "causal-allocation.json", causal_allocation)
    control_allocation_path = _write(tmp_path / "control-allocation.json", control_allocation)
    causal_root = tmp_path / "causal"
    control_root = tmp_path / "control"
    _result(causal_root, causal_allocation["allocation_sha256"], 0.045, 0.91)
    _result(control_root, control_allocation["allocation_sha256"], 0.05, 0.90)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_qwen_mcg_panel_allocations.py",
            "--causal-result",
            str(causal_root),
            "--control-result",
            str(control_root),
            "--causal-allocation",
            str(causal_allocation_path),
            "--control-allocation",
            str(control_allocation_path),
            "--output",
            str(output),
        ],
    )
    assert COMPARE.main() == 0
    result = json.loads(output.read_text())
    COMPARE._verify(result, "comparison_sha256", "comparison")
    assert result["effect"]["relative_kld_reduction"] == pytest.approx(0.1)
    assert result["effect"]["changed_matrix_choices"] == 18432
    assert result["protocol"]["positions"] == 20480
