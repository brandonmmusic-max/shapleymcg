import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_turboderp_allocation_proof",
    ROOT / "scripts" / "compare_turboderp_allocation_proof.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _seal(value, field):
    value[field] = sha256_bytes(canonical_json(value))
    return value


def _arm(root: Path, allocation: str, values: np.ndarray, top1: float):
    target = root / "turboderp-selected-k34"
    target.mkdir(parents=True)
    np.save(target / "token-kld.npy", values, allow_pickle=False)
    report = _seal(
        {
            "schema": "fixture",
            "arm": "turboderp-selected-k34",
            "metric": "KL(BF16 || quantized)",
            "summary": {"count": values.size, "mean": float(values.mean())},
            "top1_agreement": top1,
            "token_kld_sha256": sha256_file(target / "token-kld.npy"),
            "teacher_files": ["a" * 64, "b" * 64],
        },
        "report_sha256",
    )
    (target / "kld-report.json").write_text(json.dumps(report))
    summary = _seal(
        {
            "schema": "fixture",
            "allocation_sha256": allocation,
            "panel_sha256": "c" * 64,
            "expert_rate": {"k3_matrices": 2, "k4_matrices": 2},
            "fixed_nonexpert_scope": {"revision": "d" * 40},
            "turboderp_k3_revision": "e" * 40,
            "turboderp_k4_revision": "f" * 40,
            "turboderp_k3_scope": {"bits": 3},
            "turboderp_k4_scope": {"bits": 4},
            "arms": {
                "turboderp-selected-k34": {
                    "mean_kld": float(values.mean()),
                    "top1_agreement": top1,
                    "report_sha256": report["report_sha256"],
                }
            },
        },
        "summary_sha256",
    )
    (root / "summary.json").write_text(json.dumps(summary))


def test_paired_allocation_proof_requires_all_rows_and_positive_interval(tmp_path):
    stock = tmp_path / "stock"
    causal = tmp_path / "causal"
    _arm(stock, "1" * 64, np.full((10, 4), 0.04), 0.90)
    _arm(causal, "2" * 64, np.full((10, 4), 0.03), 0.92)
    result = MODULE.compare(stock, causal, seed=7, bootstrap_draws=10_000)
    assert result["effect"]["relative_mean_kld_reduction"] == pytest.approx(0.25)
    assert result["effect"]["rows_shapleymcg_better"] == 10
    assert result["effect"]["row_block_bootstrap"]["absolute_reduction_interval"] == pytest.approx([0.01, 0.01])
    assert result["interpretation"]["superiority_gate_passed"] is True
