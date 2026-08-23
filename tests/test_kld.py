import numpy as np
import pytest

from quant_pipeline.scoring.kld import summarize, token_kld


def test_kld_zero_for_identical_logits_and_positive_otherwise():
    teacher = np.asarray([[1.0, 0.0], [0.2, 0.8]])
    np.testing.assert_allclose(token_kld(teacher, teacher), 0.0, atol=1e-14)
    shifted = np.asarray([[0.0, 1.0], [0.8, 0.2]])
    assert np.all(token_kld(teacher, shifted) > 0)


def test_summary_has_tail_metrics():
    result = summarize(np.arange(100))
    assert result["p95"] > result["p50"]
    assert result["cvar95"] >= result["p95"]


def test_kld_rejects_shape_mismatch_and_non_finite_values():
    with pytest.raises(ValueError):
        token_kld(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        token_kld(np.array([[np.nan, 0.0]]), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        summarize(np.array([1.0, np.inf]))
