import numpy as np

from quant_pipeline.scoring.attribution import (
    aumann_shapley,
    conditional_quadratic_damage,
    quadratic_expert_attribution,
    reconcile_explicit_remainder,
)


def test_aumann_shapley_closes_quadratic_path():
    deltas = [np.asarray([1.0]), np.asarray([2.0])]

    def gradients(t):
        total = t * 3.0
        return [np.asarray([total]), np.asarray([total])]

    values = aumann_shapley(deltas, gradients, path_nodes=3)
    np.testing.assert_allclose(values, [1.5, 3.0], atol=1e-12)
    assert np.isclose(values.sum(), 0.5 * 3.0**2)


def test_expert_quadratic_attribution_includes_cross_terms():
    z = np.asarray([[1.0, 2.0], [-0.25, 1.0], [2.0, -0.5]])
    shares = quadratic_expert_attribution(z)
    total = np.sum(z, axis=0)
    assert np.isclose(shares.sum(), 0.5 * np.mean(total**2))


def test_reconciliation_does_not_rescale_raw_values():
    accounting = reconcile_explicit_remainder([0.2, -0.1], 0.5)
    np.testing.assert_array_equal(accounting.raw, [0.2, -0.1])
    assert np.isclose(accounting.closure_residual, 0.4)
    assert np.isclose(accounting.reconciled.sum(), 0.5)


def test_conditional_damage_matches_direct_energy_difference():
    residual = np.asarray([1.0, -1.0])
    deltas = np.asarray([[0.5, 0.0], [-1.0, 1.0]])
    predicted = conditional_quadratic_damage(residual, deltas)
    direct = np.asarray([0.5 * np.mean((residual + delta) ** 2 - residual**2) for delta in deltas])
    np.testing.assert_allclose(predicted, direct)

