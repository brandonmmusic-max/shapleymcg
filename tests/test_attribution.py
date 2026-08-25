import numpy as np
import pytest

from quant_pipeline.scoring.attribution import (
    aumann_shapley,
    conditional_quadratic_damage,
    quadratic_expert_attribution,
    reconcile_explicit_remainder,
    split_layer_damage,
    reconcile_layer_components_for_allocation,
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


def test_expert_path_attribution_uses_gauss_legendre_not_uniform_node_weights():
    _nodes, weights = np.polynomial.legendre.leggauss(3)
    weights = weights / 2.0
    # The center node has larger Gauss-Legendre mass. A nonlinear node profile
    # therefore differs materially from a uniform mean.
    z = np.asarray([[[1.0], [9.0], [2.0]], [[0.5], [-1.0], [3.0]]])
    weighted = quadratic_expert_attribution(z, weights)
    uniform = quadratic_expert_attribution(z)
    assert not np.allclose(weighted, uniform)
    total = z.sum(axis=0)
    expected_total = 0.5 * np.sum(weights[:, None] * total**2)
    assert weighted.sum() == pytest.approx(expected_total)


def test_routing_residual_is_a_joint_cross_term_participant():
    split = split_layer_damage(
        6.0,
        np.asarray([[1.0], [1.0]]),
        projected_routing_residual=np.asarray([1.0]),
    )
    np.testing.assert_allclose(split["expert_direct"], [1.5, 1.5])
    assert np.isclose(split["routing_state_shift"], 1.5)
    assert np.isclose(split["raw_total"], 4.5)
    assert np.isclose(split["unresolved_nonlinear_remainder"], 1.5)
    assert np.isclose(split["closed_total"], 6.0)


def test_reconciliation_does_not_rescale_raw_values():
    accounting = reconcile_explicit_remainder([0.2, -0.1], 0.5)
    np.testing.assert_array_equal(accounting.raw, [0.2, -0.1])
    assert np.isclose(accounting.closure_residual, 0.4)
    assert np.isclose(accounting.reconciled.sum(), 0.5)


def test_layer_reconciliation_keeps_signed_components_and_labels_redistribution():
    result = reconcile_layer_components_for_allocation(
        expert_direct=[0.3, -0.1],
        routing_state_shift=0.2,
        explicit_residual=0.1,
        raw_layer_damage=0.5,
        reconciled_layer_damage=0.75,
    )
    np.testing.assert_allclose(result["expert_direct_reconciled"], [0.45, -0.15])
    assert result["routing_state_shift_reconciled"] == pytest.approx(0.3)
    assert result["within_layer_unresolved_remainder_reconciled"] == pytest.approx(0.15)
    assert result["reconciled_component_total"] == pytest.approx(0.75)
    assert sum(result["expert_allocation_score_reconciled"]) == pytest.approx(0.75)
    assert result["expert_redistribution_policy"] == "absolute-direct-magnitude-proportional-v1"


def test_conditional_damage_matches_direct_energy_difference():
    residual = np.asarray([1.0, -1.0])
    deltas = np.asarray([[0.5, 0.0], [-1.0, 1.0]])
    predicted = conditional_quadratic_damage(residual, deltas)
    direct = np.asarray([0.5 * np.mean((residual + delta) ** 2 - residual**2) for delta in deltas])
    np.testing.assert_allclose(predicted, direct)
