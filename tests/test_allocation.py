import math

import pytest

from quant_pipeline.allocation.global_dp import Candidate, allocate, allocate_with_fixed_layer_cost, pareto_frontier


def test_global_allocator_can_trade_bytes_across_units():
    candidates = [
        Candidate("a", "a-low", 2, 10.0),
        Candidate("a", "a-high", 6, 0.0),
        Candidate("b", "b-low", 2, 2.0),
        Candidate("b", "b-high", 6, 1.0),
    ]
    result = allocate(candidates, byte_budget=8)
    assert {choice.choice_id for choice in result.choices} == {"a-high", "b-low"}
    assert result.stored_bytes == 8
    assert result.predicted_damage == 2.0


def test_frontier_removes_dominated_choices():
    values = [
        Candidate("x", "one", 1, 3.0),
        Candidate("x", "dominated", 2, 4.0),
        Candidate("x", "two", 3, 1.0),
    ]
    assert [candidate.choice_id for candidate in pareto_frontier(values)] == ["one", "two"]


def test_quantum_never_rounds_individual_candidates():
    candidates = [Candidate("a", "a", 65, 0.0), Candidate("b", "b", 63, 0.0)]
    assert allocate(candidates, 128, quantum=1).stored_bytes == 128
    with pytest.raises(ValueError, match="divisible"):
        allocate(candidates, 128, quantum=128)


def test_fixed_layer_cost_is_charged_before_expert_allocation():
    candidates = [Candidate("a", "a", 6, 0.0), Candidate("b", "b", 6, 0.0)]
    result = allocate_with_fixed_layer_cost(candidates, byte_budget=17, fixed_layer_shared_bytes=5)
    assert result.variable_payload_bytes == 12
    assert result.fixed_layer_shared_bytes == 5
    assert result.stored_bytes == 17
    with pytest.raises(ValueError, match="alone exceeds"):
        allocate_with_fixed_layer_cost(candidates, byte_budget=4, fixed_layer_shared_bytes=5)


@pytest.mark.parametrize(
    "candidate",
    [
        Candidate("", "choice", 128, 1.0),
        Candidate("unit", "", 128, 1.0),
        Candidate("unit", "choice", -128, 1.0),
        Candidate("unit", "choice", 1.5, 1.0),
        Candidate("unit", "choice", 128, math.nan),
        Candidate("unit", "choice", 128, math.inf),
    ],
)
def test_rejects_invalid_candidate(candidate):
    with pytest.raises(ValueError):
        allocate([candidate], byte_budget=128)
