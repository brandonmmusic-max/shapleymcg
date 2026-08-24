import json

import numpy as np
import pytest

from quant_pipeline.calibration.fitter import (
    ACCOUNTING_KINDS,
    ROUTE_WEIGHT_POWERS,
    CalibrationBatch,
    CalibrationFitter,
    FittedExpertStatistics,
    SearchScore,
    TransformVectorSearch,
    load_fitted_statistics,
    load_searched_transform_vectors,
    save_fitted_statistics,
    save_searched_transform_vectors,
    verify_searched_transform_vectors,
    verify_fitted_statistics,
)
from quant_pipeline.core.artifacts import sha256_file


PRED = "a" * 64
SOURCES = {
    "model_revision": "1" * 40,
    "dataset_revision": "2" * 40,
    "sealed_corpus_sha256": "b" * 64,
}


def make_fitter(**overrides):
    arguments = {
        "layer_id": 7,
        "projection": "gate",
        "hidden_size": 3,
        "predecessor_checkpoint_hash": PRED,
        "source_identities": SOURCES,
    }
    arguments.update(overrides)
    return CalibrationFitter(**arguments)


def make_batch(
    values,
    route_weights,
    *,
    expert_ids=None,
    documents=None,
    offsets=None,
    origins="natural",
    inclusion=None,
    layer_id=7,
    projection="gate",
    predecessor=PRED,
):
    rows = len(values)
    return CalibrationBatch(
        expert_inputs=np.asarray(values, dtype=np.float32),
        expert_ids=np.asarray(expert_ids if expert_ids is not None else [0] * rows, dtype=np.int64),
        route_weights=np.asarray(route_weights, dtype=np.float32),
        document_ids=documents if documents is not None else [f"doc-{index}" for index in range(rows)],
        token_offsets=np.asarray(offsets if offsets is not None else list(range(rows)), dtype=np.int64),
        layer_id=layer_id,
        predecessor_checkpoint_hash=predecessor,
        projection=projection,
        origins=origins,
        inclusion_probabilities=None if inclusion is None else np.asarray(inclusion, dtype=np.float64),
    )


def weighted_reference(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total == 0:
        return np.zeros(values.shape[1]), np.zeros((values.shape[1], values.shape[1])), np.zeros((values.shape[1], values.shape[1]))
    mean = np.einsum("n,nd->d", weights, values) / total
    second = np.einsum("n,ni,nj->ij", weights, values, values) / total
    return mean, second, second - np.outer(mean, mean)


def test_route_power_statistics_and_supplemental_accounting_match_direct_reference():
    values = np.asarray(
        [[1.0, 0.0, 2.0], [2.0, -1.0, 0.5], [3.0, 1.0, -2.0], [-1.0, 2.0, 1.5]],
        dtype=np.float64,
    )
    route = np.asarray([0.5, 0.25, 0.75, 0.125], dtype=np.float64)
    origins = ["natural", "natural", "supplemental", "supplemental"]
    inclusion = np.asarray([1.0, 1.0, 0.5, 0.25])
    fitter = make_fitter(retained_accounting=ACCOUNTING_KINDS, artifact_dtype="float64")
    fitter.update(make_batch(values, route, origins=origins, inclusion=inclusion))
    result = fitter.finalize(0)

    masks_and_corrections = {
        "natural": (np.array([True, True, False, False]), np.ones(4)),
        "supplemental_raw": (np.array([False, False, True, True]), np.ones(4)),
        "supplemental_corrected": (np.array([False, False, True, True]), 1.0 / inclusion),
        "combined": (np.ones(4, dtype=bool), np.where(np.array(origins) == "supplemental", 1.0 / inclusion, 1.0)),
    }
    for kind, (mask, correction) in masks_and_corrections.items():
        for power in ROUTE_WEIGHT_POWERS:
            expected_weights = route[mask] ** power * correction[mask]
            mean, second, covariance = weighted_reference(values[mask], expected_weights)
            np.testing.assert_allclose(result.array(kind, power, "mean"), mean, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(result.array(kind, power, "second_moment"), second, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(result.array(kind, power, "covariance"), covariance, rtol=1e-13, atol=1e-13)
            record = result.metadata["accounting"][kind]["powers"][str(power)]
            assert record["weight_sum"] == pytest.approx(expected_weights.sum())
            assert record["effective_sample_size"] == pytest.approx(expected_weights.sum() ** 2 / np.dot(expected_weights, expected_weights))

    assert result.metadata["accounting"]["natural"]["powers"]["0"]["count"] == 2
    assert result.metadata["accounting"]["supplemental_raw"]["powers"]["0"]["count"] == 2
    assert result.metadata["accounting"]["combined"]["powers"]["0"]["count"] == 4
    assert result.metadata["estimator"]["combined_accounting"] == "natural_plus_supplemental_corrected"


@pytest.mark.parametrize("projection,hidden", [("gate_up", 4), ("down", 2)])
@pytest.mark.parametrize(
    "values,route,origins,inclusion",
    [
        ([[3, -2, 1, 4]], [0.6], "natural", None),
        ([[2, 2, 2, 2], [2, 2, 2, 2]], [0.2, 0.9], "natural", None),
        ([[1, 4, -2, 3], [5, -1, 2, 7], [2, 3, 6, -4]], [0.25, 0.8, 0.5], "natural", None),
        (
            [[1, -3, 2, 5], [4, 2, -1, 6], [7, 1, 3, -2]],
            [0.4, 0.7, 0.2],
            ["natural", "supplemental", "supplemental"],
            [1.0, 0.5, 0.25],
        ),
    ],
)
def test_dense_hessian_is_exact_uncentered_route_weighted_second_moment(
    projection, hidden, values, route, origins, inclusion
):
    values = np.asarray(values, dtype=np.float64)[:, :hidden]
    # The capture contract is FP32 route weights; reference the exact values
    # accepted by the fitter rather than their decimal source literals.
    route = np.asarray(route, dtype=np.float32).astype(np.float64)
    fitter = make_fitter(
        projection=projection,
        hidden_size=hidden,
        retained_accounting=("combined",),
        artifact_dtype="float64",
    )
    fitter.update(
        make_batch(
            values,
            route,
            origins=origins,
            inclusion=inclusion,
            projection=projection,
        )
    )
    result = fitter.finalize(0)
    correction = np.ones(len(values), dtype=np.float64)
    if inclusion is not None:
        correction = np.where(
            np.asarray(origins) == "supplemental", 1.0 / np.asarray(inclusion), 1.0
        )
    for power in ROUTE_WEIGHT_POWERS:
        weights = route**power * correction
        _, expected, _ = weighted_reference(values, weights)
        np.testing.assert_allclose(
            result.dense_second_moment("combined", power), expected, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            result.dense_hessian("combined", power), expected, rtol=1e-12, atol=1e-12
        )
    if not np.allclose(values.mean(axis=0), 0):
        assert not np.allclose(
            result.dense_hessian("combined", 0),
            result.dense_covariance("combined", 0, regularized=False),
        )


def test_expert_ids_have_canonical_numeric_spelling_and_order():
    fitter = make_fitter(hidden_size=2)
    fitter.update(
        make_batch(
            [[1, 2], [3, 4], [5, 6]],
            [0.2, 0.3, 0.4],
            expert_ids=[10, 2, 1],
        )
    )
    assert fitter.expert_ids == ("1", "2", "10")
    collision = make_fitter(hidden_size=2)
    with pytest.raises(ValueError, match="duplicate"):
        collision.update(
            make_batch(
                [[1, 2], [3, 4]],
                [0.2, 0.3],
                expert_ids=np.asarray(["01", "1"]),
                documents=["same", "same"],
                offsets=[1, 1],
            )
        )


def test_experts_remain_independent_and_unrouted_zero_mass_rows_are_rejected():
    fitter = make_fitter(hidden_size=2, retained_accounting=ACCOUNTING_KINDS, artifact_dtype="float64")
    with pytest.raises(ValueError, match="strictly positive"):
        fitter.update(
            make_batch([[1, 2], [3, 4]], [0.0, 0.5], expert_ids=[0, 0])
        )
    fitter.update(
        make_batch(
            [[1, 2], [100, 200], [3, 4]],
            [0.25, 1.0, 0.5],
            expert_ids=[0, 1, 0],
            documents=["a", "b", "c"],
            offsets=[1, 2, 3],
        )
    )
    zero = fitter.finalize(0)
    one = fitter.finalize(1)
    np.testing.assert_allclose(zero.array("natural", 0, "mean"), [2, 3])
    np.testing.assert_allclose(zero.array("natural", 1, "mean"), [7 / 3, 10 / 3])
    np.testing.assert_allclose(one.array("natural", 0, "mean"), [100, 200])
    assert zero.metadata["accounting"]["natural"]["powers"]["1"]["count"] == 2
    assert zero.metadata["accounting"]["natural"]["powers"]["1"]["weight_sum"] == pytest.approx(0.75)


def test_merge_is_partition_and_order_independent_with_declared_tolerance():
    generator = np.random.default_rng(73)
    values = generator.normal(size=(60, 3)).astype(np.float32)
    route = generator.uniform(0.01, 1.0, size=60).astype(np.float32)
    documents = [f"doc-{index // 3}" for index in range(60)]
    offsets = np.arange(60, dtype=np.int64)

    direct = make_fitter()
    direct.update(make_batch(values, route, documents=documents, offsets=offsets))

    chunks = []
    for indices in np.array_split(np.arange(60), 7):
        part = make_fitter()
        part.update(
            make_batch(
                values[indices], route[indices],
                documents=[documents[index] for index in indices], offsets=offsets[indices],
            )
        )
        chunks.append(part)
    left = make_fitter()
    for part in chunks:
        left.merge(part)
    right = make_fitter()
    for part in reversed(chunks):
        right.merge(part)

    expected = direct.finalize(0)
    for actual in (left.finalize(0), right.finalize(0)):
        assert actual.metadata["estimator"]["merge_comparison_tolerance"] == {"rtol": 1e-12, "atol": 1e-12}
        assert actual.metadata["accounting"]["combined"]["powers"]["0"]["sample_keys_sha256"] == expected.metadata["accounting"]["combined"]["powers"]["0"]["sample_keys_sha256"]
        for name in expected.arrays:
            np.testing.assert_allclose(actual.arrays[name], expected.arrays[name], rtol=1e-12, atol=1e-12)


def test_oas_style_shrinkage_is_expert_local_symmetric_and_positive():
    fitter = make_fitter(
        hidden_size=2, regularization_floor=1e-9,
        retained_accounting=ACCOUNTING_KINDS, artifact_dtype="float64",
    )
    fitter.update(make_batch([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 1, 1]))
    result = fitter.finalize(0)
    record = result.metadata["accounting"]["natural"]["powers"]["0"]
    regularized = result.array("natural", 0, "regularized_covariance")
    assert 0.0 <= record["shrinkage_coefficient"] <= 1.0
    np.testing.assert_allclose(regularized, regularized.T, rtol=0, atol=0)
    assert np.linalg.eigvalsh(regularized).min() > 0.0
    assert record["shrinkage_target"] == "scaled_identity"

    singleton = make_fitter(hidden_size=2, retained_accounting=ACCOUNTING_KINDS, artifact_dtype="float64")
    singleton.update(make_batch([[7, -2]], [1]))
    singleton_result = singleton.finalize(0)
    singleton_record = singleton_result.metadata["accounting"]["natural"]["powers"]["0"]
    assert singleton_record["shrinkage_coefficient"] == 1.0
    np.testing.assert_allclose(
        singleton_result.array("natural", 0, "regularized_covariance"), np.eye(2) * 1e-12
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda batch: batch.__dict__.update(layer_id=8), "layer/projection"),
        (lambda batch: batch.__dict__.update(projection="up"), "layer/projection"),
        (lambda batch: batch.__dict__.update(predecessor_checkpoint_hash="c" * 64), "predecessor"),
        (lambda batch: batch.__dict__.update(expert_inputs=np.ones((2, 4), dtype=np.float32)), "shape"),
        (lambda batch: batch.__dict__.update(route_weights=np.ones(2, dtype=np.float64)), "FP32"),
        (lambda batch: batch.__dict__.update(token_offsets=np.ones(2, dtype=np.float32)), "integer"),
    ],
)
def test_update_rejects_identity_dtype_and_dimension_drift(mutation, match):
    batch = make_batch([[1, 2, 3], [4, 5, 6]], [0.2, 0.3])
    # Frozen dataclass mutation solely constructs malformed external input.
    mutation(batch)
    with pytest.raises((TypeError, ValueError), match=match):
        make_fitter().update(batch)


@pytest.mark.parametrize("field", ["expert_inputs", "route_weights", "inclusion_probabilities"])
def test_update_rejects_nonfinite_values(field):
    inclusion = [1.0, 1.0] if field == "inclusion_probabilities" else None
    batch = make_batch([[1, 2, 3], [4, 5, 6]], [0.2, 0.3], inclusion=inclusion)
    replacement = np.asarray(getattr(batch, field) if getattr(batch, field) is not None else [1.0, 1.0]).copy()
    replacement.reshape(-1)[0] = np.nan
    batch.__dict__[field] = replacement
    with pytest.raises(ValueError, match="finite"):
        make_fitter().update(batch)


def test_supplemental_probability_rules_and_duplicates_fail_closed():
    fitter = make_fitter()
    supplemental = make_batch([[1, 2, 3]], [0.5], origins="supplemental")
    with pytest.raises(ValueError, match="require inclusion"):
        fitter.update(supplemental)
    with pytest.raises(ValueError, match="exactly one"):
        fitter.update(make_batch([[1, 2, 3]], [0.5], inclusion=[0.5]))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        fitter.update(make_batch([[1, 2, 3]], [0.5], origins="supplemental", inclusion=[0.0]))
    with pytest.raises(ValueError, match="overflow"):
        fitter.update(make_batch([[1, 2, 3]], [0.5], origins="supplemental", inclusion=[1e-310]))

    valid = make_batch([[1, 2, 3]], [0.5], documents=["same"], offsets=[4])
    fitter.update(valid)
    with pytest.raises(ValueError, match="duplicate"):
        fitter.update(valid)


def test_supplemental_only_expert_finalizes_with_zero_natural_and_corrected_combined():
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]], dtype=np.float64)
    route = np.asarray([0.25, 0.75], dtype=np.float32)
    inclusion = np.asarray([0.5, 0.25], dtype=np.float64)
    fitter = make_fitter(artifact_dtype="float64")
    fitter.update(
        make_batch(
            values,
            route,
            expert_ids=[7, 7],
            origins=["supplemental", "supplemental"],
            inclusion=inclusion,
        )
    )
    result = fitter.finalize(7)
    assert fitter.expert_ids == ("7",)
    for power in ROUTE_WEIGHT_POWERS:
        natural = result.metadata["accounting"]["natural"]["powers"][str(power)]
        assert natural["count"] == 0
        assert natural["weight_sum"] == 0.0
        corrected_weights = route.astype(np.float64) ** power / inclusion
        expected_mean, expected_second, _ = weighted_reference(values, corrected_weights)
        np.testing.assert_allclose(
            result.array("combined", power, "mean"), expected_mean, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            result.array("combined", power, "second_moment"),
            expected_second,
            rtol=1e-13,
            atol=1e-13,
        )
        corrected = result.metadata["accounting"]["supplemental_corrected"]["powers"][str(power)]
        combined = result.metadata["accounting"]["combined"]["powers"][str(power)]
        assert corrected["weight_sum"] == pytest.approx(combined["weight_sum"])


def test_merge_rejects_identity_drift_and_overlapping_samples_transactionally():
    base = make_fitter()
    base.update(make_batch([[1, 2, 3]], [0.5], documents=["doc"], offsets=[1]))
    before = base.finalize(0)

    drifted = make_fitter(source_identities={**SOURCES, "sealed_corpus_sha256": "c" * 64})
    with pytest.raises(ValueError, match="different identities"):
        base.merge(drifted)

    duplicate = make_fitter()
    duplicate.update(make_batch([[4, 5, 6]], [0.2], documents=["doc"], offsets=[1]))
    with pytest.raises(ValueError, match="duplicate"):
        base.merge(duplicate)
    after = base.finalize(0)
    for name in before.arrays:
        np.testing.assert_array_equal(before.arrays[name], after.arrays[name])


def test_constructor_and_finalize_require_complete_strict_identities():
    with pytest.raises(ValueError, match="source_identities"):
        make_fitter(source_identities={})
    with pytest.raises(ValueError, match="must be strings"):
        make_fitter(source_identities={"model_revision": None})
    with pytest.raises(ValueError, match="64-hex"):
        make_fitter(predecessor_checkpoint_hash="not-a-hash")
    with pytest.raises(ValueError, match="positive"):
        make_fitter(hidden_size=0)
    with pytest.raises(KeyError, match="no observations"):
        make_fitter().finalize(0)


def test_hash_bound_save_load_and_byte_determinism(tmp_path):
    fitter = make_fitter()
    fitter.update(make_batch([[1, 2, 3], [3, 1, 4]], [0.2, 0.9]))
    expected = fitter.finalize(0)
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_fitted_statistics(first, expected)
    fitter.save(second, 0)

    loaded = CalibrationFitter.load(first)
    verify_fitted_statistics(loaded)
    assert loaded.metadata == expected.metadata
    for name in expected.arrays:
        np.testing.assert_array_equal(loaded.arrays[name], expected.arrays[name])

    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest == second_manifest
    for record in first_manifest["arrays"].values():
        assert sha256_file(first / record["file"]) == sha256_file(second / record["file"])


def test_load_rejects_manifest_and_array_tampering(tmp_path):
    fitter = make_fitter()
    fitter.update(make_batch([[1, 2, 3], [3, 2, 1]], [0.2, 0.9]))

    manifest_tamper = tmp_path / "manifest-tamper"
    fitter.save(manifest_tamper, 0)
    manifest = json.loads((manifest_tamper / "manifest.json").read_text())
    manifest["identity"]["layer_id"] = 99
    (manifest_tamper / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seal mismatch"):
        load_fitted_statistics(manifest_tamper)

    array_tamper = tmp_path / "array-tamper"
    fitter.save(array_tamper, 0)
    manifest = json.loads((array_tamper / "manifest.json").read_text())
    array_path = array_tamper / next(iter(manifest["arrays"].values()))["file"]
    data = bytearray(array_path.read_bytes())
    data[-1] ^= 1
    array_path.write_bytes(data)
    with pytest.raises(ValueError, match="array identity mismatch"):
        load_fitted_statistics(array_tamper)


def test_verify_rejects_nonfinite_wrong_dimension_and_unknown_arrays():
    fitter = make_fitter()
    fitter.update(make_batch([[1, 2, 3], [3, 2, 1]], [0.2, 0.9]))
    good = fitter.finalize(0)

    arrays = {name: value.copy() for name, value in good.arrays.items()}
    arrays["combined.p0.mean"][0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        verify_fitted_statistics(FittedExpertStatistics(good.metadata, arrays))

    arrays = {name: value.copy() for name, value in good.arrays.items()}
    arrays["combined.p0.second_moment"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype or dimensions"):
        verify_fitted_statistics(FittedExpertStatistics(good.metadata, arrays))

    arrays = {name: value.copy() for name, value in good.arrays.items()}
    arrays["surprise"] = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError, match="array set"):
        verify_fitted_statistics(FittedExpertStatistics(good.metadata, arrays))

    metadata = json.loads(json.dumps(good.metadata))
    metadata["accounting"]["combined"]["powers"]["0"]["weight_sum"] += 1
    with pytest.raises(ValueError, match="weight_sum mismatch"):
        verify_fitted_statistics(FittedExpertStatistics(metadata, good.arrays))

    metadata = json.loads(json.dumps(good.metadata))
    metadata["estimator"].pop("covariance")
    with pytest.raises(ValueError, match="estimator metadata"):
        verify_fitted_statistics(FittedExpertStatistics(metadata, good.arrays))


def test_save_refuses_nonempty_destination(tmp_path):
    fitter = make_fitter()
    fitter.update(make_batch([[1, 2, 3]], [0.5]))
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "foreign.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="not empty"):
        fitter.save(destination, 0)
    assert (destination / "foreign.txt").read_text() == "preserve me"


def test_finalize_schema_has_all_route_powers_accounting_and_arrays():
    fitter = make_fitter()
    fitter.update(make_batch([[1, 2, 3]], [0.5]))
    result = fitter.finalize(0)
    assert set(result.metadata["accounting"]) == set(ACCOUNTING_KINDS)
    assert set(result.arrays) == {
        f"combined.p{power}.{field}"
        for power in ROUTE_WEIGHT_POWERS
        for field in ("mean", "second_moment")
    }
    assert result.metadata["estimator"]["derived_array_fields"] == [
        "regularized_covariance", "regularized_second_moment"
    ]


def test_production_defaults_have_bounded_qwen_scale_and_no_redundant_matrices():
    gate_up = make_fitter(hidden_size=2048, projection="gate_up")
    down = make_fitter(hidden_size=768, projection="down")
    gate_estimate = gate_up.storage_estimate(expert_count=128, layer_count=48)
    down_estimate = down.storage_estimate(expert_count=128, layer_count=48)

    # Three route arms, one combined covariance each, stored as float32.
    assert gate_estimate["accumulator_bytes_one_layer"] < 14 * 1024**3
    assert down_estimate["accumulator_bytes_one_layer"] < 2 * 1024**3
    assert gate_estimate["artifact_bytes_total"] + down_estimate["artifact_bytes_total"] < 360 * 1024**3
    naive_redundant_bytes = 48 * 128 * 4 * 3 * 3 * 2048 * 2048 * 8
    assert gate_estimate["artifact_bytes_total"] < naive_redundant_bytes // 20


@pytest.mark.parametrize(
    "mode,hidden,block_size,expected_shape",
    [
        ("full", 4, 2, (4, 4)),
        ("block_diagonal", 4, 2, (2, 2, 2)),
        ("diagonal", 4, 2, (4,)),
    ],
)
def test_covariance_geometry_is_explicit_and_dense_materialization_is_correct(mode, hidden, block_size, expected_shape):
    fitter = make_fitter(hidden_size=hidden, covariance_mode=mode, block_size=block_size, artifact_dtype="float64")
    fitter.update(make_batch([[1, 2, 3, 4], [2, 0, 1, -1]], [1, 1]))
    result = fitter.finalize(0)
    covariance = result.array("combined", 0, "covariance")
    assert covariance.shape == expected_shape
    dense = result.dense_covariance("combined", 0, regularized=False)
    assert dense.shape == (hidden, hidden)
    if mode == "diagonal":
        assert np.count_nonzero(dense - np.diag(np.diag(dense))) == 0
    if mode == "block_diagonal":
        assert np.count_nonzero(dense[:2, 2:]) == 0


def test_configurable_retention_streams_scalar_accounting_but_only_persists_requested_arm():
    fitter = make_fitter(retained_accounting=("combined",), retained_powers=(2,))
    fitter.update(make_batch([[1, 2, 3], [2, 1, 0]], [0.25, 0.75]))
    result = fitter.finalize(0)
    assert set(result.arrays) == {"combined.p2.mean", "combined.p2.second_moment"}
    assert result.metadata["accounting"]["natural"]["powers"]["0"]["count"] == 2
    assert result.metadata["accounting"]["combined"]["powers"]["0"]["matrix_retained"] is False
    with pytest.raises(KeyError, match="not retained"):
        result.array("combined", 0, "covariance")


def make_vector_search(**overrides):
    arguments = {
        "layer_id": 7,
        "expert_ids": [0, 1, 2],
        "hidden_size": 256,
        "intermediate_size": 128,
        "predecessor_checkpoint_hash": PRED,
        "source_identities": SOURCES,
        "seed": 20260823,
        "draw_count": 8,
        "block_size": 128,
        "codebook_scale": 1.24371088,
        "objective_arm": "absolute_gate_squared_sse",
        "heldout_artifact_sha256": "e" * 64,
        "reference_baseline_sha256": "7" * 64,
    }
    arguments.update(overrides)
    return TransformVectorSearch(**arguments)


def score(search, method, candidate, center):
    value = float((candidate.draw_index - center) ** 2)
    return search.score_receipt(
        candidate,
        score=value,
        method=method,
        evaluator_code_sha256="c" * 64,
        codec_identity_sha256="d" * 64,
        artifact_sha256="f" * 64 if method == "exact_codec_proxy" else search.heldout_artifact_sha256,
        rows=17,
        coverage={
            "layer_id": search.layer_id,
            "expert_ids": list(search.expert_ids),
            "projection_roles": ["gate", "up", "down"],
            "row_identity_sha256": "9" * 64,
        },
    )


def test_transform_candidates_are_deterministic_multidraw_and_preserve_qwen_sharing():
    first = make_vector_search()
    second = make_vector_search()
    baseline = first.candidate(0)
    same = second.candidate(0)
    alternative = first.candidate(1)
    assert baseline.candidate_id == same.candidate_id
    assert baseline.candidate_id != alternative.candidate_id
    for name in baseline.vectors:
        np.testing.assert_array_equal(baseline.vectors[name], same.vectors[name])
        assert baseline.vectors[name].dtype == np.float16
        expected_magnitude = np.float16(1 / -first.codebook_scale) if name in {"gate_up_input_shared", "down_input"} else np.float16(1)
        assert set(np.unique(np.abs(baseline.vectors[name]))) == {np.abs(expected_magnitude)}

    expert = alternative.vectors_for_expert(1, first.expert_ids)
    np.testing.assert_array_equal(expert["gate.suh"], expert["up.suh"])
    np.testing.assert_array_equal(expert["gate.suh"], expert["up.suh"])
    np.testing.assert_array_equal(expert["gate.svh"], alternative.vectors["gate_output"][1])
    np.testing.assert_array_equal(expert["up.svh"], alternative.vectors["up_output"][1])
    np.testing.assert_array_equal(expert["down.suh"], alternative.vectors["down_input"][1])


def test_transform_proposals_are_independent_of_heldout_and_objective_identity():
    first = make_vector_search(
        objective_arm="absolute_gate_squared_sse", heldout_artifact_sha256="e" * 64
    )
    second = make_vector_search(
        objective_arm="fisher_projection", heldout_artifact_sha256="8" * 64
    )
    assert first.generator_identity == second.generator_identity
    assert first.evaluation_identity != second.evaluation_identity
    assert first.identity != second.identity
    for draw_index in range(first.draw_count):
        assert first.candidate(draw_index).candidate_id == second.candidate(draw_index).candidate_id
        for name in first.candidate(draw_index).vectors:
            np.testing.assert_array_equal(
                first.candidate(draw_index).vectors[name], second.candidate(draw_index).vectors[name]
            )


def test_transform_search_requires_exact_proxy_then_heldout_roundtrip_and_selects_by_roundtrip():
    search = make_vector_search()
    proxy_calls = []
    roundtrip_calls = []

    def proxy(candidate):
        proxy_calls.append(candidate.draw_index)
        return score(search, "exact_codec_proxy", candidate, center=5)

    def roundtrip(candidate):
        roundtrip_calls.append(candidate.draw_index)
        return score(search, "heldout_full_expert_roundtrip", candidate, center=4)

    result = search.run(proxy, roundtrip, shortlist_count=4)
    verify_searched_transform_vectors(result)
    assert proxy_calls == list(range(8))
    assert len(roundtrip_calls) == 4
    assert 0 in roundtrip_calls  # baseline is always confirmed
    assert result.metadata["winner"]["draw_index"] == 4
    assert result.metadata["winner"]["baseline"] is False
    assert result.metadata["winner"]["improves_over_baseline"] is True
    assert result.metadata["policy"]["competitive_search_complete"] is True
    mapped = result.vectors_for_expert(2)
    np.testing.assert_array_equal(mapped["gate.suh"], mapped["up.suh"])


def test_transform_search_baseline_can_win_but_only_after_multidraw_search():
    search = make_vector_search(draw_count=4)
    result = search.run(
        lambda candidate: score(search, "exact_codec_proxy", candidate, center=0),
        lambda candidate: score(search, "heldout_full_expert_roundtrip", candidate, center=0),
        shortlist_count=2,
    )
    assert result.metadata["winner"]["baseline"] is True
    assert result.metadata["winner"]["improves_over_baseline"] is False
    assert len(result.metadata["evaluations"]) == 4


@pytest.mark.parametrize(
    "bad_score,match",
    [
        (SearchScore(np.nan, {"method": "exact_codec_proxy", "receipt_sha256": "a" * 64}), "finite"),
        (SearchScore(1.0, {"method": "fake_proxy", "receipt_sha256": "a" * 64}), "canonical receipt"),
        (SearchScore(1.0, {"method": "exact_codec_proxy", "receipt_sha256": "bad"}), "canonical receipt"),
    ],
)
def test_transform_search_fails_closed_on_invalid_external_evidence(bad_score, match):
    search = make_vector_search(draw_count=3)
    with pytest.raises(ValueError, match=match):
        search.run(
            lambda candidate: bad_score,
            lambda candidate: score(search, "heldout_full_expert_roundtrip", candidate, center=1),
            shortlist_count=2,
        )


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        ("candidate_id", "0" * 64, "candidate/vector"),
        ("score", 9.0, "score mismatch"),
        ("objective_arm", "fisher_projection", "objective"),
        ("rows", 0, "rows"),
        ("receipt_sha256", "0" * 64, "content hash"),
    ],
)
def test_transform_search_receipt_content_is_strictly_recomputed(field, replacement, match):
    search = make_vector_search(draw_count=3)

    def malformed(candidate):
        valid = score(search, "exact_codec_proxy", candidate, center=1)
        evidence = dict(valid.evidence)
        evidence[field] = replacement
        return SearchScore(valid.score, evidence)

    with pytest.raises(ValueError, match=match):
        search.run(
            malformed,
            lambda candidate: score(search, "heldout_full_expert_roundtrip", candidate, center=1),
            shortlist_count=2,
        )


def test_transform_search_tie_shortlist_and_byte_real_toy_codec_are_deterministic():
    search = make_vector_search(draw_count=4)
    source = np.linspace(-2.0, 2.0, 256 * 128, dtype=np.float32).reshape(256, 128)
    packed_hashes = {}

    def byte_real(candidate, method):
        left = candidate.vectors["gate_up_input_shared"].astype(np.float32)[:, None]
        right = candidate.vectors["gate_output"][0].astype(np.float32)[None, :]
        transformed = source * left * right
        scale = max(float(np.max(np.abs(transformed))) / 7.0, 1e-12)
        packed = np.clip(np.rint(transformed / scale), -7, 7).astype(np.int8).tobytes()
        packed_hashes[(method, candidate.draw_index)] = sha256_file_bytes = __import__(
            "hashlib"
        ).sha256(packed).hexdigest()
        reconstructed = np.frombuffer(packed, dtype=np.int8).reshape(source.shape) * scale / left / right
        error = float(np.square(reconstructed - source, dtype=np.float64).mean())
        return search.score_receipt(
            candidate,
            score=error,
            method=method,
            evaluator_code_sha256="3" * 64,
            codec_identity_sha256=sha256_file_bytes,
            artifact_sha256="4" * 64 if method == "exact_codec_proxy" else search.heldout_artifact_sha256,
            rows=source.shape[0],
            coverage={
                "layer_id": search.layer_id,
                "expert_ids": list(search.expert_ids),
                "projection_roles": ["gate", "up", "down"],
                "row_identity_sha256": "5" * 64,
            },
        )

    result = search.run(
        lambda candidate: byte_real(candidate, "exact_codec_proxy"),
        lambda candidate: byte_real(candidate, "heldout_full_expert_roundtrip"),
        shortlist_count=3,
    )
    verify_searched_transform_vectors(result)
    assert len(packed_hashes) == 7
    assert result.metadata["policy"]["production_reference_chain"] == {
        "hessian": "raw_route_weighted_uncentered_second_moment",
        "normalization": "source_derived_absolute_v31",
        "gss": "selected_bit_pinned",
        "artifact_sha256": "7" * 64,
        "relationship": "transform_search_is_additive_and_requires_ablation_against_reference_chain",
    }


def test_transform_search_save_load_is_hash_bound_and_mmap_backed(tmp_path):
    search = make_vector_search(draw_count=4)
    result = search.run(
        lambda candidate: score(search, "exact_codec_proxy", candidate, center=2),
        lambda candidate: score(search, "heldout_full_expert_roundtrip", candidate, center=2),
        shortlist_count=3,
    )
    destination = tmp_path / "vectors"
    save_searched_transform_vectors(destination, result)
    loaded = load_searched_transform_vectors(destination)
    assert loaded.metadata == result.metadata
    assert all(isinstance(array, np.memmap) for array in loaded.arrays.values())
    for name in result.arrays:
        np.testing.assert_array_equal(loaded.arrays[name], result.arrays[name])

    manifest = json.loads((destination / "manifest.json").read_text())
    vector_path = destination / next(iter(manifest["arrays"].values()))["file"]
    raw = bytearray(vector_path.read_bytes())
    raw[-1] ^= 1
    vector_path.write_bytes(raw)
    with pytest.raises(ValueError, match="array identity mismatch"):
        load_searched_transform_vectors(destination)


def test_transform_search_qwen_storage_estimate_is_small_and_dimensions_are_explicit():
    search = make_vector_search(
        expert_ids=list(range(128)), hidden_size=2048, intermediate_size=768, draw_count=16
    )
    estimate = search.storage_estimate()
    assert estimate == {
        "generated_vector_bytes_per_draw": 598016,
        "generated_block_scale_working_bytes_per_draw": 18688,
        "peak_generator_bytes_upper_bound": 10203648,
        "peak_search_vector_bytes_upper_bound": 10801664,
        "persisted_winner_vector_bytes": 598016,
        "candidate_retention_policy": 0,
        "draw_count": 16,
    }
    with pytest.raises(ValueError, match="divisible"):
        make_vector_search(intermediate_size=192)
