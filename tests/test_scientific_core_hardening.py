from __future__ import annotations

import copy
import ast
import hashlib
import importlib.util
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from quant_pipeline.calibration import (
    CalibrationBatch,
    CalibrationFitter,
    RouteMassRow,
    build_route_mass_audit,
    verify_route_mass_audit,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, write_json
from quant_pipeline.normalization import (
    MatrixInput,
    PinnedGSSResult,
    coordinate_grid_scales,
    evaluate_additive_v31_candidate,
    load_absolute_v31_artifact,
    make_gss_receipt,
    make_candidate_evaluation,
    policy_permutations,
    produce_absolute_v31_artifact,
    save_absolute_v31_artifact,
    scale_family_candidates,
    verify_absolute_v31_artifact,
)
from quant_pipeline.normalization.prior_search import normalized_quarter_scales
import quant_pipeline.normalization.artifact_v31 as artifact_v31_module


AUDIT = Path("/home/brandonmusic/KLC_SANDBOXES/quant-research/prior-glm35-hf-audit")
pytestmark = pytest.mark.skipif(
    not AUDIT.is_dir(),
    reason="requires the separately sealed prior-GLM35 local audit oracle",
)
PRIOR_PERMUTATION = AUDIT / "reproducibility/r10/r7_encoder/permutation.py"
PRIOR_SEARCH = AUDIT / "reproducibility/r10/r7_encoder/search.py"
PINNED_CORE = AUDIT / "lineage/encode_tr3_v31.py"
HASH = "a" * 64


def _batch(values):
    rows = len(values)
    return CalibrationBatch(
        expert_inputs=np.asarray(values, dtype=np.float32),
        expert_ids=np.zeros(rows, dtype=np.int64),
        route_weights=np.asarray([0.125, 0.5, 0.875, 0.375][:rows], dtype=np.float32),
        document_ids=[f"d{index}" for index in range(rows)],
        token_offsets=np.arange(rows), layer_id=1, predecessor_checkpoint_hash=HASH,
        projection="gate_up",
    )


def test_codec_second_moment_is_direct_not_covariance_mean_reconstruction():
    values = np.asarray([
        [4096.0, 4096.25], [4095.75, 4096.5], [4096.5, 4095.5], [4095.5, 4096.75]
    ], dtype=np.float32)
    weights = np.asarray([0.125, 0.5, 0.875, 0.375], dtype=np.float64)
    fitter = CalibrationFitter(
        layer_id=1, projection="gate_up", hidden_size=2,
        predecessor_checkpoint_hash=HASH, source_identities={"capture": "b" * 64},
        artifact_dtype="float32",
    )
    fitter.update(_batch(values))
    result = fitter.finalize(0)
    direct = np.einsum("n,ni,nj->ij", weights, values.astype(np.float64), values.astype(np.float64)) / weights.sum()
    np.testing.assert_array_equal(result.array("combined", 1, "second_moment"), direct.astype(np.float32))
    # Simulate the removed path: independently round centered covariance and
    # mean, then reconstruct. It differs by at least one float32 ULP here.
    mean = np.einsum("n,ni->i", weights, values.astype(np.float64)) / weights.sum()
    centered = np.einsum(
        "n,ni,nj->ij", weights, values.astype(np.float64) - mean, values.astype(np.float64) - mean
    ) / weights.sum()
    old = (centered.astype(np.float32).astype(np.float64) + np.outer(mean.astype(np.float32), mean.astype(np.float32))).astype(np.float32)
    assert not np.array_equal(old, result.array("combined", 1, "second_moment"))
    assert set(result.arrays) == {
        f"combined.p{power}.{field}" for power in (0, 1, 2) for field in ("mean", "second_moment")
    }


def _load_prior_permutation_oracle():
    package = types.ModuleType("prior_perm_oracle")
    package.__path__ = []
    constants = types.ModuleType("prior_perm_oracle.constants")
    constants.HAD_K = 4
    constants.INTERMEDIATE_SIZE = 8
    sys.modules[package.__name__] = package
    sys.modules[constants.__name__] = constants
    spec = importlib.util.spec_from_file_location("prior_perm_oracle.permutation", PRIOR_PERMUTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_prior_scale_oracle():
    """Execute the two authoritative pure functions extracted from the sealed file."""

    tree = ast.parse(PRIOR_SEARCH.read_text(encoding="utf-8"), filename=str(PRIOR_SEARCH))
    selected = []
    wanted_functions = {"_normalized_quarter_scales", "_coordinate_grid_scales"}
    wanted_assignment = {"_PER128_SCALE_GRID"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
        elif isinstance(node, ast.Import) and any(alias.name == "math" for alias in node.names):
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "typing":
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignment for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)
    assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == wanted_functions
    namespace: dict[str, object] = {}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(PRIOR_SEARCH), "exec"), namespace)
    return types.SimpleNamespace(
        normalized=namespace["_normalized_quarter_scales"],
        coordinate=namespace["_coordinate_grid_scales"],
        grid=namespace["_PER128_SCALE_GRID"],
    )


def test_all_prior_policies_and_scale_families_match_authoritative_pure_functions():
    oracle = _load_prior_permutation_oracle()
    scale_oracle = _load_prior_scale_oracle()
    generator = np.random.default_rng(20260823)
    diagonals = [
        (9.0, 1.0, 7.0, 3.0, 8.0, 2.0, 6.0, 4.0),
        (0.0,) * 8,
        (4.0, 4.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0),
    ]
    diagonals.extend(tuple(generator.integers(0, 5, size=8).astype(float)) for _ in range(64))
    for diagonal in diagonals:
        actual = policy_permutations(diagonal, block=4)
        expected = {
            "identity": oracle.identity_permutation(8),
            "ldlq_visit_descending_diag": oracle.descending_diag_permutation(diagonal),
            "stored_descending_diag": oracle.stored_descending_diag_permutation(diagonal),
            "energy_balanced": oracle.energy_balanced_permutation(diagonal, block=4),
            "energy_balanced_contiguous": oracle.energy_balanced_permutation(diagonal, block=4, serpentine=False),
        }
        assert actual == expected

    midpoint = float(np.sqrt(0.8 * 0.9))
    scale_inputs = [
        (0.0, 0.0, 0.0, 0.0),
        (1e-30, 1e30, 1.0, 1.0),
        (1.0 / midpoint**4, midpoint**4),
        (0.125, 0.5, 3.0, 17.0),
    ]
    scale_inputs.extend(tuple(np.exp(generator.uniform(-30, 30, size=8))) for _ in range(128))
    assert tuple(scale_oracle.grid) == (0.5, 0.625, 0.8, 0.9, 1.0, 1.1, 1.25, 1.6, 2.0)
    for values in scale_inputs:
        assert normalized_quarter_scales(values) == scale_oracle.normalized(values)
        assert coordinate_grid_scales(values) == scale_oracle.coordinate(values)
        families = scale_family_candidates(values)
        assert tuple(families) == ("identity", "per128-grid", "inverse-per128-grid")
        assert families["per128-grid"] == scale_oracle.coordinate(values)
        assert families["inverse-per128-grid"] == tuple(
            1.0 / float(value) for value in scale_oracle.coordinate(values)
        )


def test_negative_diagonal_is_an_explicit_fail_closed_deviation_from_prior():
    oracle = _load_prior_permutation_oracle()
    diagonal = (4.0, 3.0, 2.0, -1.0, 5.0, 0.0, 1.0, 6.0)
    # The historical helper ranks this input. The production policy family
    # deliberately rejects it because a covariance/Hessian diagonal cannot
    # carry negative energy.
    assert len(oracle.energy_balanced_permutation(diagonal, block=4)) == len(diagonal)
    with pytest.raises(ValueError, match="nonnegative"):
        policy_permutations(diagonal, block=4)


class _GSS:
    def search(self, request):
        scale = 0.75 + request.bits * 0.125
        receipt = make_gss_receipt(
            request, scale=scale, evaluator_code_sha256="1" * 64,
            codec_identity_sha256="2" * 64, search_config_sha256="3" * 64,
            evaluations=9,
        )
        return PinnedGSSResult(scale, receipt)


def _core():
    incumbent = sys.modules.get("exllamav3_ext")
    if incumbent is None:
        sys.modules["exllamav3_ext"] = types.ModuleType("exllamav3_ext")
    spec = importlib.util.spec_from_file_location("hardening_pinned_core", PINNED_CORE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_matrices():
    generator = torch.Generator().manual_seed(9)
    signs8 = torch.tensor([1, -1, 1, 1, -1, 1, -1, 1], dtype=torch.float32)
    signs12 = torch.tensor([1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1], dtype=torch.float32)
    scales8 = (1.0, 1.0)
    scales12 = (1.0, 1.0, 1.0)
    return (
        MatrixInput("E0.gate_proj", "gate_proj", 3, torch.randn(8, 12, generator=generator), signs8, signs12, scales8, scales12, 2.0),
        MatrixInput("E0.up_proj", "up_proj", 3, torch.randn(8, 12, generator=generator), signs8, -signs12, scales8, scales12, 2.0),
        MatrixInput("E0.down_proj", "down_proj", 3, torch.randn(12, 8, generator=generator), signs12, signs8, scales12, scales8, 2.0),
    )


def test_tiny_all_expert_additive_v31_gss_flow_and_receipt_tamper(tmp_path):
    result = evaluate_additive_v31_candidate(
        exact_codec_evaluator=lambda artifact: make_candidate_evaluation(
            artifact, method="exact_codec_proxy", score=0.2, evaluator_sha256="a" * 64
        ),
        heldout_evaluator=lambda artifact: make_candidate_evaluation(
            artifact, method="heldout_full_expert_roundtrip", score=0.3, evaluator_sha256="b" * 64
        ),
        core=_core(), matrices=_tiny_matrices(), producer=_GSS(),
        selected_bits={key: bit for key, bit in zip((item.key for item in _tiny_matrices()), (3, 4, 5))},
        selection_decision_sha256={item.key: "4" * 64 for item in _tiny_matrices()},
        layer_id=0, predecessor_checkpoint_hash=HASH,
        source_identities={"model": "5" * 64}, core_identities={"v31": hashlib.sha256(PINNED_CORE.read_bytes()).hexdigest()},
        codebook_scale=1.24371088, block=4,
    )
    artifact = result.artifact
    assert result.exact_codec_proxy.artifact_sha256 == artifact.content_sha256
    assert result.heldout_full_expert.artifact_sha256 == artifact.content_sha256
    verify_absolute_v31_artifact(artifact)
    assert len(artifact.arrays) == 2 + 3 * 3
    for record in artifact.metadata["matrices"]:
        assert set(record["candidates"]) == {"3", "4", "5"}
    destination = tmp_path / "v31"
    save_absolute_v31_artifact(destination, artifact)
    loaded = load_absolute_v31_artifact(destination)
    assert loaded.content_sha256 == artifact.content_sha256
    tampered = copy.deepcopy(artifact.metadata)
    tampered["matrices"][0]["candidates"]["3"]["gss_receipt"]["evaluations"] += 1
    with pytest.raises(ValueError, match="receipt content hash"):
        verify_absolute_v31_artifact(type(artifact)(tampered, artifact.arrays))


def test_scalar_only_gss_and_raw_search_production_conversion_fail_closed():
    class ScalarProducer:
        def search(self, request):
            return 1.0
    with pytest.raises(TypeError, match="PinnedGSSResult"):
        produce_absolute_v31_artifact(
            core=_core(), matrices=_tiny_matrices(), producer=ScalarProducer(),
            selected_bits={item.key: 3 for item in _tiny_matrices()},
            selection_decision_sha256={item.key: "4" * 64 for item in _tiny_matrices()},
            layer_id=0, predecessor_checkpoint_hash=HASH,
            source_identities={"model": "5" * 64}, core_identities={"v31": "6" * 64},
            codebook_scale=1.24371088, block=4,
        )


def test_exact_route_mass_cold_topup_is_deterministic_and_reconciles():
    natural = [
        RouteMassRow(0, role, "n", offset, 2, "natural")
        for offset, role in enumerate(("gate_up", "down"))
    ]
    supplemental = [
        RouteMassRow(0, role, f"s-{role}-{index}", index, 2 + index, "supplemental", 1, 2)
        for role in ("gate_up", "down") for index in range(3)
    ]
    kwargs = dict(
        natural_rows=natural, supplemental_pool=supplemental, expert_ids=[0],
        unit_denominator=16, cold_expert_min_weight_units=7,
        topup_seed_sha256="7" * 64,
        role_row_identity_sha256={"gate_up": "8" * 64, "down": "9" * 64},
    )
    first = build_route_mass_audit(**kwargs)
    second = build_route_mass_audit(**kwargs)
    reordered = build_route_mass_audit(**(kwargs | {"supplemental_pool": list(reversed(supplemental))}))
    assert first.content_sha256 == second.content_sha256
    assert first.content_sha256 == reordered.content_sha256
    verify_route_mass_audit(first)
    for record in first.metadata["accounting"]:
        assert set(record["powers"]) == {"0", "1", "2"}
    with pytest.raises(ValueError, match="canonically unique"):
        build_route_mass_audit(**(kwargs | {"expert_ids": [0, "00"]}))


def test_route_mass_topup_deficit_noop_insufficient_and_exact_tamper_rejection():
    natural = [
        RouteMassRow(0, role, "natural", offset, 6, "natural")
        for offset, role in enumerate(("gate_up", "down"))
    ]
    supplemental = [
        RouteMassRow(0, role, f"supp-{role}-{index}", index, 2, "supplemental", 1, 2)
        for role in ("gate_up", "down") for index in range(2)
    ]
    kwargs = dict(
        natural_rows=natural,
        supplemental_pool=supplemental,
        expert_ids=[0],
        unit_denominator=16,
        topup_seed_sha256="7" * 64,
        role_row_identity_sha256={"gate_up": "8" * 64, "down": "9" * 64},
    )

    noop = build_route_mass_audit(**(kwargs | {"cold_expert_min_weight_units": 6}))
    assert all(record["deficit_weight_units"] == 0 for record in noop.metadata["topup"])
    assert all(record["selected_rows"] == [] for record in noop.metadata["topup"])
    assert all(
        record["selected_rows_sha256"] == sha256_bytes(canonical_json([]))
        for record in noop.metadata["topup"]
    )
    with pytest.raises(ValueError, match="cannot fill cold expert"):
        build_route_mass_audit(**(kwargs | {"cold_expert_min_weight_units": 11}))

    audit = build_route_mass_audit(**(kwargs | {"cold_expert_min_weight_units": 8}))
    inflated = copy.deepcopy(audit.metadata)
    p1 = inflated["accounting"][0]["powers"]["1"]
    natural_value = Fraction(p1["natural"]["numerator"], p1["natural"]["denominator"]) + 1
    corrected_value = Fraction(
        p1["supplemental_corrected"]["numerator"],
        p1["supplemental_corrected"]["denominator"],
    )
    p1["natural"] = {"numerator": natural_value.numerator, "denominator": natural_value.denominator}
    combined_value = natural_value + corrected_value
    p1["combined"] = {"numerator": combined_value.numerator, "denominator": combined_value.denominator}
    with pytest.raises(ValueError, match="p1 natural"):
        verify_route_mass_audit(type(audit)(inflated))

    inflated_p0 = copy.deepcopy(audit.metadata)
    p0 = inflated_p0["accounting"][0]["powers"]["0"]
    natural_count = Fraction(p0["natural"]["numerator"], p0["natural"]["denominator"]) + 1
    corrected_count = Fraction(
        p0["supplemental_corrected"]["numerator"],
        p0["supplemental_corrected"]["denominator"],
    )
    p0["natural"] = {
        "numerator": natural_count.numerator,
        "denominator": natural_count.denominator,
    }
    combined_count = natural_count + corrected_count
    p0["combined"] = {
        "numerator": combined_count.numerator,
        "denominator": combined_count.denominator,
    }
    with pytest.raises(ValueError, match="p0 natural"):
        verify_route_mass_audit(type(audit)(inflated_p0))


def test_route_mass_selected_row_payload_seal_binds_role_origin_and_identity():
    natural = [
        RouteMassRow(0, role, "natural", offset, 1, "natural")
        for offset, role in enumerate(("gate_up", "down"))
    ]
    supplemental = [
        RouteMassRow(0, role, f"supp-{role}", 0, 4, "supplemental", 1, 2)
        for role in ("gate_up", "down")
    ]
    audit = build_route_mass_audit(
        natural_rows=natural,
        supplemental_pool=supplemental,
        expert_ids=[0],
        unit_denominator=16,
        cold_expert_min_weight_units=4,
        topup_seed_sha256="7" * 64,
        role_row_identity_sha256={"gate_up": "8" * 64, "down": "9" * 64},
    )
    for field, replacement, match in (
        ("origin", "natural", "role/origin binding"),
        ("role", "down", "role/origin binding"),
        ("document_id", "different-document", "identity/payload mismatch"),
    ):
        tampered = copy.deepcopy(audit.metadata)
        topup = next(record for record in tampered["topup"] if record["role"] == "gate_up")
        topup["selected_rows"][0][field] = replacement
        topup["selected_rows_sha256"] = sha256_bytes(canonical_json(topup["selected_rows"]))
        with pytest.raises(ValueError, match=match):
            verify_route_mass_audit(type(audit)(tampered))


def _produce_tiny_artifact():
    return produce_absolute_v31_artifact(
        core=_core(), matrices=_tiny_matrices(), producer=_GSS(),
        selected_bits={item.key: 3 for item in _tiny_matrices()},
        selection_decision_sha256={item.key: "4" * 64 for item in _tiny_matrices()},
        layer_id=0, predecessor_checkpoint_hash=HASH,
        source_identities={"model": "5" * 64}, core_identities={"v31": "6" * 64},
        codebook_scale=1.24371088, block=4,
    )


def test_absolute_v31_verifier_rejects_identity_mass_selection_and_geometry_tamper():
    artifact = _produce_tiny_artifact()

    cases = []
    layer = copy.deepcopy(artifact.metadata)
    layer["identity"]["layer_id"] = True
    cases.append((layer, artifact.arrays, "layer_id"))
    block = copy.deepcopy(artifact.metadata)
    block["identity"]["block"] = 3
    cases.append((block, artifact.arrays, "block is incompatible"))
    mass = copy.deepcopy(artifact.metadata)
    mass["matrices"][0]["mass"] = 0.0
    cases.append((mass, artifact.arrays, "mass must be finite and positive"))
    selection = copy.deepcopy(artifact.metadata)
    selection["matrices"][0]["selection"]["bits"] = 6
    cases.append((selection, artifact.arrays, "selected decision malformed"))

    boolean_scale = copy.deepcopy(artifact.metadata)
    boolean_candidate = boolean_scale["matrices"][0]["candidates"]["3"]
    boolean_candidate["g_scale"] = True
    boolean_candidate["gss_receipt"]["scale"] = True
    boolean_receipt = boolean_candidate["gss_receipt"]
    boolean_receipt.pop("receipt_sha256")
    boolean_receipt["receipt_sha256"] = sha256_bytes(canonical_json(boolean_receipt))
    cases.append((boolean_scale, artifact.arrays, "not boolean"))

    shape = copy.deepcopy(artifact.metadata)
    shape_arrays = dict(artifact.arrays)
    candidate = shape["matrices"][0]["candidates"]["3"]
    name = candidate["private_vector"]["array"]
    shortened = np.ascontiguousarray(np.asarray(shape_arrays[name])[:-1])
    shape_arrays[name] = shortened
    candidate["private_vector"] = {
        "array": name,
        "dtype": str(shortened.dtype),
        "shape": list(shortened.shape),
        "sha256": sha256_bytes(shortened.tobytes(order="C")),
    }
    cases.append((shape, shape_arrays, "shapes are incompatible"))

    for metadata, arrays, match in cases:
        with pytest.raises(ValueError, match=match):
            verify_absolute_v31_artifact(type(artifact)(metadata, arrays))


def test_absolute_v31_save_is_atomic_and_loader_rejects_path_escape(tmp_path, monkeypatch):
    artifact = _produce_tiny_artifact()
    destination = tmp_path / "artifact"
    calls = []
    original_atomic_write = artifact_v31_module.atomic_write

    def recording_atomic_write(path, data):
        calls.append(Path(path).name)
        original_atomic_write(path, data)

    monkeypatch.setattr(artifact_v31_module, "atomic_write", recording_atomic_write)
    save_absolute_v31_artifact(destination, artifact)
    assert calls == [f"array-{index:05d}.npy" for index in range(len(artifact.arrays))]

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_name = next(iter(manifest["arrays"]))
    manifest["arrays"][first_name]["file"] = "../escaped.npy"
    manifest.pop("seal_sha256")
    manifest["seal_sha256"] = sha256_bytes(canonical_json(manifest))
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="filename is not local/canonical"):
        load_absolute_v31_artifact(destination)


def test_absolute_v31_loader_rejects_symlinked_root_and_array(tmp_path):
    artifact = _produce_tiny_artifact()
    destination = tmp_path / "artifact"
    save_absolute_v31_artifact(destination, artifact)

    alias = tmp_path / "artifact-link"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="regular local directory"):
        load_absolute_v31_artifact(alias)

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    filename = next(iter(manifest["arrays"].values()))["file"]
    array_path = destination / filename
    external = tmp_path / filename
    array_path.rename(external)
    array_path.symlink_to(external)
    with pytest.raises(ValueError, match="array file hash mismatch"):
        load_absolute_v31_artifact(destination)
