import hashlib
import json
import math
from dataclasses import replace

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from quant_pipeline.candidates.ledger import (
    BackendAttestation,
    CandidateJournal,
    CandidateLedgerGenerator,
    ConditionalDownFitBatch,
    ExpertCandidateInput,
    FittedProjection,
    K5Decision,
    MCGTransformArtifact,
    ProjectionTensors,
    RoutedExpertBatch,
    SCHEMA_ATTESTATION,
    all_k3_k4_triplets,
    all_k5_triplets,
    admit_k5,
    allocate_validated_records,
    allocator_handoff,
    build_expert_candidate_input,
    build_pareto_frontiers,
    reject_all_k5,
    selected_allocation_cost,
    expected_expert_inventory,
    routed_batch_sha256,
    conditional_down_fit_batch_sha256,
    validate_candidate_record,
    validate_ledger,
)
from quant_pipeline.codecs.protocols import CodecCandidate
from quant_pipeline.calibration.fitter import CalibrationBatch, CalibrationFitter
from quant_pipeline.candidates.payload_store import ExactPayloadStore
from quant_pipeline.campaign.qwen_work_units import _score_units


def _sha(tensor):
    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class DeterministicAttestedCodec:
    name = "exl3-mcg-corrected"
    codebook_scale = 1.125

    def __init__(self):
        self.calls = 0
        self.call_inputs = []

    def encode_candidates(self, *, unit_id, weight_hf, covariance, bits, input_vector, output_vector, provenance=None):
        self.calls += 1
        self.call_inputs.append((unit_id, tuple(bits), torch.as_tensor(covariance).clone()))
        result = {}
        for bit in bits:
            scale = {3: 0.91, 4: 0.96, 5: 0.985}[bit]
            reconstructed = (weight_hf.float() * scale).half()
            packed = torch.tensor([bit, len(unit_id) % 251, weight_hf.shape[0] % 251], dtype=torch.uint8)
            result[bit] = CodecCandidate(
                bits=bit,
                packed=packed,
                reconstructed=reconstructed,
                stored_bytes=packed.numel() + input_vector.numel() * 2 + output_vector.numel() * 2,
                packed_sha256=_sha(packed),
                reconstruction_sha256=_sha(reconstructed.T.contiguous().half()),
                metadata=dict(provenance or {}) | {"oracle": "deterministic-test-exact-codec"},
            )
        return result


class GenericCodec(DeterministicAttestedCodec):
    name = "uniform-symmetric-reference"


class CovarianceSensitiveCodec(DeterministicAttestedCodec):
    def encode_candidates(self, **kwargs):
        result = super().encode_candidates(**kwargs)
        covariance_sha = _sha(torch.as_tensor(kwargs["covariance"]))
        for bit, candidate in list(result.items()):
            packed = torch.tensor(list(bytes.fromhex(covariance_sha[:32])) + [bit], dtype=torch.uint8)
            result[bit] = replace(
                candidate,
                packed=packed,
                stored_bytes=packed.numel() + kwargs["input_vector"].numel() * 2 + kwargs["output_vector"].numel() * 2,
                packed_sha256=_sha(packed),
            )
        return result


def _attestation(codec, *, backend="corrected-exl3-mcg-r10"):
    identity = {
        "numeric_core_sha256": "1" * 64,
        "extension_sha256": "2" * 64,
        "python_closure_sha256": {"r10_codec.py": "3" * 64},
    }
    from quant_pipeline.core.artifacts import canonical_json

    digest = hashlib.sha256(canonical_json(identity)).hexdigest()
    return BackendAttestation(
        schema=SCHEMA_ATTESTATION,
        backend=backend,
        codec_name=codec.name,
        codec_identity=identity,
        codec_identity_sha256=digest,
        test_only=True,
    )


def _run_identity(attestation):
    from quant_pipeline.core.artifacts import canonical_json

    inventory = expected_expert_inventory([(2, 9)], profile="tiny-ledger-fixture", test_fixture=True)
    return {
        "model_revision": "a" * 40,
        "dataset_revision": "b" * 40,
        "fit_artifact_sha256": "c" * 64,
        "heldout_artifact_sha256": "d" * 64,
        "predecessor_state_sha256": "e" * 64,
        "codec_attestation_sha256": hashlib.sha256(canonical_json(attestation.as_dict())).hexdigest(),
        "search_artifact_sha256": "6" * 64,
        "capture_artifact_sha256": "7" * 64,
        "conditional_down_fit_artifact_sha256": "f" * 64,
        "fisher_probe_sha256": "8" * 64,
        "fisher_window_sha256": "9" * 64,
        "expected_inventory": inventory,
        "expected_inventory_sha256": inventory["inventory_sha256"],
    }


def _expert(k5=None, fisher=True):
    generator = torch.Generator().manual_seed(17)
    hidden, intermediate, rows, top_k, rank = 4, 6, 7, 2, 3
    gate = torch.randn(intermediate, hidden, generator=generator).bfloat16()
    up = torch.randn(intermediate, hidden, generator=generator).bfloat16()
    down = torch.randn(hidden, intermediate, generator=generator).bfloat16()
    fitted = {}
    for name, weight in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        n, k = weight.shape
        fitted[name] = FittedProjection(
            covariance=torch.eye(k),
            input_vector=torch.ones(k, dtype=torch.float16),
            output_vector=torch.ones(n, dtype=torch.float16),
            fit_identity={
                "fit_artifact_sha256": "c" * 64,
                "model_revision": "a" * 40,
                "dataset_revision": "b" * 40,
                "predecessor_state_sha256": "e" * 64,
                "route_weight_power": 2,
                "projection": name,
            },
            transform_identity={
                "hadamard": "deterministic-H128",
                "search_artifact_sha256": "6" * 64,
                "projection": name,
            },
        )
    route_weights = torch.linspace(0.2, 0.8, rows)
    source_routes = torch.tensor([[9, 1]] * rows)
    candidate_routes = source_routes.clone()
    candidate_routes[-1] = torch.tensor([9, 2])
    route_mass = torch.stack((route_weights, 1.0 - route_weights), dim=1)
    batch = RoutedExpertBatch(
        batch_id="selection-doc-7",
        hidden_states=torch.randn(rows, hidden, generator=generator),
        route_weights=route_weights,
        source_route_indices=source_routes,
        source_route_weights=route_mass,
        candidate_route_indices=candidate_routes,
        candidate_route_weights=route_mass,
        fisher_gradients=torch.randn(rank, rows, hidden, generator=generator) if fisher else None,
        identity={
            "role": "selection",
            "document_sha256": "5" * 64,
            "heldout_artifact_sha256": "d" * 64,
            "capture_artifact_sha256": "7" * 64,
            "fisher_probe_sha256": "8" * 64,
            "fisher_window_sha256": "9" * 64,
            "layer": 2,
            "expert": 9,
        },
        row_keys=[f"heldout-row-{index}" for index in range(rows)],
    )
    batch = replace(batch, identity=dict(batch.identity) | {"batch_payload_sha256": routed_batch_sha256(batch)})
    fit_batch = ConditionalDownFitBatch(
        batch_id="conditional-fit-doc-3",
        hidden_states=torch.randn(rows + 2, hidden, generator=generator),
        route_weights=torch.linspace(0.1, 0.9, rows + 2),
        sampling_weights=torch.ones(rows + 2),
        source_route_indices=torch.tensor([[9, 1]] * (rows + 2)),
        source_route_weights=torch.stack(
            (torch.linspace(0.1, 0.9, rows + 2), 1.0 - torch.linspace(0.1, 0.9, rows + 2)),
            dim=1,
        ),
        identity={
            "role": "conditional_fit",
            "conditional_down_fit_artifact_sha256": "f" * 64,
            "row_identity_sha256": "a" * 64,
            "document_sha256": "6" * 64,
            "layer": 2,
            "expert": 9,
        },
        row_keys=[f"fit-row-{index}" for index in range(rows + 2)],
    )
    fit_batch = replace(
        fit_batch,
        identity=dict(fit_batch.identity) | {"batch_payload_sha256": conditional_down_fit_batch_sha256(fit_batch)},
    )
    return ExpertCandidateInput(
        layer=2,
        expert=9,
        source=ProjectionTensors(gate, up, down),
        fitted=fitted,
        heldout_batches=[batch],
        k5_screen=reject_all_k5("K5 reserved unless K3/K4 frontier tail gate fails") if k5 is None else k5,
        conditional_down_fit_batches=[fit_batch],
    )


def test_triplet_design_is_exact_and_k5_screen_is_complete():
    assert len(all_k3_k4_triplets()) == 8
    assert set(all_k3_k4_triplets()) == set(__import__("itertools").product((3, 4), repeat=3))
    assert len(all_k5_triplets()) == 19
    assert all(5 in triplet for triplet in all_k5_triplets())
    assert len(reject_all_k5("screened out")) == 19


def test_competitive_mode_rejects_generic_and_unattested_backends():
    generic = GenericCodec()
    with pytest.raises(ValueError, match="corrected EXL3/MCG"):
        CandidateLedgerGenerator(generic, _attestation(generic, backend="uniform-reference"), allow_test_backend=True)
    corrected = DeterministicAttestedCodec()
    with pytest.raises(ValueError, match="test-only"):
        CandidateLedgerGenerator(corrected, _attestation(corrected))
    with pytest.raises(ValueError, match="BF16 arithmetic"):
        CandidateLedgerGenerator(
            corrected,
            _attestation(corrected),
            allow_test_backend=True,
            expert_compute_dtype="float32",
        )


def test_exact_candidate_generation_metrics_frontier_and_handoff(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    generator = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True)
    ledger = generator.generate([_expert()], journal=journal, output_path=tmp_path / "ledger.json")
    assert len(ledger["candidates"]) == 8
    assert {tuple(row["bit_triplet"]) for row in ledger["candidates"]} == set(all_k3_k4_triplets())
    assert codec.calls == 6  # gate/up plus four decoded-(gate,up)-conditional down calls
    for row in ledger["candidates"]:
        validate_candidate_record(row, allow_test_backend=True)
        assert row["finite_validation"] is True
        assert row["payload_bytes"] > 0
        assert row["shared_layer_payload_bytes"] > 0
        assert row["physical_payload_accounting"]["artifact_physical_bytes"] >= row["payload_bytes"]
        assert row["metrics"]["absolute_gate_squared_output_sse"] >= 0
        assert row["metrics"]["relative_output_sse"] >= 0
        assert row["metrics"]["energy_normalized_output_sse"] >= 0
        assert math.isfinite(row["metrics"]["signed_aggregate_error"])
        assert math.isfinite(row["metrics"]["interaction_term"])
        assert row["metrics"]["route_agreement"]["route_set_agreement"] < 1
        assert row["metrics"]["fisher_projection"]["rank"] == 3
    assert len(allocator_handoff(ledger["candidates"], allow_test_backend=True)) == 8
    chosen = selected_allocation_cost([ledger["candidates"][0]])
    assert chosen["allocated_payload_bytes"] == (
        ledger["candidates"][0]["payload_bytes"] + ledger["candidates"][0]["shared_layer_payload_bytes"]
    )
    assert build_pareto_frontiers(ledger["candidates"], allow_test_backend=True) == ledger["pareto_frontiers"]
    validate_ledger(json.loads((tmp_path / "ledger.json").read_text()), allow_test_backend=True)
    with pytest.raises(ValueError, match="test-only backend"):
        validate_ledger(ledger)
    # 19 explicit K5 screening records plus eight candidates.
    assert len(journal.inventory()) == 27


def test_multilayer_allocator_selected_cost_reconciliation(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    )

    def rebind_layer(raw, layer):
        row = json.loads(json.dumps(raw))
        old_unit = row["unit_id"]
        row["layer"] = layer
        row["unit_id"] = f"L{layer}.E{row['expert']}"
        row["candidate_id"] = row["candidate_id"].replace(old_unit, row["unit_id"], 1)
        row["scoring_inputs"]["unit_id"] = row["unit_id"]
        from quant_pipeline.core.artifacts import canonical_json

        row["scoring_inputs_sha256"] = hashlib.sha256(canonical_json(row["scoring_inputs"])).hexdigest()
        row["input_sha256"] = hashlib.sha256(
            canonical_json(
                {
                    "base_sha256": row["scoring_inputs_sha256"],
                    "triplet": tuple(row["bit_triplet"]),
                    "k5_admission": row["k5_admission"],
                }
            )
        ).hexdigest()
        _rehash_record(row)
        return row

    records = list(ledger["candidates"]) + [rebind_layer(row, 3) for row in ledger["candidates"]]
    expected_fixed = 2 * ledger["fixed_layer_shared_payload_bytes"]
    result = allocate_validated_records(
        records,
        byte_budget=10**9,
        competitive=True,
        allow_test_backend=True,
    )
    assert len(result.selected_records) == 2
    assert {row["layer"] for row in result.selected_records} == {2, 3}
    assert result.allocation.fixed_layer_shared_bytes == expected_fixed
    assert result.allocation.stored_bytes == result.selected_cost["allocated_payload_bytes"]
    assert sum(row["allocated_payload_bytes"] for row in result.selected_cost["selected_layer_costs"]) == result.allocation.stored_bytes


def test_explicit_k5_admission_adds_only_admitted_candidates(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    screen = reject_all_k5("screened out")
    admitted = (5, 4, 3)
    screen[admitted] = admit_k5(
        "confirmation-tail rescue",
        selection_artifact_sha256="d" * 64,
        confirmation_artifact_sha256="1" * 64,
        p99_relative_output_sse_delta=-0.02,
        mean_relative_output_sse_delta=-0.01,
        max_p99_relative_output_sse_delta=0.0,
        max_mean_relative_output_sse_delta=0.0,
    )
    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert(screen)], journal=journal
    )
    assert len(ledger["candidates"]) == 9
    row = next(value for value in ledger["candidates"] if tuple(value["bit_triplet"]) == admitted)
    assert row["k5_admission"] == screen[admitted].as_dict()
    assert row["rate_class"] == "K5-screened-admission"


def test_k5_reason_only_admission_is_rejected(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    screen = reject_all_k5("screened out")
    screen[(5, 4, 3)] = K5Decision(True, "unsupported reason only")
    with pytest.raises(ValueError, match="versioned rule evidence"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [_expert(screen)], journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation))
        )


def test_resume_reuses_atomic_candidate_records_without_codec_work(tmp_path):
    first_codec = DeterministicAttestedCodec()
    attestation = _attestation(first_codec)
    identity = _run_identity(attestation)
    first = CandidateJournal(tmp_path / "journal", identity)
    ledger_one = CandidateLedgerGenerator(first_codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=first
    )
    assert first_codec.calls == 6
    second_codec = DeterministicAttestedCodec()
    resumed = CandidateJournal(tmp_path / "journal", identity, resume=True)
    ledger_two = CandidateLedgerGenerator(second_codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=resumed
    )
    assert second_codec.calls == 0
    assert ledger_two == ledger_one


def test_resume_and_record_validation_fail_closed_on_drift_or_nonfinite(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    identity = _run_identity(attestation)
    journal = CandidateJournal(tmp_path / "journal", identity)
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=journal
    )
    drift = dict(identity)
    drift["predecessor_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="resume identity drift"):
        CandidateJournal(tmp_path / "journal", drift, resume=True)
    malformed = json.loads(json.dumps(ledger["candidates"][0]))
    malformed["metrics"]["signed_aggregate_error"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_candidate_record(malformed, allow_test_backend=True)


def test_fisher_objective_requires_fisher_inputs(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    generator = CandidateLedgerGenerator(
        codec,
        attestation,
        objective_arm="fisher_projection",
        allow_test_backend=True,
    )
    with pytest.raises(ValueError, match="requires Fisher"):
        generator.generate([_expert(fisher=False)], journal=journal)


def test_experts_and_heldout_batches_can_stream_without_model_wide_materialization(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    batches = list(base.heldout_batches)
    opens = []

    def batch_source():
        opens.append(len(opens))
        yield from batches

    streamed = replace(base, heldout_batches=batch_source)
    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        (item for item in [streamed]), journal=journal
    )
    assert len(ledger["candidates"]) == 8
    # Validation, identity capture, then one bounded pass for each triplet.
    assert len(opens) == 11
    assert codec.calls == 6


def test_streamed_batch_source_drift_is_detected(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    batches = list(base.heldout_batches)
    invocation = 0

    def drifting_source():
        nonlocal invocation
        invocation += 1
        batch = batches[0]
        if invocation >= 3:
            batch = replace(batch, hidden_states=batch.hidden_states + 0.5)
        yield batch

    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    with pytest.raises(ValueError, match="sealed payload identity"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, heldout_batches=drifting_source)], journal=journal
        )


def _fit_bridge_input(projection, dimension, *, mode="full"):
    fitter = CalibrationFitter(
        layer_id=2,
        projection=projection,
        hidden_size=dimension,
        predecessor_checkpoint_hash="e" * 64,
        source_identities={"model_revision": "a" * 40, "dataset_revision": "b" * 40},
        covariance_mode=mode,
        retained_accounting=("combined",),
        retained_powers=(0, 1, 2),
    )
    values = np.zeros((4, dimension), dtype=np.float32)
    values[:, :4] = np.asarray([[1, 0, 0, 1], [0, 2, 1, 0], [1, 1, 3, 0], [2, 1, 0, 4]], dtype=np.float32)
    fitter.update(
        CalibrationBatch(
            expert_inputs=values,
            expert_ids=np.asarray([9, 9, 9, 9], dtype=np.int64),
            route_weights=np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            document_ids=[f"doc-{index}" for index in range(4)],
            token_offsets=np.arange(4, dtype=np.int64),
            layer_id=2,
            predecessor_checkpoint_hash="e" * 64,
            projection=projection,
        )
    )
    return fitter.finalize(9)


def test_fitter_bridge_builds_covariance_and_exact_mcg_vectors_without_manual_assembly():
    generator = torch.Generator().manual_seed(91)
    source = ProjectionTensors(
        torch.randn(128, 128, generator=generator).bfloat16(),
        torch.randn(128, 128, generator=generator).bfloat16(),
        torch.randn(128, 128, generator=generator).bfloat16(),
    )
    gate_up = _fit_bridge_input("gate_up_input", 128)
    down = _fit_bridge_input("down_input", 128)
    routed = RoutedExpertBatch(
        batch_id="heldout",
        hidden_states=torch.randn(2, 128, generator=generator),
        route_weights=torch.asarray([0.3, 0.7]),
    )
    with pytest.raises(ValueError, match="requires a searched MCG transform"):
        build_expert_candidate_input(
            layer=2,
            expert=9,
            source=source,
            gate_up_statistics=gate_up,
            down_statistics=down,
            heldout_batches=[routed],
            k5_screen=reject_all_k5("not admitted"),
            route_power=2,
            accounting="combined",
            transform_seed_sha256="6" * 64,
            codebook_scale=1.125,
        )
    item = build_expert_candidate_input(
        layer=2,
        expert=9,
        source=source,
        gate_up_statistics=gate_up,
        down_statistics=down,
        heldout_batches=[routed],
        k5_screen=reject_all_k5("not admitted"),
        route_power=2,
        accounting="combined",
        transform_seed_sha256="6" * 64,
        codebook_scale=1.125,
        allow_fixed_transform_baseline=True,
    )
    assert item.fitted["gate_proj"].fit_identity["route_weight_power"] == 2
    np.testing.assert_array_equal(
        item.fitted["gate_proj"].covariance,
        item.fitted["up_proj"].covariance,
    )
    np.testing.assert_array_equal(
        item.fitted["gate_proj"].covariance,
        gate_up.dense_hessian("combined", 2, regularized=False).astype(np.float32),
    )
    assert item.fitted["gate_proj"].fit_identity["regularized"] is False
    assert item.fitted["gate_proj"].fit_identity["hessian_regularization"] == "codec-level-sigma-reg-only"
    np.testing.assert_array_equal(
        item.fitted["gate_proj"].input_vector,
        item.fitted["up_proj"].input_vector,
    )
    assert set(np.abs(item.fitted["gate_proj"].input_vector)) == {np.float32(1 / 1.125)}
    assert item.fitted["gate_proj"].transform_identity["selection_status"] == "fixed-reproducible-baseline-not-multidraw-searched"
    assert len(item.fitted["down_proj"].output_vector) == 128
    assert item.fitted["down_proj"].transform_identity["transform_sha256"]

    block = _fit_bridge_input("gate_up_input", 128, mode="block_diagonal")
    with pytest.raises(ValueError, match="requires full covariance"):
        build_expert_candidate_input(
            layer=2,
            expert=9,
            source=source,
            gate_up_statistics=block,
            down_statistics=down,
            heldout_batches=[routed],
            k5_screen=reject_all_k5("not admitted"),
            route_power=2,
            accounting="combined",
            transform_seed_sha256="6" * 64,
            codebook_scale=1.125,
            allow_fixed_transform_baseline=True,
        )


def test_fitter_bridge_rejects_legacy_raw_searched_transform_handoff():
    generator = torch.Generator().manual_seed(93)
    source = ProjectionTensors(
        torch.randn(128, 128, generator=generator).bfloat16(),
        torch.randn(128, 128, generator=generator).bfloat16(),
        torch.randn(128, 128, generator=generator).bfloat16(),
    )
    gate_up = _fit_bridge_input("gate_up_input", 128)
    down = _fit_bridge_input("down_input", 128)
    shared = torch.ones(128)
    from quant_pipeline.core.artifacts import canonical_json
    baseline = {
        "schema": "quant-pipeline.absolute-v31-gss-finalized.v1",
        "formula_id": "source-derived-absolute-v31-fp16-shared-gu-down-private-bit-gss",
        "implementation_sha256": "1" * 64,
        "source_weight_manifest_sha256": "2" * 64,
        "normalization_evidence_sha256": "3" * 64,
        "gss_evidence_sha256": "4" * 64,
    }
    baseline["artifact_sha256"] = hashlib.sha256(canonical_json(baseline)).hexdigest()
    searched = MCGTransformArtifact(
        layer=2,
        expert=9,
        gate_up_suh=shared,
        gate_svh=-shared,
        up_svh=shared,
        down_suh=-shared,
        down_svh=shared,
        codebook_scale=1.125,
        selection_method="exact-codec-multidraw-plus-block-gscale",
        objective_arm="energy_normalized_sse",
        candidates_evaluated=12,
        selected_score=0.0125,
        selection_role="selection",
        heldout_artifact_sha256="7" * 64,
        evidence_sha256="8" * 64,
        provenance={"search_revision": "9" * 40, "relationship_to_absolute_v31": "additive-ablation-only"},
        absolute_v31_baseline=baseline,
        bit_private_vectors={
            projection: {bit: torch.full((128,), float(bit)) for bit in (3, 4, 5)}
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
    )
    with pytest.raises(RuntimeError, match="canonical verified AbsoluteV31Artifact"):
        build_expert_candidate_input(
            layer=2,
            expert=9,
            source=source,
            gate_up_statistics=gate_up,
            down_statistics=down,
            heldout_batches=[
                RoutedExpertBatch("heldout", torch.randn(2, 128), torch.tensor([0.3, 0.7]))
            ],
            k5_screen=reject_all_k5("not admitted"),
            route_power=1,
            accounting="combined",
            transform_seed_sha256="6" * 64,
            codebook_scale=1.125,
            searched_transform=searched,
        )


def _rehash_record(record):
    from quant_pipeline.core.artifacts import canonical_json

    record["record_sha256"] = hashlib.sha256(
        canonical_json({key: value for key, value in record.items() if key != "record_sha256"})
    ).hexdigest()


def test_actual_fit_capture_search_and_fisher_inputs_must_match_journal(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    identity = _run_identity(attestation)
    for mutation, message in (
        ("fit", "actual fit artifact"),
        ("search", "actual transform search"),
        ("capture", "actual capture artifact"),
        ("fisher", "actual Fisher probe"),
    ):
        item = _expert()
        if mutation == "fit":
            fitted = {
                name: replace(value, fit_identity=dict(value.fit_identity) | {"fit_artifact_sha256": "0" * 64})
                for name, value in item.fitted.items()
            }
            item = replace(item, fitted=fitted)
        elif mutation == "search":
            fitted = {
                name: replace(value, transform_identity=dict(value.transform_identity) | {"search_artifact_sha256": "0" * 64})
                for name, value in item.fitted.items()
            }
            item = replace(item, fitted=fitted)
        else:
            batches = []
            for batch in item.heldout_batches:
                key = "capture_artifact_sha256" if mutation == "capture" else "fisher_probe_sha256"
                batches.append(replace(batch, identity=dict(batch.identity) | {key: "0" * 64}))
            item = replace(item, heldout_batches=batches)
        journal = CandidateJournal(tmp_path / mutation, identity)
        with pytest.raises(ValueError, match=message):
            CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate([item], journal=journal)


def test_sealed_inventory_rejects_missing_units_and_production_shape_is_exact(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    identity = _run_identity(attestation)
    inventory = expected_expert_inventory([(2, 9), (2, 10)], profile="two-unit-fixture", test_fixture=True)
    identity = dict(identity) | {
        "expected_inventory": inventory,
        "expected_inventory_sha256": inventory["inventory_sha256"],
    }
    journal = CandidateJournal(tmp_path / "missing", identity)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate([_expert()], journal=journal)
    with pytest.raises(ValueError, match="exactly 48 layers"):
        expected_expert_inventory([(0, 0)], profile="qwen3-30b-a3b-48x128x3")


def test_declared_expert_membership_and_weight_are_proven_per_row(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    batch = base.heldout_batches[0]
    bad_indices = batch.source_route_indices.clone()
    bad_indices[0, 0] = 7
    bad = replace(batch, source_route_indices=bad_indices)
    bad = replace(bad, identity=dict(bad.identity) | {"batch_payload_sha256": routed_batch_sha256(bad)})
    journal = CandidateJournal(tmp_path / "membership", _run_identity(attestation))
    with pytest.raises(ValueError, match="declared expert 9 exactly once"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, heldout_batches=[bad])], journal=journal
        )
    wrong_mass = batch.source_route_weights.clone()
    wrong_mass[0, 0] += 0.01
    bad = replace(batch, source_route_weights=wrong_mass)
    bad = replace(bad, identity=dict(bad.identity) | {"batch_payload_sha256": routed_batch_sha256(bad)})
    journal = CandidateJournal(tmp_path / "weight", _run_identity(attestation))
    with pytest.raises(ValueError, match="declared route weights do not match"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, heldout_batches=[bad])], journal=journal
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("layer", 1), ("expert", 8), ("layer", True), ("expert", False)),
)
def test_heldout_batch_identity_cannot_be_resealed_for_another_unit(tmp_path, field, value):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    batch = base.heldout_batches[0]
    wrong = replace(batch, identity=dict(batch.identity) | {field: value})
    wrong = replace(
        wrong,
        identity=dict(wrong.identity) | {"batch_payload_sha256": routed_batch_sha256(wrong)},
    )
    with pytest.raises(ValueError, match="identity must bind layer 2 and expert 9"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, heldout_batches=[wrong])],
            journal=CandidateJournal(tmp_path / field, _run_identity(attestation)),
        )


def _slice_routed_batch(batch, start, stop, batch_id):
    values = {}
    for name in (
        "hidden_states",
        "route_weights",
        "source_route_indices",
        "source_route_weights",
        "candidate_route_indices",
        "candidate_route_weights",
    ):
        value = getattr(batch, name)
        values[name] = None if value is None else value[start:stop]
    fisher = None if batch.fisher_gradients is None else batch.fisher_gradients[:, start:stop]
    result = RoutedExpertBatch(
        batch_id=batch_id,
        fisher_gradients=fisher,
        identity=dict(batch.identity),
        row_keys=list(batch.row_keys[start:stop]),
        **values,
    )
    return replace(result, identity=dict(result.identity) | {"batch_payload_sha256": routed_batch_sha256(result)})


def test_route_agreement_is_invariant_to_batch_sharding(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    item = _expert()
    one = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [item], journal=CandidateJournal(tmp_path / "one", _run_identity(attestation))
    )
    batch = item.heldout_batches[0]
    sharded = replace(item, heldout_batches=[
        _slice_routed_batch(batch, 0, 3, "selection-doc-7-a"),
        _slice_routed_batch(batch, 3, 7, "selection-doc-7-b"),
    ])
    two = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [sharded], journal=CandidateJournal(tmp_path / "two", _run_identity(attestation))
    )
    assert one["candidates"][0]["metrics"]["route_agreement"] == two["candidates"][0]["metrics"]["route_agreement"]


def test_candidate_ids_k5_policy_and_journal_inventory_are_derived(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate([_expert()], journal=journal)
    bad = json.loads(json.dumps(ledger["candidates"][0]))
    bad["candidate_id"] = "forged"
    _rehash_record(bad)
    with pytest.raises(ValueError, match="not derived"):
        validate_candidate_record(bad, allow_test_backend=True)
    ledger["k5_policy"]["L2.E9"].pop(next(iter(ledger["k5_policy"]["L2.E9"])))
    ledger["k5_policy_sha256"] = hashlib.sha256(
        __import__("quant_pipeline.core.artifacts", fromlist=["canonical_json"]).canonical_json(ledger["k5_policy"])
    ).hexdigest()
    with pytest.raises(ValueError, match="all 19 decisions"):
        validate_ledger(ledger, allow_test_backend=True)


def test_payload_store_persists_real_bytes_and_deduplicates_shared_vectors(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    journal = CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate([_expert()], journal=journal)
    record = ledger["candidates"][0]
    gate_suh = record["projections"]["gate_proj"]["exact_payload_refs"]["suh"]
    up_suh = record["projections"]["up_proj"]["exact_payload_refs"]["suh"]
    assert gate_suh["sha256"] == up_suh["sha256"]
    assert (journal.root / "payloads" / gate_suh["path"]).read_bytes()
    # Gate/up suh is one semantic group and down svh is another.  Identical
    # bytes may physically deduplicate, but each shared family is charged once.
    assert ledger["fixed_layer_shared_payload_bytes"] == 2 * gate_suh["bytes"]
    # Corruption is caught before a cached resume can be trusted.
    (journal.root / "payloads" / gate_suh["path"]).write_bytes(b"corrupt")
    resumed = CandidateJournal(tmp_path / "journal", _run_identity(attestation), resume=True)
    with pytest.raises(ValueError, match="missing or corrupt"):
        CandidateLedgerGenerator(DeterministicAttestedCodec(), attestation, allow_test_backend=True).generate(
            [_expert()], journal=resumed
        )


def test_payload_store_deduplicates_raw_bytes_without_conflating_tensor_views(tmp_path):
    store = ExactPayloadStore(tmp_path / "payloads")
    raw = torch.arange(4, dtype=torch.uint8)
    vector = store.put_tensor(raw, role="vector")
    matrix = store.put_tensor(raw.reshape(2, 2), role="matrix")
    assert vector["sha256"] == matrix["sha256"]
    assert vector["shape"] != matrix["shape"]
    manifest = store.manifest([vector, matrix])
    assert len(manifest["objects"]) == 1
    assert manifest["physical_bytes"] == 4


def test_codec_self_rehash_cannot_change_logical_cost(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    )
    record = json.loads(json.dumps(ledger["candidates"][0]))
    record["projections"]["gate_proj"]["codec_reported_payload_bytes"] += 10_000_000
    payload_identity = {
        name: {
            "bits": row["bits"],
            "codec_reported_payload_bytes": row["codec_reported_payload_bytes"],
            "packed_sha256": row["packed"]["codec_sha256"],
            "reconstruction_sha256": row["reconstruction_deployed_fp16_sha256"],
            "transform_identity": row["transform_identity"],
        }
        for name, row in record["projections"].items()
    }
    from quant_pipeline.core.artifacts import canonical_json

    record["payload_sha256"] = hashlib.sha256(canonical_json(payload_identity)).hexdigest()
    _rehash_record(record)
    with pytest.raises(ValueError, match="codec-reported payload bytes"):
        validate_candidate_record(record, allow_test_backend=True)


def test_identical_private_vectors_are_semantically_charged_per_slot(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    )
    cost = ledger["candidates"][0]["cost_breakdown"]
    private_vectors = [row for row in cost["semantic_expert_private_slots"] if row["role"] in {"suh", "svh"}]
    assert len(private_vectors) == 3
    assert len({row["sha256"] for row in private_vectors}) == 1
    assert sum(row["bytes"] for row in private_vectors) == 3 * private_vectors[0]["bytes"]
    assert len(cost["semantic_layer_shared_objects"]) == 2
    assert cost["semantic_layer_shared_bytes"] == sum(
        row["bytes"] for row in cost["semantic_layer_shared_objects"]
    )


def test_fixed_transform_baseline_cannot_validate_as_production(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    )
    record = json.loads(json.dumps(ledger["candidates"][0]))
    record["backend_attestation"]["test_only"] = False
    for projection in record["projections"].values():
        projection["transform_identity"]["selection_status"] = "fixed-reproducible-baseline-not-multidraw-searched"
    payload_identity = {
        name: {
            "bits": row["bits"],
            "codec_reported_payload_bytes": row["codec_reported_payload_bytes"],
            "packed_sha256": row["packed"]["codec_sha256"],
            "reconstruction_sha256": row["reconstruction_deployed_fp16_sha256"],
            "transform_identity": row["transform_identity"],
        }
        for name, row in record["projections"].items()
    }
    from quant_pipeline.core.artifacts import canonical_json
    record["payload_sha256"] = hashlib.sha256(canonical_json(payload_identity)).hexdigest()
    with pytest.raises(ValueError, match="fixed-transform baseline"):
        validate_candidate_record(record)


def test_down_encodes_are_conditional_on_decoded_gate_up_path(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [_expert()], journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation))
    )
    down = [(unit, bits, hessian) for unit, bits, hessian in codec.call_inputs if ".down_proj." in unit]
    assert {unit.rsplit(".", 1)[-1] for unit, _, _ in down} == {"g3u3", "g3u4", "g4u3", "g4u4"}
    assert all(torch.isfinite(hessian).all() for _, _, hessian in down)
    assert len({_sha(hessian) for _, _, hessian in down}) == 4


def test_proposal_score_and_canonical_ledger_share_exact_conditional_k4_bytes(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    _proxy, _heldout, identities = _score_units([base], codec)
    scored = replace(
        base,
        proposal_score_candidate_identity=identities[base.unit_id],
    )
    ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
        [scored],
        journal=CandidateJournal(tmp_path / "journal", _run_identity(attestation)),
    )
    uniform = next(row for row in ledger["candidates"] if row["bit_triplet"] == [4, 4, 4])
    for score_name, projection in (
        ("gate_proj", "gate_proj"),
        ("up_proj", "up_proj"),
        ("down_proj.g4u4", "down_proj"),
    ):
        expected = identities[base.unit_id][score_name]
        observed = uniform["projections"][projection]
        assert expected["packed_sha256"] == observed["packed"]["codec_sha256"]
        assert expected["reconstruction_deployed_fp16_sha256"] == observed["reconstruction_deployed_fp16_sha256"]
        assert expected["reconstruction_hf_sha256"] == observed["reconstruction_hf"]["sha256"]


def test_confirmation_role_cannot_leak_into_selection_or_conditional_fit(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    heldout = base.heldout_batches[0]
    bad_heldout = replace(heldout, identity=dict(heldout.identity) | {"role": "confirmation"})
    bad_heldout = replace(
        bad_heldout,
        identity=dict(bad_heldout.identity) | {"batch_payload_sha256": routed_batch_sha256(bad_heldout)},
    )
    with pytest.raises(ValueError, match="selection role"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, heldout_batches=[bad_heldout])],
            journal=CandidateJournal(tmp_path / "heldout-leak", _run_identity(attestation)),
        )
    conditional = base.conditional_down_fit_batches[0]
    bad_conditional = replace(
        conditional,
        identity=dict(conditional.identity) | {"role": "confirmation"},
    )
    bad_conditional = replace(
        bad_conditional,
        identity=dict(bad_conditional.identity)
        | {"batch_payload_sha256": conditional_down_fit_batch_sha256(bad_conditional)},
    )
    with pytest.raises(ValueError, match="conditional_fit, not confirmation"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, conditional_down_fit_batches=[bad_conditional])],
            journal=CandidateJournal(tmp_path / "conditional-leak", _run_identity(attestation)),
        )


def test_conditional_down_fit_is_disjoint_from_heldout_scoring_rows(tmp_path):
    attestation = _attestation(CovarianceSensitiveCodec())

    def run(name, item):
        codec = CovarianceSensitiveCodec()
        ledger = CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [item], journal=CandidateJournal(tmp_path / name, _run_identity(attestation))
        )
        return [row["projections"]["down_proj"]["packed"]["sha256"] for row in ledger["candidates"]]

    base = _expert()
    baseline = run("base", base)
    heldout = base.heldout_batches[0]
    changed_heldout = replace(heldout, hidden_states=heldout.hidden_states + 0.25)
    changed_heldout = replace(
        changed_heldout,
        identity=dict(changed_heldout.identity) | {"batch_payload_sha256": routed_batch_sha256(changed_heldout)},
    )
    assert run("heldout", replace(base, heldout_batches=[changed_heldout])) == baseline
    fit = base.conditional_down_fit_batches[0]
    changed_fit = replace(fit, hidden_states=fit.hidden_states + 0.25)
    changed_fit = replace(
        changed_fit,
        identity=dict(changed_fit.identity) | {"batch_payload_sha256": conditional_down_fit_batch_sha256(changed_fit)},
    )
    assert run("fit", replace(base, conditional_down_fit_batches=[changed_fit])) != baseline


def test_conditional_down_fit_rejects_zero_route_mass_and_row_overlap(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    fit = base.conditional_down_fit_batches[0]
    zero_weights = fit.route_weights.clone()
    zero_weights[0] = 0
    zero = replace(fit, route_weights=zero_weights)
    zero = replace(zero, identity=dict(zero.identity) | {"batch_payload_sha256": conditional_down_fit_batch_sha256(zero)})
    with pytest.raises(ValueError, match="invalid route/sampling weights"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, conditional_down_fit_batches=[zero])],
            journal=CandidateJournal(tmp_path / "zero", _run_identity(attestation)),
        )
    overlap = replace(fit, row_keys=[base.heldout_batches[0].row_keys[0], *fit.row_keys[1:]])
    overlap = replace(
        overlap,
        identity=dict(overlap.identity) | {"batch_payload_sha256": conditional_down_fit_batch_sha256(overlap)},
    )
    with pytest.raises(ValueError, match="overlap held-out selection rows"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, conditional_down_fit_batches=[overlap])],
            journal=CandidateJournal(tmp_path / "overlap", _run_identity(attestation)),
        )


def test_conditional_down_fit_rejects_wrong_expert_substitution(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    fit = base.conditional_down_fit_batches[0]
    wrong_routes = fit.source_route_indices.clone()
    wrong_routes[:, 0] = 8
    wrong = replace(fit, source_route_indices=wrong_routes)
    wrong = replace(
        wrong,
        identity=dict(wrong.identity) | {"batch_payload_sha256": conditional_down_fit_batch_sha256(wrong)},
    )
    with pytest.raises(ValueError, match="contain declared expert 9 exactly once"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, conditional_down_fit_batches=[wrong])],
            journal=CandidateJournal(tmp_path / "wrong-expert", _run_identity(attestation)),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("layer", True), ("expert", False)),
)
def test_conditional_down_fit_identity_is_bool_strict(tmp_path, field, value):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    fit = base.conditional_down_fit_batches[0]
    wrong = replace(fit, identity=dict(fit.identity) | {field: value})
    wrong = replace(
        wrong,
        identity=dict(wrong.identity)
        | {"batch_payload_sha256": conditional_down_fit_batch_sha256(wrong)},
    )
    with pytest.raises(ValueError, match="declared layer/expert identity mismatch"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, conditional_down_fit_batches=[wrong])],
            journal=CandidateJournal(tmp_path / field, _run_identity(attestation)),
        )


def test_conditional_down_fit_rejects_heldout_document_overlap(tmp_path):
    codec = DeterministicAttestedCodec()
    attestation = _attestation(codec)
    base = _expert()
    fit = base.conditional_down_fit_batches[0]
    overlap = replace(
        fit,
        identity=dict(fit.identity)
        | {"document_sha256": base.heldout_batches[0].identity["document_sha256"]},
    )
    overlap = replace(
        overlap,
        identity=dict(overlap.identity) | {"batch_payload_sha256": conditional_down_fit_batch_sha256(overlap)},
    )
    with pytest.raises(ValueError, match="documents overlap"):
        CandidateLedgerGenerator(codec, attestation, allow_test_backend=True).generate(
            [replace(base, conditional_down_fit_batches=[overlap])],
            journal=CandidateJournal(tmp_path / "document-overlap", _run_identity(attestation)),
        )
