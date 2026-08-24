from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_pipeline.campaign.qwen_services import (
    QwenAllocatorService,
    QwenCheckpointService,
    QwenCodecService,
    QwenEvaluatorService,
)
from quant_pipeline.campaign.runner import CampaignDefinition, _build_stages
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _sealed_allocation(body):
    document = {"schema": "quant-pipeline.qwen-dual-arm-allocation.v1", **body}
    document["allocation_sha256"] = sha256_bytes(canonical_json(document))
    return document


def _runtime_module(tmp_path):
    module = tmp_path / "sealed_student_runtime.py"
    module.write_text(
        """
from pathlib import Path
import numpy as np

class Runtime:
    def __init__(self, options):
        self.options = options
    def identity(self):
        return {"backend": "fixture-native-btx-runtime", "version": 1, "test_only": False}
    def capture(self, *, checkpoint_root, source_checkpoint, kld_window, output_path, model_revision):
        value = np.load(self.options["logits_source"], allow_pickle=False)
        np.save(output_path, value, allow_pickle=False)
        return {"ok": True, "checkpoint": Path(checkpoint_root).name, "model_revision": model_revision}

def build(options):
    return Runtime(options)
""".lstrip()
    )
    return module


def test_real_stage_graph_supplies_every_qwen_codec_install_dependency(tmp_path):
    definition = CampaignDefinition(
        experiment_spec="experiment.toml",
        inputs={"source_checkpoint": "checkpoint"},
        layers=(0,),
        reanchor_every_layers=1,
        reanchor_failure_policy="request_reallocation",
        retention_mode="capture-plus-ledger",
    )
    encode = next(stage for stage in _build_stages(definition) if stage.kind == "causal_encode")
    assert "causal_fit.layer_000" in encode.dependencies
    assert "causal_candidates.layer_000" in encode.dependencies

    dependency_paths = {}
    for stage_id in encode.dependencies:
        path = tmp_path / stage_id
        path.mkdir()
        dependency_paths[stage_id] = str(path)
    from quant_pipeline.campaign.qwen_services import _dependency

    context = {"dependencies": dependency_paths}
    assert _dependency(context, "causal_fit") == (tmp_path / "causal_fit.layer_000").resolve()
    assert _dependency(context, "causal_candidates") == (
        tmp_path / "causal_candidates.layer_000"
    ).resolve()


def test_pinned_student_runtime_is_source_and_identity_bound(tmp_path, monkeypatch):
    module = _runtime_module(tmp_path)
    logits_source = tmp_path / "expected.npy"
    np.save(logits_source, np.arange(12, dtype=np.float32).reshape(3, 4))
    monkeypatch.syspath_prepend(str(tmp_path))
    imported = importlib.import_module("sealed_student_runtime")
    identity = imported.build({"logits_source": str(logits_source)}).identity()
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes

    config = {
        "model_revision": "a" * 40,
        "student_runtime": {
            "factory": "sealed_student_runtime:build",
            "source_sha256": sha256_file(module),
            "identity_sha256": sha256_bytes(canonical_json(identity)),
            "options": {"logits_source": str(logits_source)},
        },
    }
    evaluator = QwenEvaluatorService(config, capturer=object())
    declared = evaluator.identity()
    assert declared["student_runtime"]["source_sha256"] == sha256_file(module)

    emission = tmp_path / "emission"
    (emission / "checkpoint").mkdir(parents=True)
    (emission / "checkpoint" / "btx-manifest.json").write_text("{}\n")
    write_json(emission / "stage-manifest.json", {
        "provider_result": {"checkpoint_manifest_file": "checkpoint/btx-manifest.json"}
    })
    output = tmp_path / "student"
    result = evaluator.capture_student({
        "output_dir": str(output),
        "dependencies": {"checkpoint_emission": str(emission)},
        "inputs": {
            "source_checkpoint": str(tmp_path / "source"),
            "kld_window": str(tmp_path / "window"),
        },
    })
    assert result["student_capture_file"] == "student-logits.npy"
    assert np.array_equal(np.load(output / result["student_capture_file"]), np.load(logits_source))
    receipt = json.loads((output / "student-capture-receipt.json").read_text())
    assert receipt["checkpoint_manifest_sha256"] == sha256_file(emission / "checkpoint" / "btx-manifest.json")

    sys.modules.pop("sealed_student_runtime", None)
    bad = dict(config) | {"student_runtime": dict(config["student_runtime"]) | {"source_sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="source is missing or drifted"):
        QwenEvaluatorService(bad, capturer=object()).identity()


def test_official_btx_filter_rejects_unexpressible_choices_before_dp():
    service = QwenAllocatorService({
        "btx_rate_structure": "per_expert_pair",
        "require_fused_btx": True,
        "intermediate_size": 768,
        "target_tp_degrees": [1],
    })
    assert service._serving_legal({"bit_triplet": [3, 4, 3]}) == (
        False, "official BTX master encodes one fc1 rate for gate/up"
    )
    assert service._serving_legal({"bit_triplet": [5, 5, 3]}) == (
        False, "official BTX per_expert_pair has no K5 vocabulary"
    )
    assert service._serving_legal({"bit_triplet": [4, 4, 4]}) == (
        False, "P44 is schema-declared but not fused on pinned master"
    )
    assert service._serving_legal({"bit_triplet": [3, 3, 3]}) == (True, None)

    tp2 = QwenAllocatorService({
        "btx_rate_structure": "per_expert_pair", "require_fused_btx": False,
        "intermediate_size": 768, "target_tp_degrees": [2],
    })
    ok, reason = tp2._serving_legal({"bit_triplet": [3, 3, 3]})
    assert ok is False and "TP2" in reason


def test_checkpoint_composition_uses_reconciled_total_and_nested_official_audit(tmp_path, monkeypatch):
    from quant_pipeline.campaign import qwen_services

    allocation_root = tmp_path / "allocation"
    allocation_root.mkdir()
    allocation = _sealed_allocation({"serving_arm": {"selected_cost": {"schema": "fixture"}}})
    write_json(allocation_root / "allocation.json", allocation)
    write_json(
        allocation_root / "stage-manifest.json",
        {"provider_result": {"allocation_file": "allocation.json"}},
    )
    observed = {}

    def reconcile(cost, installed):
        assert cost == {"schema": "fixture"}
        assert installed == ["installed-0", "installed-1"]
        return {"allocated_payload_bytes": 123, "reconciliation_sha256": "1" * 64}

    def emit(*, output_dir, expected_allocated_payload_bytes, **_kwargs):
        observed["expected"] = expected_allocated_payload_bytes
        output_dir.mkdir(parents=True)
        write_json(output_dir / "btx-manifest.json", {"fixture": True})
        return {"kind": "btx-manifest", "fixture": True}

    monkeypatch.setattr(qwen_services, "reconcile_installed_allocation", reconcile)
    monkeypatch.setattr(qwen_services, "btx_compatibility_report", lambda *_args, **_kwargs: {"compatible": True})
    monkeypatch.setattr(qwen_services, "emit_official_btx_checkpoint", emit)
    monkeypatch.setattr(
        qwen_services,
        "audit_official_btx_checkpoint",
        lambda *_args, **_kwargs: {
            "ok": True,
            "manifest_sha256": "2" * 64,
            "accounting": {"source_semantic_allocated_payload_bytes": 123},
        },
    )
    output = tmp_path / "emission"
    result = QwenCheckpointService({"target_tp_degrees": [1]}).emit({
        "output_dir": str(output),
        "installed_layer_attempts": ["installed-0", "installed-1"],
        "dependencies": {"allocation": str(allocation_root)},
    })
    assert observed["expected"] == 123
    assert result["checkpoint_manifest_sha256"] == "2" * 64
    assert result["allocation_reconciliation_file"] == "installed-allocation-reconciliation.json"
    assert json.loads((output / "emission-accounting-audit.json").read_text())["accounting"][
        "source_semantic_allocated_payload_bytes"
    ] == 123


def test_allocator_composition_persists_validated_selected_cost(tmp_path, monkeypatch):
    from quant_pipeline.candidates import ledger as ledger_module

    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    record = {
        "unit_id": "L0.E0",
        "candidate_id": "L0.E0.g3u3d3",
        "record_sha256": "3" * 64,
        "bit_triplet": [3, 3, 3],
        "predicted_damage": 0.5,
    }
    record_k4 = dict(record) | {
        "candidate_id": "L0.E0.g4u4d4",
        "record_sha256": "4" * 64,
        "bit_triplet": [4, 4, 4],
        "predicted_damage": 1.0,
    }
    write_json(candidate_root / "candidate-ledger.json", {
        "ledger_sha256": "5" * 64,
        "competitive": False,
        "candidates": [record, record_k4],
    })
    write_json(candidate_root / "stage-manifest.json", {
        "provider_result": {"candidate_ledger_file": "candidate-ledger.json"}
    })
    attribution_root = tmp_path / "attribution"
    attribution_root.mkdir()
    attribution = {
        "schema": "quant-pipeline.qwen-attribution.v1",
        "candidate_ledger_sha256": sha256_file(candidate_root / "candidate-ledger.json"),
        "layers": [{"layer_index": 0, "expert_direct": [-0.25]}],
    }
    attribution["attribution_sha256"] = sha256_bytes(canonical_json(attribution))
    write_json(attribution_root / "attribution.json", attribution)
    write_json(attribution_root / "stage-manifest.json", {
        "provider_result": {"attribution_file": "attribution.json"}
    })
    selected_cost = {
        "schema": "quant-pipeline.selected-allocation-cost.v1",
        "allocated_payload_bytes": 13,
        "selected_layer_costs": [{"layer": 0, "allocated_payload_bytes": 13}],
        "allocation_cost_sha256": "5" * 64,
    }
    calls = []

    def allocate(records, **options):
        calls.append((records, options))
        overrides = options.get("damage_overrides")
        selected = min(
            records,
            key=lambda row: (
                overrides[str(row["candidate_id"])] if overrides is not None
                else float(row["predicted_damage"])
            ),
        )
        choice = SimpleNamespace(
            unit_id="L0.E0",
            choice_id=selected["candidate_id"],
            stored_bytes=11,
            predicted_damage=(
                overrides[str(selected["candidate_id"])] if overrides is not None
                else float(selected["predicted_damage"])
            ),
            metadata=selected,
        )
        allocation = SimpleNamespace(
            choices=(choice,),
            variable_payload_bytes=11,
            fixed_layer_shared_bytes=2,
            stored_bytes=13,
            predicted_damage=choice.predicted_damage,
        )
        return SimpleNamespace(allocation=allocation, selected_cost=selected_cost)

    monkeypatch.setattr(ledger_module, "allocate_validated_records", allocate)
    monkeypatch.setattr(ledger_module, "validate_ledger", lambda *_args, **_kwargs: None)
    output = tmp_path / "allocation-output"
    QwenAllocatorService({
        "exact_payload_byte_budget": 13,
        "attribution_provisional_bit_triplet": [3, 3, 3],
        "require_fused_btx": False,
    }).allocate({
        "output_dir": str(output),
        "dependencies": {
            "candidates": str(candidate_root),
            "attribution": str(attribution_root),
        },
    })
    document = json.loads((output / "allocation.json").read_text())
    assert len(calls) == 3
    assert "damage_overrides" not in calls[0][1]
    assert calls[1][1]["damage_overrides"] == {
        "L0.E0.g3u3d3": pytest.approx(0.25),
        "L0.E0.g4u4d4": pytest.approx(0.0),
    }
    assert document["attribution_file_sha256"] == sha256_file(attribution_root / "attribution.json")
    assert document["proxy_control_arm"]["selected_cost"] == selected_cost
    assert document["research_arm"]["selected_cost"] == selected_cost
    assert document["serving_arm"]["selected_cost"] == selected_cost
    row = document["shapley_damage_calibration"]["unit_rows"][0]
    assert row["provisional_unshifted_damage"] == pytest.approx(row["expert_direct"])
    assert row["provisional_shifted_damage"] == pytest.approx(
        row["expert_direct"] + row["nonnegative_unit_offset"]
    )
    assert document["proxy_control_arm"]["choices"][0]["choice_id"] == "L0.E0.g3u3d3"
    assert document["research_arm"]["choices"][0]["choice_id"] == "L0.E0.g4u4d4"
    assert document["research_arm"]["selected_unit_offset_total"] == pytest.approx(0.5)
    assert document["research_arm"]["unshifted_predicted_damage"] == pytest.approx(-0.5)


def test_allocator_revalidates_loaded_ledger_before_consuming_it(tmp_path, monkeypatch):
    from quant_pipeline.candidates import ledger as ledger_module

    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    write_json(candidate_root / "candidate-ledger.json", {"competitive": False, "candidates": []})
    write_json(candidate_root / "stage-manifest.json", {
        "provider_result": {"candidate_ledger_file": "candidate-ledger.json"}
    })
    monkeypatch.setattr(
        ledger_module,
        "validate_ledger",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("tampered ledger")),
    )
    with pytest.raises(ValueError, match="tampered ledger"):
        QwenAllocatorService({"exact_payload_byte_budget": 1}).allocate({
            "output_dir": str(tmp_path / "output"),
            "dependencies": {"candidates": str(candidate_root)},
        })


def test_serving_filter_rejects_missing_expert_unit_before_dp(tmp_path, monkeypatch):
    from quant_pipeline.candidates import ledger as ledger_module

    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    record = {
        "unit_id": "L0.E0",
        "candidate_id": "L0.E0.g4u4d4",
        "record_sha256": "3" * 64,
        "bit_triplet": [4, 4, 4],
        "predicted_damage": 0.5,
    }
    write_json(candidate_root / "candidate-ledger.json", {
        "ledger_sha256": "4" * 64,
        "competitive": False,
        "candidates": [record],
    })
    write_json(candidate_root / "stage-manifest.json", {
        "provider_result": {"candidate_ledger_file": "candidate-ledger.json"}
    })
    attribution_root = tmp_path / "attribution"
    attribution_root.mkdir()
    attribution = {
        "schema": "quant-pipeline.qwen-attribution.v1",
        "candidate_ledger_sha256": sha256_file(candidate_root / "candidate-ledger.json"),
        "layers": [{"layer_index": 0, "expert_direct": [0.25]}],
    }
    attribution["attribution_sha256"] = sha256_bytes(canonical_json(attribution))
    write_json(attribution_root / "attribution.json", attribution)
    write_json(attribution_root / "stage-manifest.json", {
        "provider_result": {"attribution_file": "attribution.json"}
    })
    monkeypatch.setattr(ledger_module, "validate_ledger", lambda *_args, **_kwargs: None)

    def allocate(_records, **_options):
        choice = SimpleNamespace(
            unit_id="L0.E0", choice_id=record["candidate_id"], stored_bytes=1,
            predicted_damage=0.25, metadata=record,
        )
        allocation = SimpleNamespace(
            choices=(choice,), variable_payload_bytes=1, fixed_layer_shared_bytes=0,
            stored_bytes=1, predicted_damage=0.25,
        )
        return SimpleNamespace(allocation=allocation, selected_cost={})

    monkeypatch.setattr(ledger_module, "allocate_validated_records", allocate)
    with pytest.raises(ValueError, match="without a legal candidate: L0.E0"):
        QwenAllocatorService({
            "exact_payload_byte_budget": 1,
            "attribution_provisional_bit_triplet": [4, 4, 4],
            "require_fused_btx": True,
        }).allocate({
            "output_dir": str(tmp_path / "output"),
            "dependencies": {
                "candidates": str(candidate_root),
                "attribution": str(attribution_root),
            },
        })


def test_codec_install_consumes_selected_layer_cost_row(tmp_path, monkeypatch):
    from quant_pipeline.campaign import qwen_services

    candidate_root = tmp_path / "causal-candidates"
    (candidate_root / "journal" / "payloads").mkdir(parents=True)
    refs = {
        name: {"path": "unused", "sha256": "6" * 64, "bytes": 2, "dtype": "float16", "shape": [1]}
        for name in ("packed_trellis", "suh", "svh", "reconstruction_hf")
    }
    record = {
        "candidate_id": "L0.E0.g3u3d3",
        "unit_id": "L0.E0",
        "layer": 0,
        "expert": 0,
        "record_sha256": "7" * 64,
        "projections": {
            name: {"bits": 3, "exact_payload_refs": refs}
            for name in ("gate_proj", "up_proj", "down_proj")
        },
    }
    write_json(candidate_root / "candidate-ledger.json", {"candidates": [record]})
    write_json(candidate_root / "stage-manifest.json", {
        "provider_result": {"candidate_ledger_file": "candidate-ledger.json"}
    })
    layer_cost = {
        "layer": 0,
        "selected_candidate_record_sha256": ["8" * 64],
        "semantic_expert_private_bytes": 11,
        "semantic_layer_shared_objects": [],
        "semantic_layer_shared_bytes": 2,
        "allocated_payload_bytes": 13,
    }
    allocation_root = tmp_path / "allocation"
    allocation_root.mkdir()
    write_json(allocation_root / "allocation.json", _sealed_allocation({
        "serving_arm": {
            "choices": [{
                "unit_id": "L0.E0",
                "choice_id": "L0.E0.g3u3d3",
                "candidate_record_sha256": "8" * 64,
            }],
            "selected_cost": {"selected_layer_costs": [layer_cost]},
        }
    }))
    write_json(allocation_root / "stage-manifest.json", {
        "provider_result": {"allocation_file": "allocation.json"}
    })
    fit_root = tmp_path / "fit"
    fit_root.mkdir()
    write_json(fit_root / "fit.json", {"fit": True})
    write_json(fit_root / "stage-manifest.json", {"provider_result": {"fit_manifest_file": "fit.json"}})
    observed = {}

    def install(**kwargs):
        observed.update(kwargs)
        return {"installed_checkpoint_sha256": "9" * 64, "cost_breakdown": {
            key: layer_cost[key]
            for key in (
                "semantic_expert_private_bytes", "semantic_layer_shared_objects",
                "semantic_layer_shared_bytes", "allocated_payload_bytes",
            )
        }}

    monkeypatch.setattr(qwen_services, "install_layer_payloads", install)
    from quant_pipeline.candidates import ledger as ledger_module
    monkeypatch.setattr(ledger_module, "validate_ledger", lambda *_args, **_kwargs: None)
    codec = SimpleNamespace(identity={"codec": "fixture"})
    service = QwenCodecService({}, codec)
    monkeypatch.setattr(service, "_load_ref", lambda *_args: np.zeros(1, dtype=np.float16))
    result = service.install({
        "output_dir": str(tmp_path / "installed"),
        "layer": 0,
        "predecessor_state_hash": "a" * 64,
        "input_identities": {"source_checkpoint": "b" * 64},
        "dependencies": {
            "causal_candidates": str(candidate_root),
            "allocation": str(allocation_root),
            "causal_fit": str(fit_root),
        },
        "production": False,
    })
    assert observed["expected_allocated_payload_bytes"] == 13
    assert result["installed_checkpoint_sha256"] == "9" * 64


def test_codec_install_rejects_allocation_seal_drift(tmp_path, monkeypatch):
    from quant_pipeline.candidates import ledger as ledger_module

    candidate_root = tmp_path / "causal-candidates"
    candidate_root.mkdir()
    write_json(candidate_root / "candidate-ledger.json", {"competitive": False, "candidates": []})
    write_json(candidate_root / "stage-manifest.json", {
        "provider_result": {"candidate_ledger_file": "candidate-ledger.json"}
    })
    allocation_root = tmp_path / "allocation"
    allocation_root.mkdir()
    allocation = _sealed_allocation({"serving_arm": {"choices": []}})
    allocation["serving_arm"]["choices"].append({"unit_id": "L0.E0"})
    write_json(allocation_root / "allocation.json", allocation)
    write_json(allocation_root / "stage-manifest.json", {
        "provider_result": {"allocation_file": "allocation.json"}
    })
    monkeypatch.setattr(ledger_module, "validate_ledger", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="allocation seal mismatch"):
        QwenCodecService({}, SimpleNamespace(identity={})).install({
            "output_dir": str(tmp_path / "output"),
            "layer": 0,
            "dependencies": {
                "causal_candidates": str(candidate_root),
                "allocation": str(allocation_root),
            },
        })
