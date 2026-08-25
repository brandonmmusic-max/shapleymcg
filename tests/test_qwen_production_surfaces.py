# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from quant_pipeline.calibration.fitter import load_fitted_statistics
from quant_pipeline.calibration.qwen_capture import (
    CaptureWindow,
    capture_loaded_qwen,
    capture_roles_from_local_bf16,
    routed_expert_rows,
    verify_capture_manifest,
)
from quant_pipeline.checkpoint.btx_qwen import (
    InternalBTXReader,
    audit_internal_qwen_checkpoint,
    emit_internal_qwen_checkpoint,
    install_layer_payloads,
    installed_cost_breakdown,
    replay_installed_layers,
)
from quant_pipeline.checkpoint.official_btx import (
    UPSTREAM_COMMIT,
    audit_official_btx_checkpoint,
    btx_compatibility_report,
    emit_official_btx_checkpoint,
    unpack_official_btx_plane,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.campaign.qwen_adapter import _independent_kld, _verify_checkpoint_audit
from quant_pipeline.campaign.qwen_services import CAPTURE_SERVICE_SCHEMA, QwenFitterService
from quant_pipeline.campaign.runner import StageRequest


H = "1" * 64


def _bound_file(path: Path):
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "kind": "file"}


class TinyExperts(torch.nn.Module):
    def __init__(self, experts=2, hidden=4, intermediate=2):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.gate_up_proj = torch.nn.Parameter(torch.randn(experts, 2 * intermediate, hidden, generator=generator))
        self.down_proj = torch.nn.Parameter(torch.randn(experts, hidden, intermediate, generator=generator))


class TinyRouter(torch.nn.Module):
    def __init__(self, hidden=4, experts=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden, generator=torch.Generator().manual_seed(11)))

    def forward(self, hidden):
        from quant_pipeline.calibration.qwen_capture import recompute_routing

        return recompute_routing(self.weight, hidden.reshape(-1, hidden.shape[-1]), 1, True)


class TinyMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = TinyRouter()
        self.experts = TinyExperts()

    def forward(self, hidden):
        import torch.nn.functional as F

        _logits, weights, indices = self.gate(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        output = torch.zeros_like(flat)
        for expert in range(2):
            selected = torch.nonzero(indices[:, 0] == expert).flatten()
            if selected.numel():
                gate, up = self.experts.gate_up_proj[expert].chunk(2, 0)
                mid = F.silu(F.linear(flat[selected], gate)) * F.linear(flat[selected], up)
                output[selected] = F.linear(mid, self.experts.down_proj[expert]) * weights[selected]
        return output.reshape_as(hidden)


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMoe()

    def forward(self, hidden):
        return hidden + self.mlp(hidden)


class TinyConfig:
    num_experts = 2
    num_experts_per_tok = 1
    norm_topk_prob = True
    hidden_size = 4
    moe_intermediate_size = 2


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.head = torch.nn.Linear(4, 7, bias=False)
        self.config = TinyConfig()
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, **_kwargs):
        self.forward_calls += 1
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return type("Output", (), {"logits": self.head(hidden)})()


def test_final_token_is_excluded_from_routed_scoring_rows():
    experts = TinyExperts()
    hidden = torch.tensor([[1.0, 0, 0, 0], [2.0, 0, 0, 0], [999.0, 0, 0, 0]])
    ids = torch.tensor([[0], [1], [1]])
    weights = torch.ones(3, 1)
    rows = routed_expert_rows(hidden, ids, weights, experts)
    assert rows["assignment_token_offsets"].tolist() == [0, 1]
    assert 999.0 not in rows["routed_hidden_states"][:, 0].tolist()


def test_capture_resume_verifies_bytes_shapes_and_fisher_domain(tmp_path, monkeypatch):
    model = TinyModel().eval()
    backward_calls = 0
    original_grad = torch.autograd.grad

    def counted_grad(*args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    token_ids = (1, 2, 3, 31)
    token_hash = sha256_bytes(torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes())
    root = tmp_path / "capture"
    capture_loaded_qwen(
        model=model,
        windows=[CaptureWindow(token_ids, token_hash, "doc", 10)],
        role="fit",
        layers=[0, 1],
        predecessor_state_hash=H,
        output_dir=root,
        fisher_rank=3,
    )
    # One inference capture + one graph capture, then one backward per Fisher
    # probe for *all* layers together. This must not scale as layers*rank.
    assert model.forward_calls == 2
    assert backward_calls == 3
    manifest = verify_capture_manifest(root)
    assert manifest["next_token_positions_per_window"] == [3]
    # A completed call is an exact audit, not a second model execution.
    assert capture_loaded_qwen(
        model=model,
        windows=[CaptureWindow(token_ids, token_hash, "doc", 10)],
        role="fit",
        layers=[0, 1],
        predecessor_state_hash=H,
        output_dir=root,
        fisher_rank=3,
    )["capture_sha256"] == manifest["capture_sha256"]
    chunk = root / "layer-000" / "window-0000.safetensors"
    raw = bytearray(chunk.read_bytes())
    raw[-1] ^= 1
    chunk.write_bytes(raw)
    with pytest.raises(ValueError, match="byte identity"):
        verify_capture_manifest(root)


def test_capture_resume_quarantines_orphan_chunk_and_rejects_request_drift(tmp_path):
    model = TinyModel().eval()
    token_ids = (1, 2, 3, 4)
    token_hash = sha256_bytes(torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes())
    window = CaptureWindow(token_ids, token_hash, "doc", 10)
    root = tmp_path / "capture"
    capture_loaded_qwen(
        model=model,
        windows=[window],
        role="fit",
        layers=[0, 1],
        predecessor_state_hash=H,
        output_dir=root,
    )
    (root / "capture-manifest.json").unlink()
    orphan = root / "layer-000" / "window-0000.safetensors"
    orphan_receipt = orphan.with_suffix(orphan.suffix + ".receipt.json")
    orphan_sha256 = sha256_bytes(orphan.read_bytes())
    orphan_receipt.unlink()
    capture_loaded_qwen(
        model=model,
        windows=[window],
        role="fit",
        layers=[0, 1],
        predecessor_state_hash=H,
        output_dir=root,
    )
    quarantine = list((root / ".quarantine").glob("*"))
    assert len(quarantine) == 1
    assert sha256_bytes((quarantine[0] / orphan.name).read_bytes()) == orphan_sha256
    with pytest.raises(ValueError, match="different request"):
        capture_loaded_qwen(
            model=model,
            windows=[window],
            role="selection",
            layers=[0, 1],
            predecessor_state_hash=H,
            output_dir=root,
        )


def test_qwen_fitter_service_streams_one_layer_and_retains_only_objective_power(tmp_path):
    model = TinyModel().eval()
    windows = []
    for index in range(4):
        token_ids = tuple(range(1 + index * 4, 5 + index * 4))
        windows.append(CaptureWindow(
            token_ids,
            sha256_bytes(canonical_json(list(token_ids))),
            "packed-doc",
            0,
        ))
    stage = tmp_path / "fit_capture"
    capture_root = stage / "fit"
    manifest = capture_loaded_qwen(
        model=model,
        windows=windows,
        role="fit",
        layers=[0, 1],
        predecessor_state_hash=H,
        output_dir=capture_root,
        writer_workers=2,
        max_inflight_chunks=4,
    )
    service = {
        "schema": CAPTURE_SERVICE_SCHEMA,
        "predecessor_state_hash": H,
        "layers": [0, 1],
        "captures": {
            "fit": {
                "role": "fit",
                "manifest": "fit/capture-manifest.json",
                "capture_sha256": manifest["capture_sha256"],
            }
        },
        "streaming": "one-window-one-layer-chunk",
        "retention": "sealed-chunks",
    }
    service["capture_service_sha256"] = sha256_bytes(canonical_json(service))
    write_json(stage / "capture-service-manifest.json", service)
    write_json(stage / "stage-manifest.json", {
        "provider_result": {"capture_manifest_file": "capture-service-manifest.json"}
    })
    config = {
        "model_revision": "a" * 40,
        "dataset_revision": "b" * 40,
        "route_weight_power": 2,
        "retained_powers": [2],
        "retained_accounting": ["combined"],
        "covariance_mode": "full",
    }
    output = tmp_path / "fit"
    result = QwenFitterService(config | {
        "fitter_backend": "torch_full_p2",
        "fitter_device": "cpu",
    }).fit({
        "kind": "fit",
        "layer": 0,
        "output_dir": str(output),
        "dependencies": {"fit_capture": str(stage)},
        "predecessor_state_hash": H,
        "input_identities": {"source_checkpoint": H},
    })
    assert result["transient_files"]
    assert "fit-manifest.json" not in result["transient_files"]
    assert all((output / relative).is_file() for relative in result["transient_files"])
    fitted = json.loads((output / result["fit_manifest_file"]).read_text())
    assert fitted["layers"] == [0]
    assert fitted["estimator"]["retained_powers"] == [2]
    assert fitted["estimator"]["covariance_mode"] == "full"
    assert fitted["estimator"]["fitter_backend"] == "torch_full_p2"
    assert {row["layer"] for row in fitted["statistics"]} == {0}
    assert len(fitted["statistics"]) == 2
    for row in fitted["statistics"]:
        arrays = load_fitted_statistics(output / row["gate_up"]).arrays
        assert set(arrays) == {"combined.p2.mean", "combined.p2.second_moment"}
    reference_output = tmp_path / "fit-reference"
    reference = QwenFitterService(config | {"fitter_backend": "numpy_full"}).fit({
        "kind": "fit",
        "layer": 0,
        "output_dir": str(reference_output),
        "dependencies": {"fit_capture": str(stage)},
        "predecessor_state_hash": H,
        "input_identities": {"source_checkpoint": H},
    })
    reference_manifest = json.loads((reference_output / reference["fit_manifest_file"]).read_text())
    for observed_row, reference_row in zip(fitted["statistics"], reference_manifest["statistics"], strict=True):
        observed = load_fitted_statistics(output / observed_row["gate_up"]).arrays
        expected = load_fitted_statistics(reference_output / reference_row["gate_up"]).arrays
        for name in observed:
            np.testing.assert_allclose(observed[name], expected[name], rtol=1e-6, atol=1e-6)


def test_multi_role_local_capture_loads_and_replays_model_once(tmp_path, monkeypatch):
    windows = {}
    for index, role in enumerate(("fit", "selection", "confirmation", "final")):
        tokens = [index + 1, index + 2, index + 3, index + 4]
        windows[role] = [{
            "document_id": f"doc-{role}",
            "domain": "fixture",
            "token_ids": tokens,
            "token_sha256": sha256_bytes(canonical_json(tokens)),
        }]
    corpus = {
        "schema": "quant-pipeline.sealed-corpus.v1",
        "seed": 7,
        "window_tokens": 4,
        "minimum_domains": 1,
        "tokenizer": {},
        "source": {},
        "role_counts": {role: 1 for role in windows},
        "windows": windows,
    }
    corpus["seal_sha256"] = sha256_bytes(canonical_json(corpus))
    corpus_path = tmp_path / "corpus.json"
    write_json(corpus_path, corpus)
    source = tmp_path / "source"
    source.mkdir()
    model = TinyModel().eval()
    calls = 0

    def from_pretrained(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return model

    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    result = capture_roles_from_local_bf16(
        source_checkpoint=source,
        model_revision="a" * 40,
        sealed_corpus=corpus_path,
        captures=[
            {"purpose": role, "role": role, "output_dir": str(tmp_path / role), "seed": 10 + index}
            for index, role in enumerate(("fit", "selection", "confirmation"))
        ],
        layers=[0, 1],
        predecessor_state_hash=H,
        installed_layer_prefix=(),
        production_geometry=False,
    )
    assert calls == 1
    assert set(result) == {"fit", "selection", "confirmation"}
    assert all(result[role]["request"]["model_revision"] == "a" * 40 for role in result)
    assert all(result[role]["request"]["installed_replay"]["accepted_prefix_length"] == 0 for role in result)


def test_distinct_capture_checkpoint_is_sealed_into_request(tmp_path, monkeypatch):
    tokens = [1, 2, 3, 4]
    windows = {
        role: [{
            "document_id": f"doc-{role}",
            "domain": "fixture",
            "token_ids": tokens,
            "token_sha256": sha256_bytes(canonical_json(tokens)),
        }]
        for role in ("fit", "selection", "confirmation", "final")
    }
    corpus = {
        "schema": "quant-pipeline.sealed-corpus.v1",
        "seed": 7,
        "window_tokens": 4,
        "minimum_domains": 1,
        "tokenizer": {},
        "source": {},
        "role_counts": {role: 1 for role in windows},
        "windows": windows,
    }
    corpus["seal_sha256"] = sha256_bytes(canonical_json(corpus))
    corpus_path = tmp_path / "corpus.json"
    write_json(corpus_path, corpus)
    source = tmp_path / "source"
    capture_checkpoint = tmp_path / "capture-model"
    source.mkdir()
    capture_checkpoint.mkdir()
    model = TinyModel().eval()
    loaded = []

    def from_pretrained(path, **_kwargs):
        loaded.append(Path(path))
        return model

    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    identity = {
        "schema": "quant-pipeline.qwen-capture-checkpoint-identity.v1",
        "kind": "sealed-causal-reconstruction",
        "source_revision": "a" * 40,
        "model_manifest_sha256": "b" * 64,
    }
    identity["capture_checkpoint_sha256"] = sha256_bytes(canonical_json(identity))
    result = capture_roles_from_local_bf16(
        source_checkpoint=source,
        capture_checkpoint=capture_checkpoint,
        capture_checkpoint_identity=identity,
        model_revision="a" * 40,
        sealed_corpus=corpus_path,
        captures=[{
            "purpose": "fit",
            "role": "fit",
            "output_dir": str(tmp_path / "fit"),
        }],
        layers=[0, 1],
        predecessor_state_hash=H,
        production_geometry=False,
    )
    assert loaded == [capture_checkpoint.resolve()]
    assert result["fit"]["request"]["capture_checkpoint"] == identity


def test_distinct_capture_checkpoint_requires_identity(tmp_path, monkeypatch):
    source = tmp_path / "source"
    capture_checkpoint = tmp_path / "capture-model"
    source.mkdir()
    capture_checkpoint.mkdir()
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text("{}")
    with pytest.raises(ValueError, match="requires a sealed"):
        capture_roles_from_local_bf16(
            source_checkpoint=source,
            capture_checkpoint=capture_checkpoint,
            model_revision="a" * 40,
            sealed_corpus=corpus_path,
            captures=[],
            layers=[0],
            predecessor_state_hash=H,
            production_geometry=False,
        )


def test_independent_final_kld_and_pinned_checkpoint_audit_ignore_provider_assertions(tmp_path):
    campaign = tmp_path / "campaign"
    output = campaign / "final"
    output.mkdir(parents=True)
    (campaign / "events.jsonl").write_text("sealed journal fixture\n")
    windows = {}
    for index, role in enumerate(("fit", "selection", "confirmation", "final")):
        tokens = [index + 1, index + 2, index + 3]
        windows[role] = [{
            "document_id": f"doc-{role}",
            "domain": "fixture",
            "token_ids": tokens,
            "token_sha256": sha256_bytes(canonical_json(tokens)),
        }]
    corpus = {
        "schema": "quant-pipeline.sealed-corpus.v1",
        "seed": 7,
        "window_tokens": 3,
        "minimum_domains": 1,
        "tokenizer": {},
        "source": {},
        "role_counts": {role: 1 for role in windows},
        "windows": windows,
    }
    corpus["seal_sha256"] = sha256_bytes(canonical_json(corpus))
    corpus_path = tmp_path / "corpus.json"
    write_json(corpus_path, corpus)
    kld_root = tmp_path / "kld-window"
    kld_root.mkdir()
    prefix = kld_root / "source-prefix.txt"
    prefix.write_text("fixture")
    kld_tokens = [10, 11, 12]
    kld_document = {
        "schema": "quant-pipeline.kld-window.v1",
        "source_prefix": {"file": prefix.name, "bytes": prefix.stat().st_size, "sha256": sha256_file(prefix)},
        "context_length": 3,
        "prediction_positions": 2,
        "token_ids": kld_tokens,
        "token_sha256": sha256_bytes(canonical_json(kld_tokens)),
    }
    kld_document["seal_sha256"] = sha256_bytes(canonical_json(kld_document))
    write_json(kld_root / "kld-window.json", kld_document)
    teacher = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    student = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reference = output / "teacher.npy"
    capture = output / "student.npy"
    np.save(reference, teacher)
    np.save(capture, student)
    teacher_root = campaign / "teacher"
    student_root = campaign / "student"
    teacher_root.mkdir()
    student_root.mkdir()
    write_json(teacher_root / ".runner-result.json", {"metadata": {"teacher_reference_sha256": sha256_file(reference)}})
    write_json(student_root / ".runner-result.json", {
        "metadata": {
            "student_capture_sha256": sha256_file(capture),
            "checkpoint_manifest_sha256": "2" * 64,
            "checkpoint_audit_sha256": "3" * 64,
        }
    })
    request = StageRequest(
        campaign_dir=campaign,
        output_dir=output,
        stage_id="final_kld",
        kind="final_kld",
        attempt=1,
        plan_sha256=H,
        experiment_spec={
            "sha256": H,
            "document": {"objective": {"bootstrap_samples": 64}, "corpus": {"seed": 7}},
        },
        static_inputs={
            "sealed_corpus": _bound_file(corpus_path),
            "kld_window": {
                "path": str(kld_root),
                "sha256": sha256_bytes(canonical_json([])),
                "kind": "directory",
            },
        },
        dependency_artifacts={
            "teacher_capture": {"path": "teacher", "artifact_sha256": "4" * 64},
            "student_capture": {"path": "student", "artifact_sha256": "5" * 64},
        },
        predecessor_state_hash="6" * 64,
    )
    report, _, _ = _independent_kld(
        request,
        {"reference_file": reference.name, "capture_file": capture.name, "mean_kld": 999.0},
        final=True,
    )
    assert report["summary"]["mean"] < 1.0
    assert report["summary"]["max"] >= report["summary"]["mean"]
    assert report["bootstrap_mean"]["samples"] == 64

    emission = campaign / "emission"
    audit_output = campaign / "audit"
    emission.mkdir()
    audit_output.mkdir()
    write_json(emission / ".runner-result.json", {"metadata": {"checkpoint_manifest_sha256": "7" * 64}})
    reader_identity = {"reader": "pinned", "commit": "8" * 40}
    reader_result = {"ok": True, "loaded_layers": 48}
    audit_document = {
        "ok": True,
        "checkpoint_manifest_sha256": "7" * 64,
        "reader_identity": reader_identity,
        "reader_identity_sha256": sha256_bytes(canonical_json(reader_identity)),
        "reader_result": reader_result,
        "reader_result_sha256": sha256_bytes(canonical_json(reader_result)),
    }
    audit_path = audit_output / "audit.json"
    write_json(audit_path, audit_document)
    audit_request = StageRequest(
        campaign_dir=campaign,
        output_dir=audit_output,
        stage_id="checkpoint_audit",
        kind="checkpoint_audit",
        attempt=1,
        plan_sha256=H,
        experiment_spec={"sha256": H},
        static_inputs={},
        dependency_artifacts={"checkpoint_emission": {"path": "emission", "artifact_sha256": "9" * 64}},
        predecessor_state_hash="6" * 64,
    )
    verified = _verify_checkpoint_audit(
        audit_request,
        audit_path,
        {"runtime_reader_identity_sha256": audit_document["reader_identity_sha256"]},
    )
    assert verified["reader_result"]["ok"] is True


def _choice(expert: int, projection: str, shared_gu, shared_down):
    if projection in {"gate_proj", "up_proj"}:
        reconstruction = torch.full((2, 4), 0.1 + expert, dtype=torch.float16)
        suh, svh = shared_gu, torch.ones(2, dtype=torch.float16) * (expert + 1)
        topology = {"suh": "layer_shared", "svh": "expert_private"}
    else:
        reconstruction = torch.full((4, 2), 0.2 + expert, dtype=torch.float16)
        suh, svh = torch.ones(2, dtype=torch.float16) * (expert + 1), shared_down
        topology = {"suh": "expert_private", "svh": "layer_shared"}
    return {
        "expert": expert,
        "projection": projection,
        "choice_id": f"E{expert}-{projection}-k3",
        "bits": 3,
        "vector_topology": topology,
        "provenance": {"codec": "corrected-exl3-mcg-r10", "source": H},
        "tensors": {
            "trellis": torch.zeros((1, 1, 48), dtype=torch.int16),
            "suh": suh,
            "svh": svh,
            "reconstruction": reconstruction,
        },
    }


def _source_checkpoint(root: Path):
    root.mkdir()
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_size": 4,
        "moe_intermediate_size": 2,
    }
    (root / "config.json").write_text(json.dumps(config))
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 4, 4),
        "model.layers.0.mlp.experts.down_proj": torch.randn(2, 4, 2),
        "model.layers.1.mlp.experts.gate_up_proj": torch.randn(2, 4, 4),
        "model.layers.1.mlp.experts.down_proj": torch.randn(2, 4, 2),
        "model.norm.weight": torch.randn(4),
    }
    save_file(tensors, root / "model.safetensors")
    return tensors


def test_payload_install_replay_emit_reload_and_corruption(tmp_path):
    source = tmp_path / "source"
    _source_checkpoint(source)
    shared_gu = torch.tensor([1, -1, 1, -1], dtype=torch.float16)
    shared_down = torch.tensor([-1, 1, -1, 1], dtype=torch.float16)
    selected = [_choice(expert, projection, shared_gu, shared_down) for expert in range(2) for projection in ("gate_proj", "up_proj", "down_proj")]
    installed_root = tmp_path / "installed"
    installed = install_layer_payloads(
        output_dir=installed_root,
        layer=0,
        predecessor_state_hash=H,
        source_checkpoint_sha256="2" * 64,
        fit_sha256="3" * 64,
        candidate_ledger_sha256="4" * 64,
        selected_choices=selected,
    )
    # Shared gate/up input and shared down output vectors are physically deduplicated.
    assert len(installed["shared_object_sha256"]) == 2  # one gate/up input object plus one down output object
    derived_cost = installed_cost_breakdown(installed["choices"])
    assert installed["cost_breakdown"] == derived_cost
    assert installed["allocated_payload_bytes"] == derived_cost["allocated_payload_bytes"]
    assert len(derived_cost["semantic_layer_shared_objects"]) == 2
    with pytest.raises(ValueError, match="allocator cost differs"):
        emit_internal_qwen_checkpoint(
            source_checkpoint=source,
            installed_layers=[installed_root],
            output_dir=tmp_path / "bad-cost-checkpoint",
            format_version="test-v1",
            expected_allocated_payload_bytes=installed["allocated_payload_bytes"] + 1,
        )
    model = TinyModel()
    replay = replay_installed_layers(
        model,
        [installed_root],
        expected_final_state_hash="5" * 64,
        expected_prefix=[{
            "layer": 0,
            "predecessor_state_hash": H,
            "installed_checkpoint_sha256": installed["installed_checkpoint_sha256"],
            "installed_state_hash": "5" * 64,
        }],
    )
    assert replay["requested_predecessor_state_hash"] == "5" * 64
    assert replay["accepted_prefix_length"] == 1
    with pytest.raises(ValueError, match="exactly cover"):
        replay_installed_layers(
            TinyModel(), [installed_root], expected_final_state_hash="5" * 64, expected_prefix=[]
        )
    with pytest.raises(ValueError, match="checkpoint identity"):
        replay_installed_layers(
            TinyModel(),
            [installed_root],
            expected_final_state_hash="5" * 64,
            expected_prefix=[{
                "layer": 0,
                "predecessor_state_hash": H,
                "installed_checkpoint_sha256": "6" * 64,
                "installed_state_hash": "5" * 64,
            }],
        )
    expected_gate = torch.full((2, 4), 0.1, dtype=torch.float16).to(model.model.layers[0].mlp.experts.gate_up_proj.dtype)
    assert torch.equal(model.model.layers[0].mlp.experts.gate_up_proj[0, :2], expected_gate)

    checkpoint = tmp_path / "btx"
    emitted = emit_internal_qwen_checkpoint(
        source_checkpoint=source,
        installed_layers=[installed_root],
        output_dir=checkpoint,
        format_version="test-v1",
        expected_allocated_payload_bytes=installed["allocated_payload_bytes"],
    )
    assert emitted["allocated_payload_bytes"] == installed["allocated_payload_bytes"]
    audit = audit_internal_qwen_checkpoint(checkpoint, runtime_reader=InternalBTXReader())
    assert audit["ok"], audit
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.mlp.experts.gate_up_proj" not in index
    assert "model.layers.1.mlp.experts.gate_up_proj" in index
    assert "model.embed_tokens.weight" in index

    layer_file = checkpoint / "btx-experts-layer-000.safetensors"
    damaged = bytearray(layer_file.read_bytes())
    damaged[-1] ^= 1
    layer_file.write_bytes(damaged)
    corrupt = audit_internal_qwen_checkpoint(checkpoint, runtime_reader=InternalBTXReader())
    assert not corrupt["ok"]
    assert any(item.startswith("file:") or item.startswith("tensor:") for item in corrupt["failures"])


def test_production_checkpoint_audit_fails_without_runtime_reader(tmp_path):
    source = tmp_path / "source"
    _source_checkpoint(source)
    shared_gu = torch.ones(4, dtype=torch.float16)
    shared_down = torch.ones(4, dtype=torch.float16)
    selected = [_choice(expert, projection, shared_gu, shared_down) for expert in range(2) for projection in ("gate_proj", "up_proj", "down_proj")]
    installed = tmp_path / "installed"
    install_layer_payloads(output_dir=installed, layer=0, predecessor_state_hash=H, source_checkpoint_sha256="2" * 64, fit_sha256="3" * 64, candidate_ledger_sha256="4" * 64, selected_choices=selected)
    checkpoint = tmp_path / "btx"
    emit_internal_qwen_checkpoint(source_checkpoint=source, installed_layers=[installed], output_dir=checkpoint, format_version="test-v1")
    report = audit_internal_qwen_checkpoint(checkpoint)
    assert not report["ok"]
    assert "runtime-reader-unavailable" in report["failures"]


def _official_choice(expert: int, projection: str, bits: int, shared_hidden, shared_down, *, intermediate: int = 256):
    if projection in {"gate_proj", "up_proj"}:
        reconstruction = torch.full((intermediate, 16), 0.1 + expert, dtype=torch.float16)
        trellis = torch.arange(1 * (intermediate // 16) * 16 * bits, dtype=torch.int32).to(torch.int16).reshape(1, intermediate // 16, 16 * bits)
        suh = shared_hidden
        svh = torch.ones(intermediate, dtype=torch.float16) * (expert + 1)
        topology = {"suh": "layer_shared", "svh": "expert_private"}
    else:
        reconstruction = torch.full((16, intermediate), 0.2 + expert, dtype=torch.float16)
        trellis = torch.arange((intermediate // 16) * 1 * 16 * bits, dtype=torch.int32).to(torch.int16).reshape(intermediate // 16, 1, 16 * bits)
        suh = torch.ones(intermediate, dtype=torch.float16) * (expert + 1)
        svh = shared_down
        topology = {"suh": "expert_private", "svh": "layer_shared"}
    return {
        "expert": expert,
        "projection": projection,
        "choice_id": f"official-E{expert}-{projection}-k{bits}",
        "bits": bits,
        "vector_topology": topology,
        "provenance": {"codec": "corrected-exl3-mcg-r10", "source": H},
        "tensors": {"trellis": trellis, "suh": suh, "svh": svh, "reconstruction": reconstruction},
    }


def _official_install(tmp_path: Path, bits_by_projection):
    hidden = torch.ones(16, dtype=torch.float16)
    down = -torch.ones(16, dtype=torch.float16)
    choices = [
        _official_choice(expert, projection, bits_by_projection[projection], hidden, down)
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    root = tmp_path / "official-installed"
    install_layer_payloads(
        output_dir=root,
        layer=0,
        predecessor_state_hash=H,
        source_checkpoint_sha256="2" * 64,
        fit_sha256="3" * 64,
        candidate_ledger_sha256="4" * 64,
        selected_choices=choices,
    )
    return root


def test_official_btx_schema_atom_pack_and_runtime_read(tmp_path):
    installed = _official_install(tmp_path, {"gate_proj": 3, "up_proj": 3, "down_proj": 3})
    report = btx_compatibility_report([installed])
    assert report["compatible"]
    assert report["upstream_commit"] == UPSTREAM_COMMIT
    root = tmp_path / "official-btx"
    installed_doc = json.loads((installed / "manifest.json").read_text())
    with pytest.raises(ValueError, match="allocator cost differs"):
        emit_official_btx_checkpoint(
            installed_layers=[installed],
            output_dir=tmp_path / "bad-official-cost",
            expected_allocated_payload_bytes=installed_doc["allocated_payload_bytes"] + 1,
        )
    manifest = emit_official_btx_checkpoint(
        installed_layers=[installed],
        output_dir=root,
        expected_allocated_payload_bytes=installed_doc["allocated_payload_bytes"],
    )
    assert set(path.name for path in root.iterdir()) == {
        "btx-manifest.json",
        "btx-accounting.json",
        "btx-layer-00000.safetensors",
    }
    assert manifest["schema"] == "btx-atoms-v1"
    assert manifest["rates"] == {"structure": "uniform", "bits": 3}
    from quant_pipeline.checkpoint.exact_payload import ExactCodecPayloadStore

    gate = next(row for row in installed_doc["choices"] if row["expert"] == 0 and row["projection"] == "gate_proj")
    trellis = ExactCodecPayloadStore(installed / "payload-store").load_tensor(gate["objects"]["trellis"])
    assert torch.equal(unpack_official_btx_plane(root, layer=0, slot=0, expert=0, projection="gate_proj", plane="low"), trellis[:, 0, :])
    assert torch.equal(unpack_official_btx_plane(root, layer=0, slot=0, expert=0, projection="gate_proj", plane="high"), trellis[:, 1, :])
    audit = audit_official_btx_checkpoint(root, runtime_reader=lambda _root: {"ok": True, "reader": "test"})
    assert audit["ok"], audit
    assert audit["accounting"]["source_semantic_allocated_payload_bytes"] == installed_doc["allocated_payload_bytes"]
    # Corrupt row padding rather than a trellis plane and prove the strict
    # reader catches the zero-padding invariant even if the manifest SHA is
    # updated to the corrupted container.
    from safetensors import safe_open

    layer = root / "btx-layer-00000.safetensors"
    with safe_open(layer, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    tensors["atoms"][0, -1] = 1
    save_file(tensors, layer, metadata=metadata)
    document = json.loads((root / "btx-manifest.json").read_text())
    from quant_pipeline.core.artifacts import sha256_file, write_json

    document["layers"]["0"]["sha256"] = sha256_file(layer)
    write_json(root / "btx-manifest.json", document)
    damaged = audit_official_btx_checkpoint(root, runtime_reader=lambda _root: {"ok": True})
    assert not damaged["ok"]
    assert "atom-padding:0:0" in damaged["failures"]


def test_official_btx_fails_closed_for_unexpressible_allocations(tmp_path):
    mismatched = _official_install(tmp_path, {"gate_proj": 3, "up_proj": 4, "down_proj": 3})
    report = btx_compatibility_report([mismatched])
    assert not report["compatible"]
    assert any("gate/up" in item for item in report["failures"])
    with pytest.raises(ValueError, match="not representable"):
        emit_official_btx_checkpoint(installed_layers=[mismatched], output_dir=tmp_path / "must-not-emit")


def test_official_btx_preserves_master_uniform_k5_support(tmp_path):
    installed = _official_install(tmp_path, {"gate_proj": 5, "up_proj": 5, "down_proj": 5})
    report = btx_compatibility_report([installed])
    assert report["compatible"], report
    assert report["rate_structure"] == "uniform"
    assert report["uniform_bits"] == 5
    root = tmp_path / "official-k5"
    manifest = emit_official_btx_checkpoint(installed_layers=[installed], output_dir=root)
    assert manifest["rates"] == {"structure": "uniform", "bits": 5}
    audit = audit_official_btx_checkpoint(root, runtime_reader=lambda _root: {"ok": True})
    assert audit["ok"], audit


def test_official_btx_reports_tp_extent_legality(tmp_path):
    hidden = torch.ones(16, dtype=torch.float16)
    down = -torch.ones(16, dtype=torch.float16)
    # Expert 0 K3 and expert 1 K4 forces per_expert_pair on 24 Qwen slots.
    choices = []
    for expert, bits in ((0, 3), (1, 4)):
        choices.extend(_official_choice(expert, projection, bits, hidden, down, intermediate=768) for projection in ("gate_proj", "up_proj", "down_proj"))
    installed = tmp_path / "tp-installed"
    install_layer_payloads(output_dir=installed, layer=0, predecessor_state_hash=H, source_checkpoint_sha256="2" * 64, fit_sha256="3" * 64, candidate_ledger_sha256="4" * 64, selected_choices=choices)
    research = btx_compatibility_report([installed], require_fused=False, target_tp_degrees=(1, 2, 3, 4))
    assert research["tp_compatibility"]["1"]["legal"]
    assert research["tp_compatibility"]["3"]["legal"]
    assert not research["tp_compatibility"]["2"]["legal"]
    assert not research["tp_compatibility"]["4"]["legal"]
