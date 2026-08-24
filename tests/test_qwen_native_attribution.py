import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from quant_pipeline.campaign.qwen_attribution import (
    load_teacher_logits,
    load_provisional_decoded_weights,
    measure_native_causal_attribution,
    persist_provisional_winner_deltas,
    verify_attribution_inputs,
    verify_provisional_winner_deltas,
    write_attribution_inputs,
)
from quant_pipeline.candidates.ledger import ProjectionTensors
from quant_pipeline.candidates.payload_store import ExactPayloadStore
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes
from quant_pipeline.normalization.prior_search import permute_expert_hf
from quant_pipeline.scoring.attribution import split_layer_damage


def _tiny_qwen():
    config = Qwen3MoeConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=32,
        use_cache=False,
    )
    torch.manual_seed(5)
    return Qwen3MoeForCausalLM(config).eval()


def _decoded(model):
    experts = model.model.layers[0].mlp.experts
    gate, up = experts.gate_up_proj.detach().chunk(2, dim=1)
    value = {
        "gate_proj": gate.clone(),
        "up_proj": up.clone(),
        "down_proj": experts.down_proj.detach().clone(),
    }
    # A directional, decoded-function perturbation large enough to stay above
    # BF16 noise while retaining the same router/top-k control.
    value["down_proj"][0] += 0.2
    return value


def test_teacher_loader_accepts_sealed_router_captures_but_requires_logits(tmp_path):
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    teacher = tmp_path / "teacher.safetensors"
    save_file(
        {
            "logits": expected,
            "router_logits.layer_000": np.ones((4, 2), dtype=np.float32),
        },
        teacher,
    )
    assert np.array_equal(load_teacher_logits(teacher), expected)

    missing = tmp_path / "missing-logits.safetensors"
    save_file({"router_logits.layer_000": np.ones((4, 2), dtype=np.float32)}, missing)
    with pytest.raises(ValueError, match="must contain a logits tensor"):
        load_teacher_logits(missing)


def test_tiny_real_qwen_native_path_gradient_fisher_closure_and_tamper(tmp_path):
    arrays = measure_native_causal_attribution(
        model=_tiny_qwen(),
        token_ids=[1, 2, 3, 4, 5],
        decoded_by_layer={0: _decoded(_tiny_qwen())},
        path_nodes=5,
        fisher_rank=3,
        seed=19,
    )
    assert arrays["source_kld"].item() == pytest.approx(0.0, abs=1e-15)
    assert arrays["candidate_kld"].item() > 0.0
    assert np.all(arrays["path_gradients"][:, 0, 0] > 0.0)
    assert arrays["measured_layer_damage"].sum() == pytest.approx(
        arrays["measured_end_to_end_delta"].item(), rel=2e-5, abs=2e-12
    )
    split = split_layer_damage(
        float(arrays["measured_layer_damage"][0]),
        arrays["projected_expert_residuals"][0],
        projected_routing_residual=arrays["projected_routing_residuals"][0],
    )
    assert split["closed_total"] == pytest.approx(float(arrays["measured_layer_damage"][0]), abs=1e-18)
    assert split["routing_state_shift"] == pytest.approx(0.0, abs=1e-18)

    path = write_attribution_inputs(
        tmp_path / "attribution-inputs.npz",
        arrays,
        provenance={"fixture": "tiny-real-qwen", "test_only": True},
    )
    verified, receipt = verify_attribution_inputs(path)
    assert verified["candidate_kld"].item() == arrays["candidate_kld"].item()
    assert receipt["provenance"]["fixture"] == "tiny-real-qwen"
    altered = dict(arrays)
    altered["path_nodes"] = arrays["path_nodes"][::-1].copy()
    with pytest.raises(ValueError, match="canonical Gauss-Legendre"):
        write_attribution_inputs(
            tmp_path / "semantically-resealed-tamper.npz",
            altered,
            provenance={"fixture": "tiny-real-qwen", "test_only": True},
        )
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="byte identity"):
        verify_attribution_inputs(path)


def test_hf_uniform_k4_production_provenance_is_sealed_and_fail_closed(tmp_path):
    nodes, _weights = np.polynomial.legendre.leggauss(2)
    nodes = (nodes + 1.0) / 2.0
    arrays = {
        "layer_indices": np.asarray([0], dtype=np.int32),
        "layer_deltas": np.ones((1, 1), dtype=np.float64),
        "path_nodes": nodes,
        "path_gradients": np.full((2, 1, 1), 0.2, dtype=np.float64),
        "node_kld": np.asarray([0.04, 0.16], dtype=np.float64),
        "projected_expert_residuals": np.zeros((1, 2, 2, 1), dtype=np.float64),
        "projected_routing_residuals": np.zeros((1, 2, 1), dtype=np.float64),
        "measured_layer_damage": np.asarray([0.2], dtype=np.float64),
        "source_kld": np.asarray([0.0], dtype=np.float64),
        "candidate_kld": np.asarray([0.2], dtype=np.float64),
        "measured_end_to_end_delta": np.asarray([0.2], dtype=np.float64),
    }
    provenance = {
        "implementation": "native-qwen-hf-uniform-k4-mcg-blend-fisher-v2",
        "model_revision": "1" * 40,
        "kld_window_seal_sha256": "2" * 64,
        "teacher_reference_sha256": "3" * 64,
        "candidate_inventory_sha256": "4" * 64,
        "candidate_dataset_repo": "owner/reproducibility",
        "candidate_dataset_revision": "5" * 40,
        "provisional_bit_triplet": [4, 4, 4],
        "path_nodes": 2,
        "fisher_rank": 1,
        "seed": 7,
        "test_only": False,
    }
    path = write_attribution_inputs(tmp_path / "hf-native.npz", arrays, provenance=provenance)
    _verified, receipt = verify_attribution_inputs(path)
    assert receipt["provenance"] == provenance
    wrong = dict(provenance, provisional_bit_triplet=[3, 4, 4])
    with pytest.raises(ValueError, match="uniform-K4"):
        write_attribution_inputs(tmp_path / "wrong-anchor.npz", arrays, provenance=wrong)


def test_provisional_winners_persist_actual_codec_delta_and_bind_source(tmp_path):
    model = _tiny_qwen()
    experts = model.model.layers[0].mlp.experts
    gate, up = experts.gate_up_proj.detach().chunk(2, dim=1)
    decoded = {"gate_proj": [], "up_proj": [], "down_proj": []}
    permutation = tuple(reversed(range(gate.shape[1])))
    for expert in range(2):
        permuted = permute_expert_hf(gate[expert], up[expert], experts.down_proj[expert], permutation)
        for name, value in zip(("gate_proj", "up_proj", "down_proj"), permuted, strict=True):
            decoded[name].append(value)
    decoded = {name: torch.stack(values) for name, values in decoded.items()}
    decoded["down_proj"][0] += 0.2
    store = ExactPayloadStore(tmp_path / "payloads")
    units = []
    candidates = []
    refs = []
    for expert in range(2):
        source = ProjectionTensors(gate[expert].clone(), up[expert].clone(), experts.down_proj[expert].detach().clone())
        unit = SimpleNamespace(unit_id=f"L0.E{expert}", source=source)
        units.append(unit)
        projections = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            ref = store.put_tensor(decoded[name][expert], role=f"{name}.reconstruction_hf")
            refs.append(ref)
            projections[name] = {"exact_payload_refs": {"reconstruction_hf": ref}}
        record = {
            "unit_id": unit.unit_id,
            "layer": 0,
            "expert": expert,
            "candidate_id": f"{unit.unit_id}.g4u4d4",
            "bit_triplet": [4, 4, 4],
            "projections": projections,
        }
        record["record_sha256"] = sha256_bytes(canonical_json(record))
        candidates.append(record)
    payload_manifest = store.manifest(refs)
    ledger = {"candidates": candidates, "exact_payload_store": payload_manifest}
    ledger["ledger_sha256"] = sha256_bytes(canonical_json(ledger))
    manifest_path = persist_provisional_winner_deltas(
        output_dir=tmp_path / "provisional",
        ledger=ledger,
        payload_store_root=store.root,
        checkpoint_sources={(0, expert): unit.source for expert, unit in enumerate(units)},
        bit_triplet=(4, 4, 4),
    )
    manifest = verify_provisional_winner_deltas(manifest_path, payload_store_root=store.root)
    assert manifest["source_control"].startswith("alpha-zero")
    loaded = load_provisional_decoded_weights(model, manifest_path, payload_store_root=store.root)
    assert torch.equal(loaded[0]["down_proj"], decoded["down_proj"])

    document = json.loads(manifest_path.read_text())
    document["layers"][0]["winners"][0]["projections"]["down_proj"]["source_sha256"] = "0" * 64
    document["manifest_sha256"] = sha256_bytes(canonical_json({
        key: value for key, value in document.items() if key != "manifest_sha256"
    }))
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="source tensor differs"):
        load_provisional_decoded_weights(model, manifest_path, payload_store_root=store.root)
