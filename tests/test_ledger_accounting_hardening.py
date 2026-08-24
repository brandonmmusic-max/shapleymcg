# ruff: noqa: E402

import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from quant_pipeline.allocation.global_dp import Candidate, allocate_with_fixed_layer_cost
from quant_pipeline.checkpoint.btx_qwen import (
    InternalBTXReader,
    audit_internal_qwen_checkpoint,
    emit_internal_qwen_checkpoint,
    install_layer_payloads,
    reconcile_installed_allocation,
)
from quant_pipeline.checkpoint.exact_payload import (
    CHOICE_SCHEMA,
    PACKED_HASH_SCHEMA,
    ExactCodecPayloadStore,
    packed_payload_sha256,
)
from quant_pipeline.checkpoint.official_btx import (
    audit_official_btx_checkpoint,
    emit_official_btx_checkpoint,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


H = "1" * 64


def _choice(expert, projection, shared_hidden, shared_down, *, layer=None):
    hidden, intermediate, bits = 16, 256, 3
    if projection in {"gate_proj", "up_proj"}:
        reconstruction = torch.full((intermediate, hidden), 0.1 + expert, dtype=torch.float16)
        trellis = torch.arange(
            (hidden // 16) * (intermediate // 16) * 16 * bits,
            dtype=torch.int32,
        ).to(torch.int16).reshape(hidden // 16, intermediate // 16, 16 * bits)
        suh = shared_hidden
        svh = torch.ones(intermediate, dtype=torch.float16) * (expert + 1)
        topology = {"suh": "layer_shared", "svh": "expert_private"}
    else:
        reconstruction = torch.full((hidden, intermediate), 0.2 + expert, dtype=torch.float16)
        trellis = torch.arange(
            (intermediate // 16) * (hidden // 16) * 16 * bits,
            dtype=torch.int32,
        ).to(torch.int16).reshape(intermediate // 16, hidden // 16, 16 * bits)
        suh = torch.ones(intermediate, dtype=torch.float16) * (expert + 1)
        svh = shared_down
        topology = {"suh": "expert_private", "svh": "layer_shared"}
    candidate_id = f"E{expert}-k3" if layer is None else f"L{layer}.E{expert}.g3u3d3"
    provenance = {"codec": "corrected-exl3-mcg-r10", "source": H}
    if layer is not None:
        provenance["selected_candidate_record_sha256"] = sha256_bytes(
            f"L{layer}.E{expert}".encode()
        )
    return {
        "expert": expert,
        "projection": projection,
        "choice_id": candidate_id,
        "bits": bits,
        "vector_topology": topology,
        "provenance": provenance,
        "tensors": {
            "trellis": trellis,
            "suh": suh,
            "svh": svh,
            "reconstruction": reconstruction,
        },
    }


def _source_checkpoint(root):
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "hidden_size": 16,
                "moe_intermediate_size": 256,
            }
        )
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 16),
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 512, 16),
        "model.layers.0.mlp.experts.down_proj": torch.randn(2, 16, 256),
        "model.layers.1.mlp.experts.gate_up_proj": torch.randn(2, 512, 16),
        "model.layers.1.mlp.experts.down_proj": torch.randn(2, 16, 256),
        "model.norm.weight": torch.randn(16),
    }
    save_file(tensors, root / "model.safetensors")


def _allocation_cost(installed_manifests):
    layer_rows = []
    candidate_hashes = []
    for manifest in installed_manifests:
        layer = int(manifest["layer"])
        cost = manifest["cost_breakdown"]
        identities = [
            {
                "record_sha256": sha256_bytes(f"L{layer}.E{expert}".encode()),
                "candidate_id": f"L{layer}.E{expert}.g3u3d3",
                "unit_id": f"L{layer}.E{expert}",
            }
            for expert in range(2)
        ]
        identities.sort(key=lambda row: row["record_sha256"])
        hashes = [row["record_sha256"] for row in identities]
        candidate_hashes.extend(hashes)
        layer_rows.append(
            {
                "layer": layer,
                "selected_candidate_record_sha256": hashes,
                "selected_candidate_identities": identities,
                "semantic_expert_private_bytes": cost["semantic_expert_private_bytes"],
                "semantic_layer_shared_objects": cost["semantic_layer_shared_objects"],
                "semantic_layer_shared_bytes": cost["semantic_layer_shared_bytes"],
                "allocated_payload_bytes": cost["allocated_payload_bytes"],
            }
        )
    private = sum(row["semantic_expert_private_bytes"] for row in layer_rows)
    shared = sum(row["semantic_layer_shared_bytes"] for row in layer_rows)
    body = {
        "schema": "quant-pipeline.selected-allocation-cost.v1",
        "selected_candidate_record_sha256": sorted(candidate_hashes),
        "semantic_expert_private_bytes": private,
        "layer_shared_costs": [
            {
                "layer": row["layer"],
                "objects": row["semantic_layer_shared_objects"],
                "semantic_layer_shared_bytes": row["semantic_layer_shared_bytes"],
            }
            for row in layer_rows
        ],
        "selected_layer_costs": layer_rows,
        "semantic_layer_shared_bytes": shared,
        "allocated_payload_bytes": private + shared,
    }
    body["allocation_cost_sha256"] = sha256_bytes(canonical_json(body))
    return body


def test_packed_payload_identity_is_length_framed_and_legacy_choices_fail_closed(tmp_path):
    # Both triples have identical unframed concatenation: b"abc".
    left = {"trellis": b"a", "suh": b"bc", "svh": b""}
    right = {"trellis": b"ab", "suh": b"c", "svh": b""}
    assert b"".join(left.values()) == b"".join(right.values())
    assert packed_payload_sha256(left) != packed_payload_sha256(right)

    store = ExactCodecPayloadStore(tmp_path / "store")
    choice = store.put_choice(
        layer=0,
        expert=0,
        projection="gate_proj",
        choice_id="framed",
        bits=3,
        trellis=torch.zeros(3, dtype=torch.int16),
        suh=torch.ones(4, dtype=torch.float16),
        svh=torch.ones(2, dtype=torch.float16),
        reconstruction=torch.ones((2, 4), dtype=torch.float16),
        vector_topology={"suh": "layer_shared", "svh": "expert_private"},
        provenance={"test": True},
        predecessor_state_hash=H,
    )
    assert choice["schema"] == CHOICE_SCHEMA
    assert choice["packed_hash_schema"] == PACKED_HASH_SCHEMA
    legacy = dict(choice) | {"schema": "quant-pipeline.exact-codec-choice.v1"}
    with pytest.raises(ValueError, match="legacy unframed v1"):
        store.verify_choice(legacy)


def test_illegal_shared_vector_family_is_rejected_during_install(tmp_path):
    shared_hidden = torch.ones(16, dtype=torch.float16)
    shared_down = -torch.ones(16, dtype=torch.float16)
    choice = _choice(0, "gate_proj", shared_hidden, shared_down)
    choice["vector_topology"] = {"suh": "expert_private", "svh": "layer_shared"}
    with pytest.raises(ValueError, match="illegal layer-shared vector family"):
        install_layer_payloads(
            output_dir=tmp_path / "illegal",
            layer=0,
            predecessor_state_hash=H,
            source_checkpoint_sha256="2" * 64,
            fit_sha256="3" * 64,
            candidate_ledger_sha256="4" * 64,
            selected_choices=[choice],
        )


def test_two_layer_allocation_install_emit_and_audit_reconcile(tmp_path):
    shared_hidden = torch.ones(16, dtype=torch.float16)
    shared_down = -torch.ones(16, dtype=torch.float16)
    installed = []
    roots = []
    for layer in range(2):
        root = tmp_path / f"installed-{layer}"
        choices = [
            _choice(expert, projection, shared_hidden, shared_down, layer=layer)
            for expert in range(2)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]
        installed.append(
            install_layer_payloads(
                output_dir=root,
                layer=layer,
                predecessor_state_hash=H,
                source_checkpoint_sha256="2" * 64,
                fit_sha256="3" * 64,
                candidate_ledger_sha256="4" * 64,
                selected_choices=choices,
            )
        )
        roots.append(root)
    cost = _allocation_cost(installed)
    candidates = [
        Candidate(
            unit_id=f"L{row['layer']}",
            choice_id=f"selected-L{row['layer']}",
            stored_bytes=row["semantic_expert_private_bytes"],
            predicted_damage=float(row["layer"]),
        )
        for row in cost["selected_layer_costs"]
    ]
    allocated = allocate_with_fixed_layer_cost(
        candidates,
        byte_budget=cost["allocated_payload_bytes"],
        fixed_layer_shared_bytes=cost["semantic_layer_shared_bytes"],
    )
    assert allocated.stored_bytes == cost["allocated_payload_bytes"]
    reconciliation = reconcile_installed_allocation(cost, roots)
    assert reconciliation["allocated_payload_bytes"] == allocated.stored_bytes

    source = tmp_path / "source"
    _source_checkpoint(source)
    internal = tmp_path / "internal"
    emit_internal_qwen_checkpoint(
        source_checkpoint=source,
        installed_layers=roots,
        output_dir=internal,
        format_version="accounting-test-v1",
        expected_allocated_payload_bytes=reconciliation["allocated_payload_bytes"],
    )
    internal_audit = audit_internal_qwen_checkpoint(internal, runtime_reader=InternalBTXReader())
    assert internal_audit["ok"], internal_audit
    assert internal_audit["allocated_payload_bytes"] == allocated.stored_bytes

    official = tmp_path / "official"
    emit_official_btx_checkpoint(
        installed_layers=roots,
        output_dir=official,
        expected_allocated_payload_bytes=reconciliation["allocated_payload_bytes"],
    )
    official_audit = audit_official_btx_checkpoint(
        official,
        runtime_reader=lambda _root: {"ok": True, "reader": "test"},
    )
    assert official_audit["ok"], official_audit
    assert official_audit["accounting"]["source_semantic_allocated_payload_bytes"] == allocated.stored_bytes

    tampered = json.loads(json.dumps(cost))
    tampered["selected_layer_costs"][0]["allocated_payload_bytes"] += 1
    tampered["allocation_cost_sha256"] = sha256_bytes(
        canonical_json({key: value for key, value in tampered.items() if key != "allocation_cost_sha256"})
    )
    with pytest.raises(ValueError, match="layer total differs"):
        reconcile_installed_allocation(tampered, roots)

    for field, value, message in (
        ("semantic_expert_private_bytes", cost["semantic_expert_private_bytes"] + 1, "top-level private"),
        ("semantic_layer_shared_bytes", cost["semantic_layer_shared_bytes"] + 1, "top-level shared"),
        ("allocated_payload_bytes", cost["allocated_payload_bytes"] + 1, "top-level total"),
    ):
        tampered = json.loads(json.dumps(cost))
        tampered[field] = value
        tampered["allocation_cost_sha256"] = sha256_bytes(
            canonical_json({key: item for key, item in tampered.items() if key != "allocation_cost_sha256"})
        )
        with pytest.raises(ValueError, match=message):
            reconcile_installed_allocation(tampered, roots)

    tampered = json.loads(json.dumps(cost))
    tampered["layer_shared_costs"][0]["semantic_layer_shared_bytes"] += 2
    tampered["allocation_cost_sha256"] = sha256_bytes(
        canonical_json({key: value for key, value in tampered.items() if key != "allocation_cost_sha256"})
    )
    with pytest.raises(ValueError, match="layer-shared summary"):
        reconcile_installed_allocation(tampered, roots)

    tampered = json.loads(json.dumps(cost))
    tampered["selected_layer_costs"][0]["selected_candidate_identities"][0]["candidate_id"] = "forged"
    tampered["allocation_cost_sha256"] = sha256_bytes(
        canonical_json({key: value for key, value in tampered.items() if key != "allocation_cost_sha256"})
    )
    with pytest.raises(ValueError, match="choice ID differs"):
        reconcile_installed_allocation(tampered, roots)

    # The audit must also reject a canonically re-sealed illegal shared family.
    manifest_path = internal / "BTX_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    gate_unit = next(row for row in manifest["units"].values() if row["projection"] == "gate_proj")
    gate_unit["vector_topology"] = {"suh": "expert_private", "svh": "layer_shared"}
    manifest["manifest_sha256"] = sha256_bytes(
        canonical_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    write_json(manifest_path, manifest)
    (internal / "BTX_MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  BTX_MANIFEST.json\n"
    )
    damaged = audit_internal_qwen_checkpoint(internal, runtime_reader=InternalBTXReader())
    assert not damaged["ok"]
    assert any("illegal layer-shared vector family" in failure for failure in damaged["failures"])


def _reseal_installed_manifest(root, mutate):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest["installed_checkpoint_sha256"] = sha256_bytes(
        canonical_json(
            {key: value for key, value in manifest.items() if key != "installed_checkpoint_sha256"}
        )
    )
    write_json(manifest_path, manifest)


@pytest.mark.parametrize("mutation", ("layer", "predecessor", "duplicate"))
def test_resealed_installed_manifest_cannot_swap_causal_identity(tmp_path, mutation):
    shared_hidden = torch.ones(16, dtype=torch.float16)
    shared_down = -torch.ones(16, dtype=torch.float16)
    original = tmp_path / "installed-original"
    installed = install_layer_payloads(
        output_dir=original,
        layer=0,
        predecessor_state_hash=H,
        source_checkpoint_sha256="2" * 64,
        fit_sha256="3" * 64,
        candidate_ledger_sha256="4" * 64,
        selected_choices=[
            _choice(expert, projection, shared_hidden, shared_down, layer=0)
            for expert in range(2)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ],
    )
    damaged = tmp_path / f"installed-{mutation}"
    shutil.copytree(original, damaged)

    def mutate(manifest):
        if mutation == "layer":
            manifest["layer"] = 1
        elif mutation == "predecessor":
            manifest["predecessor_state_hash"] = "9" * 64
        else:
            manifest["choices"].append(json.loads(json.dumps(manifest["choices"][0])))
            manifest["choice_sha256"].append(manifest["choice_sha256"][0])

    _reseal_installed_manifest(damaged, mutate)
    message = {
        "layer": "choice layer differs",
        "predecessor": "choice predecessor differs",
        "duplicate": "unique by expert and projection",
    }[mutation]
    with pytest.raises(ValueError, match=message):
        from quant_pipeline.checkpoint.btx_qwen import verify_installed_layer

        verify_installed_layer(damaged)
    with pytest.raises(ValueError, match=message):
        reconcile_installed_allocation(_allocation_cost([installed]), [damaged])
