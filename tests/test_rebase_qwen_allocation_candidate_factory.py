from __future__ import annotations

import importlib.util
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rebase_qwen_allocation_candidate_factory",
    ROOT / "scripts" / "rebase_qwen_allocation_candidate_factory.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _seal(value, field):
    result = dict(value)
    result[field] = sha256_bytes(canonical_json(result))
    return result


def test_rebase_preserves_every_rate_choice_and_replaces_candidate_identity():
    source_choices = []
    candidates = {}
    index = 0
    for layer in range(48):
        for expert in range(128):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                bit = 3 if index % 2 == 0 else 4
                source_choices.append({
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "bits": bit,
                    "stored_bytes": bit * 10,
                    "stored_bf16_reconstruction_sha256": "a" * 64,
                })
                for candidate_bit in (3, 4):
                    candidates[(layer, expert, projection, candidate_bit)] = {
                        "stored_bytes": candidate_bit * 11,
                        "diagonal_p2_damage": float(index + candidate_bit),
                        "packed_sha256": "b" * 64,
                        "codec_fp16_reconstruction_sha256": "c" * 64,
                        "stored_bf16_reconstruction_sha256": "d" * 64,
                    }
                index += 1
    source = _seal({
        "schema": "quant-pipeline.qwen-fast-k34-allocation.v2",
        "average_weight_bits": 3.5,
        "k3_count": 9216,
        "k4_count": 9216,
        "rate_scope": "moe-expert-weight-elements",
        "choices": source_choices,
    }, "allocation_sha256")
    inventory = _seal({
        "schema": "quant-pipeline.qwen-hf-mcg-candidate-inventory.v2",
        "layers": [{"layer": layer} for layer in range(48)],
    }, "inventory_sha256")

    rebound = MODULE.rebase_allocation(source, inventory, candidates)

    assert rebound["factory_rebase_changes_rate_choices"] is False
    assert rebound["source_rate_allocation_sha256"] == source["allocation_sha256"]
    assert [row["bits"] for row in rebound["choices"]] == [
        row["bits"] for row in source_choices
    ]
    assert {row["stored_bf16_reconstruction_sha256"] for row in rebound["choices"]} == {
        "d" * 64
    }
    assert rebound["k3_count"] == rebound["k4_count"] == 9216
    assert rebound["average_weight_bits"] == 3.5
    body = {key: value for key, value in rebound.items() if key != "allocation_sha256"}
    assert rebound["allocation_sha256"] == sha256_bytes(canonical_json(body))
