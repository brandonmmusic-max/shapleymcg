from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "measure_qwen_mcg_factory_union",
    ROOT / "scripts" / "measure_qwen_mcg_factory_union.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_factory_union_requires_complete_bound_exact_rate_inventories(tmp_path):
    inventory = {
        "schema": "quant-pipeline.qwen-hf-mcg-candidate-inventory.v2",
        "repo_id": "owner/repo",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "path_prefix": "factory",
        "layers": [{"layer": layer} for layer in range(48)],
    }
    inventory["inventory_sha256"] = MODULE._hash_json(inventory)
    inventory_path = tmp_path / "inventory.json"
    _write(inventory_path, inventory)
    loaded_inventory = MODULE._load_inventory(inventory_path, "fixture")

    choices = []
    bits = (3, 4)
    index = 0
    for layer in range(48):
        for expert in range(128):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                choices.append({
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "bits": bits[index % 2],
                })
                index += 1
    allocation = {
        "schema": "quant-pipeline.qwen-fast-k34-allocation.v2",
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "average_weight_bits": 3.5,
        "k3_count": 9216,
        "k4_count": 9216,
        "choices": choices,
    }
    allocation["allocation_sha256"] = MODULE._hash_json(allocation)
    allocation_path = tmp_path / "allocation.json"
    _write(allocation_path, allocation)

    loaded, by_identity = MODULE._load_allocation(
        allocation_path, loaded_inventory, "fixture"
    )
    assert loaded["allocation_sha256"] == allocation["allocation_sha256"]
    assert len(by_identity) == 48 * 128 * 3
    assert sum(int(row["bits"]) == 3 for row in by_identity.values()) == 9216


def test_factory_union_rejects_inventory_binding_drift(tmp_path):
    inventory = {
        "layers": [{"layer": layer} for layer in range(48)],
    }
    inventory["inventory_sha256"] = MODULE._hash_json(inventory)
    allocation = {
        "candidate_inventory_sha256": "f" * 64,
        "average_weight_bits": 3.5,
        "k3_count": 9216,
        "k4_count": 9216,
        "choices": [],
    }
    allocation["allocation_sha256"] = MODULE._hash_json(allocation)
    path = tmp_path / "allocation.json"
    _write(path, allocation)
    try:
        MODULE._load_allocation(path, inventory, "drifted")
    except ValueError as error:
        assert "bound exact-3.5" in str(error)
    else:
        raise AssertionError("factory union accepted an allocation from another inventory")
