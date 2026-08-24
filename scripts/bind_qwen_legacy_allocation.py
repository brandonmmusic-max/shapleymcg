#!/usr/bin/env python3
"""Bind a sealed historical Qwen allocation to an explicit candidate inventory.

Older allocations sealed every layer receipt but predated the top-level
``candidate_inventory_sha256`` field required by the causal measurement
runners. This migration changes no matrix choice. It succeeds only when all 48
candidate tensor, receipt, size, and layer identities match the new inventory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import (
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify(document: dict[str, Any], field: str, label: str) -> None:
    if document.get(field) != _hash_json({key: value for key, value in document.items() if key != field}):
        raise ValueError(f"{label} seal mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-allocation", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    legacy = json.loads(args.legacy_allocation.read_text())
    inventory = json.loads(args.candidate_inventory.read_text())
    _verify(legacy, "allocation_sha256", "legacy allocation")
    _verify(inventory, "inventory_sha256", "candidate inventory")

    if (
        legacy.get("average_weight_bits") != 3.5
        or legacy.get("k3_count") != 9216
        or legacy.get("k4_count") != 9216
        or len(legacy.get("choices", ())) != 18432
    ):
        raise ValueError("legacy allocation is not exact 3.5 routed-expert BPW")

    legacy_receipts = {int(row["layer"]): row for row in legacy.get("encode_receipts", ())}
    inventory_layers = {int(row["layer"]): row for row in inventory.get("layers", ())}
    if set(legacy_receipts) != set(range(48)) or set(inventory_layers) != set(range(48)):
        raise ValueError("allocation or inventory does not cover all 48 layers")

    bindings = []
    for layer in range(48):
        old = legacy_receipts[layer]
        new = inventory_layers[layer]
        expected = {
            "candidate_tensor_sha256": new["candidate_sha256"],
            "candidate_tensor_bytes": int(new["candidate_bytes"]),
            "receipt_sha256": new["receipt_sha256"],
            "receipt_file_sha256": new["receipt_file_sha256"],
        }
        observed = {
            "candidate_tensor_sha256": old["candidate_tensor_sha256"],
            "candidate_tensor_bytes": int(old["candidate_tensor_bytes"]),
            "receipt_sha256": old["receipt_sha256"],
            "receipt_file_sha256": old["receipt_file_sha256"],
        }
        if observed != expected:
            raise ValueError(f"legacy allocation candidate identity differs at layer {layer}")
        bindings.append({"layer": layer, **expected})

    body = {key: value for key, value in legacy.items() if key != "allocation_sha256"}
    body.update({
        "schema": "quant-pipeline.qwen-bound-legacy-allocation.v1",
        "legacy_schema": legacy["schema"],
        "legacy_allocation_sha256": legacy["allocation_sha256"],
        "legacy_allocation_file_sha256": sha256_file(args.legacy_allocation),
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "candidate_bindings": bindings,
        "migration_effect": "identity-only; zero matrix choices changed",
    })
    body["allocation_sha256"] = _hash_json(body)

    plan = {
        "legacy_allocation": str(args.legacy_allocation.resolve()),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "output": str(args.output.resolve()),
        "bound_allocation_sha256": body["allocation_sha256"],
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "bound-allocation.json", body)
    write_json(output / "binding-plan.json", plan | {"dry_run": False})
    reread = json.loads((output / "bound-allocation.json").read_text())
    _verify(reread, "allocation_sha256", "bound allocation")
    if reread["choices"] != legacy["choices"]:
        raise RuntimeError("binding migration changed matrix choices")
    print(json.dumps({
        "ok": True,
        "allocation_sha256": body["allocation_sha256"],
        "legacy_allocation_sha256": legacy["allocation_sha256"],
        "changed_matrix_choices": 0,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
