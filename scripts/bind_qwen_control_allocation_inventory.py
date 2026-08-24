#!/usr/bin/env python3
"""Bind a sealed historical allocation to the audited candidate inventory."""

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
    observed = _hash_json({key: value for key, value in document.items() if key != field})
    if document.get(field) != observed:
        raise ValueError(f"{label} seal mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-control-inventory-binding-plan.v1",
        "allocation": str(args.allocation.resolve()),
        "allocation_file_sha256": sha256_file(args.allocation),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    allocation = json.loads(args.allocation.read_text())
    inventory = json.loads(args.candidate_inventory.read_text())
    _verify(allocation, "allocation_sha256", "historical allocation")
    _verify(inventory, "inventory_sha256", "candidate inventory")
    if allocation.get("candidate_inventory_sha256") not in (None, inventory["inventory_sha256"]):
        raise ValueError("historical allocation is bound to a different candidate inventory")
    historical_sha256 = allocation.pop("allocation_sha256")
    allocation["candidate_inventory_sha256"] = inventory["inventory_sha256"]
    allocation["historical_allocation_sha256"] = historical_sha256
    allocation["inventory_binding"] = "choice-preserving-audited-candidate-inventory-v1"
    allocation["allocation_sha256"] = _hash_json(allocation)
    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    write_json(output / "allocation.json", allocation)
    print(json.dumps({
        "ok": True,
        "allocation_sha256": allocation["allocation_sha256"],
        "historical_allocation_sha256": historical_sha256,
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "choice_count": len(allocation.get("choices", ())),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
