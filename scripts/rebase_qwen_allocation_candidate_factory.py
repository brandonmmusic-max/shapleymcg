#!/usr/bin/env python3
"""Bind one exact Qwen rate allocation to another candidate factory's bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(value: dict[str, Any], field: str, label: str) -> None:
    if value.get(field) != _hash_json({key: row for key, row in value.items() if key != field}):
        raise ValueError(f"{label} seal mismatch")


def _load_candidate_rows(
    encode_root: Path,
    inventory: dict[str, Any],
) -> dict[tuple[int, int, str, int], dict[str, Any]]:
    by_layer = {int(row["layer"]): row for row in inventory["layers"]}
    if set(by_layer) != set(range(48)):
        raise ValueError("candidate inventory must cover 48 layers")
    rows: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for layer in range(48):
        layer_root = encode_root / f"layer-{layer:03d}"
        receipt_path = layer_root / "encode-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        _verify_seal(receipt, "receipt_sha256", f"layer {layer} encode receipt")
        inventory_row = by_layer[layer]
        candidate_path = layer_root / str(receipt["candidate_tensor_file"])
        if (
            receipt.get("layer") != layer
            or receipt.get("experts") != list(range(128))
            or receipt["receipt_sha256"] != inventory_row["receipt_sha256"]
            or int(receipt["candidate_tensor_bytes"]) != int(inventory_row["candidate_bytes"])
            or receipt["candidate_tensor_sha256"] != inventory_row["candidate_sha256"]
            or candidate_path.stat().st_size != int(inventory_row["candidate_bytes"])
            or sha256_file(candidate_path) != inventory_row["candidate_sha256"]
        ):
            raise ValueError(f"layer {layer} local candidate differs from Hub inventory")
        for row in receipt["scores"]:
            key = (
                int(row["layer"]),
                int(row["expert"]),
                str(row["projection"]),
                int(row["bits"]),
            )
            if key in rows:
                raise ValueError(f"duplicate candidate score row: {key}")
            rows[key] = dict(row)
    if len(rows) != 48 * 128 * 3 * 2:
        raise ValueError("candidate score inventory is incomplete")
    return rows


def rebase_allocation(
    source: dict[str, Any],
    inventory: dict[str, Any],
    candidate_rows: dict[tuple[int, int, str, int], dict[str, Any]],
) -> dict[str, Any]:
    _verify_seal(source, "allocation_sha256", "source rate allocation")
    _verify_seal(inventory, "inventory_sha256", "candidate inventory")
    source_choices = source.get("choices", [])
    if (
        source.get("average_weight_bits") != 3.5
        or source.get("k3_count") != 9216
        or source.get("k4_count") != 9216
        or len(source_choices) != 18432
    ):
        raise ValueError("source allocation is not exact 3.5 routed-expert BPW")
    choices = []
    seen = set()
    for source_row in source_choices:
        identity = (
            int(source_row["layer"]),
            int(source_row["expert"]),
            str(source_row["projection"]),
        )
        bits = int(source_row["bits"])
        if identity in seen or bits not in {3, 4}:
            raise ValueError("source allocation has duplicate or invalid choices")
        seen.add(identity)
        candidate = candidate_rows[(*identity, bits)]
        choices.append({
            "layer": identity[0],
            "expert": identity[1],
            "projection": identity[2],
            "bits": bits,
            "stored_bytes": int(candidate["stored_bytes"]),
            "diagonal_p2_damage": float(candidate["diagonal_p2_damage"]),
            "packed_sha256": candidate["packed_sha256"],
            "codec_fp16_reconstruction_sha256": candidate["codec_fp16_reconstruction_sha256"],
            "stored_bf16_reconstruction_sha256": candidate["stored_bf16_reconstruction_sha256"],
            "source_rate_choice_sha256": _hash_json(source_row),
        })
    if len(seen) != 18432:
        raise ValueError("source allocation matrix inventory is incomplete")
    body = {
        "schema": "quant-pipeline.qwen-fast-k34-allocation.v2",
        "objective": "frozen-full-causal-rate-allocation-rebound-to-candidate-factory",
        "rate_scope": source.get("rate_scope", "moe-expert-weight-elements"),
        "average_weight_bits": 3.5,
        "unit_count": len(choices),
        "k3_count": 9216,
        "k4_count": 9216,
        "stored_payload_bytes": sum(row["stored_bytes"] for row in choices),
        "predicted_diagonal_p2_damage": sum(row["diagonal_p2_damage"] for row in choices),
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "source_rate_allocation_sha256": source["allocation_sha256"],
        "source_rate_choice_inventory_sha256": _hash_json(source_choices),
        "factory_rebase_changes_rate_choices": False,
        "choices": choices,
    }
    body["allocation_sha256"] = _hash_json(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-allocation", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "source_allocation": str(args.source_allocation.resolve()),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "encode_root": str(args.encode_root.resolve()),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    source = json.loads(args.source_allocation.read_text())
    inventory = json.loads(args.candidate_inventory.read_text())
    rows = _load_candidate_rows(args.encode_root.resolve(), inventory)
    allocation = rebase_allocation(source, inventory, rows)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, allocation)
    print(json.dumps({
        "ok": True,
        "allocation_sha256": allocation["allocation_sha256"],
        "source_rate_allocation_sha256": allocation["source_rate_allocation_sha256"],
        "stored_payload_bytes": allocation["stored_payload_bytes"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
