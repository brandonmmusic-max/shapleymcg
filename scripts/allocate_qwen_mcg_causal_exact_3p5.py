#!/usr/bin/env python3
"""Allocate sealed MCG K3/K4 matrices with native Shapley/Fisher calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    observed = _hash_json({key: value for key, value in document.items() if key != field})
    if document.get(field) != observed:
        raise ValueError(f"{label} seal mismatch")


def _load_receipt(repo: str, revision: str, path: str) -> tuple[dict[str, Any], str]:
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=path,
    ))
    document = json.loads(local.read_text())
    _verify_seal(document, "receipt_sha256", path)
    return document, sha256_file(local)


def _candidate_rows(inventory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=12) as pool:
        loaded = list(pool.map(
            lambda row: _load_receipt(
                inventory["repo_id"], inventory["revision"], row["receipt_path"]
            ),
            inventory["layers"],
        ))
    bindings = []
    scores = []
    for expected, (receipt, file_sha256) in zip(inventory["layers"], loaded, strict=True):
        layer = int(expected["layer"])
        if (
            int(receipt["layer"]) != layer
            or receipt["receipt_sha256"] != expected["receipt_sha256"]
            or file_sha256 != expected["receipt_file_sha256"]
            or receipt["candidate_tensor_sha256"] != expected["candidate_sha256"]
            or int(receipt["candidate_tensor_bytes"]) != int(expected["candidate_bytes"])
        ):
            raise ValueError(f"layer {layer} receipt differs from the attribution inventory")
        bindings.append({
            "layer": layer,
            "receipt_sha256": receipt["receipt_sha256"],
            "receipt_file_sha256": file_sha256,
            "candidate_tensor_sha256": receipt["candidate_tensor_sha256"],
            "candidate_tensor_bytes": int(receipt["candidate_tensor_bytes"]),
        })
        scores.extend(dict(row) for row in receipt["scores"])
    if len(scores) != 48 * 128 * 3 * 2:
        raise ValueError("K3/K4 candidate score inventory is incomplete")
    return bindings, scores


def _direct_scores(attribution: dict[str, Any]) -> dict[tuple[int, int], float]:
    direct: dict[tuple[int, int], float] = {}
    for layer_row in attribution.get("layers", ()):
        layer = int(layer_row["layer_index"])
        values = list(layer_row["expert_direct"])
        if len(values) != 128:
            raise ValueError(f"layer {layer} attribution does not contain 128 experts")
        for expert, value in enumerate(values):
            direct[(layer, expert)] = float(value)
    if set(direct) != {(layer, expert) for layer in range(48) for expert in range(128)}:
        raise ValueError("attribution expert inventory is incomplete")
    return direct


def _allocate(
    *,
    inventory: dict[str, Any],
    attribution: dict[str, Any],
    bindings: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    control: dict[str, Any],
) -> dict[str, Any]:
    by_matrix: dict[tuple[int, int, str], dict[int, dict[str, Any]]] = {}
    for row in scores:
        identity = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        by_matrix.setdefault(identity, {})[int(row["bits"])] = row
    expected = 48 * 128 * 3
    if len(by_matrix) != expected or any(set(value) != {3, 4} for value in by_matrix.values()):
        raise ValueError("matrix candidate inventory is incomplete")
    direct = _direct_scores(attribution)
    scale_by_expert: dict[tuple[int, int], float] = {}
    expert_rows = []
    for identity in sorted(direct):
        layer, expert = identity
        anchor = sum(
            float(by_matrix[(layer, expert, projection)][4]["diagonal_p2_damage"])
            for projection in PROJECTIONS
        )
        if anchor <= 0.0:
            raise ValueError(f"L{layer}.E{expert} uniform-K4 proxy anchor is not positive")
        scale = direct[identity] / anchor
        scale_by_expert[identity] = scale
        expert_rows.append({
            "layer": layer,
            "expert": expert,
            "expert_direct": direct[identity],
            "uniform_k4_proxy_anchor": anchor,
            "signed_scale": scale,
        })
    upgrades = []
    extra_values = set()
    for matrix, candidates in by_matrix.items():
        scale = scale_by_expert[matrix[:2]]
        k3, k4 = candidates[3], candidates[4]
        extra = int(k4["stored_bytes"]) - int(k3["stored_bytes"])
        if extra <= 0:
            raise ValueError("K4 candidate does not cost more than K3")
        extra_values.add(extra)
        benefit = scale * (
            float(k3["diagonal_p2_damage"]) - float(k4["diagonal_p2_damage"])
        )
        upgrades.append((benefit / extra, benefit, matrix, extra))
    if len(extra_values) != 1:
        raise ValueError("exact-half logical-rate allocation encountered unequal K4 increments")
    k4_count = expected // 2
    ranked = sorted(upgrades, key=lambda row: (-row[0], -row[1], row[2]))
    selected_k4 = {row[2] for row in ranked[:k4_count]}
    choices = []
    stored_bytes = 0
    raw_damage = 0.0
    calibrated_damage = 0.0
    for matrix in sorted(by_matrix):
        bits = 4 if matrix in selected_k4 else 3
        row = by_matrix[matrix][bits]
        scale = scale_by_expert[matrix[:2]]
        raw = float(row["diagonal_p2_damage"])
        calibrated = scale * raw
        stored_bytes += int(row["stored_bytes"])
        raw_damage += raw
        calibrated_damage += calibrated
        choices.append({
            "layer": matrix[0],
            "expert": matrix[1],
            "projection": matrix[2],
            "bits": bits,
            "stored_bytes": int(row["stored_bytes"]),
            "diagonal_p2_damage": raw,
            "shapley_fisher_calibrated_damage": calibrated,
            "expert_signed_scale": scale,
            "packed_sha256": row["packed_sha256"],
            "codec_fp16_reconstruction_sha256": row["codec_fp16_reconstruction_sha256"],
            "stored_bf16_reconstruction_sha256": row["stored_bf16_reconstruction_sha256"],
        })
    control_bits = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): int(row["bits"])
        for row in control["choices"]
    }
    changed = sum(control_bits[matrix] != (4 if matrix in selected_k4 else 3) for matrix in by_matrix)
    if stored_bytes != int(control["stored_payload_bytes"]):
        raise ValueError("research allocation does not preserve the control payload bytes")
    body = {
        "schema": "quant-pipeline.qwen-fast-k34-allocation.v2",
        "objective": "native-aumann-shapley-fisher-calibrated-mcg-at-exact-half-k4-weight-rate",
        "rate_scope": "moe-expert-weight-elements",
        "average_weight_bits": sum(row["bits"] for row in choices) / len(choices),
        "unit_count": len(choices),
        "k3_count": len(choices) - len(selected_k4),
        "k4_count": len(selected_k4),
        "stored_payload_bytes": stored_bytes,
        "predicted_diagonal_p2_damage": raw_damage,
        "predicted_shapley_fisher_damage": calibrated_damage,
        "candidate_inventory_sha256": inventory["inventory_sha256"],
        "attribution_sha256": attribution["attribution_sha256"],
        "control_allocation_sha256": control["allocation_sha256"],
        "changed_matrix_count_vs_control": changed,
        "calibration": {
            "method": "signed-uniform-k4-expert-anchor-ratio-v1",
            "provisional_bit_triplet": [4, 4, 4],
            "expert_rows": expert_rows,
            "negative_scale_expert_count": sum(row["signed_scale"] < 0.0 for row in expert_rows),
            "zero_scale_expert_count": sum(row["signed_scale"] == 0.0 for row in expert_rows),
        },
        "encode_receipts": bindings,
        "choices": choices,
    }
    body["allocation_sha256"] = _hash_json(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--control-allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-mcg-causal-allocation-plan.v1",
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "attribution": str(args.attribution.resolve()),
        "attribution_file_sha256": sha256_file(args.attribution),
        "control_allocation": str(args.control_allocation.resolve()),
        "control_allocation_file_sha256": sha256_file(args.control_allocation),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    inventory = json.loads(args.candidate_inventory.read_text())
    attribution = json.loads(args.attribution.read_text())
    control = json.loads(args.control_allocation.read_text())
    _verify_seal(inventory, "inventory_sha256", "candidate inventory")
    _verify_seal(attribution, "attribution_sha256", "native attribution")
    _verify_seal(control, "allocation_sha256", "control allocation")
    if attribution["candidate_inventory_sha256"] != inventory["inventory_sha256"]:
        raise ValueError("native attribution belongs to a different candidate inventory")
    bindings, scores = _candidate_rows(inventory)
    allocation = _allocate(
        inventory=inventory,
        attribution=attribution,
        bindings=bindings,
        scores=scores,
        control=control,
    )
    write_json(output / "allocation.json", allocation)
    print(json.dumps({
        "ok": True,
        "allocation_sha256": allocation["allocation_sha256"],
        "average_weight_bits": allocation["average_weight_bits"],
        "stored_payload_bytes": allocation["stored_payload_bytes"],
        "changed_matrix_count_vs_control": allocation["changed_matrix_count_vs_control"],
        "negative_scale_expert_count": allocation["calibration"]["negative_scale_expert_count"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
