#!/usr/bin/env python3
"""Reconcile sealed native path/Fisher evidence to measured endpoint KLD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pipeline.campaign.qwen_attribution import (
    build_hierarchical_attribution_document,
    verify_attribution_inputs,
)
from quant_pipeline.core.artifacts import prepare_empty_destination, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-inputs", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-mcg-attribution-reconciliation-plan.v1",
        "attribution_inputs": str(args.attribution_inputs.resolve()),
        "attribution_inputs_sha256": sha256_file(args.attribution_inputs),
        "candidate_inventory": str(args.candidate_inventory.resolve()),
        "candidate_inventory_file_sha256": sha256_file(args.candidate_inventory),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    arrays, receipt = verify_attribution_inputs(args.attribution_inputs.resolve())
    inventory = json.loads(args.candidate_inventory.read_text())
    attribution = build_hierarchical_attribution_document(
        arrays,
        inventory["inventory_sha256"],
    )
    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    write_json(output / "attribution-input-receipt.json", receipt)
    write_json(output / "attribution.json", attribution)
    print(json.dumps({
        "ok": True,
        "attribution_sha256": attribution["attribution_sha256"],
        "measured_end_to_end_delta": attribution["measured_end_to_end_delta"],
        "sum_reconciled_layer_damage": attribution["sum_reconciled_layer_damage"],
        "sum_reconciled_expert_damage": attribution["sum_reconciled_expert_damage"],
        "raw_path_remainder": attribution["raw_path_quadrature_and_nonlinear_remainder"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
