#!/usr/bin/env python3
"""Add a hash-bound attention backend to an older sealed teacher receipt.

The first post-trained teacher schema printed the backend in the immutable run
log but omitted it from the final receipt. This migration verifies both seals,
copies no logits, and creates a new receipt that binds the original receipt to
the logged launch plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    legacy = json.loads(args.legacy_receipt.read_text())
    expected = legacy.get("receipt_sha256")
    if expected != _hash_json({key: value for key, value in legacy.items() if key != "receipt_sha256"}):
        raise ValueError("legacy teacher receipt seal mismatch")
    with args.run_log.open() as handle:
        plan = json.loads(handle.readline())
    if plan.get("schema") != "quant-pipeline.turboderp-wiki2-teacher-plan.v1":
        raise ValueError("run log does not begin with the expected teacher plan")
    if (
        plan.get("model_revision") != legacy.get("model_revision")
        or Path(plan.get("output", "")).resolve() != args.legacy_receipt.resolve().parent
        or plan.get("attention_backend") not in {"eager", "sdpa"}
    ):
        raise ValueError("logged teacher plan does not bind the legacy receipt")

    body = {key: value for key, value in legacy.items() if key != "receipt_sha256"}
    body.update({
        "schema": "quant-pipeline.bound-teacher-backend-receipt.v1",
        "attention_backend": plan["attention_backend"],
        "legacy_schema": legacy["schema"],
        "legacy_receipt_sha256": expected,
        "legacy_receipt_file_sha256": sha256_file(args.legacy_receipt),
        "backend_evidence_log_sha256": sha256_file(args.run_log),
        "backend_evidence_plan": plan,
        "migration_effect": "identity-only; teacher logits unchanged",
    })
    body["receipt_sha256"] = _hash_json(body)
    print(json.dumps({
        "attention_backend": body["attention_backend"],
        "receipt_sha256": body["receipt_sha256"],
        "dry_run": not args.execute,
    }, sort_keys=True), flush=True)
    if args.execute:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
