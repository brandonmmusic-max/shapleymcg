#!/usr/bin/env python3
"""Run or resume the sealed three-role Qwen routed calibration capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pipeline.calibration.qwen_capture import capture_roles_from_local_bf16
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


ZERO_HASH = "0" * 64
MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--sealed-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--fisher-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--purposes",
        default="fit,heldout,conditional_down",
        help="comma-separated role workers to run; permits one independent process per GPU",
    )
    parser.add_argument("--writer-workers", type=int, default=8)
    parser.add_argument("--max-inflight-chunks", type=int, default=96)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    capture_specs = {
        "fit": {"role": "fit", "fisher_rank": 0},
        "heldout": {"role": "selection", "fisher_rank": args.fisher_rank},
        "conditional_down": {"role": "confirmation", "fisher_rank": 0},
    }
    purposes = tuple(item.strip() for item in args.purposes.split(",") if item.strip())
    if not purposes or len(set(purposes)) != len(purposes) or any(item not in capture_specs for item in purposes):
        parser.error("--purposes must select unique values from fit,heldout,conditional_down")
    plan = {
        "schema": "quant-pipeline.qwen-calibration-capture-plan.v1",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "sealed_corpus": str(args.sealed_corpus.resolve()),
        "output_dir": str(root),
        "model_revision": args.model_revision,
        "layers": list(range(48)),
        "captures": {purpose: capture_specs[purpose] for purpose in purposes},
        "predecessor_state_hash": ZERO_HASH,
        "device_map": args.device_map,
        "attention_backend": args.attention_backend,
        "writer_workers": args.writer_workers,
        "max_inflight_chunks": args.max_inflight_chunks,
        "seed": args.seed,
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0
    requests = [
        {
            "purpose": purpose,
            "role": row["role"],
            "output_dir": str(root / purpose),
            "fisher_rank": row["fisher_rank"],
            # A role's stochastic identity must not change when roles are
            # scheduled in separate per-GPU processes.
            "seed": args.seed + ("fit", "heldout", "conditional_down").index(purpose),
        }
        for purpose, row in plan["captures"].items()
    ]
    results = capture_roles_from_local_bf16(
        source_checkpoint=args.source_checkpoint,
        model_revision=args.model_revision,
        sealed_corpus=args.sealed_corpus,
        captures=requests,
        layers=range(48),
        predecessor_state_hash=ZERO_HASH,
        device_map=args.device_map,
        attn_implementation=args.attention_backend,
        production_geometry=True,
        writer_workers=args.writer_workers,
        max_inflight_chunks=args.max_inflight_chunks,
    )
    body = {
        **{key: value for key, value in plan.items() if key != "dry_run"},
        "sealed_corpus_file_sha256": sha256_file(args.sealed_corpus),
        "results": {
            purpose: {
                "capture_sha256": result["capture_sha256"],
                "request_sha256": result["request_sha256"],
                "manifest": f"{purpose}/capture-manifest.json",
                "manifest_sha256": sha256_file(root / purpose / "capture-manifest.json"),
            }
            for purpose, result in results.items()
        },
    }
    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    write_json(root / f"calibration-capture-{'-'.join(purposes)}-receipt.json", body)
    print(json.dumps({"ok": True, **body}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
