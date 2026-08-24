#!/usr/bin/env python3
"""Fit one full-p2 Qwen layer from a sealed routed capture.

The launcher is intentionally layer-scoped: a scheduler can run a bounded
number of independent layers in parallel, consume each fit into exact
candidate generation, and release bulky covariance intermediates without ever
materializing all 48 layers of full covariance at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pipeline.calibration.qwen_capture import verify_capture_manifest
from quant_pipeline.campaign.qwen_services import (
    CAPTURE_SERVICE_SCHEMA,
    QwenFitterService,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


ZERO_HASH = "0" * 64
MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _source_identity(path: Path) -> str:
    receipt = json.loads(path.read_text())
    expected = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    actual = sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    if expected != actual:
        raise ValueError("source-checkpoint receipt seal mismatch")
    if receipt.get("revision") != MODEL_REVISION:
        raise ValueError("source-checkpoint receipt revision mismatch")
    return str(expected)


def _capture_stage(capture_root: Path) -> dict:
    manifest_path = capture_root / "fit" / "capture-manifest.json"
    manifest = verify_capture_manifest(manifest_path.parent, verify_chunks=False)
    service = {
        "schema": CAPTURE_SERVICE_SCHEMA,
        "predecessor_state_hash": ZERO_HASH,
        "layers": list(manifest["layers"]),
        "captures": {
            "fit": {
                "role": "fit",
                "manifest": "fit/capture-manifest.json",
                "capture_sha256": manifest["capture_sha256"],
            }
        },
        "streaming": "one-window-one-layer-chunk",
        "retention": "sealed-chunks",
    }
    service["capture_service_sha256"] = sha256_bytes(canonical_json(service))
    write_json(capture_root / "capture-service-manifest.json", service)
    write_json(capture_root / "stage-manifest.json", {
        "provider_result": {"capture_manifest_file": "capture-service-manifest.json"}
    })
    return service


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--route-weight-power", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.model_revision != MODEL_REVISION or args.dataset_revision != DATASET_REVISION:
        parser.error("model and dataset revisions must match the pinned experiment")
    if args.layer not in range(48):
        parser.error("--layer must be in [0, 47]")
    capture_root = args.capture_root.resolve()
    output = args.output_dir.resolve()
    source_identity = _source_identity(args.source_receipt.resolve())
    service = _capture_stage(capture_root)
    if args.layer not in service["layers"]:
        parser.error("requested layer is absent from the sealed capture")
    config = {
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "route_weight_power": args.route_weight_power,
        "retained_powers": [args.route_weight_power],
        "retained_accounting": ["combined"],
        "covariance_mode": "full",
        "covariance_block_size": 128,
        "artifact_dtype": "float32",
        "fitter_backend": "torch_full_p2",
        "fitter_device": args.device,
        "route_weight_denominator": 1 << 24,
        "cold_expert_min_weight_units": 0,
    }
    plan = {
        "schema": "quant-pipeline.qwen-streaming-fit-plan.v1",
        "capture_root": str(capture_root),
        "capture_service_sha256": service["capture_service_sha256"],
        "source_checkpoint_identity": source_identity,
        "source_receipt_sha256": sha256_file(args.source_receipt),
        "layer": args.layer,
        "output_dir": str(output),
        "config": config,
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "fit-plan.json", plan | {"dry_run": False})
    result = QwenFitterService(config).fit({
        "kind": "fit",
        "layer": args.layer,
        "output_dir": str(output),
        "dependencies": {"fit_capture": str(capture_root)},
        "predecessor_state_hash": ZERO_HASH,
        "input_identities": {"source_checkpoint": source_identity},
    })
    manifest = output / str(result["fit_manifest_file"])
    body = {
        "schema": "quant-pipeline.qwen-streaming-fit-receipt.v1",
        "plan_sha256": sha256_file(output / "fit-plan.json"),
        "layer": args.layer,
        "fit_manifest": manifest.name,
        "fit_manifest_sha256": sha256_file(manifest),
        "fit_sha256": json.loads(manifest.read_text())["fit_sha256"],
    }
    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    write_json(output / "fit-receipt.json", body)
    print(json.dumps({"ok": True, **body}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
