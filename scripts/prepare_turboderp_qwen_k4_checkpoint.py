#!/usr/bin/env python3
"""Fetch and seal the exact TurboDerp Qwen3-30B-A3B EXL3 K4 branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


REPOSITORY = "turboderp/Qwen3-30B-A3B-exl3"
REVISION = "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"
ALLOW_PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    destination = args.destination.resolve()
    receipt_path = args.receipt.resolve()
    plan = {
        "schema": "quant-pipeline.turboderp-qwen-k4-checkpoint-plan.v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "destination": str(destination),
        "receipt": str(receipt_path),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("destination and receipt must not already exist")

    from huggingface_hub import HfApi, snapshot_download

    identity = HfApi().model_info(REPOSITORY, revision=REVISION)
    if identity.sha != REVISION:
        raise ValueError("TurboDerp revision did not resolve to the pinned commit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    required = {
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    }
    missing = {name for name in required if not (destination / name).is_file()}
    if missing:
        raise ValueError(f"TurboDerp checkpoint is incomplete: {sorted(missing)}")
    quant = json.loads((destination / "quantization_config.json").read_text())
    if (
        quant.get("bits") != 4.0
        or quant.get("head_bits") != 6
        or quant.get("calibration") != {"rows": 100, "cols": 2048}
    ):
        raise ValueError("downloaded checkpoint is not the pinned K4/K6 scope")
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".cache" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    model_shards = [row for row in files if row["path"].startswith("model-") and row["path"].endswith(".safetensors")]
    if not model_shards:
        raise ValueError("downloaded checkpoint has no model shards")
    receipt = {
        "schema": "quant-pipeline.turboderp-qwen-k4-checkpoint.v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
        "file_count": len(files),
        "model_shard_count": len(model_shards),
        "total_bytes": sum(row["bytes"] for row in files),
        "quantization_scope": {
            "body_bits": 4.0,
            "head_bits": 6,
            "calibration": {"rows": 100, "cols": 2048},
        },
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(json.dumps({"ok": True, **receipt}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
