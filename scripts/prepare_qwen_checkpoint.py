#!/usr/bin/env python3
"""Fetch and seal an exact Qwen3-30B-A3B BF16 checkpoint revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY = "Qwen/Qwen3-30B-A3B-Base"
REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
SHARDS = tuple(f"model-{index:05d}-of-00016.safetensors" for index in range(1, 17))
REQUIRED_FILES = (
    "config.json",
    "merges.txt",
    *SHARDS,
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
ALLOW_PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(value: Any) -> bytes:
    # Match quant_pipeline.core.artifacts.canonical_json.  Source receipts are
    # consumed by the fitter, encoder, and evaluators, all of which include the
    # canonical trailing newline in their identity hash.
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def verify_checkpoint(
    root: Path,
    repository: str = REPOSITORY,
    revision: str = REVISION,
) -> dict[str, Any]:
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Qwen checkpoint file is missing or symlinked: {relative}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    shard_inventory = {name: files[name] for name in SHARDS}
    body: dict[str, Any] = {
        "schema": "quant-pipeline.qwen-source-receipt.v1",
        "repository": repository,
        "revision": revision,
        "files": files,
        "config_sha256": files["config.json"]["sha256"],
        "index_sha256": files["model.safetensors.index.json"]["sha256"],
        "shard_manifest_sha256": hashlib.sha256(canonical_json(shard_inventory)).hexdigest(),
        "total_bytes": sum(int(row["bytes"]) for row in files.values()),
    }
    body["receipt_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="receipt path (default: <destination>.source-receipt.json)",
    )
    parser.add_argument("--execute", action="store_true", help="download and hash the pinned checkpoint")
    args = parser.parse_args()
    destination = args.destination.resolve()
    receipt = (
        args.receipt.resolve()
        if args.receipt is not None
        else destination.with_name(f"{destination.name}.source-receipt.json")
    )
    if destination.exists():
        raise SystemExit("destination must not exist; refusing to alter an existing checkpoint")
    if receipt.exists():
        raise SystemExit("receipt must not exist; refusing to overwrite a source receipt")
    plan = {
        "dry_run": not args.execute,
        "repository": args.repository,
        "revision": args.revision,
        "destination": str(destination),
        "receipt": str(receipt),
        "required_file_count": len(REQUIRED_FILES),
        "shard_count": len(SHARDS),
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit("huggingface-hub is required to fetch the Qwen checkpoint") from error
    identity = HfApi().model_info(args.repository, revision=args.revision)
    if identity.sha != args.revision:
        raise SystemExit(f"Hub revision mismatch: expected {args.revision}, got {identity.sha}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repository,
        revision=args.revision,
        local_dir=destination,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    sealed = verify_checkpoint(destination, args.repository, args.revision)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(canonical_json(sealed))
    print(json.dumps({"ok": True, "receipt": str(receipt), **sealed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
