#!/usr/bin/env python3
"""Build a sealed 48-layer Qwen candidate inventory at any Hub prefix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


REVISION = re.compile(r"[0-9a-f]{40}")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _lfs_sha(value: Any) -> str | None:
    lfs = getattr(value, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def _load_layer(api: Any, repo: str, revision: str, prefix: str, layer: int) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    base = f"{prefix}/candidates/layer-{layer:03d}" if prefix else f"candidates/layer-{layer:03d}"
    receipt_name = f"{base}/encode-receipt.json"
    receipt_path = Path(hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=receipt_name,
    ))
    receipt = json.loads(receipt_path.read_text())
    seal = receipt.get("receipt_sha256")
    if seal != _hash_json({key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise ValueError(f"layer {layer} encode receipt seal mismatch")
    if int(receipt.get("layer", -1)) != layer or receipt.get("experts") != list(range(128)):
        raise ValueError(f"layer {layer} encode receipt inventory is incomplete")
    candidate_name = f"{base}/{receipt['candidate_tensor_file']}"
    values = api.get_paths_info(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        paths=[candidate_name],
        expand=True,
    )
    if len(values) != 1:
        raise ValueError(f"layer {layer} candidate payload is absent")
    candidate = values[0]
    expected_bytes = int(receipt["candidate_tensor_bytes"])
    expected_sha256 = str(receipt["candidate_tensor_sha256"])
    if int(candidate.size) != expected_bytes or _lfs_sha(candidate) != expected_sha256:
        raise ValueError(f"layer {layer} candidate payload differs from its receipt")
    return {
        "layer": layer,
        "receipt_path": receipt_name,
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_sha256": seal,
        "candidate_path": candidate_name,
        "candidate_bytes": expected_bytes,
        "candidate_sha256": expected_sha256,
        "score_inventory_sha256": _hash_json(receipt["scores"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if REVISION.fullmatch(args.revision) is None:
        parser.error("--revision must be an immutable 40-hex Hub commit")
    if args.workers < 1:
        parser.error("--workers must be positive")
    prefix = "/".join(part for part in args.path_prefix.strip("/").split("/") if part)
    if any(part in {".", ".."} for part in prefix.split("/") if part):
        parser.error("--path-prefix cannot contain dot path components")
    plan = {
        "repo": args.repo,
        "revision": args.revision,
        "path_prefix": prefix,
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=args.repo, repo_type="dataset", revision=args.revision)
    if str(info.sha) != args.revision:
        raise ValueError("candidate revision did not resolve exactly")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        layers = list(pool.map(
            lambda layer: _load_layer(api, args.repo, args.revision, prefix, layer),
            range(48),
        ))
    body = {
        "schema": "quant-pipeline.qwen-hf-mcg-candidate-inventory.v2",
        "repo_id": args.repo,
        "repo_type": "dataset",
        "revision": args.revision,
        "path_prefix": prefix,
        "layers": layers,
    }
    body["inventory_sha256"] = _hash_json(body)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, body)
    print(json.dumps({
        "ok": True,
        "inventory_sha256": body["inventory_sha256"],
        "layers": len(layers),
        "total_bytes": sum(row["candidate_bytes"] for row in layers),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
