#!/usr/bin/env python3
"""Create a deterministic manifest and SHA256SUMS for a finished artifact tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json


EXCLUDED = {"MANIFEST.json", "SHA256SUMS", "PUBLICATION_RECEIPT.json"}


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains symlink: {path}")
        if not path.is_file() or path.name in EXCLUDED:
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not rows:
        raise ValueError("artifact tree contains no files")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = _inventory(root)
    manifest = {
        "schema": "quant-pipeline.artifact-tree-manifest.v1",
        "label": args.label,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    manifest["manifest_sha256"] = _hash_json(manifest)
    print(json.dumps({
        "root": str(root),
        "label": args.label,
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": manifest["manifest_sha256"],
        "dry_run": not args.execute,
    }, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()
    atomic_write(root / "SHA256SUMS", sums)
    write_json(root / "MANIFEST.json", manifest)
    if _inventory(root) != rows:
        raise RuntimeError("artifact tree changed while it was being sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
