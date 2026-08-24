#!/usr/bin/env python3
"""Verify a sealed artifact-tree publication against an immutable Hub commit."""

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
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    root = args.local_root.resolve()
    local_manifest_path = root / "MANIFEST.json"
    manifest = json.loads(local_manifest_path.read_text())
    expected_seal = manifest.get("manifest_sha256")
    if expected_seal != _hash_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("local artifact manifest seal mismatch")
    prefix = args.path_in_repo.strip("/")
    remote_manifest = Path(hf_hub_download(
        repo_id=args.repo,
        repo_type=args.repo_type,
        revision=args.revision,
        filename=f"{prefix}/MANIFEST.json",
    ))
    if sha256_file(remote_manifest) != sha256_file(local_manifest_path):
        raise ValueError("remote manifest bytes differ from the local sealed manifest")

    api = HfApi()
    expected = {f"{prefix}/{row['path']}": row for row in manifest["files"]}
    remote = {}
    paths = sorted(expected)
    for start in range(0, len(paths), 100):
        for info in api.get_paths_info(
            repo_id=args.repo,
            paths=paths[start:start + 100],
            repo_type=args.repo_type,
            revision=args.revision,
            expand=True,
        ):
            remote[info.path] = info
    if set(remote) != set(expected):
        missing = sorted(set(expected) - set(remote))
        raise ValueError(f"Hub publication is missing manifest files: {missing[:5]}")
    verified = []
    for path in paths:
        row = expected[path]
        info = remote[path]
        if int(info.size) != int(row["bytes"]):
            raise ValueError(f"Hub size differs for {path}")
        lfs = getattr(info, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if lfs_sha is not None:
            if lfs_sha != row["sha256"]:
                raise ValueError(f"Hub LFS SHA-256 differs for {path}")
            verification = "hub-lfs-sha256"
        else:
            downloaded = Path(hf_hub_download(
                repo_id=args.repo,
                repo_type=args.repo_type,
                revision=args.revision,
                filename=path,
            ))
            if sha256_file(downloaded) != row["sha256"]:
                raise ValueError(f"downloaded Hub bytes differ for {path}")
            verification = "downloaded-sha256"
        verified.append({"path": path, "bytes": int(info.size), "sha256": row["sha256"], "verification": verification})

    receipt = {
        "schema": "quant-pipeline.hf-artifact-tree-verification.v1",
        "repo": args.repo,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "path_in_repo": prefix,
        "manifest_sha256": expected_seal,
        "manifest_file_sha256": sha256_file(local_manifest_path),
        "file_count": len(verified),
        "total_bytes": sum(row["bytes"] for row in verified),
        "files": verified,
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    print(json.dumps({
        "ok": True,
        "revision": args.revision,
        "file_count": receipt["file_count"],
        "total_bytes": receipt["total_bytes"],
        "manifest_sha256": expected_seal,
        "receipt_sha256": receipt["receipt_sha256"],
        "dry_run": not args.execute,
    }, sort_keys=True), flush=True)
    if args.execute:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
