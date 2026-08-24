#!/usr/bin/env python3
"""Upload a locally sealed artifact tree to one Hugging Face Hub prefix."""

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
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--commit-message", default="Publish sealed artifact tree")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = args.local_root.resolve()
    manifest_path = root / "MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise ValueError("artifact tree must contain MANIFEST.json and SHA256SUMS")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_sha256 = manifest.get("manifest_sha256")
    if expected_manifest_sha256 != _hash_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("artifact-tree manifest seal mismatch")
    expected_sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in manifest["files"]
    ).encode()
    if sums_path.read_bytes() != expected_sums:
        raise ValueError("SHA256SUMS differs from the sealed manifest")
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"sealed artifact is absent or invalid: {row['path']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"sealed artifact size changed: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"sealed artifact hash changed: {row['path']}")

    plan = {
        "schema": "quant-pipeline.hf-sealed-tree-upload.v1",
        "repo": args.repo,
        "repo_type": args.repo_type,
        "path_in_repo": args.path_in_repo.strip("/"),
        "manifest_sha256": expected_manifest_sha256,
        "manifest_file_sha256": sha256_file(manifest_path),
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
    }
    if not args.execute:
        print(json.dumps(plan | {"dry_run": True}, sort_keys=True), flush=True)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    account = api.whoami()
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type=args.repo_type,
        folder_path=root,
        path_in_repo=plan["path_in_repo"],
        commit_message=args.commit_message,
    )
    receipt = plan | {
        "revision": str(commit.oid),
        "authenticated_account": account.get("name"),
        "remote_verification_pending": True,
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    write_json(args.receipt, receipt)
    print(json.dumps(receipt | {"ok": True}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
