#!/usr/bin/env python3
"""Upload a locally sealed artifact tree to one Hugging Face Hub prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _repo_prefix(value: str) -> str:
    prefix = value.strip("/")
    if not prefix:
        return ""
    parsed = PurePosixPath(prefix)
    if ".." in parsed.parts or parsed.as_posix() != prefix or "\\" in prefix:
        raise ValueError("path_in_repo must be a normalized relative Hub path")
    return prefix


def _sealed_upload_paths(root: Path, manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    """Return the exact, closed upload set described by the manifest.

    Walking or uploading ``root`` itself would also publish scratch logs,
    receipts, or files created after sealing.  Every data file here comes from
    the sealed manifest; only the two manifest metadata files are added.
    """

    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in manifest.get("files", ()):
        relative = str(row["path"])
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or not parsed.parts
            or parsed.as_posix() != relative
            or "\\" in relative
            or relative in {"MANIFEST.json", "SHA256SUMS"}
        ):
            raise ValueError(f"invalid sealed artifact path: {relative}")
        if relative in seen:
            raise ValueError(f"duplicate sealed artifact path: {relative}")
        seen.add(relative)
        result.append((relative, root.joinpath(*parsed.parts)))
    result.extend(
        [
            ("MANIFEST.json", root / "MANIFEST.json"),
            ("SHA256SUMS", root / "SHA256SUMS"),
        ]
    )
    return result


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
    if manifest.get("schema") != "quant-pipeline.artifact-tree-manifest.v1":
        raise ValueError("unsupported artifact-tree manifest schema")
    if int(manifest.get("file_count", -1)) != len(manifest.get("files", ())):
        raise ValueError("artifact-tree manifest file count mismatch")
    if int(manifest.get("total_bytes", -1)) != sum(
        int(row["bytes"]) for row in manifest.get("files", ())
    ):
        raise ValueError("artifact-tree manifest byte count mismatch")
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
    upload_paths = _sealed_upload_paths(root, manifest)
    for row, (_, path) in zip(manifest["files"], upload_paths):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"sealed artifact is absent or invalid: {row['path']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"sealed artifact size changed: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"sealed artifact hash changed: {row['path']}")
    for relative, path in upload_paths[-2:]:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required manifest file is absent or invalid: {relative}")

    plan = {
        "schema": "quant-pipeline.hf-sealed-tree-upload.v1",
        "repo": args.repo,
        "repo_type": args.repo_type,
        "path_in_repo": _repo_prefix(args.path_in_repo),
        "manifest_sha256": expected_manifest_sha256,
        "manifest_file_sha256": sha256_file(manifest_path),
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "uploaded_path_count": len(upload_paths),
        "upload_set_sha256": _hash_json([relative for relative, _ in upload_paths]),
    }
    if not args.execute:
        print(json.dumps(plan | {"dry_run": True}, sort_keys=True), flush=True)
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    account = api.whoami()
    prefix = plan["path_in_repo"]
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{relative}" if prefix else relative,
            path_or_fileobj=str(path),
        )
        for relative, path in upload_paths
    ]
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type=args.repo_type,
        operations=operations,
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
