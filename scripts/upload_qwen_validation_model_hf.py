#!/usr/bin/env python3
"""Publish and remotely verify the Qwen BF16 validation reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    hasher = hashlib.sha1(usedforsecurity=False)
    hasher.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_manifest(root: Path) -> tuple[dict, list[dict]]:
    manifest_path = root / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seal = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if seal != _hash_json(body):
        raise ValueError("model manifest seal mismatch")
    rows = list(manifest["files"])
    expected = {str(row["path"]) for row in rows} | {"model-manifest.json"}
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if expected != observed:
        raise ValueError("model file inventory mismatch")
    for row in rows:
        path = root / row["path"]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"local model file drifted: {row['path']}")
    return manifest, rows


def _verify_remote(api: HfApi, repo_id: str, revision: str, root: Path, rows: list[dict]) -> None:
    remote = {
        item.path: item
        for item in api.list_repo_tree(
            repo_id=repo_id,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="model",
        )
        if hasattr(item, "size")
    }
    for row in rows:
        name = str(row["path"])
        item = remote.get(name)
        if item is None or int(item.size) != int(row["bytes"]):
            raise ValueError(f"Hub model inventory mismatch for {name}")
        lfs = getattr(item, "lfs", None)
        if lfs and lfs.get("sha256"):
            if str(lfs["sha256"]) != row["sha256"]:
                raise ValueError(f"Hub model LFS hash mismatch for {name}")
        elif str(item.blob_id) != _git_blob_sha1(root / name):
            raise ValueError(f"Hub model Git-blob hash mismatch for {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.model.resolve()
    manifest, rows = _verify_manifest(root)
    manifest_row = {
        "path": "model-manifest.json",
        "bytes": (root / "model-manifest.json").stat().st_size,
        "sha256": sha256_file(root / "model-manifest.json"),
    }
    token_path = args.token_file.resolve()
    token = token_path.read_text().strip()
    token_path.unlink()
    if not token:
        raise ValueError("empty Hugging Face token")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=root,
        commit_message="Publish verified Qwen ShapleyMCG validation reconstruction",
    )
    revision = str(commit.oid)
    _verify_remote(api, args.repo_id, revision, root, rows + [manifest_row])
    receipt = {
        "schema": "quant-pipeline.qwen-validation-model-hf-publication.v1",
        "repo_id": args.repo_id,
        "repo_type": "model",
        "verified_revision": revision,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_row["sha256"],
        "file_count": len(rows) + 1,
        "total_bytes": sum(int(row["bytes"]) for row in rows) + int(manifest_row["bytes"]),
        "remote_verification": "size-all; sha256-lfs; git-blob-sha1-non-lfs",
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.receipt, receipt)
    receipt_commit = api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(args.receipt),
        path_in_repo="publication-receipt.json",
        commit_message="Record verified model publication receipt",
    )
    print(json.dumps({"ok": True, **receipt, "receipt_revision": str(receipt_commit.oid)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
