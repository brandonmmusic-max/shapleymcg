#!/usr/bin/env python3
"""Upload and independently verify a sealed Qwen control publication bundle."""

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


def _verify_manifest(bundle: Path) -> tuple[dict, list[dict]]:
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seal = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if seal != _hash_json(body):
        raise ValueError("publication bundle manifest seal mismatch")
    rows = list(manifest.get("files", []))
    expected_paths = {str(row["path"]) for row in rows}
    expected_paths.add("bundle-manifest.json")
    observed_paths = {
        path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()
    }
    if expected_paths != observed_paths:
        raise ValueError("publication bundle file inventory mismatch")
    for row in rows:
        path = bundle / str(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"publication bundle file drifted: {row['path']}")
    return manifest, rows


def _remote_files(api: HfApi, repo_id: str, revision: str, path_in_repo: str) -> dict:
    return {
        item.path: item
        for item in api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
        if hasattr(item, "size")
    }


def _verify_remote_file(item, path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if int(item.size) != expected_bytes:
        raise ValueError(f"Hub size mismatch for {item.path}")
    lfs = getattr(item, "lfs", None)
    if lfs and lfs.get("sha256"):
        if str(lfs["sha256"]) != expected_sha256:
            raise ValueError(f"Hub LFS hash mismatch for {item.path}")
    elif str(item.blob_id) != _git_blob_sha1(path):
        raise ValueError(f"Hub Git-blob hash mismatch for {item.path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--path-in-repo", default="controls/fixed-hadamard-k34-v1")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--receipt-path-in-repo",
        default="receipts/control-fixed-hadamard-k34-v1.json",
    )
    parser.add_argument(
        "--receipt-schema",
        default="quant-pipeline.qwen-control-hf-publication.v1",
    )
    parser.add_argument("--label", default="Qwen fixed-Hadamard K3/K4 control")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    manifest, rows = _verify_manifest(bundle)
    token_file = args.token_file.resolve()
    token = token_file.read_text().strip()
    token_file.unlink()
    if not token:
        raise ValueError("empty Hugging Face token")
    api = HfApi(token=token)
    folder_commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=bundle,
        path_in_repo=args.path_in_repo,
        commit_message=f"Publish sealed {args.label}",
    )
    card_commit = api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(bundle / "README.md"),
        path_in_repo="README.md",
        commit_message=f"Publish {args.label} dataset card",
    )
    revision = str(card_commit.oid)
    remote = _remote_files(api, args.repo_id, revision, args.path_in_repo)
    for row in rows:
        name = f"{args.path_in_repo}/{row['path']}"
        if name not in remote:
            raise ValueError(f"Hub inventory lacks {name}")
        _verify_remote_file(remote[name], bundle / row["path"], int(row["bytes"]), row["sha256"])
    manifest_name = f"{args.path_in_repo}/bundle-manifest.json"
    if manifest_name not in remote:
        raise ValueError("Hub inventory lacks the bundle manifest")
    _verify_remote_file(
        remote[manifest_name],
        bundle / "bundle-manifest.json",
        (bundle / "bundle-manifest.json").stat().st_size,
        sha256_file(bundle / "bundle-manifest.json"),
    )
    # The reproducibility dataset contains tens of thousands of fit/candidate
    # files.  A recursive root walk just to verify the card can take minutes or
    # exhaust pagination; inspect only the repository root here.
    root = {
        item.path: item
        for item in api.list_repo_tree(
            repo_id=args.repo_id,
            path_in_repo="",
            recursive=False,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
        if hasattr(item, "size")
    }
    if "README.md" not in root:
        raise ValueError("Hub root lacks the dataset card")
    _verify_remote_file(
        root["README.md"],
        bundle / "README.md",
        (bundle / "README.md").stat().st_size,
        sha256_file(bundle / "README.md"),
    )
    receipt = {
        "schema": args.receipt_schema,
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "path_in_repo": args.path_in_repo,
        "folder_revision": str(folder_commit.oid),
        "verified_revision": revision,
        "bundle_manifest_sha256": manifest["manifest_sha256"],
        "bundle_manifest_file_sha256": sha256_file(bundle / "bundle-manifest.json"),
        "dataset_card_sha256": sha256_file(bundle / "README.md"),
        "file_count": len(rows) + 1,
        "total_bytes": sum(int(row["bytes"]) for row in rows)
        + (bundle / "bundle-manifest.json").stat().st_size,
        "remote_verification": "size-all; sha256-lfs; git-blob-sha1-non-lfs",
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.receipt, receipt)
    receipt_commit = api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(args.receipt),
        path_in_repo=args.receipt_path_in_repo,
        commit_message=f"Record verified {args.label} publication",
    )
    print(
        json.dumps(
            {"ok": True, **receipt, "receipt_revision": str(receipt_commit.oid)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
