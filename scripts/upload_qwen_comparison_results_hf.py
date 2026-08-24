#!/usr/bin/env python3
"""Publish and remotely verify the sealed Qwen comparison panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


ARTIFACTS = (
    "hill-paper-panel-v1",
    "hill-paper-kld-v1",
    "turboderp-wiki2-kld-v1",
    "uniform-expert-controls-v1",
)


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    hasher = hashlib.sha1(usedforsecurity=False)
    hasher.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _inventory(root: Path, name: str, prefix: str) -> list[dict]:
    rows = []
    for path in sorted((root / name).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root / name).as_posix()
        rows.append(
            {
                "artifact": name,
                "local_path": str(path),
                "path": f"{prefix}/{name}/{relative}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"empty comparison artifact: {name}")
    return rows


def _verify_remote(
    api: HfApi,
    repo_id: str,
    revision: str,
    remote_root: str,
    rows: list[dict],
) -> None:
    remote = {
        item.path: item
        for item in api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=remote_root,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
        )
        if hasattr(item, "size")
    }
    for row in rows:
        item = remote.get(row["path"])
        if item is None or int(item.size) != int(row["bytes"]):
            raise ValueError(f"Hub result inventory mismatch for {row['path']}")
        lfs = getattr(item, "lfs", None)
        if lfs and lfs.get("sha256"):
            if str(lfs["sha256"]) != row["sha256"]:
                raise ValueError(f"Hub result LFS hash mismatch for {row['path']}")
        elif str(item.blob_id) != _git_blob_sha1(Path(row["local_path"])):
            raise ValueError(f"Hub result Git-blob hash mismatch for {row['path']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--path-prefix", default="results/qwen3-30b-a3b-v1")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifacts_root.resolve()
    api = HfApi()
    api.whoami()
    all_rows = []
    revisions = []
    for name in ARTIFACTS:
        rows = _inventory(root, name, args.path_prefix)
        all_rows.extend(rows)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=root / name,
            path_in_repo=f"{args.path_prefix}/{name}",
            commit_message=f"Publish sealed Qwen comparison artifact {name}",
        )
        revision = str(commit.oid)
        revisions.append({"artifact": name, "revision": revision})
        _verify_remote(
            api,
            args.repo_id,
            revision,
            f"{args.path_prefix}/{name}",
            rows,
        )
        print(
            json.dumps(
                {
                    "stage": "uploaded-and-verified",
                    "artifact": name,
                    "revision": revision,
                    "files": len(rows),
                    "bytes": sum(int(row["bytes"]) for row in rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    receipt = {
        "schema": "quant-pipeline.qwen-comparison-results-hf-publication.v1",
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "path_prefix": args.path_prefix,
        "artifact_revisions": revisions,
        "files": [
            {key: row[key] for key in ("artifact", "path", "bytes", "sha256")}
            for row in all_rows
        ],
        "file_count": len(all_rows),
        "total_bytes": sum(int(row["bytes"]) for row in all_rows),
        "remote_verification": "size-all; sha256-lfs; git-blob-sha1-non-lfs",
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.receipt, receipt)
    receipt_commit = api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(args.receipt),
        path_in_repo=f"{args.path_prefix}/publication-receipt.json",
        commit_message="Record verified Qwen comparison-result publication",
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
