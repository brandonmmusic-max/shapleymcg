#!/usr/bin/env python3
"""Verify a sealed artifact-tree publication against an immutable Hub commit."""

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


def _remote_file_inventory(api: Any, *, repo: str, repo_type: str, revision: str, prefix: str) -> set[str] | None:
    """List exact files below a Hub prefix when the installed API supports it."""

    list_tree = getattr(api, "list_repo_tree", None)
    if list_tree is None:
        return None
    files: set[str] = set()
    for entry in list_tree(
        repo_id=repo,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=prefix or None,
        recursive=True,
        expand=False,
    ):
        path = getattr(entry, "path", None)
        # RepoFolder has no byte size; RepoFile does (including zero).
        if isinstance(path, str) and getattr(entry, "size", None) is not None:
            files.add(path)
    return files


def _require_closed_namespace(observed: set[str] | None, expected: set[str]) -> None:
    if observed is None:
        return
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "Hub prefix is not a closed sealed namespace; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--publish-output-path",
        help="optional Hub path at which to publish the completed verification receipt",
    )
    parser.add_argument(
        "--publish-message",
        default="Publish remote artifact verification receipt",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.publish_output_path and not args.execute:
        parser.error("--publish-output-path requires --execute")

    from huggingface_hub import HfApi, hf_hub_download

    root = args.local_root.resolve()
    local_manifest_path = root / "MANIFEST.json"
    local_sums_path = root / "SHA256SUMS"
    if not local_manifest_path.is_file() or not local_sums_path.is_file():
        raise ValueError("local artifact tree lacks MANIFEST.json or SHA256SUMS")
    manifest = json.loads(local_manifest_path.read_text())
    if manifest.get("schema") != "quant-pipeline.artifact-tree-manifest.v1":
        raise ValueError("unsupported local artifact manifest schema")
    rows = manifest.get("files", ())
    if int(manifest.get("file_count", -1)) != len(rows):
        raise ValueError("local artifact manifest file count mismatch")
    if int(manifest.get("total_bytes", -1)) != sum(int(row["bytes"]) for row in rows):
        raise ValueError("local artifact manifest byte count mismatch")
    relative_paths = [str(row["path"]) for row in rows]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("local artifact manifest contains duplicate paths")
    for relative in relative_paths:
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
    expected_seal = manifest.get("manifest_sha256")
    if expected_seal != _hash_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("local artifact manifest seal mismatch")
    prefix = _repo_prefix(args.path_in_repo)
    remote_manifest = Path(hf_hub_download(
        repo_id=args.repo,
        repo_type=args.repo_type,
        revision=args.revision,
        filename=f"{prefix}/MANIFEST.json" if prefix else "MANIFEST.json",
    ))
    if sha256_file(remote_manifest) != sha256_file(local_manifest_path):
        raise ValueError("remote manifest bytes differ from the local sealed manifest")
    expected_sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in manifest["files"]
    ).encode()
    if local_sums_path.read_bytes() != expected_sums:
        raise ValueError("local SHA256SUMS differs from the sealed manifest")
    remote_sums = Path(hf_hub_download(
        repo_id=args.repo,
        repo_type=args.repo_type,
        revision=args.revision,
        filename=f"{prefix}/SHA256SUMS" if prefix else "SHA256SUMS",
    ))
    if remote_sums.read_bytes() != expected_sums:
        raise ValueError("remote SHA256SUMS differs from the sealed manifest")

    api = HfApi()
    expected = {
        f"{prefix}/{row['path']}" if prefix else row["path"]: row
        for row in manifest["files"]
    }
    expected_namespace = set(expected) | {
        f"{prefix}/MANIFEST.json" if prefix else "MANIFEST.json",
        f"{prefix}/SHA256SUMS" if prefix else "SHA256SUMS",
    }
    remote_inventory = _remote_file_inventory(
        api,
        repo=args.repo,
        repo_type=args.repo_type,
        revision=args.revision,
        prefix=prefix,
    )
    _require_closed_namespace(remote_inventory, expected_namespace)
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
        "sha256sums_file_sha256": sha256_file(local_sums_path),
        "namespace_inventory": (
            "exact" if remote_inventory is not None else "unavailable-in-installed-hub-api"
        ),
        "namespace_file_count": (
            len(remote_inventory) if remote_inventory is not None else None
        ),
        "file_count": len(verified),
        "total_bytes": sum(row["bytes"] for row in verified),
        "files": verified,
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    summary = {
        "ok": True,
        "revision": args.revision,
        "file_count": receipt["file_count"],
        "total_bytes": receipt["total_bytes"],
        "manifest_sha256": expected_seal,
        "receipt_sha256": receipt["receipt_sha256"],
        "dry_run": not args.execute,
    }
    if args.execute:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, receipt)
        if args.publish_output_path:
            publish_path = args.publish_output_path.strip("/")
            if not publish_path:
                raise ValueError("published verification receipt path must be non-empty")
            commit = api.upload_file(
                repo_id=args.repo,
                repo_type=args.repo_type,
                path_or_fileobj=str(args.output),
                path_in_repo=publish_path,
                commit_message=args.publish_message,
            )
            published_revision = str(commit.oid)
            published = Path(hf_hub_download(
                repo_id=args.repo,
                repo_type=args.repo_type,
                revision=published_revision,
                filename=publish_path,
            ))
            if sha256_file(published) != sha256_file(args.output):
                raise ValueError("published verification receipt bytes differ from local receipt")
            summary["verification_receipt_path"] = publish_path
            summary["verification_receipt_file_sha256"] = sha256_file(args.output)
            summary["verification_receipt_revision"] = published_revision
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
