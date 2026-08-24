#!/usr/bin/env python3
"""Upload sealed Qwen fit layers to the Hub and reclaim verified local copies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _inventory(root: Path) -> list[dict]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "hf-upload-manifest.json"
    ]


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _verify_remote(
    api: HfApi,
    *,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    root: Path,
    inventory: list[dict],
) -> None:
    remote = {
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
    for row in inventory:
        name = f"{path_in_repo}/{row['path']}"
        item = remote.get(name)
        if item is None or int(item.size) != int(row["bytes"]):
            raise ValueError(f"Hub inventory mismatch for {name}")
        lfs = getattr(item, "lfs", None)
        if lfs and lfs.get("sha256"):
            if str(lfs["sha256"]) != row["sha256"]:
                raise ValueError(f"Hub LFS hash mismatch for {name}")
            continue
        if str(item.blob_id) != _git_blob_sha1(root / row["path"]):
            raise ValueError(f"Hub Git-blob hash mismatch for {name}")
    manifest_name = f"{path_in_repo}/hf-upload-manifest.json"
    if manifest_name not in remote:
        raise ValueError(f"Hub inventory lacks {manifest_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/qwen-shapleymcg-run"))
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--delete-verified", action="store_true")
    args = parser.parse_args()
    token_file = args.token_file.resolve()
    token = token_file.read_text().strip()
    token_file.unlink()
    if not token:
        raise ValueError("empty Hugging Face token")
    api = HfApi(token=token)
    run_root = args.run_root.resolve()
    receipt_root = run_root / "artifacts" / "hf-upload" / "fits"
    receipt_root.mkdir(parents=True, exist_ok=True)

    for layer in range(args.first_layer, args.layers):
        label = f"{layer:03d}"
        fit_root = run_root / "streaming-fit" / f"layer-{label}"
        encode_receipt = run_root / "fast-encode" / f"layer-{label}" / "encode-receipt.json"
        encode_exit = run_root / "logs" / f"fast-encode-layer-{label}.exit"
        upload_receipt = receipt_root / f"layer-{label}.json"
        if upload_receipt.exists() and not fit_root.exists():
            continue
        while not (
            encode_receipt.exists()
            and encode_exit.exists()
            and encode_exit.read_text().strip() == "0"
        ):
            time.sleep(args.poll_seconds)
        if not fit_root.exists():
            raise FileNotFoundError(f"fit layer disappeared before verified upload: {fit_root}")
        inventory = _inventory(fit_root)
        manifest = {
            "schema": "quant-pipeline.qwen-fit-hf-upload-manifest.v1",
            "layer": layer,
            "encode_receipt_sha256": sha256_file(encode_receipt),
            "file_count": len(inventory),
            "total_bytes": sum(row["bytes"] for row in inventory),
            "files": inventory,
        }
        manifest["manifest_sha256"] = _hash_json(manifest)
        write_json(fit_root / "hf-upload-manifest.json", manifest)
        path_in_repo = f"fits/layer-{label}"
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=fit_root,
            path_in_repo=path_in_repo,
            commit_message=f"Upload sealed Qwen fit layer {label}",
        )
        revision = str(commit.oid)
        _verify_remote(
            api,
            repo_id=args.repo_id,
            revision=revision,
            path_in_repo=path_in_repo,
            root=fit_root,
            inventory=inventory,
        )
        receipt = {
            "schema": "quant-pipeline.qwen-fit-hf-upload-receipt.v1",
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "path_in_repo": path_in_repo,
            "layer": layer,
            "manifest_sha256": manifest["manifest_sha256"],
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "remote_verification": "size-all; sha256-lfs; git-blob-sha1-non-lfs",
            "local_deleted": bool(args.delete_verified),
        }
        receipt["receipt_sha256"] = _hash_json(receipt)
        write_json(upload_receipt, receipt)
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(upload_receipt),
            path_in_repo=f"receipts/fits/layer-{label}.json",
            commit_message=f"Record verified fit layer {label}",
        )
        if args.delete_verified:
            shutil.rmtree(fit_root)
        print(json.dumps({
            "ok": True,
            "layer": layer,
            "revision": revision,
            "bytes": manifest["total_bytes"],
            "local_deleted": bool(args.delete_verified),
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
