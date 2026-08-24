#!/usr/bin/env python3
"""Upload sealed Qwen K3/K4 candidate layers and verify their Hub identities.

Candidate tensors remain local by default because the exact KLD replay consumes
them.  ``--delete-verified`` is accepted only after a supplied KLD exit receipt
contains zero, so incremental publication cannot invalidate an unfinished run.
"""

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


def _kld_succeeded(path: Path | None) -> bool:
    return path is not None and path.is_file() and path.read_text().strip() == "0"


def _verify_sealed_document(document: dict, field: str, label: str) -> None:
    seal = document.get(field)
    body = {key: value for key, value in document.items() if key != field}
    if seal != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/qwen-shapleymcg-run"))
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--delete-verified", action="store_true")
    parser.add_argument(
        "--kld-exit",
        type=Path,
        help="required zero-valued KLD exit receipt when --delete-verified is used",
    )
    args = parser.parse_args()
    if args.delete_verified and not _kld_succeeded(args.kld_exit):
        raise ValueError("refusing candidate deletion before a successful KLD exit receipt")

    token_file = args.token_file.resolve()
    token = token_file.read_text().strip()
    token_file.unlink()
    if not token:
        raise ValueError("empty Hugging Face token")
    api = HfApi(token=token)
    run_root = args.run_root.resolve()
    receipt_root = run_root / "artifacts" / "hf-upload" / "candidates"
    receipt_root.mkdir(parents=True, exist_ok=True)

    for layer in range(args.first_layer, args.layers):
        label = f"{layer:03d}"
        candidate_root = run_root / "fast-encode" / f"layer-{label}"
        encode_receipt = candidate_root / "encode-receipt.json"
        encode_exit = run_root / "logs" / f"fast-encode-layer-{label}.exit"
        upload_receipt = receipt_root / f"layer-{label}.json"
        if upload_receipt.exists() and not candidate_root.exists():
            continue
        while not (
            encode_receipt.exists()
            and encode_exit.exists()
            and encode_exit.read_text().strip() == "0"
        ):
            time.sleep(args.poll_seconds)
        if not candidate_root.exists():
            raise FileNotFoundError(
                f"candidate layer disappeared before verified upload: {candidate_root}"
            )

        inventory = _inventory(candidate_root)
        if upload_receipt.exists():
            receipt = json.loads(upload_receipt.read_text())
            _verify_sealed_document(receipt, "receipt_sha256", f"candidate layer {label} receipt")
            manifest_path = candidate_root / "hf-upload-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            _verify_sealed_document(
                manifest,
                "manifest_sha256",
                f"candidate layer {label} upload manifest",
            )
            if (
                receipt.get("layer") != layer
                or receipt.get("path_in_repo") != f"candidates/layer-{label}"
                or receipt.get("manifest_sha256") != manifest.get("manifest_sha256")
                or manifest.get("files") != inventory
            ):
                raise ValueError(f"candidate layer {label} resume identity mismatch")
            _verify_remote(
                api,
                repo_id=args.repo_id,
                revision=str(receipt["revision"]),
                path_in_repo=str(receipt["path_in_repo"]),
                root=candidate_root,
                inventory=inventory,
            )
            if args.delete_verified:
                receipt["local_deleted"] = True
                receipt["kld_exit_sha256"] = sha256_file(args.kld_exit)
                receipt.pop("receipt_sha256", None)
                receipt["receipt_sha256"] = _hash_json(receipt)
                write_json(upload_receipt, receipt)
                api.upload_file(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    path_or_fileobj=str(upload_receipt),
                    path_in_repo=f"receipts/candidates/layer-{label}.json",
                    commit_message=f"Record safe candidate cleanup {label}",
                )
                shutil.rmtree(candidate_root)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "layer": layer,
                        "revision": receipt["revision"],
                        "bytes": manifest["total_bytes"],
                        "local_deleted": bool(args.delete_verified),
                        "resumed": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        manifest = {
            "schema": "quant-pipeline.qwen-candidate-hf-upload-manifest.v1",
            "layer": layer,
            "encode_receipt_sha256": sha256_file(encode_receipt),
            "file_count": len(inventory),
            "total_bytes": sum(row["bytes"] for row in inventory),
            "files": inventory,
        }
        manifest["manifest_sha256"] = _hash_json(manifest)
        write_json(candidate_root / "hf-upload-manifest.json", manifest)
        path_in_repo = f"candidates/layer-{label}"
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=candidate_root,
            path_in_repo=path_in_repo,
            commit_message=f"Upload sealed Qwen candidate layer {label}",
        )
        revision = str(commit.oid)
        _verify_remote(
            api,
            repo_id=args.repo_id,
            revision=revision,
            path_in_repo=path_in_repo,
            root=candidate_root,
            inventory=inventory,
        )
        receipt = {
            "schema": "quant-pipeline.qwen-candidate-hf-upload-receipt.v1",
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "path_in_repo": path_in_repo,
            "layer": layer,
            "manifest_sha256": manifest["manifest_sha256"],
            "encode_receipt_sha256": manifest["encode_receipt_sha256"],
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "remote_verification": "size-all; sha256-lfs; git-blob-sha1-non-lfs",
            "local_deleted": bool(args.delete_verified),
            "kld_exit_sha256": (
                sha256_file(args.kld_exit) if args.delete_verified and args.kld_exit else None
            ),
        }
        receipt["receipt_sha256"] = _hash_json(receipt)
        write_json(upload_receipt, receipt)
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(upload_receipt),
            path_in_repo=f"receipts/candidates/layer-{label}.json",
            commit_message=f"Record verified candidate layer {label}",
        )
        if args.delete_verified:
            shutil.rmtree(candidate_root)
        print(
            json.dumps(
                {
                    "ok": True,
                    "layer": layer,
                    "revision": revision,
                    "bytes": manifest["total_bytes"],
                    "local_deleted": bool(args.delete_verified),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
