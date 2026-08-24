#!/usr/bin/env python3
"""Batch-publish remaining Qwen fit or candidate layers in bounded commits.

The incremental publishers intentionally make one data commit and one receipt
commit per layer.  This recovery publisher preserves the same manifests and
receipts while batching still-local layers, avoiding Hub repository commit
rate and operation-count limits without weakening verification or deletion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

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
    size = path.stat().st_size
    hasher = hashlib.sha1(usedforsecurity=False)
    hasher.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _commit_with_rate_retry(
    api: HfApi,
    *,
    repo_id: str,
    operations: list[CommitOperationAdd],
    message: str,
    retry_minutes: float,
):
    deadline = time.monotonic() + retry_minutes * 60.0
    attempt = 0
    while True:
        attempt += 1
        try:
            return api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=message,
            )
        except HfHubHTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in {429, 500, 502, 503, 504} or time.monotonic() >= deadline:
                raise
            delay = min(
                300.0 if status == 429 else 60.0,
                max(1.0, deadline - time.monotonic()),
            )
            print(
                json.dumps(
                    {
                        "stage": "commit-retry",
                        "status": status,
                        "attempt": attempt,
                        "retry_seconds": delay,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)


def _remote_files(api: HfApi, repo_id: str, revision: str, prefix: str) -> dict:
    return {
        item.path: item
        for item in api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=prefix,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
        if hasattr(item, "size")
    }


def _verify_file(item, path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if item is None or int(item.size) != expected_bytes:
        raise ValueError(f"Hub inventory mismatch for {getattr(item, 'path', path)}")
    lfs = getattr(item, "lfs", None)
    if lfs and lfs.get("sha256"):
        if str(lfs["sha256"]) != expected_sha256:
            raise ValueError(f"Hub LFS hash mismatch for {item.path}")
    elif str(item.blob_id) != _git_blob_sha1(path):
        raise ValueError(f"Hub Git-blob hash mismatch for {item.path}")


def _verify_remote_layer(
    api: HfApi,
    repo_id: str,
    revision: str,
    plural: str,
    layer_row: dict,
) -> None:
    root = layer_row["root"]
    remote = _remote_files(
        api,
        repo_id,
        revision,
        f"{plural}/{root.name}",
    )
    for row in layer_row["upload_rows"]:
        name = f"{plural}/{root.name}/{row['path']}"
        _verify_file(
            remote.get(name),
            root / row["path"],
            int(row["bytes"]),
            row["sha256"],
        )


def _batch_is_remote(
    api: HfApi,
    repo_id: str,
    revision: str,
    plural: str,
    batch: list[dict],
) -> bool:
    try:
        for layer_row in batch:
            _verify_remote_layer(api, repo_id, revision, plural, layer_row)
    except (ValueError, FileNotFoundError):
        return False
    return True


def _kld_succeeded(path: Path | None) -> bool:
    return path is not None and path.is_file() and path.read_text().strip() == "0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/qwen-shapleymcg-run"))
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kind", choices=("fit", "candidate"), required=True)
    parser.add_argument("--kld-exit", type=Path)
    parser.add_argument("--delete-verified", action="store_true")
    parser.add_argument(
        "--include-receipted",
        action="store_true",
        help="re-upload layers that already have a locally verified Hub receipt",
    )
    parser.add_argument("--retry-minutes", type=float, default=75.0)
    parser.add_argument(
        "--batch-layers",
        type=int,
        default=4,
        help="maximum layers per data commit; bounds Hub commit-operation count",
    )
    args = parser.parse_args()
    if args.retry_minutes < 0:
        parser.error("--retry-minutes must be nonnegative")
    if args.batch_layers < 1:
        parser.error("--batch-layers must be positive")
    if args.kind == "candidate" and args.delete_verified and not _kld_succeeded(args.kld_exit):
        raise ValueError("refusing candidate deletion before successful KLD")

    token_file = args.token_file.resolve()
    token = token_file.read_text().strip()
    token_file.unlink()
    if not token:
        raise ValueError("empty Hugging Face token")
    api = HfApi(token=token)
    run_root = args.run_root.resolve()
    if args.kind == "fit":
        data_root = run_root / "streaming-fit"
        plural = "fits"
    else:
        data_root = run_root / "fast-encode"
        plural = "candidates"
    receipt_root = run_root / "artifacts" / "hf-upload" / plural
    receipt_root.mkdir(parents=True, exist_ok=True)
    layer_roots = sorted(
        path for path in data_root.glob("layer-[0-9][0-9][0-9]") if path.is_dir()
    )
    if not args.include_receipted:
        layer_roots = [
            path for path in layer_roots if not (receipt_root / f"{path.name}.json").is_file()
        ]
    if not layer_roots:
        raise ValueError(f"no local {args.kind} layers remain")

    layers: list[dict] = []
    for root in layer_roots:
        layer = int(root.name.removeprefix("layer-"))
        if args.kind == "fit":
            completion_receipt = root / "fit-receipt.json"
            completion_exit = (
                run_root / "logs" / f"streaming-fit-layer-{layer:03d}.exit"
            )
            completion_key = "fit_receipt_sha256"
            completion_label = "fit"
        else:
            completion_receipt = root / "encode-receipt.json"
            completion_exit = (
                run_root / "logs" / f"fast-encode-layer-{layer:03d}.exit"
            )
            completion_key = "encode_receipt_sha256"
            completion_label = "encode"
        if not completion_receipt.is_file() or not _kld_succeeded(completion_exit):
            raise ValueError(
                f"layer {layer} lacks a successful {completion_label} gate"
            )
        inventory = _inventory(root)
        manifest = {
            "schema": (
                "quant-pipeline.qwen-fit-hf-upload-manifest.v2"
                if args.kind == "fit"
                else "quant-pipeline.qwen-candidate-hf-upload-manifest.v1"
            ),
            "layer": layer,
            completion_key: sha256_file(completion_receipt),
            "file_count": len(inventory),
            "total_bytes": sum(int(row["bytes"]) for row in inventory),
            "files": inventory,
        }
        manifest["manifest_sha256"] = _hash_json(manifest)
        manifest_path = root / "hf-upload-manifest.json"
        write_json(manifest_path, manifest)
        upload_rows = inventory + [
            {
                "path": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            }
        ]
        upload_operations = []
        for row in upload_rows:
            upload_operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{plural}/{root.name}/{row['path']}",
                    path_or_fileobj=str(root / row["path"]),
                )
            )
        layers.append(
            {
                "layer": layer,
                "root": root,
                "manifest": manifest,
                "upload_rows": upload_rows,
                "upload_operations": upload_operations,
            }
        )
        print(
            json.dumps(
                {"stage": "inventory", "kind": args.kind, "layer": layer, "bytes": manifest["total_bytes"]},
                sort_keys=True,
            ),
            flush=True,
        )

    changed_receipts = []
    data_revisions = []
    operation_count = 0
    existing_revision = str(api.dataset_info(args.repo_id).sha)
    for start in range(0, len(layers), args.batch_layers):
        batch = layers[start : start + args.batch_layers]
        operations = [
            operation
            for layer_row in batch
            for operation in layer_row["upload_operations"]
        ]
        operation_count += len(operations)
        already_remote = _batch_is_remote(
            api,
            args.repo_id,
            existing_revision,
            plural,
            batch,
        )
        if already_remote:
            revision = existing_revision
        else:
            commit = _commit_with_rate_retry(
                api,
                repo_id=args.repo_id,
                operations=operations,
                message=(
                    f"Batch-publish Qwen {plural} layers "
                    f"{batch[0]['layer']:03d}-{batch[-1]['layer']:03d}"
                ),
                retry_minutes=args.retry_minutes,
            )
            revision = str(commit.oid)
        data_revisions.append(
            {
                "layers": [row["layer"] for row in batch],
                "revision": revision,
                "operations": len(operations),
            }
        )
        for layer_row in batch:
            root = layer_row["root"]
            _verify_remote_layer(api, args.repo_id, revision, plural, layer_row)
            layer = int(layer_row["layer"])
            manifest = layer_row["manifest"]
            receipt = {
                "schema": (
                    "quant-pipeline.qwen-fit-hf-upload-receipt.v2"
                    if args.kind == "fit"
                    else "quant-pipeline.qwen-candidate-hf-upload-receipt.v1"
                ),
                "repo_id": args.repo_id,
                "repo_type": "dataset",
                "revision": revision,
                "path_in_repo": f"{plural}/layer-{layer:03d}",
                "layer": layer,
                "manifest_sha256": manifest["manifest_sha256"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "remote_verification": "size-all; sha256-lfs; git-blob-sha1-non-lfs",
                "local_deleted": bool(args.delete_verified),
            }
            if args.kind == "candidate":
                receipt["encode_receipt_sha256"] = manifest["encode_receipt_sha256"]
                receipt["kld_exit_sha256"] = (
                    sha256_file(args.kld_exit)
                    if args.delete_verified and args.kld_exit
                    else None
                )
            else:
                receipt["fit_receipt_sha256"] = manifest["fit_receipt_sha256"]
            receipt["receipt_sha256"] = _hash_json(receipt)
            receipt_path = receipt_root / f"layer-{layer:03d}.json"
            write_json(receipt_path, receipt)
            changed_receipts.append(receipt_path)
        print(
            json.dumps(
                {
                    "stage": "data-batch-verified",
                    "kind": args.kind,
                    "layers": [row["layer"] for row in batch],
                    "revision": revision,
                    "operations": len(operations),
                    "already_remote": already_remote,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    receipt_operations = [
        CommitOperationAdd(
            path_in_repo=f"receipts/{plural}/{path.name}",
            path_or_fileobj=str(path),
        )
        for path in sorted(receipt_root.glob("layer-[0-9][0-9][0-9].json"))
    ]
    receipt_commit = _commit_with_rate_retry(
        api,
        repo_id=args.repo_id,
        operations=receipt_operations,
        message=f"Batch-record verified Qwen {plural}",
        retry_minutes=args.retry_minutes,
    )
    receipt_revision = str(receipt_commit.oid)
    receipt_remote = _remote_files(api, args.repo_id, receipt_revision, f"receipts/{plural}")
    for path in changed_receipts:
        name = f"receipts/{plural}/{path.name}"
        _verify_file(receipt_remote.get(name), path, path.stat().st_size, sha256_file(path))

    if args.delete_verified:
        for layer_row in layers:
            shutil.rmtree(layer_row["root"])
    print(
        json.dumps(
            {
                "ok": True,
                "kind": args.kind,
                "layers": [row["layer"] for row in layers],
                "data_revisions": data_revisions,
                "receipt_revision": receipt_revision,
                "file_count": operation_count,
                "total_bytes": sum(row["manifest"]["total_bytes"] for row in layers),
                "local_deleted": bool(args.delete_verified),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
