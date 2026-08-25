#!/usr/bin/env python3
"""Bind the exact GLM calibration and KLD-procedure lineage to a Qwen result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from quant_pipeline.core.artifacts import (
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)


REVISION = re.compile(r"[0-9a-f]{40}")
SELECTED_CONTROL_PATHS = (
    "calibration/qwen-sealed-corpus.json",
    "calibration/reap-recall-packing-receipt.json",
    "calibration/reap_recall_calib.jsonl",
    "calibration/reap_recall_calib.role-safe-packed.jsonl",
    "kld/kld-window.json",
    "kld/source-prefix.txt",
    "kld/teacher-capture-receipt.json",
    "kld/teacher-logits.safetensors",
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _lfs_sha(info: Any) -> str | None:
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def _verify_remote_rows(
    *,
    api: Any,
    repo: str,
    repo_type: str,
    revision: str,
    prefix: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    expected = {
        f"{prefix.strip('/')}/{row['path']}".strip("/"): row
        for row in rows
    }
    values = api.get_paths_info(
        repo_id=repo,
        repo_type=repo_type,
        revision=revision,
        paths=sorted(expected),
        expand=True,
    )
    remote = {value.path: value for value in values}
    if set(remote) != set(expected):
        raise ValueError("immutable lineage repository lacks one or more required paths")
    verified = []
    for path in sorted(expected):
        row = expected[path]
        info = remote[path]
        if int(info.size) != int(row["bytes"]):
            raise ValueError(f"remote lineage size differs: {path}")
        lfs_sha = _lfs_sha(info)
        if lfs_sha is not None:
            if lfs_sha != row["sha256"]:
                raise ValueError(f"remote lineage LFS hash differs: {path}")
            method = "hub-lfs-sha256"
        else:
            downloaded = Path(hf_hub_download(
                repo_id=repo,
                repo_type=repo_type,
                revision=revision,
                filename=path,
            ))
            if sha256_file(downloaded) != row["sha256"]:
                raise ValueError(f"downloaded lineage bytes differ: {path}")
            method = "downloaded-sha256"
        verified.append({
            "path": path,
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "verification": method,
        })
    return verified


def _hardlink_tree(source: Path, destination: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"fast KLD tree contains a symlink: {path}")
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        rows.append({
            "path": target.relative_to(destination.parent).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    if not rows:
        raise ValueError("fast KLD result tree is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-bundle", type=Path, required=True)
    parser.add_argument("--fast-kld-root", type=Path, required=True)
    parser.add_argument("--qwen-dataset-repo", required=True)
    parser.add_argument("--qwen-dataset-revision", required=True)
    parser.add_argument("--qwen-control-prefix", required=True)
    parser.add_argument("--glm-model-repo", required=True)
    parser.add_argument("--glm-model-revision", required=True)
    parser.add_argument(
        "--glm-calibration-path",
        default="calibration/reap_recall_calib.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for label, revision in (
        ("Qwen dataset", args.qwen_dataset_revision),
        ("GLM model", args.glm_model_revision),
    ):
        if REVISION.fullmatch(revision) is None:
            parser.error(f"{label} revision must be immutable 40-hex")
    plan = {
        "schema": "quant-pipeline.qwen-glm-lineage-plan.v1",
        "control_bundle": str(args.control_bundle.resolve()),
        "fast_kld_root": str(args.fast_kld_root.resolve()),
        "qwen_dataset_repo": args.qwen_dataset_repo,
        "qwen_dataset_revision": args.qwen_dataset_revision,
        "qwen_control_prefix": args.qwen_control_prefix.strip("/"),
        "glm_model_repo": args.glm_model_repo,
        "glm_model_revision": args.glm_model_revision,
        "glm_calibration_path": args.glm_calibration_path.strip("/"),
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    bundle = args.control_bundle.resolve()
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != _hash_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("control-bundle manifest seal mismatch")
    by_path = {row["path"]: row for row in manifest["files"]}
    if any(path not in by_path for path in SELECTED_CONTROL_PATHS):
        raise ValueError("control-bundle manifest lacks required lineage files")
    selected = [dict(by_path[path]) for path in SELECTED_CONTROL_PATHS]
    for row in selected:
        path = bundle / row["path"]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"local control-bundle lineage file drifted: {row['path']}")

    from huggingface_hub import HfApi

    api = HfApi()
    qwen_info = api.dataset_info(args.qwen_dataset_repo, revision=args.qwen_dataset_revision)
    if str(qwen_info.sha) != args.qwen_dataset_revision:
        raise ValueError("Qwen lineage dataset revision did not resolve exactly")
    qwen_remote = _verify_remote_rows(
        api=api,
        repo=args.qwen_dataset_repo,
        repo_type="dataset",
        revision=args.qwen_dataset_revision,
        prefix=args.qwen_control_prefix,
        rows=selected,
    )
    glm_info = api.model_info(args.glm_model_repo, revision=args.glm_model_revision)
    if str(glm_info.sha) != args.glm_model_revision:
        raise ValueError("GLM lineage model revision did not resolve exactly")
    original_calibration = by_path["calibration/reap_recall_calib.jsonl"]
    glm_remote = _verify_remote_rows(
        api=api,
        repo=args.glm_model_repo,
        repo_type="model",
        revision=args.glm_model_revision,
        prefix="",
        rows=[{
            "path": args.glm_calibration_path.strip("/"),
            "bytes": original_calibration["bytes"],
            "sha256": original_calibration["sha256"],
        }],
    )

    fast_root = args.fast_kld_root.resolve()
    fast_report = json.loads((fast_root / "kld-report.json").read_text())
    if fast_report.get("report_sha256") != _hash_json(
        {key: value for key, value in fast_report.items() if key != "report_sha256"}
    ):
        raise ValueError("progressive fast-KLD report seal mismatch")
    if fast_report.get("teacher_sha256") != by_path["kld/teacher-logits.safetensors"]["sha256"]:
        raise ValueError("progressive fast-KLD teacher differs from GLM-lineage Qwen teacher")

    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for relative in (
        "calibration/qwen-sealed-corpus.json",
        "calibration/reap-recall-packing-receipt.json",
        "kld/kld-window.json",
        "kld/source-prefix.txt",
        "kld/teacher-capture-receipt.json",
    ):
        target = evidence / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / relative, target)
    fast_rows = _hardlink_tree(fast_root, output / "fast-progressive-kld")
    receipt = {
        "schema": "quant-pipeline.qwen-glm-lineage.v1",
        "historical_relationship": (
            "The GLM calibration JSONL bytes are identical. The KLD procedure uses the "
            "same pinned WikiText-2 raw-test source-prefix construction, while Qwen token "
            "IDs and BF16 logits are necessarily regenerated with the Qwen tokenizer/model."
        ),
        "control_bundle_manifest_sha256": manifest["manifest_sha256"],
        "qwen_dataset": {
            "repo": args.qwen_dataset_repo,
            "revision": args.qwen_dataset_revision,
            "prefix": args.qwen_control_prefix.strip("/"),
            "verified_files": qwen_remote,
        },
        "original_glm_model": {
            "repo": args.glm_model_repo,
            "revision": args.glm_model_revision,
            "verified_files": glm_remote,
        },
        "progressive_fast_kld": {
            "report_sha256": fast_report["report_sha256"],
            "mean_kld": fast_report["summary"]["mean"],
            "teacher_sha256": fast_report["teacher_sha256"],
            "student_sha256": fast_report["student_sha256"],
            "token_kld_sha256": fast_report["token_kld_sha256"],
            "files": fast_rows,
        },
    }
    receipt["lineage_sha256"] = _hash_json(receipt)
    write_json(output / "lineage.json", receipt)
    print(json.dumps({
        "ok": True,
        "lineage_sha256": receipt["lineage_sha256"],
        "glm_calibration_sha256": original_calibration["sha256"],
        "qwen_teacher_sha256": fast_report["teacher_sha256"],
        "progressive_fast_kld_mean": fast_report["summary"]["mean"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
