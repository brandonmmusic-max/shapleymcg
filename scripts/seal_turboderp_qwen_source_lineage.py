#!/usr/bin/env python3
"""Seal the source-revision evidence for the TurboDerp Qwen K4 comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


SOURCE_REPO = "Qwen/Qwen3-30B-A3B"
SOURCE_REVISION = "4c446470ba0aec43e22ac1128f9ffd915f338ba3"
REFERENCE_REPO = "turboderp/Qwen3-30B-A3B-exl3"
REFERENCE_REVISION = "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _tree_identity(api, repo: str, revision: str, names: set[str]) -> dict[str, dict[str, Any]]:
    rows = {}
    for item in api.list_repo_tree(
        repo_id=repo,
        revision=revision,
        recursive=False,
        expand=True,
    ):
        if item.path not in names:
            continue
        lfs = getattr(item, "lfs", None) or {}
        rows[item.path] = {
            "bytes": int(item.size),
            "git_blob_sha1": str(item.blob_id),
            "lfs_sha256": lfs.get("sha256"),
        }
    missing = names - set(rows)
    if missing:
        raise ValueError(f"missing Hub files for {repo}@{revision}: {sorted(missing)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--reference-repo", default=REFERENCE_REPO)
    parser.add_argument("--reference-revision", default=REFERENCE_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.turboderp-qwen-source-lineage-plan.v1",
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "reference_repo": args.reference_repo,
        "reference_revision": args.reference_revision,
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    source_info = api.model_info(args.source_repo, revision=args.source_revision)
    reference_info = api.model_info(args.reference_repo, revision=args.reference_revision)
    if source_info.sha != args.source_revision or reference_info.sha != args.reference_revision:
        raise ValueError("Hub revision did not resolve to the requested immutable commit")

    reference_commits = api.list_repo_commits(
        args.reference_repo,
        revision=args.reference_revision,
    )
    reference_commit = next(
        row for row in reference_commits if row.commit_id == args.reference_revision
    )
    source_commits = api.list_repo_commits(args.source_repo)
    eligible = [row for row in source_commits if row.created_at <= reference_commit.created_at]
    if not eligible:
        raise ValueError("no source commit predates the reference quant upload")
    latest_available = max(eligible, key=lambda row: row.created_at)
    if latest_available.commit_id != args.source_revision:
        raise ValueError(
            "requested source revision was not the latest source commit available "
            "when the reference quant was uploaded"
        )

    source_names = set(TOKENIZER_FILES) | {"config.json"} | {
        f"model-{index:05d}-of-00016.safetensors" for index in range(1, 17)
    }
    reference_names = set(TOKENIZER_FILES) | {"config.json", "quantization_config.json"}
    source_tree = _tree_identity(api, args.source_repo, args.source_revision, source_names)
    reference_tree = _tree_identity(
        api,
        args.reference_repo,
        args.reference_revision,
        reference_names,
    )
    tokenizer_matches = {
        name: source_tree[name] == reference_tree[name] for name in TOKENIZER_FILES
    }
    if not all(tokenizer_matches.values()):
        raise ValueError("reference quant tokenizer files do not match the inferred source")

    source_config_path = Path(
        hf_hub_download(args.source_repo, "config.json", revision=args.source_revision)
    )
    reference_config_path = Path(
        hf_hub_download(args.reference_repo, "config.json", revision=args.reference_revision)
    )
    quant_config_path = Path(
        hf_hub_download(
            args.reference_repo,
            "quantization_config.json",
            revision=args.reference_revision,
        )
    )
    source_config = json.loads(source_config_path.read_text())
    reference_config = json.loads(reference_config_path.read_text())
    embedded_quant = reference_config.pop("quantization_config", None)
    standalone_quant = json.loads(quant_config_path.read_text())
    if reference_config != source_config:
        raise ValueError("reference config minus quantization metadata differs from source")
    if not isinstance(embedded_quant, dict):
        raise ValueError("reference config omits embedded quantization metadata")
    expected_scope = {
        "bits": 4.0,
        "head_bits": 6,
        "calibration": {"rows": 100, "cols": 2048},
    }
    if any(standalone_quant.get(key) != value for key, value in expected_scope.items()):
        raise ValueError("reference quantization scope differs from the pinned K4 comparison")

    receipt = {
        "schema": "quant-pipeline.turboderp-qwen-source-lineage.v1",
        "inference": (
            "The pinned Qwen commit was the latest source commit available when the "
            "TurboDerp K4 content commit was uploaded; copied config and tokenizer "
            "identities match. The reference repository does not itself record an "
            "upstream source commit, so this is strong lineage evidence, not an "
            "upstream-authored source-revision declaration."
        ),
        "source": {
            "repo": args.source_repo,
            "revision": args.source_revision,
            "commit_time": _time(latest_available.created_at),
            "config_sha256": sha256_file(source_config_path),
            "tree": source_tree,
        },
        "reference": {
            "repo": args.reference_repo,
            "revision": args.reference_revision,
            "commit_time": _time(reference_commit.created_at),
            "config_sha256": sha256_file(reference_config_path),
            "quantization_config_sha256": sha256_file(quant_config_path),
            "tree": reference_tree,
            "quantization_scope": expected_scope,
        },
        "checks": {
            "source_was_latest_available_at_reference_upload": True,
            "config_equal_after_removing_embedded_quantization_config": True,
            "tokenizer_file_identities_equal": tokenizer_matches,
            "source_model_shards": 16,
        },
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, receipt)
    print(json.dumps({"ok": True, **receipt}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
