#!/usr/bin/env python3
"""Fetch and verify the pinned corrected R10/EXL3 Python source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "quant-pipeline.corrected-exl3-source-manifest.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "qwen3-30b-a3b-b200"
    / "corrected-exl3-source-manifest.json"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError("corrected EXL3 source manifest schema mismatch")
    if not REVISION.fullmatch(str(document.get("revision", ""))):
        raise ValueError("corrected EXL3 source revision must be an immutable commit")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("corrected EXL3 source manifest file inventory is empty")
    for relative, expected in files.items():
        candidate = Path(relative)
        if (
            not isinstance(relative, str)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not SHA256.fullmatch(str(expected))
        ):
            raise ValueError("corrected EXL3 source manifest contains an unsafe entry")
    required = {
        str(document.get("numeric_core", "")),
        str(Path(str(document.get("source_root", ""))) / "r7_encoder" / "r10_codec.py"),
        str(document.get("license_file", "")),
    }
    if not required <= set(files):
        raise ValueError("corrected EXL3 source manifest omits a required closure file")
    return document


def verify_checkout(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"corrected EXL3 source file is missing or symlinked: {relative}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"corrected EXL3 source hash mismatch: {relative}")
        observed[relative] = actual
    return {
        "ok": True,
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "file_count": len(observed),
        "source_root": str((root / manifest["source_root"]).resolve()),
        "numeric_core": str((root / manifest["numeric_core"]).resolve()),
        "license_file": str((root / manifest["license_file"]).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true", help="download the pinned files from Hugging Face")
    args = parser.parse_args()
    destination = args.destination.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if destination.exists():
        raise SystemExit("destination must not exist; refusing to alter an existing source closure")
    plan = {
        "dry_run": not args.execute,
        "repository": manifest["repository"],
        "repository_type": manifest["repository_type"],
        "revision": manifest["revision"],
        "destination": str(destination),
        "manifest": str(manifest_path),
        "file_count": len(manifest["files"]),
        "source_root": str(destination / manifest["source_root"]),
        "numeric_core": str(destination / manifest["numeric_core"]),
        "license_file": str(destination / manifest["license_file"]),
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("huggingface-hub is required to fetch the corrected EXL3 source closure") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=manifest["repository"],
        repo_type=manifest["repository_type"],
        revision=manifest["revision"],
        local_dir=destination,
        allow_patterns=sorted(manifest["files"]),
    )
    print(json.dumps(verify_checkout(destination, manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
