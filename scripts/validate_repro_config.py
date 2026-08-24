#!/usr/bin/env python3
"""Validate the sealed-example configuration without launching model work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any


PLACEHOLDER = re.compile(r"^__REQUIRED_[A-Z0-9_]+__$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


def placeholders(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(placeholders(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(placeholders(item, f"{path}[{index}]"))
    elif isinstance(value, str) and PLACEHOLDER.fullmatch(value):
        found.append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()
    root = args.config_dir.resolve()
    names = ("artifact-lock.json", "adapter-config.json", "campaign.json")
    documents = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in names}
    documents["experiment.toml"] = tomllib.loads((root / "experiment.toml").read_text(encoding="utf-8"))
    artifact = documents["artifact-lock.json"]
    adapter = documents["adapter-config.json"]
    campaign = documents["campaign.json"]
    if not REVISION.fullmatch(artifact["model"]["revision"]):
        raise SystemExit("model revision is not immutable 40-hex")
    if not REVISION.fullmatch(artifact["corpus"]["dataset_revision"]):
        raise SystemExit("dataset revision is not immutable 40-hex")
    if artifact["b12x"]["commit"] != "36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8":
        raise SystemExit("B12X commit drift")
    if any(not SHA256.fullmatch(value) for value in artifact["b12x"]["closure"].values()):
        raise SystemExit("B12X closure contains a non-SHA256 identity")
    hash_values = (
        artifact["model"]["config_sha256"],
        artifact["model"]["index_sha256"],
        artifact["model"]["shard_manifest_sha256"],
        artifact["corpus"]["calibration_jsonl_sha256"],
        artifact["corpus"]["sealed_corpus_sha256"],
        artifact["corpus"]["kld_window_sha256"],
        artifact["exl3"]["corrected_source_manifest_sha256"],
        artifact["exl3"]["source_closure_sha256"],
        artifact["exl3"]["numeric_core_sha256"],
        artifact["exl3"]["r10_codec_sha256"],
        artifact["exl3"]["sm100_extension_sha256"],
    )
    if any(not (SHA256.fullmatch(value) or PLACEHOLDER.fullmatch(value)) for value in hash_values):
        raise SystemExit("artifact identity must be an exact SHA256 or explicit required placeholder")
    if artifact["exl3"]["upstream_commit"] != "c5d9c657966ffeeaa9353f0cc899f18629da4a13":
        raise SystemExit("ExLlamaV3 source commit drift")
    source_manifest = root / "corrected-exl3-source-manifest.json"
    if artifact["exl3"]["corrected_source_manifest_sha256"] != hashlib.sha256(source_manifest.read_bytes()).hexdigest():
        raise SystemExit("corrected EXL3 source manifest identity drift")
    if campaign["layers"] != list(range(48)) or campaign["retention_mode"] != "capture-plus-ledger":
        raise SystemExit("campaign geometry or retention policy drift")
    unresolved = {name: placeholders(document) for name, document in documents.items()}
    unresolved = {name: rows for name, rows in unresolved.items() if rows}
    print(json.dumps({"ok": not unresolved, "unresolved": unresolved}, sort_keys=True))
    if unresolved and not args.allow_placeholders:
        raise SystemExit("required placeholders remain; refusing production use")
    if not unresolved:
        if ":" not in adapter["service_factory"]:
            raise SystemExit("service_factory must be module:attribute")
        if isinstance(adapter["exact_payload_byte_budget"], bool) or int(adapter["exact_payload_byte_budget"]) <= 0:
            raise SystemExit("exact payload byte budget must be positive")
        if float(adapter["reanchor_kld_threshold"]) < 0:
            raise SystemExit("reanchor KLD threshold must be non-negative")
        if not SHA256.fullmatch(adapter["runtime_reader_identity_sha256"]):
            raise SystemExit("runtime reader identity must be an exact SHA256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
