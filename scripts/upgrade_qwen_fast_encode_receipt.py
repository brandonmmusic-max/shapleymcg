#!/usr/bin/env python3
"""Upgrade a v1 fast-encode receipt with exact stored-BF16 identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.normalization.artifact_v31 import tensor_sha256


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("layer_root", type=Path)
    args = parser.parse_args()
    root = args.layer_root.resolve()
    path = root / "encode-receipt.json"
    receipt = json.loads(path.read_text())
    seal = receipt.pop("receipt_sha256", None)
    if seal != _hash_json(receipt):
        raise ValueError("source encode receipt seal mismatch")
    tensor_path = root / str(receipt["candidate_tensor_file"])
    if sha256_file(tensor_path) != receipt["candidate_tensor_sha256"]:
        raise ValueError("candidate tensor file drifted")
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        for row in receipt["scores"]:
            old = row.pop("reconstruction_sha256", None)
            if old is None:
                old = row["codec_fp16_reconstruction_sha256"]
            row["codec_fp16_reconstruction_sha256"] = old
            key = (
                f"K{int(row['bits'])}.E{int(row['expert']):03d}."
                f"{row['projection']}.reconstruction_hf"
            )
            row["stored_bf16_reconstruction_sha256"] = tensor_sha256(handle.get_tensor(key))
    receipt["schema"] = "quant-pipeline.qwen-fast-k34-encode.v2"
    receipt["receipt_sha256"] = _hash_json(receipt)
    write_json(path, receipt)
    print(json.dumps({
        "ok": True,
        "layer": receipt["layer"],
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_file_sha256": sha256_file(path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
