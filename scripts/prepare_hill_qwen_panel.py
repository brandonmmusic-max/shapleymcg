#!/usr/bin/env python3
"""Build a sealed reconstruction of Hill's Qwen3-30B BFCL/RULER panel.

The paper identifies the source categories and reports 16 shared 2,048-token
evaluation sequences, but does not publish the selected row IDs or token IDs.
This script therefore makes the missing choices explicit and reproducible.  It
creates disjoint calibration and evaluation halves, each containing eight BFCL
and eight RULER sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from quant_pipeline.core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
BFCL_FILES = {
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    # BFCL v4 renamed the v3 `simple` category to `simple_python`.
    "simple_python": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
}
RULER_REPO = "llamastack/ruler"
RULER_REVISION = "ab8c79838dbf818e5f19a7eb2d536ad05b4d7b5c"
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _role(identity: str) -> str:
    return "calibration" if hashlib.sha256(identity.encode()).digest()[0] & 1 == 0 else "evaluation"


def _ordered(records: list[dict[str, Any]], role: str, source: str) -> list[dict[str, Any]]:
    chosen = [row for row in records if row["role"] == role]
    return sorted(
        chosen,
        key=lambda row: hashlib.sha256(
            f"hill-qwen-panel-v1\0{source}\0{role}\0{row['identity']}".encode()
        ).digest(),
    )


def _pack(
    records: list[dict[str, Any]],
    *,
    role: str,
    source: str,
    sequence_count: int,
    sequence_length: int,
    separator: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ordered = _ordered(records, role, source)
    needed = sequence_count * sequence_length
    tokens: list[int] = []
    contributors: list[dict[str, Any]] = []
    for row in ordered:
        start = len(tokens)
        values = row["token_ids"]
        tokens.extend(values)
        tokens.extend(separator)
        contributors.append(
            {
                "identity": row["identity"],
                "source": source,
                "token_start": start,
                "token_stop": min(len(tokens), needed),
                "prompt_token_count": len(values),
                "prompt_token_sha256": _hash_json(values),
            }
        )
        if len(tokens) >= needed:
            break
    if len(tokens) < needed:
        raise ValueError(f"{source}/{role} has {len(tokens)} tokens; {needed} required")
    contributors = [row for row in contributors if row["token_start"] < needed]
    array = np.asarray(tokens[:needed], dtype=np.int32).reshape(sequence_count, sequence_length)
    return array, contributors


def _normalise_bfcl_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for function in functions:
        item = json.loads(json.dumps(function))
        parameters = item.get("parameters")
        if isinstance(parameters, dict) and parameters.get("type") == "dict":
            parameters["type"] = "object"
        tools.append({"type": "function", "function": item})
    return tools


def _messages_from_ruler(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("chat_completion_input"):
        value = json.loads(row["chat_completion_input"])
    else:
        value = row.get("messages")
    if not isinstance(value, list) or not value:
        raise ValueError("RULER row has no usable message list")
    return value


def _token_ids(encoded: Any) -> list[int]:
    """Normalize Transformers 4.x/5.x chat-template return shapes."""
    if hasattr(encoded, "get"):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not encoded or not all(
        isinstance(value, int) for value in encoded
    ):
        raise TypeError(f"unexpected chat-template token shape: {type(encoded).__name__}")
    return encoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-count", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.sequence_count != 16 or args.sequence_length != 2048:
        parser.error("the paper reconstruction is fixed at 16 sequences of 2,048 tokens")
    plan = {
        "schema": "quant-pipeline.hill-qwen-panel-plan.v1",
        "model": str(args.model.resolve()),
        "output": str(args.output.resolve()),
        "sequence_count_per_role": args.sequence_count,
        "sequence_length": args.sequence_length,
        "bfcl_repo": BFCL_REPO,
        "bfcl_revision": BFCL_REVISION,
        "ruler_repo": RULER_REPO,
        "ruler_revision": RULER_REVISION,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    source_files = []
    bfcl_records: list[dict[str, Any]] = []
    for category, filename in BFCL_FILES.items():
        path = Path(
            hf_hub_download(BFCL_REPO, filename, repo_type="dataset", revision=BFCL_REVISION)
        )
        source_files.append(
            {"repo": BFCL_REPO, "revision": BFCL_REVISION, "path": filename, "sha256": sha256_file(path)}
        )
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        for row in rows:
            identity = f"bfcl/{category}/{row['id']}"
            messages = row["question"][0]
            tokens = _token_ids(tokenizer.apply_chat_template(
                messages,
                tools=_normalise_bfcl_tools(row["function"]),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            bfcl_records.append(
                {"identity": identity, "role": _role(identity), "token_ids": tokens}
            )

    ruler_records: list[dict[str, Any]] = []
    for task in RULER_TASKS:
        filename = f"data/validation_8192_{task}-00000-of-00001.parquet"
        path = Path(
            hf_hub_download(RULER_REPO, filename, repo_type="dataset", revision=RULER_REVISION)
        )
        source_files.append(
            {"repo": RULER_REPO, "revision": RULER_REVISION, "path": filename, "sha256": sha256_file(path)}
        )
        for row in pq.read_table(path).to_pylist():
            identity = f"ruler/{task}/{row['id']}"
            tokens = _token_ids(tokenizer.apply_chat_template(
                _messages_from_ruler(row),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            ruler_records.append(
                {"identity": identity, "role": _role(identity), "token_ids": tokens}
            )

    separator = [int(tokenizer.eos_token_id)]
    arrays: dict[str, np.ndarray] = {}
    inventory: dict[str, Any] = {}
    half = args.sequence_count // 2
    for role in ("calibration", "evaluation"):
        bfcl, bfcl_inventory = _pack(
            bfcl_records,
            role=role,
            source="bfcl",
            sequence_count=half,
            sequence_length=args.sequence_length,
            separator=separator,
        )
        ruler, ruler_inventory = _pack(
            ruler_records,
            role=role,
            source="ruler",
            sequence_count=half,
            sequence_length=args.sequence_length,
            separator=separator,
        )
        arrays[role] = np.concatenate((bfcl, ruler), axis=0)
        inventory[role] = {"bfcl": bfcl_inventory, "ruler": ruler_inventory}

    token_path = output / "panel-token-ids.npz"
    buffer = io.BytesIO()
    np.savez(buffer, calibration=arrays["calibration"], evaluation=arrays["evaluation"])
    atomic_write(token_path, buffer.getvalue())
    panel = {
        "schema": "quant-pipeline.hill-qwen-panel-reconstruction.v1",
        "status": "paper-protocol-reconstruction-not-author-provided-token-panel",
        "paper": {
            "title": "Saturation Makes Quantization Error Additive: A Coverage Model with a Certificate",
            "arxiv": "2607.12266v1",
            "documented_protocol": "BFCL-v3 live_simple/live_multiple/simple_python/multiple plus RULER validation prompts; disjoint evaluation half; 16x2048 tokens",
        },
        "model": "Qwen/Qwen3-30B-A3B-Base",
        "tokenizer_path": str(args.model.resolve()),
        "tokenizer_files": {
            name: sha256_file(args.model.resolve() / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "selection": {
            "role": "low bit of sha256(source/category/id): 0 calibration, 1 evaluation",
            "order": "sha256(hill-qwen-panel-v1\\0source\\0role\\0identity)",
            "packing": "8 BFCL then 8 RULER sequences per role; EOS between prompts; take first 8x2048 tokens per source",
            "chat_template": "checkpoint chat template; add_generation_prompt=true; enable_thinking=false",
        },
        "shape": [args.sequence_count, args.sequence_length],
        "source_files": source_files,
        "inventory": inventory,
        "calibration_token_sha256": _hash_json(arrays["calibration"].tolist()),
        "evaluation_token_sha256": _hash_json(arrays["evaluation"].tolist()),
        "token_file": token_path.name,
        "token_file_sha256": sha256_file(token_path),
    }
    panel["panel_sha256"] = _hash_json(panel)
    write_json(output / "panel.json", panel)
    print(json.dumps({"ok": True, **panel}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
