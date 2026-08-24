#!/usr/bin/env python3
"""Seal the exact WikiText-2 10x2048 panel and BF16 teacher logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from measure_turboderp_wiki2_kld import _capture, _hash_json, _load_model, _prepare_panel
from quant_pipeline.core.artifacts import sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.turboderp-wiki2-teacher-plan.v1",
        "model": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "source_receipt": str(args.source_receipt.resolve()),
        "output": str(args.output.resolve()),
        "rows": 10,
        "row_length": 2048,
        "attention_backend": args.attention_backend,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch

    source = json.loads(args.source_receipt.read_text())
    source_seal = source.get("receipt_sha256")
    if source.get("revision") != args.model_revision or source_seal != _hash_json(
        {key: value for key, value in source.items() if key != "receipt_sha256"}
    ):
        raise ValueError("teacher source receipt revision or seal mismatch")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    panel, token_ids = _prepare_panel(args.model.resolve(), output)
    teacher = _load_model(args.model.resolve(), args.attention_backend)
    teacher_paths = _capture(
        teacher,
        token_ids,
        output / "teacher-logits",
        "posttrained-bf16-reference",
    )
    del teacher
    torch.cuda.empty_cache()
    receipt = {
        "schema": "quant-pipeline.turboderp-wiki2-teacher.v1",
        "model_revision": args.model_revision,
        "source_receipt_sha256": source_seal,
        "source_receipt_file_sha256": sha256_file(args.source_receipt),
        "panel_sha256": panel["panel_sha256"],
        "teacher_files": [
            {"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)}
            for path in teacher_paths
        ],
        "teacher_dtype": "BF16 model execution; float32 stored logits",
        "kv_cache": "disabled (use_cache=False)",
        "elapsed_seconds": time.monotonic() - started,
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    write_json(output / "teacher-receipt.json", receipt)
    print(json.dumps({"ok": True, **receipt}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
