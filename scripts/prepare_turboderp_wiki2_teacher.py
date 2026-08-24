#!/usr/bin/env python3
"""Seal the exact WikiText-2 10x2048 panel and BF16 teacher logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from measure_turboderp_wiki2_kld import _capture, _hash_json, _load_model, _prepare_panel
from quant_pipeline.core.artifacts import sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-panel-root", type=Path)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.turboderp-wiki2-teacher-plan.v1",
        "model": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "source_receipt": str(args.source_receipt.resolve()),
        "output": str(args.output.resolve()),
        "existing_panel_root": (
            str(args.existing_panel_root.resolve()) if args.existing_panel_root else None
        ),
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
    if args.existing_panel_root:
        existing = args.existing_panel_root.resolve()
        panel = json.loads((existing / "panel.json").read_text())
        if panel.get("panel_sha256") != _hash_json(
            {key: value for key, value in panel.items() if key != "panel_sha256"}
        ):
            raise ValueError("existing panel seal mismatch")
        token_source = existing / panel["token_file"]
        if sha256_file(token_source) != panel["token_file_sha256"]:
            raise ValueError("existing panel token file drifted")
        shutil.copy2(existing / "panel.json", output / "panel.json")
        shutil.copy2(token_source, output / panel["token_file"])
        with np.load(output / panel["token_file"]) as handle:
            token_ids = np.asarray(handle["input_ids"], dtype=np.int32)
        if list(token_ids.shape) != [10, 2048]:
            raise ValueError("existing panel is not 10x2048")
    else:
        panel, token_ids = _prepare_panel(args.model.resolve(), output)
    teacher = _load_model(args.model.resolve(), args.attention_backend)
    teacher_paths = _capture(
        teacher,
        token_ids,
        output / "teacher-logits",
        "posttrained-bf16-reference",
        metadata={
            "attention_backend": args.attention_backend,
            "model_revision": args.model_revision,
            "panel_sha256": panel["panel_sha256"],
        },
    )
    del teacher
    torch.cuda.empty_cache()
    receipt = {
        "schema": "quant-pipeline.turboderp-wiki2-teacher.v1",
        "model_revision": args.model_revision,
        "attention_backend": args.attention_backend,
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
