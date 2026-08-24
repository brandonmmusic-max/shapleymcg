#!/usr/bin/env python3
"""Capture a sealed BF16 teacher for any sealed rank-two Qwen token panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from measure_qwen_mcg_causal_allocation import _hash_json, _verify_seal
from measure_qwen_mcg_panel_allocation import _load_panel
from quant_pipeline.core.artifacts import prepare_empty_destination, sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--token-key", default="input_ids")
    parser.add_argument("--drop-last-logit", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    panel, token_ids = _load_panel(args.panel_root.resolve(), args.token_key)
    plan = {
        "schema": "quant-pipeline.qwen-sealed-panel-teacher-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "model_revision": args.model_revision,
        "panel_root": str(args.panel_root.resolve()),
        "panel_sha256": panel["panel_sha256"],
        "token_key": args.token_key,
        "token_shape": list(token_ids.shape),
        "drop_last_logit": args.drop_last_logit,
        "attention_backend": args.attention_backend,
        "device_map": args.device_map,
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    output = prepare_empty_destination(args.output.resolve())
    write_json(output / "plan.json", plan | {"dry_run": False})
    logits_root = output / "teacher-logits"
    logits_root.mkdir()
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model.resolve(),
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    files = []
    started = time.monotonic()
    for index, values in enumerate(token_ids):
        ids = torch.from_numpy(values.astype(np.int64, copy=False)).unsqueeze(0).to(device)
        row_started = time.monotonic()
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
            if args.drop_last_logit:
                logits = logits[:, :-1]
        stored = logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()
        path = logits_root / f"row-{index:02d}.safetensors"
        save_file({"logits": stored}, path, metadata={
            "role": "kld-teacher",
            "row": str(index),
            "model_revision": args.model_revision,
            "panel_sha256": panel["panel_sha256"],
            "attention_backend": args.attention_backend,
            "token_key": args.token_key,
            "drop_last_logit": str(args.drop_last_logit).lower(),
        })
        files.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
        print(json.dumps({
            "stage": "capture",
            "row": index,
            "positions": int(stored.shape[0]),
            "sha256": files[-1]["sha256"],
            "elapsed_seconds": time.monotonic() - row_started,
        }, sort_keys=True), flush=True)
    receipt = {
        "schema": "quant-pipeline.qwen-sealed-panel-teacher.v1",
        "model_revision": args.model_revision,
        "attention_backend": args.attention_backend,
        "panel_sha256": panel["panel_sha256"],
        "token_key": args.token_key,
        "token_shape": list(token_ids.shape),
        "drop_last_logit": args.drop_last_logit,
        "teacher_files": files,
        "teacher_dtype": "BF16 model execution; float32 stored logits",
        "kv_cache": "disabled (use_cache=False)",
        "elapsed_seconds": time.monotonic() - started,
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    write_json(output / "teacher-receipt.json", receipt)
    _verify_seal(receipt, "receipt_sha256", "panel teacher")
    print(json.dumps({"ok": True, **receipt}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
