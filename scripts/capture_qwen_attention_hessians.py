#!/usr/bin/env python3
"""Capture exact Qwen attention-input Gram matrices on one corpus shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _atomic_save(path: Path, tensors: dict, metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(
        {key: value.detach().contiguous().cpu() for key, value in tensors.items()},
        temporary,
        metadata=metadata,
    )
    os.replace(temporary, path)


def _load_corpus(path: Path, role: str, shard_index: int, shard_count: int) -> tuple[dict, list[dict]]:
    corpus = json.loads(path.read_text())
    expected = corpus.get("seal_sha256")
    if expected != _hash_json({key: value for key, value in corpus.items() if key != "seal_sha256"}):
        raise ValueError("sealed corpus hash mismatch")
    windows = corpus.get("windows", {}).get(role)
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"sealed corpus has no {role!r} windows")
    selected = [row for index, row in enumerate(windows) if index % shard_count == shard_index]
    if not selected:
        raise ValueError("corpus shard is empty")
    return corpus, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sealed-corpus", type=Path, required=True)
    parser.add_argument("--role", default="fit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")

    corpus, windows = _load_corpus(
        args.sealed_corpus.resolve(), args.role, args.shard_index, args.shard_count
    )
    output = args.output.resolve()
    plan = {
        "schema": "quant-pipeline.qwen-attention-hessian-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "source_revision": args.source_revision,
        "sealed_corpus": str(args.sealed_corpus.resolve()),
        "sealed_corpus_sha256": sha256_file(args.sealed_corpus),
        "sealed_corpus_seal": corpus["seal_sha256"],
        "role": args.role,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "window_tokens": int(corpus["window_tokens"]),
        "windows": [
            {
                "document_id": row["document_id"],
                "offset": int(row["offset"]),
                "token_sha256": row["token_sha256"],
            }
            for row in windows
        ],
        "attention_backend": args.attention_backend,
        "accumulator_dtype": "float32",
        "matmul_precision": "highest",
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "capture-plan.json", plan | {"dry_run": False})

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model.resolve(),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.layers
    if len(layers) != 48:
        raise ValueError(f"expected 48 layers, found {len(layers)}")

    grams: dict[int, dict[str, torch.Tensor]] = {}
    rows: dict[int, dict[str, int]] = {}
    handles = []

    def hook(layer: int, kind: str):
        def capture(_module, values):
            hidden = values[0].detach().reshape(-1, values[0].shape[-1]).float()
            state = grams.setdefault(layer, {})
            if kind not in state:
                state[kind] = torch.zeros(
                    (hidden.shape[-1], hidden.shape[-1]),
                    dtype=torch.float32,
                    device=hidden.device,
                )
                rows.setdefault(layer, {})[kind] = 0
            state[kind].addmm_(hidden.T, hidden)
            rows[layer][kind] += int(hidden.shape[0])
        return capture

    for layer, block in enumerate(layers):
        handles.append(block.self_attn.q_proj.register_forward_pre_hook(hook(layer, "qkv")))
        handles.append(block.self_attn.o_proj.register_forward_pre_hook(hook(layer, "o")))

    started = time.monotonic()
    input_device = model.get_input_embeddings().weight.device
    try:
        for index, window in enumerate(windows):
            token_ids = torch.tensor(window["token_ids"], dtype=torch.long, device=input_device).unsqueeze(0)
            if token_ids.shape[1] != int(corpus["window_tokens"]):
                raise ValueError("calibration window length mismatch")
            with torch.inference_mode():
                result = model.model(input_ids=token_ids, use_cache=False, return_dict=True)
            del result, token_ids
            print(
                json.dumps(
                    {
                        "stage": "capture",
                        "shard_index": args.shard_index,
                        "window": index,
                        "windows": len(windows),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    records = []
    expected_rows = len(windows) * int(corpus["window_tokens"])
    for layer in range(48):
        if set(grams.get(layer, {})) != {"qkv", "o"}:
            raise RuntimeError(f"layer {layer} did not produce both attention captures")
        if rows[layer] != {"qkv": expected_rows, "o": expected_rows}:
            raise RuntimeError(f"layer {layer} row counts differ: {rows[layer]}")
        path = output / f"layer-{layer:03d}.safetensors"
        _atomic_save(
            path,
            {
                "qkv_gram": grams[layer]["qkv"],
                "o_gram": grams[layer]["o"],
            },
            {
                "schema": "quant-pipeline.qwen-attention-hessian-shard.v1",
                "layer": str(layer),
                "rows": str(expected_rows),
                "shard_index": str(args.shard_index),
                "shard_count": str(args.shard_count),
            },
        )
        records.append(
            {
                "layer": layer,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": expected_rows,
                "qkv_dimension": int(grams[layer]["qkv"].shape[0]),
                "o_dimension": int(grams[layer]["o"].shape[0]),
            }
        )
        del grams[layer]
    del model
    torch.cuda.empty_cache()
    manifest = {
        "schema": "quant-pipeline.qwen-attention-hessian-shard.v1",
        "plan_sha256": sha256_file(output / "capture-plan.json"),
        "source_revision": args.source_revision,
        "sealed_corpus_seal": corpus["seal_sha256"],
        "role": args.role,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "window_count": len(windows),
        "rows": expected_rows,
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest["manifest_sha256"] = _hash_json(manifest)
    write_json(output / "capture-manifest.json", manifest)
    print(json.dumps({"ok": True, **manifest}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
