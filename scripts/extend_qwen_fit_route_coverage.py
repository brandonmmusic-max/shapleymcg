#!/usr/bin/env python3
"""Extend only the fit role until every routed expert has natural support.

Selection is router-only and restricted to unused windows from the already
assigned fit documents.  It never reads quantization error, teacher logits, or
evaluation roles, so document isolation and the final KLD gate remain intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pipeline.calibration.windows import (
    build_windows,
    read_documents,
    verify_sealed_corpus,
)
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _window_identity(row: dict) -> tuple[str, int, str]:
    return str(row["document_id"]), int(row["offset"]), str(row["token_sha256"])


def _missing_experts(capture_root: Path) -> tuple[dict, dict[int, set[int]]]:
    from safetensors import safe_open

    manifest = json.loads((capture_root / "capture-manifest.json").read_text())
    experts = set(range(int(manifest["geometry"]["experts"])))
    missing: dict[int, set[int]] = {}
    for layer in manifest["layers"]:
        observed: set[int] = set()
        for record in manifest["records"][str(layer)]:
            with safe_open(capture_root / record["file"], framework="numpy") as handle:
                observed.update(map(int, handle.get_tensor("expert_ids").reshape(-1)))
        absent = experts - observed
        if absent:
            missing[int(layer)] = absent
    return manifest, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--sealed-corpus", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    plan = {
        "schema": "quant-pipeline.qwen-fit-route-coverage-plan.v1",
        "model": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "sealed_corpus": str(args.sealed_corpus.resolve()),
        "capture_root": str(args.capture_root.resolve()),
        "output": str(args.output.resolve()),
        "max_candidates": args.max_candidates,
        "device_map": args.device_map,
        "attention_backend": args.attention_backend,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a route-coverage corpus")
    sealed = json.loads(args.sealed_corpus.read_text())
    verify_sealed_corpus(sealed)
    if sha256_file(sealed["source"]["path"]) != sealed["source"]["sha256"]:
        raise ValueError("sealed corpus source JSONL drifted")
    capture_manifest, missing = _missing_experts(args.capture_root.resolve())
    if not missing:
        raise ValueError("fit capture already covers every expert")

    fit_rows = list(sealed["windows"]["fit"])
    fit_document_ids = {str(row["document_id"]) for row in fit_rows}
    documents = [
        row
        for row in read_documents(sealed["source"]["path"])
        if row["id"] in fit_document_ids
    ]
    if {row["id"] for row in documents} != fit_document_ids:
        raise ValueError("fit documents are missing from the sealed source")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quant_pipeline.calibration.qwen_capture import qwen_moe_layers

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.resolve(), local_files_only=True, use_fast=True
    )
    candidates = build_windows(
        documents,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        int(sealed["window_tokens"]),
        len(fit_rows) + args.max_candidates,
        int(sealed["seed"]),
    )
    prefix = [_window_identity(row) for row in candidates[: len(fit_rows)]]
    if prefix != [_window_identity(row) for row in fit_rows]:
        raise ValueError("reconstructed fit-window order differs from the sealed prefix")

    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(),
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = qwen_moe_layers(model)
    input_device = model.get_input_embeddings().weight.device
    observed: dict[int, set[int]] = {}
    handles = []
    for layer in sorted(missing):
        def hook(module, inputs, output, *, layer=layer):
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise ValueError("Qwen router hook expected logits, weights, and indices")
            observed[layer] = set(map(int, output[2].detach().reshape(-1).cpu().tolist()))

        handles.append(blocks[layer].gate.register_forward_hook(hook))

    uncovered = {(layer, expert) for layer, experts in missing.items() for expert in experts}
    selected = []
    try:
        for index, candidate in enumerate(candidates[len(fit_rows) :], len(fit_rows)):
            observed.clear()
            input_ids = torch.tensor(
                [candidate["token_ids"]], dtype=torch.long, device=input_device
            )
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False, return_dict=True)
            covered = sorted(
                [layer, expert]
                for layer, expert in uncovered
                if expert in observed.get(layer, set())
            )
            print(
                json.dumps(
                    {
                        "stage": "route-scan",
                        "candidate_index": index,
                        "document_id": candidate["document_id"],
                        "offset": candidate["offset"],
                        "covered": covered,
                        "remaining": len(uncovered) - len(covered),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if covered:
                selected.append(candidate)
                uncovered -= {tuple(row) for row in covered}
                if not uncovered:
                    break
    finally:
        for handle in handles:
            handle.remove()
    if uncovered:
        raise ValueError(f"candidate limit did not cover routed experts: {sorted(uncovered)}")

    extended = {key: value for key, value in sealed.items() if key != "seal_sha256"}
    extended["windows"] = {role: list(rows) for role, rows in sealed["windows"].items()}
    extended["windows"]["fit"] = fit_rows + selected
    extended["role_counts"] = dict(sealed["role_counts"])
    extended["role_counts"]["fit"] = len(extended["windows"]["fit"])
    extended["route_coverage_extension"] = {
        "schema": "quant-pipeline.qwen-fit-route-coverage.v1",
        "parent_seal_sha256": sealed["seal_sha256"],
        "capture_sha256": capture_manifest["capture_sha256"],
        "model_revision": args.model_revision,
        "selection_scope": "unused-windows-from-existing-fit-documents-only",
        "selection_signal": "router-topk-coverage-only",
        "missing_before": {
            str(layer): sorted(experts) for layer, experts in sorted(missing.items())
        },
        "selected_windows": [
            {
                "document_id": row["document_id"],
                "domain": row["domain"],
                "offset": row["offset"],
                "token_sha256": row["token_sha256"],
            }
            for row in selected
        ],
    }
    extended["seal_sha256"] = sha256_bytes(canonical_json(extended))
    verify_sealed_corpus(extended)
    write_json(args.output, extended)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "parent_seal_sha256": sealed["seal_sha256"],
                "seal_sha256": extended["seal_sha256"],
                "selected_windows": len(selected),
                "fit_windows": len(extended["windows"]["fit"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
