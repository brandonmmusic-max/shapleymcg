#!/usr/bin/env python3
"""Assemble and verify the exact BF16 Qwen validation reconstruction.

The experiment allocates K3/K4 independently for gate, up, and down matrices.
Official BTX currently couples gate/up bit rates, so this publisher deliberately
emits a normal Transformers BF16 checkpoint.  It is the exact expanded model
used for KLD validation, not a compact or runtime-qualified 3.5-bpw checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import time

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
EXPERT_KEY = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
WEIGHT_SHARD = re.compile(r"^model-\d+-of-\d+\.safetensors$")


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_seal(document: dict, field: str, label: str) -> None:
    expected = document.get(field)
    body = {key: value for key, value in document.items() if key != field}
    if expected != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def _load_inputs(kld_root: Path) -> tuple[dict, dict, dict, dict[tuple[int, int, str], dict]]:
    allocation = json.loads((kld_root / "allocation.json").read_text())
    report = json.loads((kld_root / "kld-report.json").read_text())
    verification = json.loads((kld_root / "independent-verification.json").read_text())
    _verify_seal(allocation, "allocation_sha256", "allocation")
    _verify_seal(report, "report_sha256", "KLD report")
    _verify_seal(verification, "verification_sha256", "independent verification")
    choices = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): row
        for row in allocation["choices"]
    }
    if (
        len(choices) != 48 * 128 * 3
        or allocation.get("average_weight_bits") != 3.5
        or allocation.get("k3_count") != len(choices) // 2
        or allocation.get("k4_count") != len(choices) // 2
        or verification.get("allocation_sha256") != allocation["allocation_sha256"]
        or verification.get("kld_report_sha256") != report["report_sha256"]
    ):
        raise ValueError("KLD inputs are not the independently verified exact-half-K4 control")
    return allocation, report, verification, choices


def _candidate_key(choice: dict) -> str:
    return (
        f"K{int(choice['bits'])}.E{int(choice['expert']):03d}."
        f"{choice['projection']}.reconstruction_hf"
    )


def _copy_support_files(source: Path, output: Path) -> None:
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if path.name == "README.md" or path.name == "model.safetensors.index.json":
            continue
        if WEIGHT_SHARD.fullmatch(path.name):
            continue
        shutil.copy2(path, output / path.name)


def _model_card(report: dict, allocation: dict, verification: dict) -> str:
    summary = report["summary"]
    return f"""---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B-Base
tags:
- qwen3
- mixture-of-experts
- quantization-research
- shapleymcg
---

# Qwen3-30B-A3B ShapleyMCG K3/K4 validation reconstruction

This repository is the **exact expanded BF16 validation model** used for the
reported WikiText KLD experiment. It installs the selected K3/K4 reconstructed
expert matrices into the original Transformers checkpoint so the measured
student logits can be independently reproduced.

## Important format statement

The allocation has a logical average rate of **3.5 bits per MoE expert-weight
element** ({allocation['k3_count']:,} K3 and {allocation['k4_count']:,} K4
matrix choices). This repository stores those reconstructions in BF16 and is
therefore about the size of the BF16 base model. It is **not** a compact
3.5-bpw checkpoint and is **not runtime-qualified**. The current official BTX
format couples gate/up bit rates, while this experiment selects gate and up
independently; publishing it as official BTX would misrepresent the measured
model.

## Exact KLD result

Against sealed BF16 teacher logits on 2,047 next-token positions from the
immutable WikiText-2-raw-v1 test-prefix window:

| Metric | Value |
|---|---:|
| Mean KLD | {summary['mean']:.12g} |
| Median KLD | {summary['p50']:.12g} |
| P95 KLD | {summary['p95']:.12g} |
| Maximum KLD | {summary['max']:.12g} |

Allocation SHA256: `{allocation['allocation_sha256']}`  
KLD report seal: `{report['report_sha256']}`  
Independent verification seal: `{verification['verification_sha256']}`

The full candidates, calibration/Hessian artifacts, teacher and student logits,
token window, manifests, hashes, and receipts are published in
[`brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility).
The executable pipeline and methodology are in
[`brandonmmusic-max/shapleymcg`](https://github.com/brandonmmusic-max/shapleymcg).

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
```

## Attribution

- Base model: the Qwen team, [`Qwen/Qwen3-30B-A3B-Base`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base).
- EXL3/TRELLIS codec lineage: turboderp and ExLlamaV3 contributors.
- Shapley/value-attribution and Hessian-guided allocation references are
  enumerated with links in the GitHub methodology and sealed dataset card.
- Experiment, integration, and publication: Brandon Music / ShapleyMCG.
"""


def _write_sha256s(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"SHA256SUMS", "model-manifest.json"}:
            rows.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def _seal_manifest(output: Path, provenance: dict) -> dict:
    rows = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "model-manifest.json"
    ]
    body = {
        "schema": "quant-pipeline.qwen-validation-reconstruction-model.v1",
        "storage_dtype": "bfloat16",
        "logical_expert_weight_bits": 3.5,
        "runtime_qualified": False,
        "compact_quantized_checkpoint": False,
        "source_revision": MODEL_REVISION,
        "allocation_sha256": provenance["allocation_sha256"],
        "kld_report_sha256": provenance["kld_report_sha256"],
        "model_logit_verification_sha256": provenance["model_logit_verification_sha256"],
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    body["manifest_sha256"] = _hash_json(body)
    write_json(output / "model-manifest.json", body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--kld-root", type=Path, required=True)
    parser.add_argument("--kld-window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "schema": "quant-pipeline.qwen-validation-model-assembly-plan.v1",
        "source_model": str(args.source_model.resolve()),
        "source_revision": MODEL_REVISION,
        "encode_root": str(args.encode_root.resolve()),
        "kld_root": str(args.kld_root.resolve()),
        "kld_window": str(args.kld_window.resolve()),
        "output": str(args.output.resolve()),
        "attention_backend": args.attention_backend,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    from quant_pipeline.normalization.absolute_v31 import tensor_sha256

    started = time.monotonic()
    source = args.source_model.resolve()
    encode_root = args.encode_root.resolve()
    kld_root = args.kld_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing nonempty/reused model output: {output}")
    output.mkdir(parents=True)
    write_json(output / "assembly-plan.json", plan | {"dry_run": False})
    allocation, report, verification, choices = _load_inputs(kld_root)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = dict(index["weight_map"])
    replacement_keys = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight": choice
        for (layer, expert, projection), choice in choices.items()
    }
    if not set(replacement_keys).issubset(weight_map):
        missing = sorted(set(replacement_keys) - set(weight_map))[:5]
        raise ValueError(f"source checkpoint lacks selected expert tensors: {missing}")
    _copy_support_files(source, output)
    replacement_count = 0
    shard_records = []
    for shard_name in sorted(set(weight_map.values())):
        source_shard = source / shard_name
        output_shard = output / shard_name
        shard_keys = sorted(key for key, value in weight_map.items() if value == shard_name)
        layers = sorted({int(EXPERT_KEY.fullmatch(key)[1]) for key in shard_keys if EXPERT_KEY.fullmatch(key)})
        tensors = {}
        with ExitStack() as stack:
            candidate_handles = {
                layer: stack.enter_context(
                    safe_open(
                        encode_root / f"layer-{layer:03d}" / "k34-candidates.safetensors",
                        framework="pt",
                        device="cpu",
                    )
                )
                for layer in layers
            }
            source_handle = stack.enter_context(safe_open(source_shard, framework="pt", device="cpu"))
            if set(source_handle.keys()) != set(shard_keys):
                raise ValueError(f"source index/key mismatch in {shard_name}")
            metadata = source_handle.metadata()
            for key in shard_keys:
                choice = replacement_keys.get(key)
                if choice is None:
                    tensors[key] = source_handle.get_tensor(key)
                    continue
                match = EXPERT_KEY.fullmatch(key)
                assert match is not None
                tensor = candidate_handles[int(match[1])].get_tensor(_candidate_key(choice)).contiguous()
                if tensor.dtype != torch.bfloat16 or tensor_sha256(tensor) != choice["stored_bf16_reconstruction_sha256"]:
                    raise ValueError(f"selected reconstruction identity mismatch for {key}")
                tensors[key] = tensor
                replacement_count += 1
            temporary = output_shard.with_suffix(output_shard.suffix + ".partial")
            save_file(tensors, temporary, metadata=metadata)
            os.replace(temporary, output_shard)
        del tensors
        with safe_open(output_shard, framework="pt", device="cpu") as check:
            if set(check.keys()) != set(shard_keys):
                raise ValueError(f"output key mismatch in {shard_name}")
            for key in set(shard_keys) & set(replacement_keys):
                if tensor_sha256(check.get_tensor(key)) != replacement_keys[key]["stored_bf16_reconstruction_sha256"]:
                    raise ValueError(f"persisted reconstruction mismatch for {key}")
        shard_records.append({"path": shard_name, "bytes": output_shard.stat().st_size, "sha256": sha256_file(output_shard)})
        print(json.dumps({"stage": "shard", "shard": shard_name, "replacements": replacement_count}), flush=True)
    if replacement_count != len(choices):
        raise ValueError(f"installed {replacement_count} reconstructions, expected {len(choices)}")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )

    # Reload the persisted checkpoint and require exact equality with the logits
    # whose KLD was independently verified.  This makes the HF model itself part
    # of the result gate rather than merely a plausible reconstruction.
    model = AutoModelForCausalLM.from_pretrained(
        output,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    window = json.loads((args.kld_window.resolve() / "kld-window.json").read_text())
    ids = torch.tensor(
        [window["token_ids"]],
        dtype=torch.long,
        device=model.get_input_embeddings().weight.device,
    )
    with torch.inference_mode():
        observed = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
    observed = observed.float().cpu().reshape(-1, observed.shape[-1]).contiguous()
    with safe_open(kld_root / "student-logits.safetensors", framework="pt", device="cpu") as handle:
        expected = handle.get_tensor("logits").contiguous()
    exact = bool(torch.equal(observed, expected))
    max_abs_delta = float((observed - expected).abs().max().item())
    logit_verification = {
        "schema": "quant-pipeline.qwen-validation-model-logit-verification.v1",
        "allocation_sha256": allocation["allocation_sha256"],
        "token_sha256": str(window["token_sha256"]),
        "shape": list(observed.shape),
        "expected_raw_sha256": tensor_sha256(expected),
        "observed_raw_sha256": tensor_sha256(observed),
        "exact_tensor_equality": exact,
        "max_abs_delta": max_abs_delta,
    }
    logit_verification["verification_sha256"] = _hash_json(logit_verification)
    write_json(output / "model-logit-verification.json", logit_verification)
    del observed, expected, model
    torch.cuda.empty_cache()
    if not exact:
        raise ValueError(f"assembled model logits differ from measured student; max abs {max_abs_delta}")

    provenance = {
        "schema": "quant-pipeline.qwen-validation-model-provenance.v1",
        "source_model": "Qwen/Qwen3-30B-A3B-Base",
        "source_revision": MODEL_REVISION,
        "source_index_sha256": sha256_file(index_path),
        "allocation_sha256": allocation["allocation_sha256"],
        "allocation_file_sha256": sha256_file(kld_root / "allocation.json"),
        "kld_report_sha256": report["report_sha256"],
        "kld_report_file_sha256": sha256_file(kld_root / "kld-report.json"),
        "independent_verification_sha256": verification["verification_sha256"],
        "independent_verification_file_sha256": sha256_file(kld_root / "independent-verification.json"),
        "model_logit_verification_sha256": logit_verification["verification_sha256"],
        "replacement_count": replacement_count,
        "storage_dtype": "bfloat16",
        "logical_expert_weight_bits": 3.5,
        "runtime_qualified": False,
        "compact_quantized_checkpoint": False,
        "shards": shard_records,
        "elapsed_seconds": time.monotonic() - started,
    }
    provenance["provenance_sha256"] = _hash_json(provenance)
    write_json(output / "provenance.json", provenance)
    (output / "README.md").write_text(_model_card(report, allocation, verification))
    _write_sha256s(output)
    manifest = _seal_manifest(output, provenance)
    print(json.dumps({"ok": True, "manifest_sha256": manifest["manifest_sha256"], **provenance}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
