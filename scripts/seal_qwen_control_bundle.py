#!/usr/bin/env python3
"""Seal the complete Qwen fixed-Hadamard K3/K4 control publication bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


REVISION = re.compile(r"^[0-9a-f]{40}$")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    expected = document.get(field)
    body = {key: value for key, value in document.items() if key != field}
    if expected != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def _require_zero(path: Path, label: str) -> None:
    if not path.is_file() or path.read_text().strip() != "0":
        raise ValueError(f"{label} did not finish with exit zero")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"publication source is missing or symlinked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _capture(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset({"bundle-manifest.json"})) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    ]


def _upload_receipts(root: Path, kind: str, layers: int) -> list[dict[str, Any]]:
    result = []
    for layer in range(layers):
        path = root / f"layer-{layer:03d}.json"
        receipt = _read_json(path)
        _verify_seal(receipt, "receipt_sha256", f"{kind} upload layer {layer}")
        if receipt.get("layer") != layer or receipt.get("repo_type") != "dataset":
            raise ValueError(f"{kind} upload receipt layer identity mismatch: {layer}")
        result.append(
            {
                "layer": layer,
                "path_in_repo": receipt["path_in_repo"],
                "revision": receipt["revision"],
                "manifest_sha256": receipt["manifest_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "total_bytes": receipt["total_bytes"],
            }
        )
    return result


def _resolved_lock(code_root: Path, run_root: Path, artifact_root: Path) -> dict[str, Any]:
    template = _read_json(code_root / "configs/qwen3-30b-a3b-b200/artifact-lock.json")
    source = _read_json(run_root / "artifacts/qwen-source-receipt.json")
    corrected_manifest_path = (
        code_root / "configs/qwen3-30b-a3b-b200/corrected-exl3-source-manifest.json"
    )
    corrected = _read_json(corrected_manifest_path)
    packed_input = run_root / "inputs/reap_recall_calib.role-safe-packed.jsonl"
    sealed_corpus = run_root / "artifacts/qwen-sealed-corpus.json"
    kld_window = artifact_root / "kld-window/kld-window.json"
    extension = run_root / "encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so"
    template["model"].update(
        {
            "config_sha256": source["config_sha256"],
            "index_sha256": source["index_sha256"],
            "shard_manifest_sha256": source["shard_manifest_sha256"],
        }
    )
    template["corpus"].update(
        {
            "calibration_jsonl_sha256": sha256_file(packed_input),
            "sealed_corpus_sha256": sha256_file(sealed_corpus),
            "kld_window_sha256": sha256_file(kld_window),
        }
    )
    template["exl3"].update(
        {
            "source_closure_sha256": _hash_json(corrected["files"]),
            "sm100_extension_sha256": sha256_file(extension),
        }
    )
    template["required_placeholders"] = []
    template["resolution"] = {
        "schema": "quant-pipeline.qwen-artifact-lock-resolution.v1",
        "model_source_receipt_sha256": sha256_file(
            run_root / "artifacts/qwen-source-receipt.json"
        ),
        "original_calibration_jsonl_sha256": sha256_file(
            run_root / "inputs/reap_recall_calib.jsonl"
        ),
        "packed_calibration_jsonl_sha256": sha256_file(packed_input),
        "sealed_corpus_file_sha256": sha256_file(sealed_corpus),
        "kld_window_file_sha256": sha256_file(kld_window),
        "corrected_source_manifest_file_sha256": sha256_file(corrected_manifest_path),
        "corrected_source_closure_method": "sha256(canonical_json(path-to-sha256 map))",
        "extension_file_sha256": sha256_file(extension),
    }
    template["lock_sha256"] = _hash_json(template)
    return template


def _result_readme(report: dict[str, Any], allocation: dict[str, Any], git_revision: str) -> str:
    summary = report["summary"]
    return f"""---
license: other
pretty_name: ShapleyMCG Qwen3-30B-A3B reproducibility artifacts
---

# Qwen3-30B-A3B fixed-Hadamard K3/K4 control

This dataset preserves the calibration statistics, exact corrected-R10 K3/K4
candidate artifacts, source and corpus identities, BF16 teacher logits,
selected reconstructed-BF16 student logits, and tokenwise KLD for the first
full Qwen control run of the ShapleyMCG pipeline.

- Source model revision: `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`
- Pipeline Git revision: `{git_revision}`
- MoE expert-weight rate: `{allocation['average_weight_bits']:.6f}` bits
- Selected units: `{allocation['k3_count']}` K3 and `{allocation['k4_count']}` K4
- Mean next-token KLD: `{summary['mean']:.12g}`
- Token positions: `{summary['count']}`
- Allocation SHA256: `{allocation['allocation_sha256']}`
- KLD report SHA256: `{report['report_sha256']}`

## Same-parent expert-rate controls

The primary allocation-quality comparison keeps the Base parent, ten-by-2,048
WikiText token panel, BF16 teacher logits, and source-BF16 attention, routers,
and head fixed:

| Expert allocation | Expert logical BPW | Mean KLD | Top-1 agreement |
|---|---:|---:|---:|
| Uniform K3 | 3.0 | 0.09943217778983483 | 0.8728515625 |
| **ShapleyMCG mixed K3/K4** | **3.5** | **0.05005581795647327** | **0.908447265625** |
| Uniform K4 | 4.0 | 0.033991548914098856 | 0.922509765625 |

The selected mix is 24.9671% below the linear K3/K4 KLD midpoint and 13.8995%
below the geometric midpoint. A separate PyTorch `kl_div` implementation
recomputed the mixed mean as 0.05005581997721873 and matched top-1 exactly.
The sealed panels, logits, reports, and verification are under
`results/qwen3-30b-a3b-v1` in this dataset.

The exact expanded BF16 validation model is published at
[`brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction`](https://huggingface.co/brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction).
Its verified publication receipt is included in this control bundle.

The KLD gate replays exact codec-produced BF16 reconstructions in Transformers;
it is not a packed-runtime, CUDA-graph, or throughput qualification. The
historical GLM-style control is the sealed 2,048-token Qwen-tokenized prefix of
the pinned WikiText-2 raw test split. Its source text, token IDs, dataset
revision, tokenizer revision, hashes, and BF16 logits are included.

The capture-time manifests recorded `start_token: 0` for calibration chunks.
The exact token IDs and token hashes were nevertheless sealed, and sample
identity includes `document_id@token_sha256`, so this metadata defect did not
change captured tensors, fitting, or duplicate separation. The published code
records the true sealed offset for future captures.

See `bundle-manifest.json`, `resolved-artifact-lock.json`, and `SHA256SUMS` for
the complete identity closure. Large fit and candidate layers live at
`fits/layer-NNN` and `candidates/layer-NNN`; their verified revisions and
manifest hashes are enumerated by this bundle.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/qwen-shapleymcg-run"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/artifacts/shapleymcg/qwen3-30b-a3b-v1"),
    )
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--model-publication-receipt", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not REVISION.fullmatch(args.git_revision):
        parser.error("--git-revision must be an immutable 40-hex commit")
    run_root = args.run_root.resolve()
    artifact_root = args.artifact_root.resolve()
    code_root = args.code_root.resolve()
    output = args.output.resolve()
    plan = {
        "schema": "quant-pipeline.qwen-control-bundle-plan.v1",
        "run_root": str(run_root),
        "artifact_root": str(artifact_root),
        "code_root": str(code_root),
        "output": str(output),
        "git_revision": args.git_revision,
        "model_publication_receipt": str(args.model_publication_receipt.resolve()),
        "layers": args.layers,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if output.exists():
        raise FileExistsError("publication output already exists; refusing an in-place reseal")
    _require_zero(run_root / "logs/fast-encode-waves.exit", "encode scheduler")
    _require_zero(run_root / "logs/fast-kld.exit", "exact KLD")
    for layer in range(args.layers):
        _require_zero(
            run_root / "logs" / f"fast-encode-layer-{layer:03d}.exit",
            f"encode layer {layer}",
        )

    kld_root = artifact_root / "fast-k34-kld"
    report = _read_json(kld_root / "kld-report.json")
    allocation = _read_json(kld_root / "allocation.json")
    independent = _read_json(kld_root / "independent-verification.json")
    model_publication = _read_json(args.model_publication_receipt.resolve())
    _verify_seal(report, "report_sha256", "KLD report")
    _verify_seal(allocation, "allocation_sha256", "allocation")
    _verify_seal(independent, "verification_sha256", "independent KLD verification")
    _verify_seal(model_publication, "receipt_sha256", "model publication receipt")
    if (
        independent.get("allocation_sha256") != allocation["allocation_sha256"]
        or independent.get("kld_report_sha256") != report["report_sha256"]
        or independent.get("selected_reconstruction_count") != 48 * 128 * 3
    ):
        raise ValueError("independent verification is not bound to the complete selected control")
    if (
        model_publication.get("repo_type") != "model"
        or model_publication.get("repo_id")
        != "brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction"
    ):
        raise ValueError("model publication receipt identifies the wrong repository")
    fit_refs = _upload_receipts(run_root / "artifacts/hf-upload/fits", "fit", args.layers)
    candidate_refs = _upload_receipts(
        run_root / "artifacts/hf-upload/candidates", "candidate", args.layers
    )

    mappings = {
        run_root / "inputs/reap_recall_calib.jsonl": Path("calibration/reap_recall_calib.jsonl"),
        run_root / "inputs/reap_recall_calib.role-safe-packed.jsonl": Path(
            "calibration/reap_recall_calib.role-safe-packed.jsonl"
        ),
        run_root / "artifacts/qwen-source-receipt.json": Path(
            "identities/qwen-source-receipt.json"
        ),
        run_root / "artifacts/reap-recall-packing-receipt.json": Path(
            "calibration/reap-recall-packing-receipt.json"
        ),
        run_root / "artifacts/qwen-sealed-corpus.json": Path(
            "calibration/qwen-sealed-corpus.json"
        ),
        artifact_root / "kld-window/kld-window.json": Path("kld/kld-window.json"),
        artifact_root / "kld-window/source-prefix.txt": Path("kld/source-prefix.txt"),
        artifact_root / "teacher-kld/capture-receipt.json": Path(
            "kld/teacher-capture-receipt.json"
        ),
        artifact_root / "teacher-kld/window-0000.safetensors": Path(
            "kld/teacher-logits.safetensors"
        ),
        run_root / "encoding-site/exllamav3_ext.cpython-311-x86_64-linux-gnu.so": Path(
            "software/exllamav3_ext-sm100-py311.so"
        ),
        code_root / "configs/qwen3-30b-a3b-b200/corrected-exl3-source-manifest.json": Path(
            "software/corrected-exl3-source-manifest.json"
        ),
        code_root / "environments/b200-cu132.lock.json": Path("software/b200-cu132.lock.json"),
        code_root / "environments/requirements-b200-cu132.txt": Path(
            "software/requirements-b200-cu132.txt"
        ),
        args.model_publication_receipt.resolve(): Path(
            "publication/model-publication-receipt.json"
        ),
    }
    for source in sorted(kld_root.iterdir()):
        if source.is_file():
            mappings[source] = Path("kld/result") / source.name
    capture_files = (
        (run_root / "calibration-capture/calibration-capture-fit-receipt.json", "fit-receipt.json"),
        (
            run_root / "calibration-capture/calibration-capture-conditional_down-receipt.json",
            "confirmation-receipt.json",
        ),
        (
            run_root / "calibration-capture-base/calibration-capture-heldout-receipt.json",
            "selection-receipt.json",
        ),
        (run_root / "calibration-capture/fit/capture-manifest.json", "fit-manifest.json"),
        (
            run_root / "calibration-capture/conditional_down/capture-manifest.json",
            "confirmation-manifest.json",
        ),
        (
            run_root / "calibration-capture-base/heldout/capture-manifest.json",
            "selection-manifest.json",
        ),
    )
    for source, name in capture_files:
        mappings[source] = Path("calibration/captures") / name
    for source, relative in mappings.items():
        _copy(source, output / relative)

    write_json(output / "resolved-artifact-lock.json", _resolved_lock(code_root, run_root, artifact_root))
    write_json(
        output / "software/actual-environment.json",
        {
            "schema": "quant-pipeline.qwen-b200-environment.v1",
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": args.git_revision,
            "pip_freeze": _capture([sys.executable, "-m", "pip", "freeze"]),
            "nvidia_smi": _capture(["nvidia-smi", "-q"]),
        },
    )
    (output / "README.md").write_text(_result_readme(report, allocation, args.git_revision))
    inventory = _inventory(output, exclude=frozenset({"bundle-manifest.json", "SHA256SUMS"}))
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory)
    (output / "SHA256SUMS").write_text(sums)
    manifest = {
        "schema": "quant-pipeline.qwen-control-publication-bundle.v1",
        "pipeline_git_revision": args.git_revision,
        "model_revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "files": _inventory(output),
        "fit_layers": fit_refs,
        "candidate_layers": candidate_refs,
        "allocation_sha256": allocation["allocation_sha256"],
        "kld_report_sha256": report["report_sha256"],
        "kld_summary": report["summary"],
        "validation_model": {
            "repo_id": model_publication["repo_id"],
            "verified_revision": model_publication["verified_revision"],
            "manifest_sha256": model_publication["manifest_sha256"],
            "receipt_sha256": model_publication["receipt_sha256"],
        },
    }
    manifest["manifest_sha256"] = _hash_json(manifest)
    write_json(output / "bundle-manifest.json", manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "files": len(manifest["files"]),
                "bytes": sum(row["bytes"] for row in manifest["files"]),
                "manifest_sha256": manifest["manifest_sha256"],
                "kld_summary": report["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
