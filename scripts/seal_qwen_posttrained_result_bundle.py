#!/usr/bin/env python3
"""Seal the post-trained Qwen/TurboDerp matched-comparison publication bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


SOURCE_REVISION = "4c446470ba0aec43e22ac1128f9ffd915f338ba3"
TURBO_REVISION = "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _read_sealed(path: Path, field: str, label: str) -> dict:
    value = json.loads(path.read_text())
    expected = value.get(field)
    if expected != _hash_json({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"{label} seal mismatch: {path}")
    return value


def _require_zero(path: Path, label: str) -> None:
    if not path.is_file() or path.read_text().strip() != "0":
        raise ValueError(f"{label} did not finish successfully: {path}")


def _link_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"bundle source is missing or symlinked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"bundle tree is missing or symlinked: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _link_file(path, destination / path.relative_to(source))


def _inventory(root: Path, exclude: frozenset[str] = frozenset()) -> list[dict]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    ]


def _upload_receipts(root: Path, kind: str) -> list[dict]:
    rows = []
    for layer in range(48):
        path = root / f"layer-{layer:03d}.json"
        value = _read_sealed(path, "receipt_sha256", f"{kind} upload layer {layer}")
        if value.get("layer") != layer or value.get("repo_type") != "dataset":
            raise ValueError(f"{kind} upload receipt identity mismatch: layer {layer}")
        rows.append(
            {
                "layer": layer,
                "path_in_repo": value["path_in_repo"],
                "revision": value["revision"],
                "manifest_sha256": value["manifest_sha256"],
                "receipt_sha256": value["receipt_sha256"],
                "total_bytes": value["total_bytes"],
            }
        )
    return rows


def _card(summary: dict, naive: dict, exact_3p5: dict, git_revision: str) -> str:
    arms = summary["arms"]
    order = (
        ("ours-selected-k34", "ShapleyMCG selected K3/K4 experts; dense BF16"),
        ("ours-expert-k4", "ShapleyMCG K4 experts; dense BF16"),
        ("turboderp-full-k4", "TurboDerp full K4 body / K6 head"),
        ("hybrid-ours-experts", "TurboDerp dense K4/K6 + ShapleyMCG K4 experts"),
    )
    table = "\n".join(
        f"| {label} | {arms[key]['mean_kld']:.12g} | {arms[key]['top1_agreement']:.12g} |"
        for key, label in order
    )
    return f"""---
license: apache-2.0
pretty_name: Qwen3-30B-A3B post-trained ShapleyMCG matched K4 evaluation
tags:
- quantization
- qwen3
- mixture-of-experts
- shapleymcg
---

# Qwen3-30B-A3B post-trained matched comparison

This sealed result reruns the complete ShapleyMCG calibration, full-p2 fit,
K3/K4 encoding, and end-to-end evaluation on the immutable post-trained parent
`Qwen/Qwen3-30B-A3B@{SOURCE_REVISION}`. The historical TurboDerp comparison is
`turboderp/Qwen3-30B-A3B-exl3@{TURBO_REVISION}`. The source relation is recorded
as strong inferred lineage evidence, not as an upstream-authored source pin.

All arms below use the same 10 x 2,048 WikiText-2 panel, the same post-trained
BF16 teacher logits, the same Transformers BF16 replay, and no KV cache.

The original 32-window fit split left five expert/layer pairs unrouted. A
router-only coverage pass over unused windows from the same fit documents
selected two additional windows covering `(9,120)`, `(20,112)`, `(32,112)`,
`(35,44)`, and `(36,13)`. No selection, confirmation, teacher, student, or KLD
signal entered that choice. The final fit corpus therefore contains 34 windows.

| Arm | Mean tokenwise KLD | Top-1 agreement |
|---|---:|---:|
{table}

The strict component-attribution arm fixes TurboDerp dense K4/K6 components,
router, evaluator, and panel, replacing only the expert reconstructions. Its
relative KLD change versus the locally reconstructed TurboDerp arm is
`{summary['matched_hybrid_kld_reduction_vs_turboderp']:.12g}` and its top-1
change is `{summary['matched_hybrid_top1_gain_vs_turboderp']:.12g}`.

Five direct score-blind allocations at exactly 3.5 logical expert BPW measured
mean KLD `{naive['naive_mean_kld']:.12g}` (sample SD
`{naive['naive_sample_std_kld']:.12g}`, range `{naive['naive_min_kld']:.12g}`
to `{naive['naive_max_kld']:.12g}`). The selected allocation measured
`{naive['selected_mean_kld']:.12g}`; `{naive['naive_seeds_beating_selected_kld']}`
of five score-blind seeds beat it.

The exact matched 3.5-BPW comparison fixes TurboDerp K4 non-expert weights,
K6 head, parent, panel, and the same half-K3/half-K4 matrix allocation. The
TurboDerp reconstruction measured KLD
`{exact_3p5['arms']['turboderp-selected-k34']['mean_kld']:.12g}`; replacing
only those expert reconstructions with ShapleyMCG measured
`{exact_3p5['arms']['hybrid-ours-selected-k34']['mean_kld']:.12g}`. The
ShapleyMCG relative KLD reduction at identical expert rate is
`{exact_3p5['ours_kld_reduction_vs_turboderp_at_exact_3p5']:.12g}`.

The bundle includes source/corpus seals, BF16 teacher and student logits,
per-token KLD arrays, all arm reports, five naive controls, capture manifests,
and the 48 fit plus 48 candidate Hub receipts. The large fit and candidate
artifacts live at their receipt paths in this same dataset repository.

Pipeline Git revision: `{git_revision}`. The executable methodology is in
[`brandonmmusic-max/shapleymcg`](https://github.com/brandonmmusic-max/shapleymcg).
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    plan = {
        "schema": "quant-pipeline.qwen-posttrained-result-bundle-plan.v1",
        "run_root": str(run_root),
        "artifact_root": str(artifact_root),
        "code_root": str(args.code_root.resolve()),
        "output": str(output),
        "git_revision": args.git_revision,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if output.exists():
        raise FileExistsError("refusing to overwrite a post-trained result bundle")
    if len(args.git_revision) != 40:
        raise ValueError("pipeline Git revision must be immutable 40-hex")
    for name, label in (
        ("calibration-parallel.exit", "calibration"),
        ("route-complete-capture.exit", "route-complete fit recapture"),
        ("streaming-fit-waves.exit", "fit"),
        ("encode-publish-waves.exit", "encode and fit publication"),
        ("matched-k4-evaluation.exit", "matched K4 evaluation"),
        ("naive-controls.exit", "score-blind controls"),
        ("exact-3p5-comparison.exit", "exact 3.5 BPW comparison"),
        ("hf-candidates.exit", "candidate publication"),
    ):
        _require_zero(run_root / "logs" / name, label)

    matched_root = artifact_root / "matched-k4-comparison"
    exact_root = artifact_root / "matched-3p5-comparison"
    naive_root = artifact_root / "naive-3p5-controls-v1"
    summary = _read_sealed(matched_root / "summary.json", "summary_sha256", "matched result")
    naive = _read_sealed(naive_root / "summary.json", "summary_sha256", "naive controls")
    exact_3p5 = _read_sealed(
        exact_root / "summary.json", "summary_sha256", "exact 3.5 BPW comparison"
    )
    if summary.get("source_revision") != SOURCE_REVISION or naive.get("selected_mean_kld") != summary["arms"]["ours-selected-k34"]["mean_kld"]:
        raise ValueError("post-trained result components are not bound to one selected arm")
    fit_refs = _upload_receipts(run_root / "artifacts/hf-upload/fits", "fit")
    candidate_refs = _upload_receipts(run_root / "artifacts/hf-upload/candidates", "candidate")

    output.mkdir(parents=True)
    _link_tree(run_root / "artifacts/source", output / "provenance/source")
    _link_file(run_root / "experiment.resolved.toml", output / "provenance/experiment.resolved.toml")
    _link_file(run_root / "inputs/reap_recall_calib.jsonl", output / "calibration/reap_recall_calib.jsonl")
    _link_file(run_root / "inputs/reap_recall_calib.role-safe-packed.jsonl", output / "calibration/reap_recall_calib.role-safe-packed.jsonl")
    _link_file(run_root / "artifacts/reap-recall-packing-receipt.json", output / "calibration/reap-recall-packing-receipt.json")
    _link_file(run_root / "artifacts/qwen-sealed-corpus.json", output / "calibration/qwen-sealed-corpus-initial.json")
    _link_file(
        run_root / "artifacts/qwen-sealed-corpus-route-complete.json",
        output / "calibration/qwen-sealed-corpus-route-complete.json",
    )
    for source, relative in (
        (run_root / "calibration-capture/calibration-capture-fit-conditional_down-receipt.json", "initial-fit-confirmation-receipt.json"),
        (run_root / "calibration-capture-route-complete/calibration-capture-fit-receipt.json", "fit-route-complete-receipt.json"),
        (run_root / "calibration-capture-base/calibration-capture-heldout-receipt.json", "selection-receipt.json"),
        (run_root / "calibration-capture/fit/capture-manifest.json", "initial-fit-manifest.json"),
        (run_root / "calibration-capture-route-complete/fit/capture-manifest.json", "fit-route-complete-manifest.json"),
        (run_root / "calibration-capture/conditional_down/capture-manifest.json", "confirmation-manifest.json"),
        (run_root / "calibration-capture-base/heldout/capture-manifest.json", "selection-manifest.json"),
    ):
        _link_file(source, output / "calibration/captures" / relative)
    _link_tree(artifact_root / "turboderp-wiki2-teacher", output / "evaluation/teacher-panel")
    _link_tree(matched_root, output / "evaluation/matched-k4-comparison")
    _link_tree(exact_root, output / "evaluation/matched-3p5-comparison")
    _link_tree(naive_root, output / "evaluation/naive-3p5-controls")
    _link_tree(run_root / "artifacts/hf-upload/fits", output / "publication/fit-receipts")
    _link_tree(run_root / "artifacts/hf-upload/candidates", output / "publication/candidate-receipts")
    (output / "README.md").write_text(
        _card(summary, naive, exact_3p5, args.git_revision)
    )

    inventory = _inventory(output, frozenset({"bundle-manifest.json", "SHA256SUMS"}))
    (output / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory)
    )
    manifest = {
        "schema": "quant-pipeline.qwen-posttrained-result-bundle.v1",
        "pipeline_git_revision": args.git_revision,
        "source_revision": SOURCE_REVISION,
        "turboderp_revision": TURBO_REVISION,
        "matched_summary_sha256": summary["summary_sha256"],
        "naive_summary_sha256": naive["summary_sha256"],
        "exact_3p5_summary_sha256": exact_3p5["summary_sha256"],
        "fit_layers": fit_refs,
        "candidate_layers": candidate_refs,
        "files": _inventory(output),
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
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
