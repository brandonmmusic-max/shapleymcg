#!/usr/bin/env python3
"""Seal a paired TurboDerp-v0.0.1 versus ShapleyMCG allocation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _load_sealed(path: Path, seal: str) -> dict:
    value = json.loads(path.read_text())
    expected = value.get(seal)
    body = {key: item for key, item in value.items() if key != seal}
    if not expected or expected != _hash_json(body):
        raise ValueError(f"seal mismatch: {path}")
    return value


def _arm(root: Path) -> tuple[dict, dict, np.ndarray]:
    summary = _load_sealed(root / "summary.json", "summary_sha256")
    report_path = root / "turboderp-selected-k34" / "kld-report.json"
    report = _load_sealed(report_path, "report_sha256")
    token_path = root / "turboderp-selected-k34" / "token-kld.npy"
    if report.get("token_kld_sha256") != sha256_file(token_path):
        raise ValueError(f"tokenwise KLD hash mismatch: {token_path}")
    values = np.load(token_path, allow_pickle=False).astype(np.float64, copy=False)
    if values.size != int(report["summary"]["count"]):
        raise ValueError("tokenwise KLD count disagrees with report")
    if float(values.mean()) != float(report["summary"]["mean"]):
        raise ValueError("tokenwise KLD mean disagrees with report")
    arm = summary["arms"].get("turboderp-selected-k34")
    if not arm or arm["report_sha256"] != report["report_sha256"]:
        raise ValueError("summary arm does not bind the KLD report")
    return summary, report, values


def compare(stock_root: Path, causal_root: Path, *, seed: int, bootstrap_draws: int) -> dict:
    stock_summary, stock_report, stock = _arm(stock_root)
    causal_summary, causal_report, causal = _arm(causal_root)
    invariants = (
        "panel_sha256",
        "expert_rate",
        "fixed_nonexpert_scope",
        "turboderp_k3_revision",
        "turboderp_k4_revision",
        "turboderp_k3_scope",
        "turboderp_k4_scope",
    )
    for field in invariants:
        if stock_summary.get(field) != causal_summary.get(field):
            raise ValueError(f"comparison invariant drifted: {field}")
    if stock_report.get("teacher_files") != causal_report.get("teacher_files"):
        raise ValueError("teacher-logit files differ between allocation arms")
    if stock.shape != causal.shape or stock.ndim != 2:
        raise ValueError("allocation arms do not share the same row/token shape")

    delta = stock - causal
    row_delta = delta.mean(axis=1)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, delta.shape[0], size=(bootstrap_draws, delta.shape[0]))
    bootstrap = row_delta[sampled].mean(axis=1)
    stock_mean = float(stock.mean())
    causal_mean = float(causal.mean())
    body = {
        "schema": "quant-pipeline.turboderp-v001-shapleymcg-allocation-proof.v1",
        "claim_scope": "allocator-only over identical published TurboDerp K3/K4 reconstructed candidates",
        "stock_allocator": {
            "name": "ExLlamaV3 v0.0.1 carried-surplus fused-projection rule",
            "allocation_sha256": stock_summary["allocation_sha256"],
            "mean_kld": stock_mean,
            "top1_agreement": stock_report["top1_agreement"],
            "summary_file_sha256": sha256_file(stock_root / "summary.json"),
            "report_file_sha256": sha256_file(stock_root / "turboderp-selected-k34" / "kld-report.json"),
            "token_kld_file_sha256": stock_report["token_kld_sha256"],
        },
        "shapleymcg_allocator": {
            "name": "full causal Aumann-Shapley/Fisher allocation",
            "allocation_sha256": causal_summary["allocation_sha256"],
            "mean_kld": causal_mean,
            "top1_agreement": causal_report["top1_agreement"],
            "summary_file_sha256": sha256_file(causal_root / "summary.json"),
            "report_file_sha256": sha256_file(causal_root / "turboderp-selected-k34" / "kld-report.json"),
            "token_kld_file_sha256": causal_report["token_kld_sha256"],
        },
        "fixed": {
            field: stock_summary[field] for field in invariants
        }
        | {
            "teacher_files": stock_report["teacher_files"],
            "metric": stock_report["metric"],
            "token_shape": list(stock.shape),
        },
        "effect": {
            "absolute_mean_kld_reduction": float(delta.mean()),
            "relative_mean_kld_reduction": float(delta.mean() / stock_mean),
            "top1_agreement_gain": float(
                causal_report["top1_agreement"] - stock_report["top1_agreement"]
            ),
            "rows_shapleymcg_better": int(np.count_nonzero(row_delta > 0)),
            "rows_total": int(delta.shape[0]),
            "per_row_stock_mean_kld": [float(value) for value in stock.mean(axis=1)],
            "per_row_shapleymcg_mean_kld": [float(value) for value in causal.mean(axis=1)],
            "per_row_absolute_reduction": [float(value) for value in row_delta],
            "row_block_bootstrap": {
                "seed": seed,
                "draws": bootstrap_draws,
                "confidence": 0.95,
                "absolute_reduction_interval": [
                    float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
                ],
            },
        },
        "interpretation": {
            "superiority_gate_passed": bool(
                causal_mean < stock_mean
                and np.quantile(bootstrap, 0.025) > 0
                and np.count_nonzero(row_delta > 0) == delta.shape[0]
            ),
            "native_mixed_checkpoint_claim": False,
            "note": (
                "This isolates allocation over a common reconstruction pool. It does not compare "
                "against a native v0.0.1 3.5-BPW conversion, which TurboDerp did not publish."
            ),
        },
    }
    body["comparison_sha256"] = _hash_json(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-root", type=Path, required=True)
    parser.add_argument("--shapleymcg-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    args = parser.parse_args()
    if args.bootstrap_draws < 10_000:
        raise SystemExit("bootstrap-draws must be at least 10,000")
    document = compare(
        args.stock_root.resolve(),
        args.shapleymcg_root.resolve(),
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    write_json(args.output.resolve(), document)
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
