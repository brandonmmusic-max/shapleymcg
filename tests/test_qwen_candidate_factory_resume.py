from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "measure_qwen_candidate_factory_union",
    ROOT / "scripts" / "measure_qwen_candidate_factory_union.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resume_reconstructs_and_verifies_all_layer_swap_rows(tmp_path):
    baseline = np.zeros(64, dtype=np.float64)
    inventory = {"layers": []}
    lines = ["non-json diagnostic line"]
    for layer in range(48):
        token = np.full(64, (layer + 1) / 10_000, dtype=np.float64)
        delta = token - baseline
        token_path = tmp_path / "single-layer-token-kld" / f"layer-{layer:03d}.npy"
        delta_path = tmp_path / "single-layer-token-delta" / f"layer-{layer:03d}.npy"
        token_sha = MODULE._save_npy(token_path, token)
        delta_sha = MODULE._save_npy(delta_path, delta)
        candidate_sha = format((layer % 15) + 1, "x") * 64
        inventory["layers"].append({"layer": layer, "candidate_sha256": candidate_sha})
        row = {
            "stage": "single-layer-swap",
            "layer": layer,
            "mcg_mean_kld": float(token.mean()),
            "delta_mean_kld_vs_turbo": float(delta.mean()),
            "relative_delta_vs_turbo": 0.0,
            "paired_block_95_interval": MODULE._paired_block_interval(
                delta, seed=20260824 + layer
            ),
            "single_swap_improves_mean": False,
            "single_swap_interval_below_zero": False,
            "candidate_file_sha256": candidate_sha,
            "installed_layer_sha256": "f" * 64,
            "token_kld_sha256": token_sha,
            "token_delta_sha256": delta_sha,
            "elapsed_seconds": 1.0,
        }
        lines.append(json.dumps(row, sort_keys=True))
    log = tmp_path / "producer.log"
    log.write_text("\n".join(lines) + "\n")

    rows = MODULE._resume_layer_rows(
        log,
        tmp_path,
        baseline,
        inventory,
        seed=20260824,
    )
    assert [row["layer"] for row in rows] == list(range(48))
