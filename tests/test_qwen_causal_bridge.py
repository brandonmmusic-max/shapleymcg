from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shapley_fisher_bridge_preserves_exact_rate_and_payload_budget():
    module = _script("allocate_qwen_mcg_causal_exact_3p5.py")
    scores = []
    identities = []
    for layer in range(48):
        for expert in range(128):
            for projection_index, projection in enumerate(module.PROJECTIONS):
                identity = (layer, expert, projection)
                identities.append(identity)
                importance = 1.0 + ((layer * 128 + expert + projection_index) % 97) / 97.0
                for bits, damage, stored in ((3, 2.0 * importance, 300), (4, importance, 400)):
                    scores.append({
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "bits": bits,
                        "diagonal_p2_damage": damage,
                        "stored_bytes": stored,
                        "packed_sha256": f"{bits}" * 64,
                        "codec_fp16_reconstruction_sha256": f"{bits + 2}" * 64,
                        "stored_bf16_reconstruction_sha256": f"{bits + 4}" * 64,
                    })
    control_k4 = set(sorted(identities)[: len(identities) // 2])
    control_choices = [
        {"layer": layer, "expert": expert, "projection": projection, "bits": 4 if identity in control_k4 else 3}
        for identity in identities
        for layer, expert, projection in (identity,)
    ]
    control = {
        "allocation_sha256": "a" * 64,
        "stored_payload_bytes": sum(400 if identity in control_k4 else 300 for identity in identities),
        "choices": control_choices,
    }
    attribution = {
        "attribution_sha256": "b" * 64,
        "layers": [
            {
                "layer_index": layer,
                "expert_direct_reconciled": [float(1 + ((layer + expert) % 13)) for expert in range(128)],
            }
            for layer in range(48)
        ],
    }
    result = module._allocate(
        inventory={"inventory_sha256": "c" * 64},
        attribution=attribution,
        bindings=[],
        scores=scores,
        control=control,
    )
    assert result["average_weight_bits"] == 3.5
    assert result["k3_count"] == result["k4_count"] == 9216
    assert result["stored_payload_bytes"] == control["stored_payload_bytes"]
    assert result["changed_matrix_count_vs_control"] > 0
    assert result["objective"].startswith("native-aumann-shapley-fisher")


def test_causal_measurement_kld_is_zero_for_identical_logits():
    module = _script("measure_qwen_mcg_causal_allocation.py")
    rng = np.random.default_rng(17)
    logits = rng.standard_normal((9, 23), dtype=np.float32)
    values = module._token_kld(logits, logits.copy(), chunk=3)
    assert np.max(np.abs(values)) < 1e-14


def test_panel_allocation_plan_binds_teacher_receipt_and_defaults_to_sdpa(tmp_path):
    allocation = tmp_path / "allocation.json"
    inventory = tmp_path / "inventory.json"
    teacher_receipt = tmp_path / "teacher-receipt.json"
    for path in (allocation, inventory, teacher_receipt):
        path.write_text("{}\n")
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/measure_qwen_mcg_panel_allocation.py"),
            "--source-model",
            str(tmp_path / "model"),
            "--allocation",
            str(allocation),
            "--candidate-inventory",
            str(inventory),
            "--candidate-cache",
            str(tmp_path / "cache"),
            "--panel-root",
            str(tmp_path / "panel"),
            "--teacher-root",
            str(tmp_path / "teacher"),
            "--teacher-receipt",
            str(teacher_receipt),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["attention_backend"] == "sdpa"
    assert plan["teacher_receipt_file_sha256"]
    assert not output.exists()


def test_causal_control_comparison_requires_exact_3p5_choices():
    module = _script("compare_qwen_mcg_causal_control.py")
    rows = []
    for layer in range(48):
        for expert in range(128):
            for projection_index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                ordinal = (layer * 128 + expert) * 3 + projection_index
                rows.append({
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "bits": 3 if ordinal < 9216 else 4,
                })
    choices = module._choice_map({"choices": rows})
    assert len(choices) == 18432
    assert sum(bit == 3 for bit in choices.values()) == 9216
