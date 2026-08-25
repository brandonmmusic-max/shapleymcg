from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


ROOT = Path(__file__).parents[1]


def _run(script: str, env: dict[str, str]):
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / script)],
        cwd=ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=False,
    )


def test_progressive_plan_modes_do_not_create_run_roots_or_publish(tmp_path):
    experiment_root = tmp_path / "experiment"
    result = _run(
        "run_qwen_progressive_candidate_experiment.sh",
        {"ACTION": "plan", "RUN_ROOT": str(experiment_root)},
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["publication"] is False
    assert not experiment_root.exists()

    validation_root = tmp_path / "validation-run"
    result = _run(
        "run_qwen_progressive_candidate_validation.sh",
        {"ACTION": "plan", "RUN_ROOT": str(validation_root)},
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["publish"] == 0
    assert not validation_root.exists()


def test_progressive_execute_requires_or_launches_capture_explicitly(tmp_path):
    result = _run(
        "run_qwen_progressive_candidate_experiment.sh",
        {
            "ACTION": "execute",
            "CAPTURE_MODE": "require",
            "RUN_ROOT": str(tmp_path / "run"),
            "CODE_ROOT": str(ROOT),
        },
    )
    assert result.returncode == 1
    assert "required capture receipt is absent" in result.stderr


def test_local_candidate_inventory_needs_no_hub_publication(tmp_path):
    encode = tmp_path / "encode"
    for layer in range(48):
        root = encode / f"layer-{layer:03d}"
        root.mkdir(parents=True)
        candidate = root / "k34-candidates.safetensors"
        candidate.write_bytes(f"layer-{layer}".encode())
        receipt = {
            "layer": layer,
            "experts": list(range(128)),
            "candidate_tensor_file": candidate.name,
            "candidate_tensor_bytes": candidate.stat().st_size,
            "candidate_tensor_sha256": sha256_file(candidate),
            "scores": [],
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
        write_json(root / "encode-receipt.json", receipt)
    output = tmp_path / "inventory.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_qwen_candidate_inventory.py"),
            "--local-root",
            str(encode),
            "--output",
            str(output),
            "--workers",
            "4",
            "--execute",
        ],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    inventory = json.loads(output.read_text())
    assert inventory["source"]["kind"] == "local-sealed-fast-encode-tree"
    assert inventory["source"]["publication_side_effect"] is False
    assert len(inventory["layers"]) == 48
