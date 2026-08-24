import json
import os
import subprocess
import sys
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]


def run(name: str, *args: str):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
    )


def test_turboderp_lineage_plan_pins_historical_source(tmp_path):
    output = tmp_path / "lineage.json"
    result = run("seal_turboderp_qwen_source_lineage.py", "--output", str(output))
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["source_revision"] == "4c446470ba0aec43e22ac1128f9ffd915f338ba3"
    assert plan["reference_revision"] == "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"
    assert not output.exists()


def test_hybrid_k4_plan_is_nonmutating_and_names_three_arms(tmp_path):
    output = tmp_path / "comparison"
    result = run(
        "measure_qwen_turboderp_hybrid_k4.py",
        "--source-model",
        str(tmp_path / "source"),
        "--source-revision",
        "4c446470ba0aec43e22ac1128f9ffd915f338ba3",
        "--encode-root",
        str(tmp_path / "encode"),
        "--panel-root",
        str(tmp_path / "panel"),
        "--turboderp-model",
        str(tmp_path / "turbo"),
        "--exllamav3-root",
        str(tmp_path / "exllamav3"),
        "--output",
        str(output),
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["arms"] == [
        "ours-expert-k4",
        "turboderp-full-k4",
        "hybrid-ours-experts",
    ]
    assert not output.exists()


def test_checked_in_naive_control_record_is_sealed():
    path = ROOT / "results/qwen3-30b-a3b-base/naive-3p5-controls-summary.json"
    record = json.loads(path.read_text())
    seal = record.pop("record_sha256")
    assert seal == sha256_bytes(canonical_json(record))
    assert len(record["controls"]) == 5
    assert record["naive_seeds_beating_selected_kld"] == 0
    assert record["selected_kld_reduction_vs_naive_mean"] > 0.28
