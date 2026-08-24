import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]


def _write_sealed(path: Path, value: dict, field: str) -> dict:
    value = dict(value)
    value[field] = sha256_bytes(canonical_json(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value


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


def test_streaming_fit_accepts_repository_canonical_source_seal(tmp_path):
    pytest.importorskip("torch")
    revision = "4c446470ba0aec43e22ac1128f9ffd915f338ba3"
    receipt = _write_sealed(
        tmp_path / "source.json",
        {"schema": "fixture", "revision": revision},
        "receipt_sha256",
    )
    namespace = runpy.run_path(str(ROOT / "scripts/run_qwen_streaming_fit.py"))
    assert namespace["_source_identity"](tmp_path / "source.json", revision) == receipt["receipt_sha256"]


def test_hybrid_k4_plan_is_nonmutating_and_names_three_arms(tmp_path):
    output = tmp_path / "comparison"
    result = run(
        "measure_qwen_turboderp_hybrid_k4.py",
        "--source-model",
        str(tmp_path / "source"),
        "--source-revision",
        "4c446470ba0aec43e22ac1128f9ffd915f338ba3",
        "--source-receipt",
        str(tmp_path / "source-receipt.json"),
        "--encode-root",
        str(tmp_path / "encode"),
        "--panel-root",
        str(tmp_path / "panel"),
        "--turboderp-model",
        str(tmp_path / "turbo"),
        "--turboderp-receipt",
        str(tmp_path / "turbo-receipt.json"),
        "--lineage-receipt",
        str(tmp_path / "lineage-receipt.json"),
        "--exllamav3-root",
        str(tmp_path / "exllamav3"),
        "--output",
        str(output),
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["arms"] == [
        "ours-selected-k34",
        "ours-expert-k4",
        "turboderp-full-k4",
        "hybrid-ours-experts",
    ]
    assert not output.exists()


def test_exact_turboderp_3p5_plan_is_matched_and_nonmutating(tmp_path):
    output = tmp_path / "exact-3p5"
    result = run(
        "measure_qwen_turboderp_exact_3p5.py",
        "--source-model",
        str(tmp_path / "source"),
        "--encode-root",
        str(tmp_path / "encode"),
        "--allocation",
        str(tmp_path / "allocation.json"),
        "--panel-root",
        str(tmp_path / "panel"),
        "--turboderp-k3-model",
        str(tmp_path / "turbo-k3"),
        "--turboderp-k3-revision",
        "1" * 40,
        "--turboderp-k4-model",
        str(tmp_path / "turbo-k4"),
        "--exllamav3-root",
        str(tmp_path / "exllamav3"),
        "--output",
        str(output),
    )
    plan = json.loads(result.stdout)
    assert plan["expert_rate"] == "exact half K3 / half K4 = 3.5 logical BPW"
    assert plan["arms"] == [
        "turboderp-selected-k34",
        "hybrid-ours-selected-k34",
    ]
    assert not output.exists()


def test_hybrid_k4_resume_adopts_only_fully_sealed_arm(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        namespace = runpy.run_path(
            str(ROOT / "scripts/measure_qwen_turboderp_hybrid_k4.py")
        )
    finally:
        sys.path.pop(0)
    teacher = []
    for row in range(10):
        path = tmp_path / f"teacher-{row:02d}.bin"
        path.write_bytes(f"teacher-{row}".encode())
        teacher.append(path)
    arm = "ours-selected-k34"
    arm_root = tmp_path / arm
    students = arm_root / "student-logits"
    students.mkdir(parents=True)
    student_paths = []
    for row in range(10):
        path = students / f"row-{row:02d}.safetensors"
        path.write_bytes(f"student-{row}".encode())
        student_paths.append(path)
    token = arm_root / "token-kld.npy"
    token.write_bytes(b"sealed-token-kld")
    report = {
        "schema": "quant-pipeline.qwen-posttrained-k4-arm.v1",
        "arm": arm,
        "summary": {"mean": 0.041},
        "top1_agreement": 0.92,
        "token_kld_sha256": namespace["sha256_file"](token),
        "teacher_files": [namespace["sha256_file"](path) for path in teacher],
        "student_files": [namespace["sha256_file"](path) for path in student_paths],
    }
    report["report_sha256"] = namespace["_hash_json"](report)
    (arm_root / "kld-report.json").write_text(json.dumps(report))
    adopted = namespace["_load_existing_arm"](tmp_path, arm, teacher)
    assert adopted["report_sha256"] == report["report_sha256"]
    token.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="token-KLD artifact mismatch"):
        namespace["_load_existing_arm"](tmp_path, arm, teacher)


def test_hybrid_k4_resume_adopts_sealed_allocation_without_tensor_rehash(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        namespace = runpy.run_path(
            str(ROOT / "scripts/measure_qwen_turboderp_hybrid_k4.py")
        )
    finally:
        sys.path.pop(0)
    receipts = [
        {
            "layer": layer,
            "receipt_sha256": f"receipt-{layer}",
            "candidate_tensor_sha256": f"tensor-{layer}",
            "candidate_tensor_bytes": layer + 100,
        }
        for layer in range(48)
    ]
    allocation = {
        "schema": "quant-pipeline.qwen-fast-k34-allocation.v2",
        "average_weight_bits": 3.5,
        "encode_receipts": receipts,
        "choices": [
            {
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "bits": 3,
            }
            for layer in range(48)
            for expert in range(128)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ],
    }
    allocation["allocation_sha256"] = namespace["_hash_json"](allocation)
    (tmp_path / "selected-allocation.json").write_text(json.dumps(allocation))
    adopted = namespace["_load_existing_allocation"](tmp_path, receipts)
    assert adopted["allocation_sha256"] == allocation["allocation_sha256"]
    receipts[0]["candidate_tensor_bytes"] += 1
    with pytest.raises(ValueError, match="disagrees with layer 0"):
        namespace["_load_existing_allocation"](tmp_path, receipts)


def test_fit_route_coverage_extension_defaults_to_dry_run(tmp_path):
    output = tmp_path / "extended.json"
    result = run(
        "extend_qwen_fit_route_coverage.py",
        "--model",
        str(tmp_path / "model"),
        "--model-revision",
        "4c446470ba0aec43e22ac1128f9ffd915f338ba3",
        "--sealed-corpus",
        str(tmp_path / "corpus.json"),
        "--capture-root",
        str(tmp_path / "capture"),
        "--output",
        str(output),
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["max_candidates"] == 256
    assert not output.exists()


def test_checked_in_naive_control_record_is_sealed():
    path = ROOT / "results/qwen3-30b-a3b-base/naive-3p5-controls-summary.json"
    record = json.loads(path.read_text())
    seal = record.pop("record_sha256")
    assert seal == sha256_bytes(canonical_json(record))
    assert len(record["controls"]) == 5
    assert record["naive_seeds_beating_selected_kld"] == 0
    assert record["selected_kld_reduction_vs_naive_mean"] > 0.28


def test_posttrained_teacher_capture_defaults_to_dry_run(tmp_path):
    output = tmp_path / "teacher"
    result = run(
        "prepare_turboderp_wiki2_teacher.py",
        "--model",
        str(tmp_path / "model"),
        "--model-revision",
        "4c446470ba0aec43e22ac1128f9ffd915f338ba3",
        "--source-receipt",
        str(tmp_path / "source.json"),
        "--output",
        str(output),
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["rows"] == 10
    assert plan["row_length"] == 2048
    assert plan["existing_panel_root"] is None
    assert not output.exists()


def test_posttrained_teacher_accepts_and_labels_legacy_source_seal(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = runpy.run_path(str(ROOT / "scripts/prepare_turboderp_wiki2_teacher.py"))
    revision = "4c446470ba0aec43e22ac1128f9ffd915f338ba3"
    body = {"schema": "source.v1", "revision": revision, "files": {}}
    legacy = hashlib.sha256(canonical_json(body).removesuffix(b"\n")).hexdigest()
    seal, scheme = namespace["_verify_source_receipt"](
        {**body, "receipt_sha256": legacy}, revision
    )
    assert seal == legacy
    assert scheme == "legacy-canonical-json-without-trailing-newline"


def test_posttrained_result_bundle_seals_every_hub_layer(tmp_path):
    run_root = tmp_path / "run"
    artifact_root = tmp_path / "artifacts"
    output = tmp_path / "bundle"
    for name in (
        "calibration-parallel.exit",
        "route-complete-capture.exit",
        "streaming-fit-waves.exit",
        "encode-publish-waves.exit",
        "matched-k4-evaluation.exit",
        "naive-controls.exit",
        "exact-3p5-comparison.exit",
        "hf-candidates.exit",
    ):
        path = run_root / "logs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0\n")
    source = run_root / "artifacts/source/source-receipt.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}\n")
    for relative in (
        "experiment.resolved.toml",
        "inputs/reap_recall_calib.jsonl",
        "inputs/reap_recall_calib.role-safe-packed.jsonl",
        "artifacts/reap-recall-packing-receipt.json",
        "artifacts/qwen-sealed-corpus.json",
        "artifacts/qwen-sealed-corpus-route-complete.json",
        "calibration-capture/calibration-capture-fit-conditional_down-receipt.json",
        "calibration-capture-route-complete/calibration-capture-fit-receipt.json",
        "calibration-capture-base/calibration-capture-heldout-receipt.json",
        "calibration-capture/fit/capture-manifest.json",
        "calibration-capture-route-complete/fit/capture-manifest.json",
        "calibration-capture/conditional_down/capture-manifest.json",
        "calibration-capture-base/heldout/capture-manifest.json",
    ):
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n")
    teacher = artifact_root / "turboderp-wiki2-teacher/teacher.json"
    teacher.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_text("{}\n")

    arms = {
        key: {"mean_kld": value, "top1_agreement": 0.9, "report_sha256": key}
        for key, value in (
            ("ours-selected-k34", 0.03),
            ("ours-expert-k4", 0.02),
            ("turboderp-full-k4", 0.025),
            ("hybrid-ours-experts", 0.02),
        )
    }
    _write_sealed(
        artifact_root / "matched-k4-comparison/summary.json",
        {
            "schema": "fixture",
            "source_revision": "4c446470ba0aec43e22ac1128f9ffd915f338ba3",
            "arms": arms,
            "matched_hybrid_kld_reduction_vs_turboderp": 0.2,
            "matched_hybrid_top1_gain_vs_turboderp": 0.01,
        },
        "summary_sha256",
    )
    _write_sealed(
        artifact_root / "naive-3p5-controls-v1/summary.json",
        {
            "schema": "fixture",
            "naive_mean_kld": 0.04,
            "naive_sample_std_kld": 0.001,
            "naive_min_kld": 0.039,
            "naive_max_kld": 0.041,
            "selected_mean_kld": 0.03,
            "naive_seeds_beating_selected_kld": 0,
        },
        "summary_sha256",
    )
    _write_sealed(
        artifact_root / "matched-3p5-comparison/summary.json",
        {
            "schema": "fixture",
            "arms": {
                "turboderp-selected-k34": {
                    "mean_kld": 0.035,
                    "top1_agreement": 0.91,
                },
                "hybrid-ours-selected-k34": {
                    "mean_kld": 0.03,
                    "top1_agreement": 0.92,
                },
            },
            "ours_kld_reduction_vs_turboderp_at_exact_3p5": 1 - 0.03 / 0.035,
        },
        "summary_sha256",
    )
    for kind in ("fits", "candidates"):
        for layer in range(48):
            _write_sealed(
                run_root / f"artifacts/hf-upload/{kind}/layer-{layer:03d}.json",
                {
                    "layer": layer,
                    "repo_type": "dataset",
                    "path_in_repo": f"{kind}/layer-{layer:03d}",
                    "revision": f"revision-{layer}",
                    "manifest_sha256": f"manifest-{layer}",
                    "total_bytes": layer + 1,
                },
                "receipt_sha256",
            )

    result = run(
        "seal_qwen_posttrained_result_bundle.py",
        "--run-root",
        str(run_root),
        "--artifact-root",
        str(artifact_root),
        "--code-root",
        str(ROOT),
        "--output",
        str(output),
        "--git-revision",
        "a" * 40,
        "--execute",
    )
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["ok"] is True
    manifest = json.loads((output / "bundle-manifest.json").read_text())
    seal = manifest.pop("manifest_sha256")
    assert seal == sha256_bytes(canonical_json(manifest))
    assert len(manifest["fit_layers"]) == 48
    assert len(manifest["candidate_layers"]) == 48
    assert (output / "evaluation/matched-k4-comparison/summary.json").is_file()
    assert (output / "evaluation/matched-3p5-comparison/summary.json").is_file()
    assert (output / "evaluation/naive-3p5-controls/summary.json").is_file()
