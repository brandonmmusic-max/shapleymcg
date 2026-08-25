from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import save_file

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_candidate_factory_union",
    ROOT / "scripts" / "verify_qwen_candidate_factory_union.py",
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def _seal(body: dict, field: str) -> dict:
    result = dict(body)
    result[field] = sha256_bytes(canonical_json(body))
    return result


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _logits(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"logits": value.astype(np.float32)}, path)
    return sha256_file(path)


def _summary(value: np.ndarray) -> dict:
    return {"mean": float(value.mean())}


def test_union_verifier_replays_all_endpoint_arrays_without_model_forward(
    tmp_path: Path, monkeypatch
) -> None:
    panel_root = tmp_path / "panel"
    experiment = tmp_path / "experiment"
    teacher_root = panel_root / "teacher-logits"
    generator = np.random.default_rng(41)
    teachers: list[Path] = []
    baseline_tokens = []
    union_tokens = []
    baseline_records = []
    union_records = []

    for row in range(10):
        teacher = generator.normal(size=(64, 7)).astype(np.float32)
        baseline = teacher + generator.normal(scale=0.13, size=teacher.shape).astype(np.float32)
        union = teacher + generator.normal(scale=0.08, size=teacher.shape).astype(np.float32)
        teacher_path = teacher_root / f"row-{row:02d}.safetensors"
        _logits(teacher_path, teacher)
        teachers.append(teacher_path)
        if row == 0:
            baseline_path = experiment / "selection-baseline-student-logits" / "row-00.safetensors"
            union_path = experiment / "selection-union-student-logits" / "row-00.safetensors"
            baseline_sha = _logits(baseline_path, baseline)
            union_sha = _logits(union_path, union)
            selection_baseline = VERIFY._token_kld_torch(teacher.astype(np.float64), baseline.astype(np.float64))
            selection_union = VERIFY._token_kld_torch(teacher.astype(np.float64), union.astype(np.float64))
            np.save(experiment / "selection-baseline.token-kld.npy", selection_baseline, allow_pickle=False)
            np.save(experiment / "selection-union.token-kld.npy", selection_union, allow_pickle=False)
            continue
        baseline_path = experiment / "validation-baseline-student-logits" / f"row-{row:02d}.safetensors"
        union_path = experiment / "validation-union-student-logits" / f"row-{row:02d}.safetensors"
        baseline_sha_row = _logits(baseline_path, baseline)
        union_sha_row = _logits(union_path, union)
        baseline_token = VERIFY._token_kld_torch(teacher.astype(np.float64), baseline.astype(np.float64))
        union_token = VERIFY._token_kld_torch(teacher.astype(np.float64), union.astype(np.float64))
        baseline_tokens.append(baseline_token)
        union_tokens.append(union_token)
        common = {"row": row, "teacher_sha256": sha256_file(teacher_path)}
        baseline_records.append(common | {
            "student_logits_sha256": baseline_sha_row,
            "mean_kld": float(baseline_token.mean()),
            "top1_agreement": float(np.mean(np.argmax(teacher, axis=-1) == np.argmax(baseline, axis=-1))),
        })
        union_records.append(common | {
            "student_logits_sha256": union_sha_row,
            "mean_kld": float(union_token.mean()),
            "top1_agreement": float(np.mean(np.argmax(teacher, axis=-1) == np.argmax(union, axis=-1))),
        })

    panel = _seal({"token_file": "tokens.npz"}, "panel_sha256")
    _write(panel_root / "panel.json", panel)
    teacher_receipt = _seal(
        {
            "panel_sha256": panel["panel_sha256"],
            "teacher_files": [
                {"path": path.relative_to(panel_root).as_posix(), "sha256": sha256_file(path)}
                for path in teachers
            ],
        },
        "receipt_sha256",
    )
    _write(panel_root / "teacher-receipt.json", teacher_receipt)
    baseline_array = np.stack(baseline_tokens)
    union_array = np.stack(union_tokens)
    np.save(experiment / "validation-baseline.token-kld.npy", baseline_array, allow_pickle=False)
    np.save(experiment / "validation-union.token-kld.npy", union_array, allow_pickle=False)
    allocation = _seal(
        {
            "selection_row": 0,
            "selected_mcg_layers": [3, 8],
        },
        "allocation_sha256",
    )
    _write(experiment / "factory-allocation.json", allocation)
    seed = 20260824
    _write(
        experiment / "plan.json",
        {"selection_row": 0, "validation_rows": list(range(1, 10)), "seed": seed},
    )
    delta = baseline_array - union_array
    report = _seal(
        {
            "factory_allocation_sha256": allocation["allocation_sha256"],
            "panel_sha256": panel["panel_sha256"],
            "teacher_receipt_sha256": teacher_receipt["receipt_sha256"],
            "selection": {
                "row": 0,
                "baseline_mean_kld": float(selection_baseline.mean()),
                "baseline_student_logits_sha256": baseline_sha,
                "baseline_token_kld_sha256": sha256_file(experiment / "selection-baseline.token-kld.npy"),
                "union_mean_kld": float(selection_union.mean()),
                "union_student_logits_sha256": union_sha,
                "union_token_kld_sha256": sha256_file(experiment / "selection-union.token-kld.npy"),
            },
            "untouched_validation": {
                "baseline_records": baseline_records,
                "union_records": union_records,
                "baseline_summary": _summary(baseline_array),
                "union_summary": _summary(union_array),
                "baseline_token_kld_sha256": sha256_file(experiment / "validation-baseline.token-kld.npy"),
                "union_token_kld_sha256": sha256_file(experiment / "validation-union.token-kld.npy"),
                "paired_block_95_interval_for_baseline_minus_union": VERIFY._paired_block_interval(
                    delta.reshape(-1), seed=seed + 10_000
                ),
            },
        },
        "report_sha256",
    )
    _write(experiment / "report.json", report)
    output = experiment / "independent-verification.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_qwen_candidate_factory_union.py",
            "--experiment-root",
            str(experiment),
            "--panel-root",
            str(panel_root),
            "--output",
            str(output),
        ],
    )
    assert VERIFY.main() == 0
    verification = json.loads(output.read_text())
    assert verification["verified"] is True
    assert verification["factory_allocation_sha256"] == allocation["allocation_sha256"]
    assert verification["replayed_file_count"] == 24
