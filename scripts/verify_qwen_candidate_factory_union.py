#!/usr/bin/env python3
"""Independently verify a completed Qwen candidate-factory union experiment."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from quant_pipeline.core.artifacts import (
    atomic_write,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json,
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _verify_json_seal(document: dict[str, Any], field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def _load_logits(path: Path) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise ValueError(f"unexpected tensor inventory in {path}")
        value = np.asarray(handle.get_tensor(keys[0]), dtype=np.float64)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid logits in {path}")
    return value


def _token_kld_torch(teacher: np.ndarray, student: np.ndarray, chunk: int = 8) -> np.ndarray:
    """Use Torch float64 softmax/log-softmax, independent of the producer path."""

    import torch

    if teacher.shape != student.shape:
        raise ValueError("teacher/student logit geometry mismatch")
    result = []
    for start in range(0, teacher.shape[0], chunk):
        target = torch.from_numpy(teacher[start : start + chunk])
        observed = torch.from_numpy(student[start : start + chunk])
        target_logp = torch.log_softmax(target, dim=-1)
        observed_logp = torch.log_softmax(observed, dim=-1)
        result.append(torch.sum(target_logp.exp() * (target_logp - observed_logp), dim=-1).numpy())
    return np.concatenate(result).astype(np.float64, copy=False)


def _load_npy_verified(path: Path, expected_sha256: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"hash mismatch for {path}")
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)


def _paired_block_interval(
    delta: np.ndarray,
    *,
    seed: int,
    block_tokens: int = 64,
    draws: int = 20_000,
) -> list[float]:
    if delta.ndim != 1 or len(delta) % block_tokens:
        raise ValueError("paired block bootstrap geometry mismatch")
    blocks = delta.reshape(-1, block_tokens).mean(axis=1)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(blocks), size=(draws, len(blocks)))
    return [float(value) for value in np.quantile(blocks[sampled].mean(axis=1), (0.025, 0.975))]


def _close(actual: float, expected: float, label: str, tolerance: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _save_npy(path: Path, value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_write(path, buffer.getvalue())
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=2e-11)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    panel_root = args.panel_root.resolve()
    output = (args.output or root / "independent-verification.json").resolve()

    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    _verify_json_seal(report, "report_sha256", "union report")
    plan = json.loads((root / "plan.json").read_text())
    panel = json.loads((panel_root / "panel.json").read_text())
    _verify_json_seal(panel, "panel_sha256", "evaluation panel")
    if report["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("report and panel identities differ")

    teacher_paths = sorted((panel_root / "teacher-logits").glob("row-*.safetensors"))
    if len(teacher_paths) != 10:
        raise ValueError("evaluation panel lacks ten teacher rows")

    selection_row = int(report["selection"]["row"])
    teacher = _load_logits(teacher_paths[selection_row])
    selection_values: dict[str, np.ndarray] = {}
    for arm in ("baseline", "union"):
        path = root / f"selection-{arm}-student-logits" / f"row-{selection_row:02d}.safetensors"
        expected = report["selection"][f"{arm}_student_logits_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"selection {arm} student-logit hash mismatch")
        selection_values[arm] = _token_kld_torch(teacher, _load_logits(path))
    _close(
        float(selection_values["baseline"].mean()),
        float(report["selection"]["baseline_mean_kld"]),
        "selection baseline mean KLD",
        args.tolerance,
    )
    _close(
        float(selection_values["union"].mean()),
        float(report["selection"]["union_mean_kld"]),
        "selection union mean KLD",
        args.tolerance,
    )

    validation = report["untouched_validation"]
    validation_rows = [int(row) for row in plan["validation_rows"]]
    by_arm: dict[str, np.ndarray] = {}
    for arm in ("baseline", "union"):
        values = []
        records = {int(row["row"]): row for row in validation[f"{arm}_records"]}
        for row in validation_rows:
            teacher = _load_logits(teacher_paths[row])
            student_path = root / f"validation-{arm}-student-logits" / f"row-{row:02d}.safetensors"
            record = records[row]
            if sha256_file(student_path) != record["student_logits_sha256"]:
                raise ValueError(f"validation {arm} row {row} student-logit hash mismatch")
            student = _load_logits(student_path)
            token = _token_kld_torch(teacher, student)
            _close(float(token.mean()), float(record["mean_kld"]), f"{arm} row {row} KLD", args.tolerance)
            top1 = float(np.mean(np.argmax(teacher, axis=-1) == np.argmax(student, axis=-1)))
            _close(top1, float(record["top1_agreement"]), f"{arm} row {row} top1", 0.0)
            values.append(token)
        by_arm[arm] = np.stack(values)
        _close(
            float(by_arm[arm].mean()),
            float(validation[f"{arm}_summary"]["mean"]),
            f"validation {arm} mean KLD",
            args.tolerance,
        )
        produced = _load_npy_verified(
            root / f"validation-{arm}.token-kld.npy",
            validation[f"{arm}_token_kld_sha256"],
        )
        if not np.allclose(produced, by_arm[arm], rtol=0.0, atol=args.tolerance):
            raise ValueError(f"validation {arm} token KLD differs from independent recomputation")

    delta = by_arm["baseline"] - by_arm["union"]
    interval = _paired_block_interval(delta.reshape(-1), seed=int(plan["seed"]) + 10_000)
    expected_interval = validation["paired_block_95_interval_for_baseline_minus_union"]
    for index, value in enumerate(interval):
        _close(value, float(expected_interval[index]), f"validation interval endpoint {index}", args.tolerance)

    body = {
        "schema": "quant-pipeline.qwen-candidate-factory-union-independent-verification.v1",
        "report_sha256": report["report_sha256"],
        "report_file_sha256": sha256_file(report_path),
        "panel_sha256": panel["panel_sha256"],
        "implementation": "torch.float64 log_softmax independent recomputation",
        "selection_baseline_mean_kld": float(selection_values["baseline"].mean()),
        "selection_union_mean_kld": float(selection_values["union"].mean()),
        "validation_baseline_mean_kld": float(by_arm["baseline"].mean()),
        "validation_union_mean_kld": float(by_arm["union"].mean()),
        "validation_absolute_reduction": float(delta.mean()),
        "validation_relative_reduction": float(delta.mean() / by_arm["baseline"].mean()),
        "paired_block_95_interval_for_baseline_minus_union": interval,
        "rows_union_better": int(np.count_nonzero(
            by_arm["union"].mean(axis=1) < by_arm["baseline"].mean(axis=1)
        )),
        "row_count": len(validation_rows),
        "tolerance": args.tolerance,
        "verified": True,
    }
    body["verification_sha256"] = _hash_json(body)
    write_json(output, body)
    print(json.dumps(body, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
