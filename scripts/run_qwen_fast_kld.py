#!/usr/bin/env python3
"""Allocate the sealed K3/K4 candidates, replay them in BF16, and measure KLD."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.calibration.qwen_capture import qwen_moe_layers
from quant_pipeline.core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.evaluation.kld_window import verify_kld_window
from quant_pipeline.scoring.kld import summarize


MODEL_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _load_candidates(root: Path) -> tuple[list[dict], list[dict]]:
    receipts = []
    rows = []
    for layer in range(48):
        layer_root = root / f"layer-{layer:03d}"
        receipt_path = layer_root / "encode-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        seal = receipt.get("receipt_sha256")
        if seal != _hash_json({key: value for key, value in receipt.items() if key != "receipt_sha256"}):
            raise ValueError(f"layer {layer} encode receipt seal mismatch")
        if receipt.get("layer") != layer or receipt.get("experts") != list(range(128)):
            raise ValueError(f"layer {layer} encode inventory is incomplete")
        tensor_path = layer_root / str(receipt["candidate_tensor_file"])
        if sha256_file(tensor_path) != receipt["candidate_tensor_sha256"]:
            raise ValueError(f"layer {layer} candidate tensor file drifted")
        receipts.append({
            "layer": layer,
            "receipt_sha256": seal,
            "receipt_file_sha256": sha256_file(receipt_path),
            "candidate_tensor_sha256": receipt["candidate_tensor_sha256"],
            "candidate_tensor_bytes": receipt["candidate_tensor_bytes"],
        })
        rows.extend(dict(row) for row in receipt["scores"])
    expected = 48 * 128 * 3 * 2
    if len(rows) != expected:
        raise ValueError(f"candidate score inventory has {len(rows)} rows, expected {expected}")
    return receipts, rows


def _allocate(receipts: list[dict], rows: list[dict]) -> dict:
    by_unit: dict[tuple[int, int, str], dict[int, dict]] = {}
    for row in rows:
        identity = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        by_unit.setdefault(identity, {})[int(row["bits"])] = row
    expected_units = 48 * 128 * 3
    if len(by_unit) != expected_units or any(set(value) != {3, 4} for value in by_unit.values()):
        raise ValueError("K3/K4 allocation inventory is incomplete")
    upgrades = []
    for identity, candidates in by_unit.items():
        k3, k4 = candidates[3], candidates[4]
        extra = int(k4["stored_bytes"]) - int(k3["stored_bytes"])
        if extra <= 0:
            raise ValueError("K4 candidate does not cost more bytes than K3")
        benefit = float(k3["diagonal_p2_damage"]) - float(k4["diagonal_p2_damage"])
        upgrades.append((benefit / extra, benefit, identity, extra))
    k4_count = expected_units // 2
    ranked = sorted(upgrades, key=lambda row: (-row[0], -row[1], row[2]))
    selected_k4 = {row[2] for row in ranked[:k4_count]}
    choices = []
    total_bytes = 0
    total_damage = 0.0
    for identity in sorted(by_unit):
        bit = 4 if identity in selected_k4 else 3
        row = by_unit[identity][bit]
        total_bytes += int(row["stored_bytes"])
        total_damage += float(row["diagonal_p2_damage"])
        choices.append({
            "layer": identity[0],
            "expert": identity[1],
            "projection": identity[2],
            "bits": bit,
            "stored_bytes": int(row["stored_bytes"]),
            "diagonal_p2_damage": float(row["diagonal_p2_damage"]),
            "packed_sha256": row["packed_sha256"],
            "codec_fp16_reconstruction_sha256": row["codec_fp16_reconstruction_sha256"],
            "stored_bf16_reconstruction_sha256": row["stored_bf16_reconstruction_sha256"],
        })
    body = {
        "schema": "quant-pipeline.qwen-fast-k34-allocation.v2",
        "objective": "minimum-diagonal-routed-p2-damage-at-exact-half-k4-weight-rate",
        "rate_scope": "moe-expert-weight-elements",
        "average_weight_bits": sum(row["bits"] for row in choices) / len(choices),
        "unit_count": len(choices),
        "k3_count": len(choices) - len(selected_k4),
        "k4_count": len(selected_k4),
        "stored_payload_bytes": total_bytes,
        "predicted_diagonal_p2_damage": total_damage,
        "encode_receipts": receipts,
        "choices": choices,
    }
    body["allocation_sha256"] = _hash_json(body)
    return body


def _token_kld_chunked(teacher: np.ndarray, student: np.ndarray, chunk: int = 16) -> np.ndarray:
    if teacher.shape != student.shape:
        raise ValueError("teacher and student logits differ in shape")
    result = np.empty(teacher.shape[0], dtype=np.float64)
    for start in range(0, teacher.shape[0], chunk):
        stop = min(start + chunk, teacher.shape[0])
        t = np.asarray(teacher[start:stop], dtype=np.float64)
        s = np.asarray(student[start:stop], dtype=np.float64)
        if not np.isfinite(t).all() or not np.isfinite(s).all():
            raise ValueError("non-finite logits")
        t -= np.max(t, axis=-1, keepdims=True)
        s -= np.max(s, axis=-1, keepdims=True)
        t -= np.log(np.sum(np.exp(t), axis=-1, keepdims=True))
        s -= np.log(np.sum(np.exp(s), axis=-1, keepdims=True))
        p = np.exp(t)
        result[start:stop] = np.sum(p * (t - s), axis=-1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encode-root", type=Path, required=True)
    parser.add_argument("--kld-window", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--allocate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    plan = {
        "schema": "quant-pipeline.qwen-fast-k34-kld-plan.v1",
        "model": str(args.model.resolve()),
        "model_revision": MODEL_REVISION,
        "encode_root": str(args.encode_root.resolve()),
        "kld_window": str(args.kld_window.resolve()),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256_file(args.teacher),
        "output": str(output),
        "attention_backend": args.attention_backend,
        "allocate_only": args.allocate_only,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "kld-plan.json", plan | {"dry_run": False})
    receipts, rows = _load_candidates(args.encode_root.resolve())
    allocation = _allocate(receipts, rows)
    write_json(output / "allocation.json", allocation)
    print(json.dumps({
        "stage": "allocation",
        "average_weight_bits": allocation["average_weight_bits"],
        "k3_count": allocation["k3_count"],
        "k4_count": allocation["k4_count"],
        "allocation_sha256": allocation["allocation_sha256"],
    }, sort_keys=True), flush=True)
    if args.allocate_only:
        return 0

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model.resolve(),
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = qwen_moe_layers(model)
    choices = {
        (int(row["layer"]), int(row["expert"]), str(row["projection"])): int(row["bits"])
        for row in allocation["choices"]
    }
    for layer in range(48):
        candidate_file = args.encode_root.resolve() / f"layer-{layer:03d}" / "k34-candidates.safetensors"
        gate_up = []
        down = []
        with safe_open(candidate_file, framework="pt", device="cpu") as handle:
            for expert in range(128):
                gate_bit = choices[(layer, expert, "gate_proj")]
                up_bit = choices[(layer, expert, "up_proj")]
                down_bit = choices[(layer, expert, "down_proj")]
                gate = handle.get_tensor(f"K{gate_bit}.E{expert:03d}.gate_proj.reconstruction_hf")
                up = handle.get_tensor(f"K{up_bit}.E{expert:03d}.up_proj.reconstruction_hf")
                gate_up.append(torch.cat((gate, up), dim=0))
                down.append(handle.get_tensor(
                    f"K{down_bit}.E{expert:03d}.down_proj.reconstruction_hf"
                ))
        experts = blocks[layer].experts
        gate_up_tensor = torch.stack(gate_up).to(
            device=experts.gate_up_proj.device,
            dtype=experts.gate_up_proj.dtype,
        )
        down_tensor = torch.stack(down).to(
            device=experts.down_proj.device,
            dtype=experts.down_proj.dtype,
        )
        with torch.no_grad():
            experts.gate_up_proj.copy_(gate_up_tensor)
            experts.down_proj.copy_(down_tensor)
        del gate_up, down, gate_up_tensor, down_tensor
        print(json.dumps({"stage": "install", "layer": layer}), flush=True)

    window_root = args.kld_window.resolve()
    window = json.loads((window_root / "kld-window.json").read_text())
    verify_kld_window(window, window_root)
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([window["token_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
    student = logits.float().cpu().reshape(-1, logits.shape[-1]).contiguous()
    student_path = output / "student-logits.safetensors"
    save_file({"logits": student}, student_path, metadata={
        "role": "kld-student",
        "allocation_sha256": allocation["allocation_sha256"],
        "token_sha256": str(window["token_sha256"]),
    })
    del logits, model
    torch.cuda.empty_cache()

    with safe_open(args.teacher.resolve(), framework="np") as handle:
        teacher = handle.get_tensor("logits")
    student_np = student.numpy()
    values = _token_kld_chunked(teacher, student_np)
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    values_path = output / "token-kld.npy"
    atomic_write(values_path, buffer.getvalue())
    report = {
        "schema": "quant-pipeline.qwen-fast-k34-kld.v1",
        "allocation_sha256": allocation["allocation_sha256"],
        "teacher_sha256": sha256_file(args.teacher),
        "student_sha256": sha256_file(student_path),
        "student_shape": list(student.shape),
        "token_kld_sha256": sha256_file(values_path),
        "summary": summarize(values),
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = _hash_json(report)
    write_json(output / "kld-report.json", report)
    print(json.dumps({"ok": True, **report}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
