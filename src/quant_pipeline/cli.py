from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from .allocation.global_dp import Candidate, allocate
from .calibration.windows import seal_corpus
from .checkpoint.reference_pack import audit_packed_checkpoint, encode_reference_checkpoint
from .core.artifacts import require_execute, sha256_file, write_json
from .models.hf_capture import capture_logits
from .models.inventory import load_inventory
from .scoring.attribution import split_layer_damage
from .scoring.kld import summarize, token_kld
from .spec import ExperimentSpec


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_inspect(args) -> None:
    spec = ExperimentSpec.load(args.spec)
    _print({"spec": dataclasses.asdict(spec), "spec_sha256": spec.digest})


def command_inventory(args) -> None:
    units = load_inventory(args.config, args.family)
    result = {"family": args.family, "unit_count": len(units), "units": [dataclasses.asdict(unit) for unit in units]}
    if args.output:
        write_json(args.output, result)
    _print({"family": args.family, "unit_count": len(units), "output": args.output})


def command_seal(args) -> None:
    spec = ExperimentSpec.load(args.spec)
    try:
        from transformers import AutoTokenizer
    except Exception as error:
        raise RuntimeError("install quant-pipeline[hf] to seal with the model tokenizer") from error
    tokenizer = AutoTokenizer.from_pretrained(spec.corpus.tokenizer_id, revision=spec.model.revision)
    limits = {
        "fit": spec.corpus.fit_windows,
        "selection": spec.corpus.selection_windows,
        "confirmation": spec.corpus.confirmation_windows,
        "final": spec.corpus.final_windows,
    }
    artifact = seal_corpus(
        spec.corpus.input_jsonl,
        args.output,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        spec.corpus.window_tokens,
        limits,
        spec.corpus.seed,
        {"id": spec.corpus.tokenizer_id, "revision": spec.model.revision, "class": type(tokenizer).__name__},
        spec.corpus.minimum_domains,
    )
    _print({"output": str(Path(args.output).resolve()), "seal_sha256": artifact["seal_sha256"], "windows": {key: len(value) for key, value in artifact["windows"].items()}})


def command_capture(args) -> None:
    result = capture_logits(args.model, args.sealed_corpus, args.role, args.output_dir, args.execute, args.dtype, args.device_map, args.model_revision)
    _print({"output_dir": str(Path(args.output_dir).resolve()), "windows": len(result["records"])})


def command_kld(args) -> None:
    try:
        from safetensors import safe_open
    except Exception as error:
        raise RuntimeError("safetensors is required for KLD evaluation") from error
    teacher = Path(args.teacher_dir)
    student = Path(args.student_dir)
    teacher_receipt_path = teacher / "capture-receipt.json"
    student_receipt_path = student / "capture-receipt.json"
    teacher_receipt = json.loads(teacher_receipt_path.read_text())
    student_receipt = json.loads(student_receipt_path.read_text())
    if teacher_receipt["sealed_corpus_sha256"] != student_receipt["sealed_corpus_sha256"]:
        raise ValueError("teacher and student captures bind different sealed corpora")
    if teacher_receipt["role"] != student_receipt["role"]:
        raise ValueError("teacher and student captures use different corpus roles")
    teacher_records = {row["file"]: row for row in teacher_receipt["records"]}
    student_records = {row["file"]: row for row in student_receipt["records"]}
    if teacher_records.keys() != student_records.keys():
        raise ValueError("teacher/student capture file sets differ")
    actual_teacher = {path.name for path in teacher.glob("window-*.safetensors")}
    actual_student = {path.name for path in student.glob("window-*.safetensors")}
    if actual_teacher != teacher_records.keys() or actual_student != student_records.keys():
        raise ValueError("capture directory contents differ from their receipts")
    per_window = []
    per_token = []
    for filename in sorted(teacher_records):
        teacher_path = teacher / filename
        student_path = student / teacher_path.name
        with safe_open(teacher_path, framework="np") as reference, safe_open(student_path, framework="np") as candidate:
            teacher_metadata = reference.metadata() or {}
            student_metadata = candidate.metadata() or {}
            expected_token_sha = teacher_records[filename]["token_sha256"]
            if student_records[filename]["token_sha256"] != expected_token_sha:
                raise ValueError(f"receipt token identity mismatch for {filename}")
            if teacher_metadata.get("token_sha256") != expected_token_sha or student_metadata.get("token_sha256") != expected_token_sha:
                raise ValueError(f"safetensors token identity mismatch for {filename}")
            teacher_logits = reference.get_tensor("logits")
            student_logits = candidate.get_tensor("logits")
            if teacher_logits.shape != student_logits.shape:
                raise ValueError(f"logit shape mismatch for {filename}")
            values = token_kld(teacher_logits, student_logits)
        if sha256_file(teacher_path) != teacher_records[filename]["sha256"]:
            raise ValueError(f"teacher capture hash mismatch for {filename}")
        if sha256_file(student_path) != student_records[filename]["sha256"]:
            raise ValueError(f"student capture hash mismatch for {filename}")
        per_token.append(values)
        per_window.append(float(np.mean(values)))
    if not per_window:
        raise ValueError("no matched window files")
    report = {
        "schema": "quant-pipeline.kld-report.v1",
        "teacher_receipt_sha256": sha256_file(teacher_receipt_path),
        "student_receipt_sha256": sha256_file(student_receipt_path),
        "sealed_corpus_sha256": teacher_receipt["sealed_corpus_sha256"],
        "window": summarize(np.asarray(per_window)),
        "token": summarize(np.concatenate(per_token)),
        "per_window": per_window,
    }
    write_json(args.output, report)
    _print(report)


def command_attribute_experts(args) -> None:
    data = np.load(args.projected_residuals)
    report = split_layer_damage(args.measured_damage, data[args.key], args.routing_state_shift)
    write_json(args.output, report)
    _print(report)


def command_allocate(args) -> None:
    raw = json.loads(Path(args.candidates).read_text())
    candidates = [Candidate(**row) for row in raw["candidates"]]
    result = allocate(candidates, args.byte_budget, args.quantum)
    document = {
        "schema": "quant-pipeline.allocation.v1",
        "byte_semantics": "codec-payload-including-codec-vectors-excluding-container",
        "byte_budget": args.byte_budget,
        "stored_bytes": result.stored_bytes,
        "predicted_damage": result.predicted_damage,
        "choices": {
            choice.unit_id: {
                **(choice.metadata or {}),
                "choice_id": choice.choice_id,
                "stored_bytes": choice.stored_bytes,
                "predicted_damage": choice.predicted_damage,
            }
            for choice in result.choices
        },
    }
    write_json(args.output, document)
    _print({key: value for key, value in document.items() if key != "choices"} | {"unit_count": len(document["choices"])})


def command_encode(args) -> None:
    require_execute(args.execute, "encode and write a packed checkpoint")
    result = encode_reference_checkpoint(args.model_path, args.family, args.allocation, args.output_dir, args.group_size)
    _print({"output_dir": str(Path(args.output_dir).resolve()), "stored_bytes": result["stored_bytes"], "units": len(result["units"])})


def command_audit(args) -> None:
    result = audit_packed_checkpoint(args.packed_dir)
    _print(result)
    if not result["ok"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="validate and display an experiment spec")
    inspect.add_argument("spec")
    inspect.set_defaults(func=command_inspect)

    inventory = sub.add_parser("inventory", help="enumerate expert quantization units")
    inventory.add_argument("--config", required=True)
    inventory.add_argument("--family", required=True, choices=("qwen3_moe", "gemma4"))
    inventory.add_argument("--output")
    inventory.set_defaults(func=command_inventory)

    seal = sub.add_parser("seal", help="seal document-disjoint token windows")
    seal.add_argument("spec")
    seal.add_argument("--output", required=True)
    seal.set_defaults(func=command_seal)

    capture = sub.add_parser("capture", help="capture logits and router logits from a sealed role")
    capture.add_argument("--model", required=True)
    capture.add_argument("--sealed-corpus", required=True)
    capture.add_argument("--role", required=True, choices=("fit", "selection", "confirmation", "final"))
    capture.add_argument("--output-dir", required=True)
    capture.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    capture.add_argument("--device-map", default="auto")
    capture.add_argument("--model-revision", required=True)
    capture.add_argument("--execute", action="store_true")
    capture.set_defaults(func=command_capture)

    kld = sub.add_parser("kld", help="measure next-token KLD for matched captures")
    kld.add_argument("--teacher-dir", required=True)
    kld.add_argument("--student-dir", required=True)
    kld.add_argument("--output", required=True)
    kld.set_defaults(func=command_kld)

    experts = sub.add_parser("attribute-experts", help="close projected expert residuals to measured layer damage")
    experts.add_argument("--projected-residuals", required=True)
    experts.add_argument("--key", default="projected_residuals")
    experts.add_argument("--measured-damage", required=True, type=float)
    experts.add_argument("--routing-state-shift", default=0.0, type=float)
    experts.add_argument("--output", required=True)
    experts.set_defaults(func=command_attribute_experts)

    allocation = sub.add_parser("allocate", help="solve the exact global codec-payload-byte allocation")
    allocation.add_argument("--candidates", required=True)
    allocation.add_argument("--byte-budget", required=True, type=int)
    allocation.add_argument("--quantum", default=1, type=int)
    allocation.add_argument("--output", required=True)
    allocation.set_defaults(func=command_allocate)

    encode = sub.add_parser("encode-reference", help="write an auditable reference packed checkpoint")
    encode.add_argument("--model-path", required=True)
    encode.add_argument("--family", required=True, choices=("qwen3_moe", "gemma4"))
    encode.add_argument("--allocation", required=True)
    encode.add_argument("--output-dir", required=True)
    encode.add_argument("--group-size", default=128, type=int)
    encode.add_argument("--execute", action="store_true")
    encode.set_defaults(func=command_encode)

    audit = sub.add_parser("audit", help="verify packed checkpoint hashes and byte accounting")
    audit.add_argument("--packed-dir", required=True)
    audit.set_defaults(func=command_audit)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
