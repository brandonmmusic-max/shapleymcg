#!/usr/bin/env python3
"""Encode Qwen q/k/v/o projections with the corrected R10/MCG K4 path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.campaign.qwen_services import CorrectedPinnedGSSProducer
from quant_pipeline.campaign.qwen_work_units import _CheckpointWeights, _block_values, _sign
from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.normalization.absolute_v31 import MatrixInput, fit_layer_absolute_normalization
from quant_pipeline.normalization.artifact_v31 import PinnedGSSRequest, tensor_identity_sha256, tensor_sha256
from quant_pipeline.normalization.prior_search import scale_family_candidates


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
TRANSFORM_SEED = "eeb56e706c26ef3d84d6fc317fceea61181944107a2295ee2f6d34cc959250aa"


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _load_capture_roots(paths: list[Path]) -> list[tuple[Path, dict]]:
    result = []
    identities = set()
    for path in paths:
        root = path.resolve()
        manifest = json.loads((root / "capture-manifest.json").read_text())
        expected = manifest.get("manifest_sha256")
        if expected != _hash_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
            raise ValueError(f"attention capture seal mismatch: {root}")
        if manifest.get("schema") != "quant-pipeline.qwen-attention-hessian-shard.v1":
            raise ValueError(f"unexpected attention capture schema: {root}")
        identities.add(
            (
                manifest["source_revision"],
                manifest["sealed_corpus_seal"],
                manifest["role"],
                int(manifest["shard_count"]),
            )
        )
        result.append((root, manifest))
    if len(identities) != 1:
        raise ValueError("attention capture shards have different identities")
    shard_count = next(iter(identities))[3]
    if len(result) != shard_count or {int(row[1]["shard_index"]) for row in result} != set(range(shard_count)):
        raise ValueError("attention capture shard inventory is incomplete")
    return sorted(result, key=lambda row: int(row[1]["shard_index"]))


def _load_covariances(captures: list[tuple[Path, dict]], layer: int):
    import torch
    from safetensors import safe_open

    qkv = None
    o = None
    rows = 0
    source_hashes = []
    for root, manifest in captures:
        record = next(row for row in manifest["records"] if int(row["layer"]) == layer)
        path = root / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"attention capture layer {layer} hash mismatch")
        with safe_open(path, framework="pt", device="cpu") as handle:
            this_qkv = handle.get_tensor("qkv_gram").float()
            this_o = handle.get_tensor("o_gram").float()
        qkv = this_qkv if qkv is None else qkv.add_(this_qkv)
        o = this_o if o is None else o.add_(this_o)
        rows += int(record["rows"])
        source_hashes.append(record["sha256"])
    if rows <= 0 or qkv is None or o is None:
        raise ValueError(f"attention capture layer {layer} is empty")
    return (
        np.ascontiguousarray((qkv / rows).numpy()),
        np.ascontiguousarray((o / rows).numpy()),
        rows,
        source_hashes,
    )


def _source_identity(path: Path) -> tuple[str, str]:
    receipt = json.loads(path.read_text())
    expected = receipt.get("receipt_sha256")
    if expected != _hash_json({key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise ValueError("source receipt seal mismatch")
    return str(expected), str(receipt["revision"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--capture-shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--numeric-core", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=TRANSFORM_SEED)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.first_layer not in range(48) or args.layers < 1 or args.first_layer + args.layers > 48:
        parser.error("requested layer range escapes [0, 48)")

    captures = _load_capture_roots(args.capture_shard)
    source_identity, source_revision = _source_identity(args.source_receipt.resolve())
    output = args.output.resolve()
    plan = {
        "schema": "quant-pipeline.qwen-attention-k4-encode-plan.v1",
        "model": str(args.model.resolve()),
        "source_revision": source_revision,
        "source_checkpoint_identity": source_identity,
        "source_receipt_sha256": sha256_file(args.source_receipt),
        "capture_manifests": [sha256_file(root / "capture-manifest.json") for root, _ in captures],
        "output": str(output),
        "first_layer": args.first_layer,
        "layers": args.layers,
        "bits": 4,
        "policy": "energy_balanced",
        "scale_family": "per128-grid",
        "transform_seed_sha256": args.seed,
        "device": args.device,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import save_file

    reader = _CheckpointWeights(args.model.resolve())
    codec = Exl3MCGCodec(
        source_root=args.source_root,
        numeric_core=args.numeric_core,
        extension=args.extension,
        device=args.device,
    )
    backend = codec._codec()
    producer = CorrectedPinnedGSSProducer(codec)
    all_records = []
    started = time.monotonic()
    for layer in range(args.first_layer, args.first_layer + args.layers):
        layer_started = time.monotonic()
        destination = output / f"layer-{layer:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        qkv_covariance, o_covariance, rows, capture_hashes = _load_covariances(captures, layer)
        prefix = f"model.layers.{layer}.self_attn"
        source = {
            projection: reader.tensor(f"{prefix}.{projection}.weight").contiguous()
            for projection in ATTENTION
        }
        qkv_diagonal = np.diag(qkv_covariance).copy()
        o_diagonal = np.diag(o_covariance).copy()
        qkv_k_scales = scale_family_candidates(_block_values(qkv_diagonal, 128))["per128-grid"]
        o_k_scales = scale_family_candidates(_block_values(o_diagonal, 128))["per128-grid"]
        shared_qkv_sign = _sign(source["q_proj"].shape[1], args.seed, layer, "attention", "qkv-suh")
        shared_o_output_sign = _sign(source["o_proj"].shape[0], args.seed, layer, "attention", "o-svh")
        matrices = []
        projection_roles = {
            "q_proj": "gate_proj",
            "k_proj": "gate_proj",
            "v_proj": "up_proj",
            "o_proj": "down_proj",
        }
        for projection in ATTENTION:
            weight = source[projection]
            is_o = projection == "o_proj"
            matrices.append(
                MatrixInput(
                    key=projection,
                    projection=projection_roles[projection],
                    bits=4,
                    weight_kn=weight.T.to(args.device).contiguous(),
                    suh_sign=(
                        _sign(weight.shape[1], args.seed, layer, "attention", projection, "suh")
                        if is_o
                        else shared_qkv_sign
                    ),
                    svh_sign=(
                        shared_o_output_sign
                        if is_o
                        else _sign(weight.shape[0], args.seed, layer, "attention", projection, "svh")
                    ),
                    k_block_scales=o_k_scales if is_o else qkv_k_scales,
                    n_block_scales=scale_family_candidates(
                        _block_values(weight.float().pow(2).mean(dim=1).numpy(), 128)
                    )["per128-grid"],
                    mass=float(rows),
                )
            )
        fit = fit_layer_absolute_normalization(
            backend.core,
            matrices,
            codebook_scale=float(backend.codebook_scale),
            block=128,
        )
        scales = {}
        gss_receipts = {}
        for key, target in sorted(fit.gss_targets().items()):
            request = PinnedGSSRequest(
                matrix_key=f"L{layer}.self_attn.{key}",
                bits=4,
                target=target,
                target_sha256=tensor_sha256(target),
                source_weight_identity_sha256=tensor_identity_sha256(fit.matrices[key].source.weight_kn),
                predecessor_checkpoint_hash="0" * 64,
            )
            result = producer.search(request)
            scales[key] = float(result.scale)
            gss_receipts[key] = dict(result.receipt)
        finalized = fit.finalize(scales)
        requests = []
        for projection in ATTENTION:
            value = finalized.matrices[projection]
            weight = source[projection]
            requests.append(
                {
                    "tensor_id": codec._parse_unit(
                        f"L{layer}.self_attn.{projection}", tuple(weight.shape)
                    ),
                    "weight_hf": weight,
                    "covariance": o_covariance if projection == "o_proj" else qkv_covariance,
                    "bits": (4,),
                    "suh": value.stored_suh,
                    "svh": value.stored_svh,
                    "sigma_reg": codec.sigma_reg,
                    "provenance": {
                        "attention_k4_arm": True,
                        "layer": layer,
                        "projection": projection,
                        "capture_sha256": capture_hashes,
                    },
                }
            )
        encoded = backend.encode_group(requests)
        tensors = {}
        records = []
        for projection, item in zip(ATTENTION, encoded, strict=True):
            candidate = item[4]
            reconstructed = candidate.reconstructed_kn.T.detach().to(torch.bfloat16).cpu().contiguous()
            tensors[f"K4.{projection}.reconstruction_hf"] = reconstructed
            tensors[f"K4.{projection}.packed_trellis"] = candidate.trellis.detach().cpu().contiguous()
            tensors[f"K4.{projection}.suh"] = finalized.matrices[projection].stored_suh.detach().cpu().contiguous()
            tensors[f"K4.{projection}.svh"] = finalized.matrices[projection].stored_svh.detach().cpu().contiguous()
            records.append(
                {
                    "projection": projection,
                    "bits": 4,
                    "source_shape": list(source[projection].shape),
                    "packed_sha256": str(candidate.packed_sha256),
                    "codec_fp16_reconstruction_sha256": str(candidate.reconstruction_sha256),
                    "stored_bf16_reconstruction_sha256": tensor_sha256(reconstructed),
                    "gss_receipt_sha256": gss_receipts[projection]["receipt_sha256"],
                }
            )
        tensor_path = destination / "attention-k4.safetensors"
        save_file(tensors, tensor_path)
        receipt = {
            "schema": "quant-pipeline.qwen-attention-k4-encode.v1",
            "layer": layer,
            "source_revision": source_revision,
            "source_checkpoint_identity": source_identity,
            "capture_sha256": capture_hashes,
            "capture_rows": rows,
            "bits": 4,
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": sha256_file(tensor_path),
            "tensor_file_bytes": tensor_path.stat().st_size,
            "records": records,
            "codec_identity_sha256": sha256_bytes(canonical_json(codec.identity)),
            "elapsed_seconds": time.monotonic() - layer_started,
        }
        receipt["receipt_sha256"] = _hash_json(receipt)
        write_json(destination / "encode-receipt.json", receipt)
        all_records.append(
            {
                "layer": layer,
                "receipt_sha256": receipt["receipt_sha256"],
                "tensor_file_sha256": receipt["tensor_file_sha256"],
                "tensor_file_bytes": receipt["tensor_file_bytes"],
            }
        )
        reader._cache.clear()
        print(
            json.dumps(
                {
                    "stage": "attention-k4-encode",
                    "layer": layer,
                    "elapsed_seconds": receipt["elapsed_seconds"],
                    "total_elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    range_receipt = {
        "schema": "quant-pipeline.qwen-attention-k4-range.v1",
        "plan": plan | {"dry_run": False},
        "records": all_records,
        "elapsed_seconds": time.monotonic() - started,
    }
    range_receipt["receipt_sha256"] = _hash_json(range_receipt)
    write_json(
        output / f"range-{args.first_layer:03d}-{args.first_layer + args.layers - 1:03d}.json",
        range_receipt,
    )
    print(json.dumps({"ok": True, **range_receipt}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
