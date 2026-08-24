#!/usr/bin/env python3
"""Encode one fitted Qwen layer into sealed K3/K4 reconstructed candidates.

This is the speed-to-KLD arm.  It retains the calibrated full-p2 Hessians,
source-derived absolute-v31 normalization, per-matrix pinned GSS, and corrected
R10 EXL3/MCG codec, while deferring the 15-arm proposal search and K5 research
arm.  The two reconstructions let a later global allocator choose an exact
3.5-bpw K3/K4 mixture without repeating an encode.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import time

import numpy as np

from quant_pipeline.campaign.qwen_services import CorrectedPinnedGSSProducer
from quant_pipeline.campaign.qwen_work_units import _CheckpointWeights, _fit_rows, _proposal
from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec
from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.normalization.absolute_v31 import fit_layer_absolute_normalization
from quant_pipeline.normalization.artifact_v31 import PinnedGSSRequest, tensor_identity_sha256, tensor_sha256
from quant_pipeline.normalization.prior_search import PERMUTATION_POLICIES, SCALE_FAMILIES


ZERO_HASH = "0" * 64
SOURCE_IDENTITY = "373a35d7f97f49444f376a9fd0ce4b2c0a0020754a43c887eb7b7035834ff476"
_KEY = re.compile(r"E(\d+)\.(gate_proj|up_proj|down_proj)")


def _hash_json(value) -> str:
    return sha256_bytes(canonical_json(value))


def _matrix_vectors(value):
    if value.projection == "down_proj":
        return value.stored_suh, value.stored_svh
    return value.stored_suh, value.stored_svh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--numeric-core", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy", choices=PERMUTATION_POLICIES, default="energy_balanced")
    parser.add_argument("--scale-family", choices=SCALE_FAMILIES, default="per128-grid")
    parser.add_argument("--seed", default="eeb56e706c26ef3d84d6fc317fceea61181944107a2295ee2f6d34cc959250aa")
    parser.add_argument("--expert-limit", type=int)
    parser.add_argument("--encode-group-size", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.layer not in range(48):
        parser.error("--layer must be in [0, 47]")
    fit_path = args.fit_root.resolve() / f"layer-{args.layer:03d}" / "fit-manifest.json"
    output = args.output.resolve()
    plan = {
        "schema": "quant-pipeline.qwen-fast-k34-encode-plan.v1",
        "model": str(args.model.resolve()),
        "fit_manifest": str(fit_path),
        "fit_manifest_sha256": sha256_file(fit_path),
        "layer": args.layer,
        "output": str(output),
        "policy": args.policy,
        "scale_family": args.scale_family,
        "transform_seed_sha256": args.seed,
        "bits": [3, 4],
        "device": args.device,
        "expert_limit": args.expert_limit,
        "encode_group_size": args.encode_group_size,
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0
    if args.encode_group_size < 1:
        parser.error("--encode-group-size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "encode-plan.json", plan | {"dry_run": False})

    import torch
    from safetensors.torch import save_file

    started = time.monotonic()
    fits = _fit_rows(fit_path, args.layer)
    if args.expert_limit is not None:
        if args.expert_limit < 1:
            parser.error("--expert-limit must be positive")
        fits = {expert: fits[expert] for expert in sorted(fits)[: args.expert_limit]}
    reader = _CheckpointWeights(args.model)
    source = {expert: reader.expert(args.layer, expert) for expert in sorted(fits)}
    matrices, proposal_source, proposal_fits = _proposal(
        layer=args.layer,
        policy=args.policy,
        family=args.scale_family,
        source=source,
        fits=fits,
        seed=args.seed,
    )
    # The absolute-v31 primitives and pinned GSS extension operate on CUDA.
    # Keep the source/checkpoint carrier on CPU, but place only the current
    # layer's normalization matrices on the selected B200.
    matrices = [replace(item, weight_kn=item.weight_kn.to(args.device)) for item in matrices]
    print(json.dumps({
        "stage": "proposal-ready", "layer": args.layer, "matrices": len(matrices),
        "elapsed_seconds": time.monotonic() - started,
    }, sort_keys=True), flush=True)
    codec = Exl3MCGCodec(
        source_root=args.source_root,
        numeric_core=args.numeric_core,
        extension=args.extension,
        device=args.device,
    )
    backend = codec._codec()
    producer = CorrectedPinnedGSSProducer(codec)
    tensors: dict[str, torch.Tensor] = {}
    scores = []
    bit_receipts = {}
    for bit in (3, 4):
        bit_started = time.monotonic()
        fit = fit_layer_absolute_normalization(
            backend.core,
            [replace(item, bits=bit) for item in matrices],
            codebook_scale=float(backend.codebook_scale),
            block=128,
        )
        targets = fit.gss_targets()
        scales = {}
        receipts = {}
        for key in sorted(targets):
            target = targets[key]
            source_identity = tensor_identity_sha256(fit.matrices[key].source.weight_kn)
            request = PinnedGSSRequest(
                matrix_key=key,
                bits=bit,
                target=target,
                target_sha256=tensor_sha256(target),
                source_weight_identity_sha256=source_identity,
                predecessor_checkpoint_hash=ZERO_HASH,
            )
            result = producer.search(request)
            scales[key] = float(result.scale)
            receipts[key] = dict(result.receipt)
        finalized = fit.finalize(scales)
        print(json.dumps({
            "stage": "gss-ready", "layer": args.layer, "bits": bit,
            "matrices": len(scales), "elapsed_seconds": time.monotonic() - bit_started,
        }, sort_keys=True), flush=True)
        ordered_keys = sorted(finalized.matrices)
        requests = []
        request_metadata = []
        for key in ordered_keys:
            match = _KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"malformed matrix key {key}")
            expert = int(match[1])
            projection = match[2]
            source_weight = getattr(proposal_source[expert], projection)
            fitted = proposal_fits[expert][0 if projection != "down_proj" else 1]
            value = finalized.matrices[key]
            suh, svh = _matrix_vectors(value)
            unit_id = f"L{args.layer}.E{expert}.{projection}"
            # The corrected R10 bridge consumes the raw uncentered routed p2
            # Gram.  Damping belongs exclusively to codec-level sigma_reg;
            # applying OAS here would change both the experiment and result.
            covariance = fitted.dense_hessian("combined", 2, regularized=False)
            damage_diagonal = torch.as_tensor(
                np.diag(covariance).copy(),
                device=args.device,
                dtype=torch.float32,
            )
            requests.append({
                "tensor_id": codec._parse_unit(unit_id, tuple(source_weight.shape)),
                "weight_hf": source_weight,
                "covariance": covariance,
                "bits": (bit,),
                "suh": suh,
                "svh": svh,
                "sigma_reg": codec.sigma_reg,
                "provenance": {
                    "fast_kld_arm": True,
                    "policy": args.policy,
                    "scale_family": args.scale_family,
                    "fit_manifest_sha256": plan["fit_manifest_sha256"],
                },
            })
            request_metadata.append((key, expert, projection, suh, svh, damage_diagonal))
        # R10's native group encoder prepares/factorizes each matrix once and
        # locksteps all equal-bit LDLQ walks.  This is the intended high-GPU-
        # occupancy path and avoids hundreds of tiny serial codec launches.
        encoded_group = []
        for start in range(0, len(requests), args.encode_group_size):
            stop = min(start + args.encode_group_size, len(requests))
            encoded_group.extend(backend.encode_group(requests[start:stop]))
            if stop == len(requests) or stop % (args.encode_group_size * 8) == 0:
                print(json.dumps({
                    "stage": "encode", "layer": args.layer, "bits": bit,
                    "completed_matrices": stop, "matrix_count": len(requests),
                    "elapsed_seconds": time.monotonic() - bit_started,
                }, sort_keys=True), flush=True)
        for metadata, encoded in zip(request_metadata, encoded_group, strict=True):
            key, expert, projection, suh, svh, damage_diagonal = metadata
            candidate = encoded[bit]
            reconstructed_hf = candidate.reconstructed_kn.T.detach().to(torch.bfloat16)
            reconstructed = reconstructed_hf.to(device="cpu").contiguous()
            packed = candidate.trellis.detach().cpu().contiguous()
            # The source matrix already lives on the selected GPU as the
            # normalization input.  Score the exact stored-BF16 replay there;
            # avoid a BF16 CPU round trip and a second dense-Hessian rebuild.
            source_hf = fit.matrices[key].source.weight_kn.T
            residual = source_hf.float() - reconstructed_hf.float()
            damage = float(
                (residual.square() * damage_diagonal.unsqueeze(0)).sum().item()
            )
            prefix = f"K{bit}.E{expert:03d}.{projection}"
            tensors[prefix + ".reconstruction_hf"] = reconstructed
            tensors[prefix + ".packed_trellis"] = packed
            tensors[prefix + ".suh"] = suh.detach().cpu().contiguous()
            tensors[prefix + ".svh"] = svh.detach().cpu().contiguous()
            scores.append({
                "layer": args.layer,
                "expert": expert,
                "projection": projection,
                "bits": bit,
                "diagonal_p2_damage": damage,
                "stored_bytes": int(
                    packed.numel() * packed.element_size()
                    + suh.numel() * suh.element_size()
                    + svh.numel() * svh.element_size()
                ),
                "packed_sha256": str(candidate.packed_sha256),
                # The corrected R10 codec preserves its historical FP16
                # reconstruction identity.  Replay uses the exact BF16 tensor
                # persisted above, so seal that representation independently.
                "codec_fp16_reconstruction_sha256": str(candidate.reconstruction_sha256),
                "stored_bf16_reconstruction_sha256": tensor_sha256(reconstructed),
                "gss_receipt_sha256": receipts[key]["receipt_sha256"],
            })
            del residual, reconstructed_hf, damage_diagonal
        bit_receipts[str(bit)] = {
            "receipt_sha256": _hash_json(receipts),
            "matrix_count": len(receipts),
            "elapsed_seconds": time.monotonic() - bit_started,
        }
    tensor_path = output / "k34-candidates.safetensors"
    save_file(tensors, tensor_path)
    body = {
        "schema": "quant-pipeline.qwen-fast-k34-encode.v2",
        "plan_sha256": sha256_file(output / "encode-plan.json"),
        "source_checkpoint_identity": SOURCE_IDENTITY,
        "layer": args.layer,
        "experts": sorted(fits),
        "matrix_count": len(matrices),
        "candidate_count": len(scores),
        "candidate_tensor_file": tensor_path.name,
        "candidate_tensor_sha256": sha256_file(tensor_path),
        "candidate_tensor_bytes": tensor_path.stat().st_size,
        "bit_receipts": bit_receipts,
        "scores": scores,
        "elapsed_seconds": time.monotonic() - started,
    }
    body["receipt_sha256"] = _hash_json(body)
    write_json(output / "encode-receipt.json", body)
    print(json.dumps({
        "ok": True,
        "layer": args.layer,
        "experts": len(fits),
        "candidate_count": len(scores),
        "elapsed_seconds": body["elapsed_seconds"],
        "receipt_sha256": body["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
