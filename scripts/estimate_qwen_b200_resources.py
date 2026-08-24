#!/usr/bin/env python3
"""Conservative storage/VRAM/CPU estimator for the 48x128 Qwen pilot."""

from __future__ import annotations

import argparse
import json


GIB = 1024**3


def estimate(*, retention: str, fit: int, selection: int, confirmation: int, final: int, tokens: int) -> dict:
    layers, experts, top_k, hidden, intermediate = 48, 128, 8, 2048, 768
    parameters = 30_500_000_000
    source_bf16 = parameters * 2
    # Three retained p0/p1/p2 raw-second-moment arms in float64 for both input geometries.
    covariance_all_layers = layers * experts * (hidden**2 + intermediate**2) * 8 * 3
    covariance_one_layer = experts * (hidden**2 + intermediate**2) * 8 * 3
    routed_windows = fit + selection + confirmation
    routed_rows = routed_windows * tokens * layers * top_k
    routed_capture = routed_rows * (hidden + intermediate) * 2
    # The rank sketch is downstream per token/layer, not duplicated for every
    # routed expert. capture-plus-ledger retains one layer's working sketch;
    # full mode retains the complete model-wide sketch.
    fisher_rows = selection * tokens * layers
    fisher_rank = 32
    fisher_all_layers = fisher_rows * hidden * fisher_rank * 2
    fisher = fisher_all_layers if retention == "full" else fisher_all_layers // layers
    teacher_student_logits = 2 * final * tokens * 151_936 * 2
    exact_payload_and_checkpoint = 2 * 24 * GIB
    stage_overhead = 32 * GIB
    retained_covariance = covariance_all_layers if retention == "full" else covariance_one_layer
    peak_disk = (
        source_bf16
        + retained_covariance
        + routed_capture
        + fisher
        + teacher_student_logits
        + exact_payload_and_checkpoint
        + stage_overhead
    )
    per_gpu = source_bf16 + 20 * GIB + 12 * GIB
    host_ram = source_bf16 + 2 * covariance_one_layer + 48 * GIB
    return {
        "schema": "quant-pipeline.qwen-resource-estimate.v1",
        "geometry": {
            "layers": layers,
            "experts": experts,
            "top_k": top_k,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "parameters": parameters,
        },
        "windows": {
            "fit": fit,
            "selection": selection,
            "confirmation": confirmation,
            "final": final,
            "tokens": tokens,
        },
        "retention_mode": retention,
        "components_bytes": {
            "source_bf16": source_bf16,
            "retained_covariance": retained_covariance,
            "routed_capture": routed_capture,
            "fisher_rank32": fisher,
            "teacher_student_logits": teacher_student_logits,
            "exact_payload_checkpoint_staging": exact_payload_and_checkpoint,
            "stage_overhead": stage_overhead,
        },
        "recommended": {
            "estimated_peak_disk_bytes": peak_disk,
            "safety_margin_bytes": max(100 * GIB, peak_disk // 6),
            "minimum_free_vram_bytes_per_gpu": per_gpu,
            "minimum_available_host_ram_bytes": host_ram,
            "minimum_cpu_count": 32,
        },
        "assumptions": [
            "conservative uncompressed routed captures",
            "rank-32 FP16 Fisher sketches",
            "one BF16 replica plus 32 GiB workspace per B200",
            "container padding and temporary compiler cache excluded from model bitrate but included by safety margin"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention", choices=("full", "capture-plus-ledger"), default="capture-plus-ledger")
    parser.add_argument("--fit-windows", type=int, default=32)
    parser.add_argument("--selection-windows", type=int, default=16)
    parser.add_argument("--confirmation-windows", type=int, default=16)
    parser.add_argument("--final-windows", type=int, default=25)
    parser.add_argument("--window-tokens", type=int, default=2048)
    args = parser.parse_args()
    values = (args.fit_windows, args.selection_windows, args.confirmation_windows, args.final_windows, args.window_tokens)
    if any(value <= 0 for value in values):
        raise SystemExit("window counts and token count must be positive")
    print(json.dumps(estimate(
        retention=args.retention,
        fit=args.fit_windows,
        selection=args.selection_windows,
        confirmation=args.confirmation_windows,
        final=args.final_windows,
        tokens=args.window_tokens,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
