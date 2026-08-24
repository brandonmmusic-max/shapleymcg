#!/usr/bin/env python3
"""Reproduce ExLlamaV3 v0.0.1's routed-expert allocation at 3.5 BPW.

The legacy allocator treats every expert in a projection as one fused group.
For Qwen3-30B-A3B the gate, up and down expert matrices have equal numbers of
weights, so its carried-surplus rule alternates K3/K3/K4 and K3/K4/K4 across
successive transformer layers.  This script executes the original arithmetic
rather than hard-coding that consequence and emits the same per-matrix choice
shape consumed by the matched K3/K4 reconstruction evaluator.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, write_json


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PERMUTATIONS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
UPSTREAM_REVISION = "ae04741f22324cc746ab78c27365e53e3f9f1cf4"


def allocate_layer(*, bpw: float, matrix_numel: int, surplus_bits: int) -> tuple[tuple[int, int, int], int, int]:
    """Execute v0.0.1 ``allocate_transformer`` for the expert-only scope."""

    numels = (matrix_numel, matrix_numel, matrix_numel)
    total = sum(numels)
    budget = int(bpw * total) + surplus_bits + 1
    base_bpw = max(int(math.floor(budget / total)), 1)
    permutations = [tuple(min(8, offset + base_bpw) for offset in row) for row in PERMUTATIONS]
    options = sorted((sum(bit * size for bit, size in zip(row, numels)), row) for row in permutations)
    index = max(0, bisect.bisect_right(options, (budget,)) - 1)
    used, selected = options[index]
    return selected, budget - used, used


def build(*, layers: int, experts: int, matrix_numel: int, bpw: float) -> dict:
    surplus = 0
    choices = []
    layer_rows = []
    total_used = 0
    for layer in range(layers):
        selected, next_surplus, used = allocate_layer(
            bpw=bpw,
            matrix_numel=matrix_numel * experts,
            surplus_bits=surplus,
        )
        for expert in range(experts):
            for projection, bits in zip(PROJECTIONS, selected):
                choices.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "bits": bits,
                    }
                )
        layer_rows.append(
            {
                "layer": layer,
                "projection_bits": dict(zip(PROJECTIONS, selected)),
                "surplus_bits_before": surplus,
                "surplus_bits_after": next_surplus,
                "used_bits": used,
            }
        )
        surplus = next_surplus
        total_used += used

    k3 = sum(row["bits"] == 3 for row in choices)
    k4 = sum(row["bits"] == 4 for row in choices)
    unit_count = len(choices)
    logical_bpw = sum(row["bits"] for row in choices) / unit_count
    body = {
        "schema": "quant-pipeline.turboderp-v001-expert-allocation.v1",
        "objective": "verbatim-exllamav3-v0.0.1-carried-surplus-rule",
        "upstream_repository": "https://github.com/turboderp-org/exllamav3",
        "upstream_revision": UPSTREAM_REVISION,
        "target_bpw": bpw,
        "average_weight_bits": logical_bpw,
        "rate_scope": "moe-expert-weight-elements",
        "layers": layers,
        "experts_per_layer": experts,
        "matrix_numel": matrix_numel,
        "unit_count": unit_count,
        "k3_count": k3,
        "k4_count": k4,
        "total_used_bits": total_used,
        "terminal_surplus_bits": surplus,
        "allocator_granularity": "one fused all-expert group per projection per layer",
        "layer_allocations": layer_rows,
        "choices": choices,
    }
    if bpw == 3.5 and layers % 2 == 0 and (k3 != unit_count // 2 or k4 != unit_count // 2):
        raise ValueError("v0.0.1 Qwen 3.5 allocation did not reach exact half K3/half K4")
    body["allocation_sha256"] = sha256_bytes(canonical_json(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--matrix-numel", type=int, default=2048 * 768)
    parser.add_argument("--bpw", type=float, default=3.5)
    args = parser.parse_args()
    if args.layers < 1 or args.experts < 1 or args.matrix_numel < 1:
        raise SystemExit("layers, experts, and matrix-numel must be positive")
    document = build(
        layers=args.layers,
        experts=args.experts,
        matrix_numel=args.matrix_numel,
        bpw=args.bpw,
    )
    write_json(args.output.resolve(), document)
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
