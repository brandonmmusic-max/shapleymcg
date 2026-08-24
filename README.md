# Quant Pipeline

Auditable foundations for a calibration, attribution, allocation, encoding and
validation pipeline for low-bit dense and MoE language models.

## Status: first full Qwen/B200 control measured and independently verified

This repository now contains local implementations of the calibration fitter,
canonical source-derived absolute-v31/GSS artifacts, exact-codec candidate
ledger, fixed-cost allocator, causal campaign runner, Qwen routed capture,
internal checkpoint assembly, upstream BTX writer/auditor, and independent KLD
checks. These modules have local synthetic/tiny-model tests. The first full
48-layer two-B200 Qwen fixed-Hadamard K3/K4 control has now completed at an
exact 3.5-bit MoE expert-weight rate. Its reconstructed-BF16 replay measured
mean next-token KLD `0.06335949321311507` over 2,047 sealed WikiText positions.
Every selected matrix payload and the token KLD were independently reverified.
See the [complete result and provenance ledger](docs/QWEN_B200_CONTROL_RESULT.md).

This is a validation-model result, not SM100 packed-runtime qualification. The
published model stores the measured reconstructions in BF16 because the
current official BTX format cannot express this control's independent gate/up
choices. No CUDA-graph, throughput, or compact-checkpoint claim is made.

The production example under `configs/qwen3-30b-a3b-b200` now names the
in-repository concrete Qwen service factory. Machine-specific corrected-codec,
B12X, budget, threshold, driver, and artifact hashes remain required
placeholders. They are deliberately fail-closed: a clone is a reproducibility
template, not an unreviewed launch configuration. The
[two-B200 runbook](docs/B200_RUNBOOK.md) stops
after a read-only preflight and requires fresh owner approval before execution.

The primary pilot is `Qwen/Qwen3-30B-A3B-Base`. It tests model-level
Aumann-Shapley path attribution and route-aware expert attribution in one MoE,
then compares the predicted choices with exact next-token KLD. Gemma 4 26B-A4B
is the portability model after the Qwen control is established.

The central rule is simple: proxy scores may propose candidates, but sealed
end-to-end KLD decides. The package preserves raw proxy predictions and their
closure residual instead of normalizing expert scores to look exact.

Implemented now:

- immutable experiment specs and canonical SHA256 receipts;
- a sealed, target-tokenized reproduction of the GLM WikiText-2 2,048-token KLD procedure;
- document-disjoint calibration/selection/confirmation/final corpus sealing;
- Qwen3-MoE and Gemma 4 expert inventory adapters;
- arbitrary 2-8 bit deterministic reference packing and reconstruction;
- a hash-bound adapter to the pinned corrected EXL3/MCG K3/K4/K5 encoder;
- multi-window KLD with tail metrics and paired bootstrap support;
- Aumann-Shapley quadrature and module-blend execution helpers;
- exact routed Qwen expert residuals with downstream Fisher/Jacobian sketches;
- cross-expert quadratic attribution with exact surrogate closure;
- explicit unresolved-remainder accounting;
- exact codec-payload-byte Pareto pruning and global multiple-choice knapsack allocation;
- a Shapley/Fisher-anchored allocation arm with an uncalibrated proxy control;
- a causal, resumable campaign runner with rollback/reallocation generations;
- canonical absolute-v31 plus pinned K3/K4/K5 GSS candidate artifacts;
- exact candidate payload/accounting ledgers and Qwen internal/official BTX emission;
- packed-shard manifests, hashes, reconstruction and audits; and
- guarded GPU/model commands requiring explicit `--execute`;
- independently hash-verified incremental Hugging Face publication of fitted
  Hessian and exact candidate layers; and
- a fail-closed Qwen control-bundle sealer that requires all 48 encode exits,
  exact KLD, and all remote layer receipts before publication.

The concrete service factory captures independent fit, held-out, conditional
down-fit, and optional supplemental streams; fits direct routed raw second
moments with per-layer exact route-mass receipts; evaluates the five historical
permutation controls crossed with the three historical scale families through
canonical absolute-v31 and pinned K3/K4/K5 GSS; emits exact candidate payloads;
keeps research and official-BTX-compatible allocation arms separate; and
installs layers causally before pinned BTX emission/audit. These paths are
implemented. The fast fixed-Hadamard K3/K4 control is now qualified for
reconstructed-BF16 KLD on real Qwen/B200 hardware; the full Shapley allocation
and native BTX/runtime arms remain separate future gates.

The native causal attribution producer is checked in. The candidate stage
persists an explicit actual-codec provisional control and exact decoded-minus-
source deltas; the attribution stage independently loads the pinned local Qwen,
sealed KLD window, and sealed teacher logits. At every Gauss-Legendre node it
substitutes decoded full-expert outputs into the complete model under
differentiable per-layer blends, measures next-token KL gradients, and projects
exact routed expert residuals through score-function Fisher VJPs. Cross-expert
and routing/backend residuals remain explicit. A tiny real PyTorch Qwen MoE
fixture validates direction, closure, source control, non-identity permutations,
and tamper failure. The full native causal-attribution arm remains pending a
real-Qwen comparison against the published fixed-Hadamard control.

Reconstructed-BF16 re-anchor KLD and native-BTX final student KLD remain
separate acceptance gates.

The reference uniform codec proves the pipeline mechanics. It is not presented
as a quality competitor. Competitive experiments use `Exl3MCGCodec`, which
loads the explicitly supplied, hash-bound corrected encoder/numeric core and
SM-specific extension. It requires matrix dimensions divisible by 128, which
Qwen3-30B-A3B satisfies and Gemma 4's 704-wide experts do not. Native runtime
qualification remains a separate gate.

See [the full pipeline](docs/PIPELINE.md), [scientific gates](docs/SCIENTIFIC_METHOD.md),
[model decision](docs/MODEL_SELECTION.md), [scientific lineage and references](docs/REFERENCES.md),
and [two-B200 runbook](docs/B200_RUNBOOK.md).

## Reproduce from source to sealed preflight

1. Clone a specific Git commit and create the pinned Python environment in
   [B200 runbook section 1](docs/B200_RUNBOOK.md#1-create-an-isolated-environment).
2. Fetch and seal the exact Qwen BF16 revision, fetch and hash-verify the pinned
   corrected R10/EXL3 source closure, prepare the pinned B12X checkout, and
   build the hash-bound encoder extension using sections 2-3 of the runbook.
3. Generate and preserve the resource estimate; copy
   `configs/qwen3-30b-a3b-b200` to durable storage.
4. Resolve every `__REQUIRED_...__` value from measured local artifacts. Run
   `scripts/validate_repro_config.py` without `--allow-placeholders`; unresolved
   identities are a hard failure.
5. Seal the target-tokenized WikiText control, the documented `id`/`domain`/
   `text` JSONL corpus into document-disjoint fit/selection/confirmation/final
   roles, and BF16 teacher logits as described in
   [pipeline stage 0](docs/PIPELINE.md#stage-0-freeze-identity-and-repair-the-instrument).
6. Create and audit the immutable campaign plan. Run the read-only host/GPU
   preflight and preserve `plan.json`, artifact locks, environment locks,
   resource estimate, and `preflight-review.json`.
7. Stop at the owner-approval barrier. When separately approved, execute the
   causal campaign and preserve every journal, exact payload, checkpoint audit,
   student-logit capture, KLD report, and SHA256 receipt.

This ordering is intentional. Tiny-model/unit tests establish implementation
closure; only a sealed full-model run establishes Qwen quality or deployability.

## Development

```bash
python3 -m pip install -e '.[hf,test]'
pytest -q
```

Clean-node configuration and non-launching checks are documented in
`docs/B200_RUNBOOK.md`. Packaging includes the example configurations,
environment manifests, scripts, and documentation for offline review.

Source-available under the repository's attribution-required license. See
`LICENSE`, `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_LICENSES`, and
`docs/REFERENCES.md`. The Apache-2.0 notice on the derived BTX writer controls
that module.
