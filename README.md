# Quant Pipeline

Auditable foundations for a calibration, attribution, allocation, encoding and
validation pipeline for low-bit dense and MoE language models.

## Post-trained Qwen/TurboDerp matched comparison

The same-parent, same-panel predecessor-pipeline full-scope experiment is
complete. Its candidates use the ShapleyMCG MCG encoder, but its exact-rate
selection uses the historical diagonal routed-p2 Hessian/router allocator—not
the full Aumann–Shapley/Fisher allocator. At exactly 3.5 logical bits per
routed-expert weight, that predecessor allocation reduced
KLD by 34.7513% versus five score-blind allocations, but its independently
encoded R10/MCG reconstruction did not beat the reconstruction assembled from
published TurboDerp checkpoints. Adding ShapleyMCG K4 attention
increased KLD from `0.04112263218133531` to `0.046834114392727964`; the matched
TurboDerp-checkpoint-reconstruction arm measured `0.030274917976982833`. At
uniform K4 the corresponding full-body values were `0.03776677825098351` and
`0.02137210911467856`. These results establish allocation quality when the
candidate reconstructions are held fixed. They do **not** isolate a codebook:
the R10 arm also changes calibration tokens, Hessian state, rotations, scaling,
and encoding policy relative to the published TurboDerp checkpoints. See the
[matched comparison report](docs/QWEN_POSTTRAINED_TURBODERP_COMPARISON.md) and
[compact sealed record](results/qwen3-30b-a3b-posttrained/fullscope-summary.json).

The full post-trained causal allocator has now been measured separately. On
the identical 20,480-position eager panel at the identical exact 3.5 expert
BPW, it lowered MCG KLD from the predecessor's `0.04112263218133531` to
`0.040368551745534186`, a **1.833736% reduction**, while top-1 agreement rose
by **0.141602 percentage points**. Independent float64 replay had zero maximum
tokenwise difference. This is evidence for the complete allocation method; it
does not turn the separately encoded R10-versus-TurboDerp pipeline comparison
into a codec-only ablation. See the
[complete result ledger](docs/QWEN_COMPLETE_RESULTS_LEDGER.md).

The causal choices also transfer to TurboDerp's unchanged K3/K4 checkpoint
reconstructions: with
the same post-trained parent, 20,480-position panel, K4 body/K6 head, and exact
3.5 expert rate, they lower KLD from `0.030274917976982833` to
`0.029269076647285147` (**3.322359%**) and gain **0.273438 percentage points**
of top-1 agreement. Replacing those experts with reconstructions produced by
the independently calibrated R10/MCG pipeline gives KLD
`0.04579310254291688`, 56.455576% higher. The allocator is therefore a
demonstrated improvement. The latter gap belongs to the complete encoding
pipelines and cannot be attributed to MCG, calibration, scaling, rotations, or
any other single component without another controlled experiment.

TurboDerp v0.0.1's own carried-surplus expert allocation has now been
reproduced over that same published K3/K4 candidate pool. At the identical
9,216 K3 plus 9,216 K4 matrix count, fixed K4 body/K6 head, BF16 teacher,
20,480-token panel, and evaluator, it measured KLD `0.033941535180377104`
versus ShapleyMCG's `0.029269076647285147`. ShapleyMCG therefore reduced KLD
by **13.766197%**, gained **0.600586 percentage points** of top-1 agreement,
and won all 10 evaluation rows. A 200,000-draw seeded row-block bootstrap gave
an absolute-KLD reduction interval of `[0.0035278602, 0.0059290927]`. This is
an allocator-only win over TurboDerp's rule on common reconstructed candidate
bytes; it is not a claim about a native TurboDerp 3.5 checkpoint, which was not
published. The sealed comparison is under
[`results/qwen3-30b-a3b-posttrained/turboderp-v001-allocation-proof`](results/qwen3-30b-a3b-posttrained/turboderp-v001-allocation-proof/comparison.json).

Candidate production is now being isolated from allocation. The controlled
factory-union protocol freezes the full causal bit allocation and tests
same-bit whole-layer reconstructions from the published TurboDerp pool and the
existing R10/MCG pool on a selection row, followed by nine untouched validation
rows. A separate progressive-state arm captures calibration activations from a
sealed causal reconstruction while continuing to encode immutable BF16 source
weights. The executable protocol and the planned matrix-level multi-factory
ledger are documented in
[candidate-factory selection](docs/CANDIDATE_FACTORY_SELECTION.md). The
model-agnostic `quant_pipeline.candidates.factory_union` boundary makes native
MCG coverage mandatory, admits upstream/ModelOpt-style proposals additively,
common-scores every reconstruction, and hands `(factory, rate)` alternatives
to the exact-byte allocator. The compatibility-safe
`quant_pipeline.candidates.factory_allocation` path first constructs a Pareto
frontier inside each layer/shared-transform domain, charges each shared payload
once, and then selects one complete domain/profile per layer globally. This
allows matrix-level factory mixing only when the sealed shared-transform
identity matches and prevents an undeployable mixture of incompatible MCG
rotations. A new model therefore never depends on prepublished upstream
candidates. No result claim is made until the untouched validation endpoint is
sealed.

## Base-model causal allocation result

The full two-level Aumann–Shapley/Fisher arm is now measured. At the identical
3.5 routed-expert BPW and with the parent, MCG candidates, BF16 teacher, tokens,
SDPA arithmetic, and all non-expert weights fixed, the causal allocation lowers
mean KLD from `0.04908888647295481` to `0.04529370272688347` over 20,480
positions: a **7.731248% reduction** with a **0.268555 percentage-point** top-1
gain. Independent NumPy float64 replay reproduced both per-token vectors with
zero maximum difference. A separate 2,047-position window gives the same
direction with an 11.896262% reduction. See the
[causal result and interpretation](docs/QWEN_CAUSAL_ALLOCATION_RESULT.md) and
the checked-in [20k comparison](results/qwen3-30b-a3b-base/causal-vs-historical-sdpa-20k.json).

The raw five-node layer attribution explained 69.58% of the measured uniform-K4
endpoint; the remaining 30.42% is published explicitly and reconciled in a
separate step. Exact ledger closure is not presented as raw proxy additivity.

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
Two broader comparison panels are also sealed: the reconstructed Hill-paper
BFCL/RULER panel measured `0.018260861970005038` over 32,752 positions, and the
TurboDerp/ExLlamaV3 WikiText-2 20k protocol measured `0.05005581795647327`
over 20,480 positions with `0.908447265625` top-1 agreement. The paper panel is
not an author-provided token panel, and the published TurboDerp quant uses the
post-trained Qwen parent rather than Base, so neither is labeled a strict
head-to-head result. The strict same-parent controls keep attention, routers,
and the head in BF16: uniform expert K3 measured `0.09943217778983483`, the
3.5-bpw selected mix measured `0.05005581795647327`, and uniform expert K4
measured `0.033991548914098856`. The selected mix is 24.9671% below the linear
K3/K4 endpoint midpoint and 13.8995% below the geometric endpoint midpoint;
those are descriptive interpolations, not measured naive allocations. Five
direct score-blind 3.5-bpw allocations measured mean KLD
`0.06997419465008492` (sample SD `0.010737570027276615`, range
`0.05595689276383418` to `0.08057943718460085`). The selected allocation is
28.4653% lower than that measured mean, improves top-1 agreement by 1.6816
percentage points, and was better than all five seeds. An independent PyTorch
`kl_div` replay agrees with the primary mixed score to about `2e-9` in the
mean. The compact five-seed record is checked in under
[`results/qwen3-30b-a3b-base`](results/qwen3-30b-a3b-base/naive-3p5-controls-summary.json);
full logits and tokenwise KLD are published on Hugging Face.
See the [complete result and provenance ledger](docs/QWEN_B200_CONTROL_RESULT.md).
The [current-state reproducibility audit](docs/QWEN_BASE_REPRODUCIBILITY_AUDIT.md)
maps each Base-control requirement to immutable GitHub and Hugging Face evidence.

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
implemented. The fixed-Hadamard K3/K4 control and full causal allocation are
now qualified for reconstructed-BF16 KLD on real Qwen/B200 hardware. Native
BTX/runtime qualification remains a separate future gate.

The native causal attribution producer is checked in. The candidate stage
persists an explicit actual-codec provisional control and exact decoded-minus-
source deltas; the attribution stage independently loads the pinned local Qwen,
sealed KLD window, and sealed teacher logits. At every Gauss-Legendre node it
substitutes decoded full-expert outputs into the complete model under
differentiable per-layer blends, measures next-token KL gradients, and projects
exact routed expert residuals through score-function Fisher VJPs. Cross-expert
and routing/backend residuals remain explicit. A tiny real PyTorch Qwen MoE
fixture validates direction, closure, source control, non-identity permutations,
and tamper failure. The full native causal-attribution arm has now run on real
Qwen, produced an exact-rate allocation, and improved on the historical
Hessian/router control in both sealed SDPA comparisons. The full attribution
and its unresolved remainder are published rather than normalized away.

Reconstructed-BF16 re-anchor KLD and native-BTX final student KLD remain
separate acceptance gates.

The reference uniform codec proves the pipeline mechanics. It is not presented
as a quality competitor. Competitive experiments use `Exl3MCGCodec`, which
loads the explicitly supplied, hash-bound corrected encoder/numeric core and
SM-specific extension. It requires matrix dimensions divisible by 128, which
Qwen3-30B-A3B satisfies and Gemma 4's 704-wide experts do not. Native runtime
qualification remains a separate gate.

See [the complete ShapleyMCG method](docs/SHAPLEYMCG_METHOD.md),
[the controlled TurboDerp and Shapley-paper superiority protocol](docs/TURBODERP_SHAPLEY_SUPERIORITY_PROTOCOL.md),
[the full pipeline](docs/PIPELINE.md), [scientific gates](docs/SCIENTIFIC_METHOD.md),
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
