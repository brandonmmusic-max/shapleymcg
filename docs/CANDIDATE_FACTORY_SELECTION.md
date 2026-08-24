# Candidate-factory union protocol

## Question and current evidence boundary

The matched post-trained Qwen result proves that the full ShapleyMCG allocator
beats TurboDerp v0.0.1's carried-surplus rule when both allocate the identical
published TurboDerp K3/K4 reconstruction pool. It does not prove that the
existing ShapleyMCG candidate generator produces better reconstructions.

Candidate generation and candidate allocation are therefore tested as separate
causal factors. The upstream EXL3 factory is an additional candidate source,
not a baseline to discard and not an attribution shortcut.

## Production architecture: candidate union is part of the method

The production pipeline does not choose one encoder family in advance. Its
candidate stage registers the native MCG factory as required and may register
ModelOpt, upstream EXL3, or future factories as additional proposers. For every
independently allocatable weight unit and requested rate, all available
factories write exact-byte, hash-addressed reconstruction proposals into one
ledger. The pipeline then:

1. scores every reconstruction through the same Shapley/Fisher/Jacobian
   instrument owned by this method;
2. applies only cross-fitted causal calibration factors learned outside the
   final evaluation rows;
3. gives the exact-byte allocator all `(factory, rate)` alternatives; and
4. accepts the reconstructed allocation only through untouched end-to-end KLD.

Factory-reported MSE, Hessian loss, or KLD may be retained as provenance but is
never an allocation objective. `quant_pipeline.candidates.factory_union`
defines and seals this model-agnostic boundary. The required native MCG factory
must cover every unit/rate pair, so a new model remains fully quantizable when
no upstream checkpoint or model-specific factory exists. Optional factories
are additive arrows in the candidate-selection quiver, not dependencies.

Every registered factory must also attest the same runtime payload-format hash
as the required native factory. An upstream MCG implementation can therefore
contribute directly. A ModelOpt or other foreign-format proposal must first be
materialized through a `modelopt-guided-mcg` adapter (so the resulting packed
candidate is still emitted by the pinned MCG runtime); otherwise it remains a
diagnostic oracle and is barred from allocation. This prevents a statistically
attractive but undeployable mixed checkpoint.

Runtime-format compatibility is necessary but not sufficient. MCG transforms
also create layer-shared payloads: gate/up and down candidates must agree on the
sealed shared-transform domain. Every proposal therefore carries a coupling
group, a shared-domain hash, and the exact fixed bytes for that domain.
`quant_pipeline.candidates.factory_allocation` builds the exact internal Pareto
frontier for each `(layer, shared domain)`, charging shared bytes once, then
runs the global exact-byte DP across layers. The allocator may combine native
and optional factory proposals within a layer only when they attest the same
shared-domain hash. Otherwise it chooses one complete factory/transform domain
for that layer. A flat per-matrix knapsack is explicitly unsafe when more than
one domain exists.

## Whole-layer factory-union experiment

`scripts/measure_qwen_candidate_factory_union.py` freezes all of the following:

- the post-trained parent and BF16 teacher;
- the 9,216 K3 plus 9,216 K4 routed-expert matrix choices produced by the full
  causal allocator;
- TurboDerp K4 non-expert body weights and K6 head;
- the evaluator, attention backend, token panel, and per-token KL definition;
- the exact bit assigned to every expert projection.

The TurboDerp reconstruction pool is installed first. On selection row 0, each
routed-expert layer is independently replaced with the same-bit reconstructions
from the existing R10/MCG pool. Each swap records its direct end-to-end tokenwise
KLD delta and a seeded paired 64-token block-bootstrap interval. Negative-delta
swaps are ranked and greedily admitted only when the recomputed mixture lowers
selection KLD.

Rows 1 through 9 are never used for factory selection. They measure the final
mixture and the reconstructed TurboDerp baseline. The held-out paired result,
not the selection-row improvement, decides whether the union generalized.

Factory choice is presently whole-layer. This is intentionally a low-dimensional
first test: it can show that one factory dominates, or that the pools are
complementary, without searching 18,432 individual matrix choices on one row.

### Measured result

The selection row admitted MCG layers 5, 32, 34, 40, and 46. Its diagnostic
mean KLD fell from `0.02013594786940779` to `0.016742801006509656`
(16.851190%), but that row was used for selection and is not an endpoint.

Across the nine untouched rows, the reconstructed TurboDerp-candidate baseline
measured `0.030283841053315566`; the selected union measured
`0.02936814480556666`, an absolute reduction of `0.0009156962477489026`
(3.023712%). The union won 5 of 9 rows. A seeded paired 64-token block bootstrap
gave `[-0.00011704741100369084, 0.0019815815832019573]` for baseline minus
union. Because that interval crosses zero, this is directional held-out
evidence that the pools contain complementary layer reconstructions. It does
not establish candidate-factory superiority at the 95% interval gate. The
correct production response is to retain both factories in the candidate
quiver and validate the eventual coupled allocation on a larger untouched
panel.

The report seal is
`1b8897431f07768405afc1d4d98d970acdec84733d2e3bacc49da301978492a1`;
the exact bit-allocation seal is
`51bf1caf86292968055a054b6b5afcdb36354fad05e1ede109303634fa7edc6e`;
and the factory-allocation seal is
`1cd0759dd810b137e701d32a36d04e3ac673e7588c4180aff655410fff25bc34`.
Independent Torch float64 replay verified the tokenwise result at `2e-11`
absolute tolerance; verification seal:
`865d8f696b29cd5b84e51910fbf4fa4632082b44d63e67c973507d3c7dce3397`.
The complete 39.64 GB, 136-file
[artifact tree](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility/tree/2de2d5e87e3493739784eec74b1446991b910614/results/qwen3-30b-a3b-posttrained-v1/candidate-factory-union-v1)
is remotely verified at Hugging Face dataset revision
`2de2d5e87e3493739784eec74b1446991b910614`; tree-manifest seal:
`c6029ca5c7d0cd54c00f633b498256605de2f7c97c73d82fe6f4a99c8ccae2c5`;
remote-verification receipt seal:
`0d6bc93ad6430524cf535159e3cabb8850d9faa10b8beba064cda0a87e31b730`.

## Progressive-state candidate arm

TurboDerp v0.0.1 gathers later-layer calibration states after earlier modules
have already been quantized. The original ShapleyMCG candidate pool was fitted
from source-model states. `scripts/run_qwen_progressive_candidate_experiment.sh`
therefore runs a separate controlled arm in which routed states come from a
sealed causal reconstruction while every encode still reads the immutable BF16
source weights. The capture checkpoint identity, source revision, allocation,
KLD report, reconstruction manifest, and logit verification are sealed into the
capture request.

This arm estimates the effect of calibration state separately from the
whole-layer factory union. It does not silently replace the original control.

The first progressive-state score uses the fast diagonal allocator and is
therefore a candidate-pipeline diagnostic, not the full method. The next gate
uses `rebase_qwen_allocation_candidate_factory.py` to preserve every K3/K4
choice from the already validated full causal allocation while rebinding its
payload bytes and reconstruction hashes to the progressive factory. This
frozen-rate rebase isolates candidate quality at the Shapley-selected rates.
Only after that ablation may the progressive candidates enter common scoring
and joint factory-plus-rate allocation. `build_qwen_candidate_inventory.py`
constructs the immutable, prefix-aware Hub inventory needed for both steps.

## Matrix-level candidate ledger

The competitive ledger keys each candidate by
`(layer, expert, projection, bits, factory, reconstruction_sha256)` and retain
the exact packed-byte cost. For candidates at the same bit rate:

1. Decode every candidate to the common BF16 comparison representation.
2. Score decoded-minus-source residuals with the same routed Hessian and
   downstream Fisher/Jacobian instrument, regardless of factory.
3. Use score-blind, cross-fitted whole-layer KLD swaps to estimate a shrunk
   factory/layer calibration factor for the local proxy. Never fit and report
   that factor on the same held-out rows.
4. Preserve raw proxy, calibrated proxy, uncertainty, factory identity, packed
   bytes, and reconstruction hash as separate ledger fields.
5. Build per-layer, per-shared-domain frontiers, then let the coupled global
   allocator choose both rate and factory under the exact byte budget.
   Candidates with equal rate and higher calibrated damage are dominated;
   different-rate candidates remain on the Pareto frontier. Shared payload
   costs are charged once per selected layer domain.
6. Reconstruct the chosen mixture and adjudicate it with sealed end-to-end KLD.

The direct whole-layer experiment is stronger evidence for those 48 aggregate
factory decisions than a local proxy. The proxy is still needed to make
fine-grained selection computationally tractable; measured aggregate effects
calibrate it and the untouched final panel verifies it.

`quant_pipeline.candidates.factory_calibration` implements the calibration as
leave-one-layer-out ridge regression of measured challenger-minus-reference
KLD on the common proxy delta. The map shrinks toward the identity (not toward
either factory), preserves out-of-fold residual uncertainty, and never consumes
the final evaluation rows. Strong measured effects can therefore influence
fine-grained candidate scores while noisy layer ties receive limited leverage.

## Claim gates

- **Allocator claim:** requires common candidate bytes and common rate.
- **Candidate-factory claim:** requires common allocator, bit choices, parent,
  body, teacher, tokens, and evaluator.
- **Progressive-state claim:** requires common source weights, encoder, policy,
  corpus, and evaluator; only the state-producing checkpoint may differ.
- **Combined-pipeline claim:** may be reported only after the selected union is
  reconstructed and wins on rows excluded from every selection or calibration
  fit.

Selection-row improvements, local Hessian damage, and reconstructed-BF16 replay
are diagnostic evidence. None substitutes for the untouched end-to-end KLD
gate.
