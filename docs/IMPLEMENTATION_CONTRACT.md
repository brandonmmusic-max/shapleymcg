# Full-pipeline implementation contract

This contract freezes interfaces for the three parallel implementation streams.
It is local-code-only until the owner explicitly authorizes B200 execution and
publication.

## Shared invariants

- Every persisted artifact is schema-versioned, canonical-JSON hash bound, and
  written only into an empty destination or by atomic replacement.
- Non-finite numeric values, missing identities, malformed dimensions, and
  backend substitutions fail closed.
- A real campaign requires immutable 40-hex model and dataset revisions.
- The historical control is the sealed Qwen-tokenized 2,048-token WikiText
  window with role `kld`; it is never calibration data.
- Fit, selection, confirmation, and final windows remain document-disjoint.
- `plan` performs no model/GPU writes. `execute` requires explicit approval.
- Resume is permitted only when the spec, code, source checkpoint, corpus, and
  predecessor-state identities match the journal.

## Fitter handoff

Module owner: `quant_pipeline.calibration.fitter`.

Input records are batches containing expert inputs `[rows, hidden]`, expert
IDs, FP32 route weights, document IDs, token offsets, layer ID, and predecessor
checkpoint hash. The fitter streams mergeable float64 sufficient statistics
for route-weight powers 0, 1, and 2. It emits one artifact per
layer/expert/projection with count, document count, weight sums, mean,
second-moment/covariance, shrinkage coefficient, regularized covariance,
natural-versus-supplemental accounting, and source identities.

Public operations: update, merge, finalize, save, load/verify. Merge must be
order-independent within declared floating tolerance and reject identity drift.

## Exact-codec candidate-ledger handoff

Module owner: `quant_pipeline.candidates.ledger`.

The generator accepts fitted statistics, exact BF16 projection tensors,
routed held-out expert-function batches, and a `QuantCodec`. Competitive mode
must require a backend attestation identifying the corrected EXL3/MCG encoder;
the uniform codec is test-only and cannot produce a competitive ledger.

Each candidate records layer/expert, `(gate, up, down)` bit triplet, objective
arm, exact codec payload bytes, packed and reconstruction hashes, transform
identities, absolute gate-squared output SSE, relative and energy-normalized
SSE, signed aggregate error, interaction term, route agreement, Fisher/Jacobian
projection if present, and finite-value validation. K3/K4 enumeration is all
eight triplets; K5 admission is explicit and journaled. Per-layer Pareto
frontiers and the global allocator handoff use the existing `Candidate` fields
plus complete metadata.

## Causal runner handoff

Module owner: `quant_pipeline.campaign.runner` and its CLI wiring.

Modes are `plan`, `execute`, `resume`, `status`, and `audit`. The append-only
event journal records stage start/completion/failure, input and output hashes,
predecessor state, and attempt number. Stages are identity, teacher capture,
fit capture, fit, candidates, attribution, allocation, causal encode, KLD
re-anchor, checkpoint emission/audit, student capture, and final KLD.

Every successor layer depends on the installed predecessor-state hash. A
re-anchor is required at least every four accepted layers. The declared policy
must either continue, rollback, or request reallocation; it may not silently
ignore a failed gate. External stage adapters may invoke real model/codec work,
but tests use deterministic local adapters. The runner never fabricates model
outputs.

## Integration boundary

Implementers may add package `__init__.py` files and their own tests but must
not edit another stream's owned module. The integration agent reconciles CLI,
spec fields, schemas, documentation, and end-to-end tests after all three
handoffs freeze.
