# Scientific-core hardening status

This implementation treats the proven GLM-5.2 3.5-bpw procedure as the
control. Shapley/MCG, Fisher, new transform draws, and Qwen-specific geometry
are additive. A control mechanism can become replaceable only after a matched
exact-codec and sealed end-to-end KLD ablation wins.

## Implemented contracts

- Calibration schema `quant-pipeline.calibration-fit.v2` independently
  accumulates and persists the float64 uncentered weighted second-moment
  numerator. The persisted raw second moment is the only full codec matrix;
  centered covariance is an on-demand diagnostic derived from it and the
  mean, avoiding a second production-size matrix.
- `calibration.route_mass` records router mass as exact integer-rational
  p0/p1/p2 totals. Natural, supplemental-raw, supplemental-corrected, and
  combined accounting remain separate. Cold-expert top-up is deterministic,
  seed-bound, row-disjoint, role-bound, and auditable. Each selected
  supplemental row seals its full canonical payload (expert, role, origin,
  document/token location, integer weight, and inclusion probability), and
  verification recomputes both its row identity and the ordered payload seal.
- `normalization.prior_search` directly ports the five prior permutation
  policies and three scale families. Geometry is parameterized; equations,
  tie-breaking, the nine-point grid, and Python-float64 geometric
  normalization are unchanged. Permutation applies to gate/up rows, down
  columns, down inputs, and both down-Hessian coordinates. A functional
  inverse oracle checks expert-function and byte round trips.
- `normalization.artifact_v31` is the one canonical v31 artifact. Each matrix
  seals source identity and mass, deduplicated shared FP16 bytes, K3/K4/K5 GSS
  targets, evaluator/codec/search receipts, selected scalar, final private
  FP16 vector, and a separate selected-bit decision.
- Artifact verification rejects invalid layer/block/mass identities, missing
  selected-bit candidates, incompatible shared/private vector geometry, and
  non-FP16 or non-finite vectors. Array files and the manifest are atomically
  written with file and parent-directory `fsync`; loading permits only bound,
  canonical local filenames and rejects symlinks and traversal.
- GSS is an injected `PinnedGSSProducer`. A positive scalar is not evidence:
  production requires a content-sealed multi-point receipt bound to matrix,
  bit, target, source weight, predecessor checkpoint, evaluator, codec, and
  search configuration.
- `evaluate_additive_v31_candidate` enforces the causal order: exact layer
  v31 fit, receipt-bound per-matrix/per-bit GSS, exact-codec proxy, then
  held-out full-expert round trip. Both scores bind the finalized artifact.
  Raw `TransformVectorSearch` output cannot be converted into ledger fields.

## Deterministic route-mass estimand

Coldness is the natural routed p1 mass expressed once as integer weight units.
For each `(expert, role)`, supplemental candidates are sorted by the
SHA256 of the sealed top-up seed and canonical row identity, without
replacement or dependence on input-pool order. Rows are selected until raw
p1 units first reach the natural-mass deficit. A zero deficit is a strict
no-op; an insufficient pool fails closed. Inverse-inclusion weighting changes
the supplemental estimator only after selection—it does not change the
subselection rule. Verification independently reconciles p1 natural mass to
the top-up integer units and recomputes p0/p1/p2 raw and corrected totals from
the sealed selected-row payloads. Natural p0 is independently reconciled to
the number of natural row identities. Natural p2 is not self-contained in this
compact receipt because natural-row weights are not repeated there; it must be
re-derived from the sealed capture identified by `role_row_identity_sha256`.

## FP16 composition and fit-cost boundaries

The absolute-v31 arithmetic follows the frozen control's staged FP16
materializations. Gate/up relative `svh` is rounded before row-RMS fitting,
then its beta-adjusted private vector is rounded, and GSS division produces a
third FP16 materialization. Down shared `svh` is rounded before row-RMS
fitting; private `suh` is rounded before and after GSS. Regularization and
decode consume the final stored FP16 bytes. These intermediate casts are
numerically observable and are not collapsed into one composition/cast.

The prior permutation policies are source-exact for valid covariance/Hessian
diagonals. A negative diagonal is an explicit fail-closed deviation: the
historical helpers can rank negative values, but this pipeline rejects them as
invalid energy rather than allowing them to influence policy selection.

K3, K4, and K5 are three independent full layer fits. Shared vectors must be
byte-identical across those fits, but that invariant does not make the work a
single fit: production pays three v31 fits plus one receipt-bound GSS search
per matrix per bit before selecting a candidate.

## Source comparison and local gates

`tests/test_scientific_core_hardening.py` compares all five permutation pure
functions with the authoritative files under the sealed prior-HF audit,
executes the authoritative normalized and coordinate scale functions extracted
from the sealed file, and fuzzes exact float results including zero, tie,
midpoint, and clamp cases. It proves the direct second-moment
path with an adversarial ULP case, exercises deterministic exact route-mass
top-up including deficit/no-op/insufficient/pool-order and inflated-natural
tamper cases, rejects scalar-only/tampered GSS evidence, and runs a tiny complete
expert through v31, all K3/K4/K5 candidates, exact-codec and held-out receipt
binding, atomic save/load, geometry/identity tampering, and path traversal.

The artifact API is intentionally fail-closed. The former selected-bit-only
adapter is rejected because it cannot prove the unselected K3/K4/K5 options.
The raw transform search remains useful as a proposal generator, but its
vectors are not checkpoint vectors.
