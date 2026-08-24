# Prior GLM-5.2 3.5-bpw additive port contract

Status: binding implementation control for the Qwen3-30B-A3B pilot. The
reviewed source is Hugging Face commit
`7c73450f05a151439d0f184f216b1eefcc394a31`, with
`reproducibility/local-corrected-v1` authoritative over the historical R10
snapshot. This document is not authorization to execute a GPU campaign.

The prior corrected build is the numeric and causal baseline. New
Aumann-Shapley, Fisher/Jacobian, p0/p1/p2, searched-transform, and global
allocation methods are additive experiments. A prior mechanism may be removed
from the competitive default only after a matched ablation improves sealed
end-to-end KLD without breaking byte/runtime parity.

## Proven control

- 75 causally rebuilt routed GLM layers, layers 3 through 77.
- Exactly 3.5 bpw and 2,688 selected bit units per routed layer.
- Five 2,047-position KLD runs: `0.0616431846`, `0.0623442891`,
  `0.0626262729`, `0.0604832642`, and `0.0593142137`.
- Mean KLD `0.0612822449`, sample SD `0.0013762398`.

These numbers are a GLM compatibility control, not a Qwen quality claim.

## Mandatory numeric-core ports

1. Preserve source-derived absolute v31 normalization, its FP16 storage
   boundary, layer-shared gate/up `suh`, layer-shared down `svh`, and
   selected-bit GSS folded into the private side. Searched transforms operate
   on top of this control and must not silently replace it.
2. Feed the exact codec an uncentered route-weighted second moment/Hessian,
   `sum(w*x*x^T)/sum(w)`. Centered covariance and weighted OAS-style shrinkage
   remain named experimental arms, never aliases for the v31 Hessian.
3. For every decoded `(gate, up)` candidate pair, recompute the SwiGLU down
   inputs, rebuild a fresh down Hessian, refactor, and re-encode down at each
   admitted bit. Cache identity includes both gate/up packed and reconstruction
   hashes. A static down encoding reused across triplets is forbidden.
4. Port the prior five permutation policies and hierarchical shared-layer /
   private-expert search as a control. The stronger exact-codec proxy plus
   held-out full-expert shortlist is retained as an additive validation layer.
5. Seal exact float32 router mass using integer units, reject duplicate expert
   IDs and malformed top-k rows, and bind the route audit to captures, fits,
   ledgers, and allocation.
6. Preserve deterministic cold-expert support with disjoint row identities and
   explicit blending/accounting. Never manufacture an identity Hessian for an
   unsupported expert.

## Mandatory byte and checkpoint ports

1. Persist trellis bytes, private vectors, layer-shared vectors, deployed FP16
   reconstruction, and their hashes for every admitted choice. Hash-only JSON
   is insufficient.
2. Charge layer-shared gate/up `suh` and down `svh` once per layer. Separate
   fixed shared-vector bytes, variable trellis/private-vector bytes, and
   container overhead. Allocator totals must equal emitted payload totals.
3. Emit and independently read/audit the target Qwen B12X/BTX checkpoint.
   Copy every non-quantized source tensor, bind tensor inventory, shapes,
   dtypes, selected bits, permutations, payloads, reconstructions, source
   identities, predecessor chain, and runtime-reader identity.
4. Resume from official BF16 and replay completed installed payloads in causal
   order. Mutable in-memory model state and manifest-shaped hashes are not
   authoritative.
5. Use atomic file writes, directory fsync, sealed successor state, and retire
   predecessor/covariance artifacts only after their consumer seal is durable.

## Additive improvements retained

- document-disjoint fit, selection, confirmation, and final roles;
- p0/p1/p2 route-power objectives and inverse-inclusion accounting;
- byte-real full-expert triplet scoring and nonlinear interaction terms;
- explicit all-eight K3/K4 triplets and all nineteen K5 decisions;
- exact-codec closure and numeric-environment attestation;
- Aumann-Shapley layer attribution, Fisher expert split, cross terms, and an
  explicit unreconciled nonlinear remainder;
- exact global multiple-choice byte allocation;
- hash-chain journaling, causal re-anchors, corruption recovery, and rollback;
- the historical 2,048-token compatibility window plus additional
  document-disjoint final windows, bootstrap intervals, and tail metrics.

## Required go/no-go gates

1. Prior GLM layer-3 vector, selected-bit GSS, conditional-down, packed-byte,
   and reconstruction replay oracle.
2. Synthetic two-layer Qwen MoE campaign with kill/resume, torn journal,
   corrupted payload, retirement-window, rollback/reallocation, emitter/read,
   and byte-identical logit tests.
3. One real Qwen layer with all 128 experts: exact route audit, fit/search,
   conditional triplets, payload reload, target-reader parity, and measured
   teacher-forced KLD.
4. Only after those gates: full Qwen campaign and final emitted-checkpoint KLD.

No B200 command may run until all required production modules exist, the local
test and adversarial-review gates are green, and the owner explicitly approves
the reviewed execution commands.

## Current BTX compatibility boundary

The target runtime container is the upstream B12X `btx-atoms-v1` contract at
commit `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8`.  A custom safetensors index
whose tensor names merely resemble EXL3 payloads is not a BTX checkpoint.
Production emission must round-trip through the upstream schema and reader.

The current upstream container stores one `rates_fc1` byte for gate and up and
one `rates_fc2` byte for down per 256-channel pair.  Consequently, arbitrary
per-projection `(gate, up, down)` decisions from the corrected GLM control are
not all expressible: gate and up must share the BTX-v1 rate at each pair, and
the declared dynamic pair vocabulary currently excludes mixed K5 pair codes.
The pipeline therefore keeps two explicit allocation surfaces:

1. an unconstrained research/control ledger that preserves every independently
   selected projection and exact payload; and
2. a `btx-atoms-v1` serving arm that filters to choices the upstream reader can
   represent and fails closed with a compatibility report otherwise.

No quality result from the constrained serving arm may be substituted for the
unconstrained control without reporting the allocation constraint and matched
end-to-end KLD.  Extending BTX for independent gate/up rates or mixed K5 is a
separate, versioned B12X schema/runtime change, not something this writer may
silently invent.
