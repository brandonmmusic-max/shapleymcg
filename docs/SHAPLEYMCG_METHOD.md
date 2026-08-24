# ShapleyMCG: complete method specification

This document defines the full ShapleyMCG method and distinguishes it from the
historical Hessian/router allocation used in the first Qwen controls. A result
may be labeled **full ShapleyMCG** only when every stage below is executed and
its identities are sealed. Fixed-Hadamard MCG encoding plus diagonal routed-p2
allocation is the **predecessor pipeline**, not the full method.

## 1. Target and experimental boundaries

The target is teacher-to-student next-token KL divergence on sealed token
windows. Proxy losses propose choices; they are not substitutes for final KLD.
The model revision, tokenizer, token IDs, teacher logits, attention backend,
arithmetic dtype, expert-weight rate, non-expert weight scope, codec, and
candidate inventory are fixed within a comparison.

Documents are separated by role:

- `fit`: Hessian/covariance, routing, scale, transform, and sketch fitting;
- `selection`: candidate and allocation decisions;
- `confirmation`: prospective sign, ranking, closure, and regret checks; and
- `final`: untouched KLD and downstream validation.

Every corpus and artifact records content hashes. Final tokens, logits, KLD,
allocations, candidate tensors, and reports are independently replayable.

## 2. Calibration and exact-codec candidates

For every routed expert and each gate, up, and down projection, the calibration
stage captures route IDs, route weights, inputs, and the full raw second moment
`X^T X`; the cross-coordinate terms are retained. Candidate fitting uses the
actual routed subset and records the source tensor, predecessor state, corpus,
fit, and implementation identities.

The current Qwen experiment uses deterministic fixed-Hadamard/source-derived
absolute-v31 and GSS search primitives to create corrected EXL3/MCG K3 and K4
candidates. The reconstructed tensor and exact packed payload are both hashed.
The candidate ledger records actual stored bytes; nominal bit labels never
replace byte accounting.

This candidate generator is intentionally separable from allocation. A better
allocation can coexist with an inferior codec, and a better codec can coexist
with an inferior allocation. Results therefore name both.

## 3. Full-model Aumann-Shapley layer attribution

An explicit uniform-K4 actual-codec model is the provisional endpoint. In every
MoE block, the implementation recomputes the decoded expert function at the
current path hidden state and substitutes

```
source_block_output + alpha[layer] * (decoded_block_output - source_block_output)
```

Thus `alpha=0` is the exact source model and `alpha=1` is the decoded provisional
model. All layers move simultaneously along this source-to-decoded path. At five
canonical Gauss-Legendre nodes, the pipeline evaluates teacher-to-path
next-token KLD and differentiates it with respect to every layer coefficient.
Quadrature integrates those derivatives into signed Aumann-Shapley layer
contributions.

This is a full-model forward/backward calculation, not a layer-local “KLD” or
an inference from Hessian loss. Signed effects are retained because cancellation
is real information.

## 4. Routed expert attribution inside each layer

At each path node the implementation computes the actual routed residual of
each expert's complete gate -> SiLU -> up -> down function. Downstream
score-function Fisher/Jacobian vector-Jacobian products project those residuals
with a sealed rank-8 sketch. For projected expert residuals `z_e`, the quadratic
share is

```
psi_e = 0.5 * mean(z_e * sum_j(z_j))
```

This symmetrically allocates cross-expert interactions without one downstream
model pass per expert. Direct expert-codec damage, route/state-shift damage,
cross-expert terms, and unresolved nonlinear/backend remainder remain separate
in the raw ledger.

## 5. Completeness and reconciliation

The five-node raw expert and layer attributions need not naturally equal the
measured endpoint. The pipeline publishes:

1. the raw attribution sum;
2. the exact uniform-K4 endpoint KLD;
3. the signed unresolved remainder and its fraction of the endpoint; and
4. a hierarchical accounting reconciliation in which expert entries sum to
   their layer entry and layer entries plus the explicit remainder sum to the
   measured endpoint.

Reconciliation is bookkeeping for exact accounting. It is never described as
proof that the raw proxy was additive. The raw values are retained unchanged.

## 6. Exact-rate global allocation

For each expert, its reconciled causal share is divided by that expert's
uniform-K4 routed-p2 proxy anchor. The resulting signed scale maps the actual
K3-versus-K4 candidate proxy delta onto the end-to-end causal attribution. A
global fixed-cost allocator then chooses exactly 9,216 K3 and 9,216 K4 matrix
payloads across 18,432 Qwen routed-expert matrices: exactly 3.5 logical bits per
routed-expert weight, with the candidate inventory and stored-byte budget held
fixed.

This allocation changes choices only. It does not claim to improve the MCG
codec, calibration corpus, scale policy, attention encoding, or runtime kernel.

## 7. Causal installation and final measurement

Selected decoded tensors are hash-verified before installation. Layers are
installed in order and the student is re-anchored against sealed teacher logits
at least every four layers. Final evaluation uses the identical teacher,
tokens, attention backend, arithmetic, and non-expert scope for all compared
allocations. Student logits and tokenwise KLD are saved and replayed in float64.

Allocation is conditional on the surrounding model state. If attention,
dense, embedding, head, or activation arithmetic will also be quantized, the
causal path and final allocation must be re-anchored with that body installed.
A source-BF16 allocation win does not license assuming the same ranking after
the non-expert body changes; that interaction is measured, not interpolated.

The validation hierarchy is:

1. a 2,047-position GLM-lineage WikiText window for a fast directional gate;
2. a 20,480-position matched WikiText panel for the main allocation comparison;
3. score-blind exact-rate allocations and uniform K3/K4 controls;
4. a reconstructed Hill-style BFCL/RULER panel, labeled reconstruction unless
   the authors' exact tokens and W4A4 execution are available; and
5. same-parent TurboDerp codec/body controls that vary one factor at a time.

Expanded BF16 reconstructed-weight evaluation is not packed-runtime
qualification. CUDA graphs, native BTX/EXL3 readers, KV-cache formats, and
serving throughput require separate parity and runtime gates.

## 8. Names used in result ledgers

- **Predecessor allocation**: fixed-Hadamard/MCG candidates selected by minimum
  diagonal routed-p2 Hessian/router damage. It may be a useful control, but it
  is not the full causal method.
- **Full ShapleyMCG allocation**: all stages 1-7 above, including full-model
  five-node Aumann-Shapley attribution, routed Fisher/Jacobian expert split,
  explicit remainder, causal-score calibration, and exact-rate allocation.
- **ShapleyMCG codec**: the corrected MCG candidate encoder used in this study;
  this label alone says nothing about which allocator selected the candidates.
- **TurboDerp checkpoint-reconstruction arm**: an exact-rate construction that
  selects matrices from the published EXL3 K3 and K4 checkpoints. It tests the
  allocator on a second fixed candidate pool; it does not isolate the codec or
  calibration stack. Credit belongs to turboderp and the ExLlamaV3
  contributors.

## 9. Claims that the present evidence can support

A full-method improvement claim requires a matched predecessor-versus-causal
comparison. A codec superiority claim requires the same allocation, parent,
panel, non-expert scope, and evaluator with only the codec changed. A comparison
with Hill et al. must be called reconstructed unless their exact token panel,
parent, quantized scope, and W4A4 arithmetic are reproduced. Interpolation
between K3 and K4 is descriptive and is never reported as a measured 3.5-bpw
result.

## 10. Attribution

ShapleyMCG combines original integration and experiment engineering with prior
work. The direct modern quantization precedent for Aumann-Shapley/additivity is
Joshua Hill, *Saturation Makes Quantization Error Additive: A Coverage Model
with a Certificate*, and NVIDIA Model Optimizer PR #2183. Aumann-Shapley value
theory originates with Robert J. Aumann and Lloyd S. Shapley. EXL3/TRELLIS and
its quantization tooling are by turboderp and ExLlamaV3 contributors. Qwen3 and
its model weights are by the Qwen team. The repository's full references and
scope-specific credits are in [REFERENCES.md](REFERENCES.md).

The implementation entry points are
`scripts/run_qwen_mcg_native_attribution.py`,
`scripts/allocate_qwen_mcg_causal_exact_3p5.py`,
`scripts/measure_qwen_mcg_causal_allocation.py`, and
`scripts/measure_qwen_mcg_panel_allocation.py`.
