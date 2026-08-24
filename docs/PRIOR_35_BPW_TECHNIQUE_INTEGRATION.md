# Prior 3.5-bpw technique integration

The Qwen pipeline is an additive hardening of the proven GLM-5.2 3.5-bpw
workflow preserved under `quant-research/prior-glm35-hf-audit`; it is not a
replacement with a generic quantizer.

## Preserved invariants

- The official source checkpoint remains the carrier. Every non-replaced
  tensor is copied into the emitted checkpoint and audited by tensor-payload
  SHA-256, following `r7_encoder/assemble.py`.
- Packed trellis bytes and decoded FP16 reconstruction bytes are both hard
  gates. The candidate selected by the allocator is the payload emitted; no
  later re-encode is allowed.
- Calibration is mandatory. There is no identity-Hessian production escape
  hatch, matching `lineage/encode_tr3_v31.py`.
- Gate/up share the input-side layer vector and down shares the output-side
  layer vector when the selected B12X topology declares them broadcast. The
  format also records expert-private vectors explicitly instead of silently
  assuming one topology.
- Writes are temporary-file, fsync, replace transactions. Resume verifies the
  entire safetensors file, every tensor payload, dtype, shape, finiteness, and
  causal identity before reuse.
- Logical codec bytes are reported separately from container bytes and
  deduplicated physical object bytes.

## Additive improvements

- Qwen routing is captured with a bit-exact linear/FP32-softmax/top-k
  recomputation check. Stored records include pre-gate/up hidden inputs,
  post-SwiGLU down inputs, route weights, document/offset identity, and Fisher
  sketches.
- All residual/Fisher rows use the same next-token domain as KLD: offsets
  `0..T-2`. The unscored final input token is excluded before route expansion.
- Route-power 0/1/2 fitted statistics, deterministic searched transform
  vectors, exact K3/K4 triplets, explicit K5 decisions, routed full-expert
  scoring, Shapley-style reconciliation, global byte allocation, and
  per-layer causal re-fitting are layered on top of the preserved calibrated
  EXL3/MCG baseline.
- Installed layers are self-contained content-addressed stores containing
  trellis, FP16 `suh`/`svh`, and FP16 reconstructions. Replay always starts
  from official BF16 and verifies every installed choice.
- The causal installed-payload/checkpoint directory is an internal assembly
  format, not BTX. Runtime emission separately transforms compatible choices
  into upstream `btx-atoms-v1` (`btx-manifest.json` plus slot-major
  `btx-layer-<NNNNN>.safetensors`) and binds upstream commit
  `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8`.
- Current BTX master cannot express independent gate/up rates or K5/K6
  `per_expert_pair` kinds. Uniform K5/K6 remains legal. Other allocations stay
  losslessly preserved in the internal store, but BTX emission fails with an
  actionable compatibility report instead of relabeling them as compatible.
  With Qwen's 768-channel intermediate axis (24 atom slots), current
  per-expert-pair extents are legal for TP1 and TP3, but not TP2 or TP4 without
  a declared padding/repartition extension. P44 is schema-declared but is
  rejected when a fully fused production path is required.
- Final checkpoint audit requires a target runtime reader in production. The
  internal reader is labelled reference-only and cannot satisfy that gate.

## Model-specific translation

The prior GLM encoder quantized per-TP rank slices and used 256 experts across
layers 3 through 77. Qwen3-30B-A3B uses 48 MoE layers, 128 experts, top-8
routing, hidden size 2,048, and MoE intermediate size 768. Those geometry and
tensor-layout changes are explicit production checks; GLM constants or rank
slice assumptions are never inherited implicitly.
