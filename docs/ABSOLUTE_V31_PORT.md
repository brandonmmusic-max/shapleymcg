# Absolute-v31 normalization control

This package ports the deployed `local-corrected-v1` normalization equations
without changing their arithmetic.  It is the compatibility control beneath
new search, attribution, and allocation experiments; those experiments may
select signs, block factors, permutations, bits, or a GSS scalar, but they do
not silently substitute a different amplitude normalization.

## Frozen equations and boundaries

- Sign/block-factor bases are materialized by Python float64 division and then
  cross once into float32.
- Output-channel RMS is `block_rms(W, dim=0)` divided by its arithmetic mean.
- Gate/up share the mass-weighted geometric mean of post-right-Hadamard row
  RMS.  The stored shared `suh` is FP16 and includes division by the negative
  codebook scale; each gate/up matrix retains a private FP16 `svh` amplitude.
- Down projections share the mass-weighted geometric mean of relative output
  RMS, renormalized to arithmetic mean one and stored as FP16 `svh`; absolute
  row amplitude remains in each private FP16 `suh`.
- Selected-bit GSS is bound to `(matrix key, bits)`.  It divides private `svh`
  for gate/up and private `suh` for down.  Shared bytes are invariant to bits.
- K3, K4, and K5 each execute a complete layer-level absolute-v31 fit followed
  by a per-matrix pinned GSS search. This is three full fit passes, not one fit
  with three inexpensive scalar variants. The separately computed shared FP16
  vectors must nevertheless be byte-identical across all three passes.
- Regularization and decode consume only the FP16-rounded vectors and the
  injected pinned numeric core.

The frozen control intentionally has multiple staged FP16 casts. Gate/up first
stores relative private `svh`; that rounded value is used to fit row RMS, then
is multiplied by beta and stored again, and is divided by GSS and stored a
third time. Down stores shared `svh` before row-RMS fitting, stores private
absolute `suh`, and stores private `suh` again after GSS. The decoder consumes
the final stored vectors. Collapsing these stages into a single FP16 cast would
change the compatibility control.

## Production identity and memory contract

`FitSamplePlan` binds order, key, role, selected bits, geometry, route mass,
base-vector hashes, and a dtype/shape-aware source-weight hash.  The production
streaming fitter additionally requires the lowercase SHA-256 of the injected
numeric core.  Replay fails if any planned matrix changes.  Its retained fit
state is two float64 log accumulators plus shared/base vectors: O(K + N), with
one source matrix live at a time and no dependency on expert count.

The formulas require only the semantic projection roles `gate_proj`,
`up_proj`, and `down_proj`.  Matrix keys and counts are unconstrained, and
dimensions are checked from each matrix, so the same implementation covers
GLM and Qwen MoE expert inventories without a frozen 256-expert assumption.

The production artifact verifier also binds nonnegative `layer_id`, positive
block and route mass, exact K3/K4/K5 candidate inventory, selected-bit
existence, and consistent hidden/intermediate vector geometry. Persistence is
power-loss hardened: every `.npy` payload and the manifest use atomic replace
with file and directory `fsync`. Loading accepts only canonical local array
filenames, rejects symlinks/path traversal/unbound files, and verifies file,
array-byte, manifest, and complete artifact hashes before use.

## Oracle status

`tests/test_absolute_v31.py` imports the frozen authoritative source and the
pinned `encode_tr3_v31.py` numeric core from the local audit snapshot.  It
compares shared vectors, every pre/post-GSS private vector, GSS target, and
regularized/decode tensors byte-for-byte on deterministic rectangular
synthetic expert matrices.  It also tests strict source/core identity failure.
The frozen corrected 4.23 GB layer-3 shard is installed beside the local model.
The test suite binds the exact manifest identity and its declared shard
identity, then reads the shared vectors plus representative private/GSS-folded
vectors and packed trellises back through `safetensors`, comparing their raw
bytes with the frozen payload hashes.  This avoids a 4.23 GB whole-file read in
every unit-test run while still detecting payload drift in the normalization
control.  BF16 source weights needed to recompute all vectors are not retained
in that quantized shard, so the full source-to-vector replay remains a later
checkpoint gate rather than being simulated from receipts.
