# Exact payload cost accounting

The candidate ledger keeps four byte quantities separate. They are not
interchangeable and none is reconstructed from a self-reported total.

1. **Codec logical bytes** are the exact trellis, `suh`, and `svh` tensor
   lengths for each projection. The codec-reported value must equal this sum.
2. **Semantic expert-private bytes** count every private runtime slot. Equal
   private tensor contents do not reduce the allocator cost.
3. **Semantic layer-shared bytes** count each declared shared family once per
   layer: gate/up `suh` and down `svh`. Cross-layer equality does not remove a
   layer's fixed cost.
4. **Physical object bytes** deduplicate content-addressed objects and are an
   observed storage quantity only. Reconstruction oracles are separately
   included in artifact-store physical bytes and excluded from deployment cost.

`Candidate.payload_bytes` is the semantic expert-private allocator increment.
The ledger's `fixed_layer_shared_costs` is the layer-fixed term. A selected
allocation is therefore the sum of one candidate increment per expert plus one
fixed shared term per selected layer. `selected_allocation_cost()` seals that
calculation.

The competitive composed chain is:

1. `allocator_handoff()` validates every candidate record before it becomes a
   DP candidate.
2. `allocate_validated_records()` derives the complete layer-fixed term, calls
   `allocate_with_fixed_layer_cost()`, selects one record per unit, calls
   `selected_allocation_cost()`, and requires private, shared, and total byte
   equality between the DP result and selected records.
3. `install_layer_payloads()` reconstructs exact v2 choice objects from the
   selected causal ledger payloads and independently derives the selected
   layer's cost.
4. `verify_installed_layer()` reopens every object and choice from disk. It
   independently requires each choice's layer and predecessor to equal the
   manifest, a non-empty unique canonical expert/projection inventory, legal
   shared families, and derived semantic and physical totals.
5. `reconcile_installed_allocation()` re-derives global aggregates from the
   selected layer rows, binds available selected candidate hashes and choice
   identities to installed provenance, and requires every installed layer and
   aggregate total to match.
6. `emit_internal_qwen_checkpoint()` or `emit_official_btx_checkpoint()`
   receives the reconciled `allocated_payload_bytes` as its mandatory expected
   total. The corresponding audit then checks emitted tensor/container bytes;
   the official audit reports its source total at
   `accounting.source_semantic_allocated_payload_bytes`.

Initial allocation and causal re-encoding have distinct record identities.
Installed provenance therefore carries the selected allocation record hash and
the current causal candidate record hash separately; the stable `candidate_id`
binds the selected bit triplet across that causal regeneration.

`install_layer_payloads()` independently derives the same model from exact
choice objects. Production geometry requires the allocator's expected byte
total and fails if it differs. Internal and official emitters also expose an
expected-total parity gate before writing output. Container headers, alignment,
and official BTX row padding remain physical format overhead, not model bitrate.

K5 admission is independent of byte accounting. An admitted K5 triplet must
carry the `k5-confirmation-tail-rescue-v1` rule, exact mean and p99 metrics and
thresholds, a sealed selection artifact, and a different sealed confirmation
artifact with the `disjoint-confirmation` role. A reason string alone cannot
admit K5.
