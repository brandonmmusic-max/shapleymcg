# Qwen3-MoE production campaign adapter contract

Status: frozen runner-side integration contract. This is not authorization to
run a GPU campaign. The concrete module must be
`quant_pipeline.campaign.qwen_adapter:QwenCampaignAdapter`; the campaign loader
must not require an owner-written adapter or shell command template.

The fitter and candidate-ledger APIs were still changing while the causal
runner was implemented. This contract fixes the exact boundary the integration
agent must satisfy before any B200 execution is allowed.

## Immutable inputs

The campaign definition must bind these local paths. URLs, Hub IDs, SSH paths,
and implicit caches are forbidden:

| input key | required content |
| --- | --- |
| `source_checkpoint` | complete local Qwen3-30B-A3B-Base BF16 snapshot |
| `sealed_corpus` | verified `quant-pipeline.sealed-corpus.v1` JSON |
| `kld_window` | directory containing the verified 2,048-token Qwen WikiText control and source prefix |
| `adapter_config` | canonical JSON configuration described below |
| `exl3_source_root` | corrected R10 Python encoder closure |
| `exl3_numeric_core` | pinned numeric-core source file |
| `exl3_extension` | locally built CUDA extension |

The adapter configuration contains only scientific/resource parameters: model
and dataset 40-hex revisions, devices, BF16 dtype, route-power/accounting arm,
full-covariance fitter settings, transform seed, codebook scale, explicit K5
screen policy, exact byte budget, KLD re-anchor threshold, minimum GPU count and
compute capability, disk headroom, and checkpoint-format version. Paths belong
in the campaign `inputs` map so the runner binds their complete byte identity.

Production configuration must also carry this exact fail-closed declaration:

```json
{
  "scientific_contract": {
    "normalization": "source-derived-absolute-v31",
    "gss": "per-matrix-selected-bit-k3-k4-k5",
    "transform_search": "additive-ablation-against-v31-baseline",
    "candidate_payloads": "exact-packed-vectors-reconstruction",
    "checkpoint": "upstream-btx-atoms-v1-pinned"
  }
}
```

This is not descriptive metadata. Preflight rejects a missing or altered
contract before model loading. Raw sign/block proposals from
`TransformVectorSearch` are not codec vectors: each proposal must be applied
to source BF16 weights through absolute-v3.1 normalization and independently
selected-bit K3/K4/K5 GSS, then exact-codec and held-out scored. Search is an
additive ablation against the proven source-derived v3.1 chain, not a
replacement for it.

`identity()` hashes the source closure of the adapter, route capturer, fitter,
candidate bridge/generator, EXL3 codec adapter, attribution, allocator,
checkpoint emitter/auditor, HF capture, and KLD scorer. The external EXL3
closure is separately bound by the immutable inputs.

The sealed adapter identity also resolves the configured `service_factory`
without calling it and hashes every Python source file in that factory's
package closure. Production preflight then resolves the services and rejects a
capturer, fitter, ledger, codec, evaluator, allocator, or checkpoint provider
whose implementation file is outside that sealed closure. Dynamic Python
dispatch is therefore not an unbound code-loading escape hatch.

## Resource preflight

`preflight(plan)` is read-only and returns exactly the runner preflight schema.
It must:

1. Reject every non-local input and report `local_only=true` with an empty
   `remote_endpoints` list.
2. Validate all input seals and revisions before importing or loading a model.
3. Require the configured CUDA device count and minimum capability (SM100 for
   the 2x B200 campaign), successful extension import on each selected device,
   BF16 support, and enough aggregate/free VRAM for the declared placement.
4. Estimate peak disk use for routed captures, full covariance artifacts,
   exact codec payloads, reconstructed predecessor state, BF16/student logits,
   and checkpoint staging; compare it with `statvfs` free space plus configured
   reserve.
5. Record Python, Torch, CUDA, driver, Transformers, Safetensors, NumPy, fitter,
   ledger, and codec versions/hashes. It performs no forward pass, encoding,
   remote access, publication, or output creation.

Production resource estimates are never allowed to default to zero. The
configuration must bind campaign-volume peak bytes and reserve, per-device
minimum free VRAM, CPU count, available RAM, and the required numeric
environment. Preflight measures the filesystem containing `campaign_dir`, each
selected GPU, host CPUs/RAM, and exact environment variable values.

`plan` and `status` never call preflight. `execute` and every `resume` call it
before the first pending stage; a failure is journaled and no stage adapter is
invoked.

The campaign definition explicitly selects `retention_mode`. Under `full`,
preflight budgets the approximately 352.9 GB three-arm fit store, approximately
206 GB rank-32 Fisher selection store, 40+ GB conditional codec payloads, 61 GB
source model, final checkpoint, captures, and a safety margin; it fails closed
if that peak does not fit. Under `capture-plus-ledger`, processing is streamed
per layer. Routed source captures, selected transform/vector artifacts,
ledgers, exact payloads, receipts, and hashes remain permanent. Dense
covariance intermediates may be removed only after their payload/ledger
consumer is sealed, and each removal is journaled with producer, consumer,
relative file, byte count, and hash. Source captures, transform evidence,
ledgers, and payloads are never transient.

## Required concrete stage wiring

Every `run(request)` call writes real artifacts in `request.output_dir` and
returns only metadata describing those files. Metadata is never accepted as a
substitute for model/codec output.

### Initial provisional stages

- `identity`: load/verify config, safetensors index and every source shard;
  enumerate all Qwen expert gate/up/down tensors and emit the inventory and
  complete source receipt.
- `teacher_capture`: call local-only HF capture on `kld_window` in BF16 and emit
  all 2,047 next-token logits, router logits, receipt, and hashes.
- `fit_capture`: run the sealed `fit` windows through the exact BF16 source
  checkpoint. A Qwen MoE hook emits, for every layer and routed expert, the
  pre-gate/up hidden vectors, post-SwiGLU down-projection inputs, FP32 router
  weights, expert IDs, document IDs, and token offsets. Records are sharded and
  hash sealed; fit and held-out roles remain separate.
- `fit`: stream the captures through `CalibrationBatch` and one
  `CalibrationFitter` for each layer/projection geometry. Use full float64
  accumulators and the configured retained arms; finalize/save one verified
  artifact per layer/expert for gate-up input and down input. Never materialize
  all experts' dense covariances simultaneously. Then run
  `TransformVectorSearch` per layer (128 experts, hidden 2,048, intermediate
  768, default 16 draws) with exact-codec proxy evaluation for every draw and
  held-out full-expert roundtrip evaluation for the baseline and shortlist;
  seal the winning `MCGTransformArtifact` and its selection evidence.
- `candidates`: construct `Exl3MCGCodec` from the three bound EXL3 inputs,
  attest it with `attest_corrected_exl3_mcg`, bridge verified fitter artifacts
  with `build_expert_candidate_input`, stream held-out routed batches, and call
  `CandidateLedgerGenerator.generate` with a resumable `CandidateJournal`.
  Enumerate all eight K3/K4 triplets and explicitly record every K5 decision.
  Competitive generation requires a searched `MCGTransformArtifact` from the
  selection role and leaves `allow_fixed_transform_baseline=False`; the fixed
  deterministic transform is accepted only as a labeled test/research
  baseline. The ledger is provisional because later layers have not yet
  observed quantized predecessors.
- `attribution`: compute measured end-to-end path anchors, Aumann-Shapley layer
  shares, full routed expert residuals, downstream Fisher/Jacobian projections,
  cross-expert terms, route shifts, and explicit nonlinear remainder. Reconcile
  expert totals to each measured layer and layer totals to measured KLD. Emit
  both raw and reconciled values; never rescale away a remainder silently.
- `allocation`: validate the exact-codec allocator handoff, bind the sealed
  attribution to that exact ledger, and anchor each expert's candidate proxy
  ratios to its signed provisional Aumann-Shapley/Fisher share. A per-unit
  constant makes the optimization scores non-negative without changing any
  within-unit ordering. Preserve the uncalibrated proxy allocation as a
  control, then run exact global byte allocation and emit choice IDs, exact
  payload bytes, shifted and reconstructed-unshifted predicted damage, offsets,
  budget slack, and all source hashes. Revalidate the complete sealed ledger at
  each consumer, and reject an official-BTX arm before DP if any sealed expert
  unit lacks a legal serving candidate.

#### Native attribution implementation

The candidate stage now persists the explicit configured provisional bit
triplet, its exact decoded payload references, and decoded-minus-original-
checkpoint deltas. This source binding is independent of a winning internal
expert permutation. The later attribution stage—not the candidate stage—loads
the pinned source model and KLD window, verifies freshly reproduced source
logits against the sealed teacher reference, and runs the full model at every
configured Gauss-Legendre path node. Each MoE hook returns the exact source
block output at `alpha=0` and the decoded block output cast into the model
compute dtype at `alpha=1`.

Next-token teacher-to-path KL is differentiated with respect to every layer
blend coefficient. Exact contemporaneously routed expert residuals are
projected through score-function Fisher VJPs, including symmetric cross-expert
terms. A separate projected routing/backend residual and explicit per-layer and
global nonlinear/quadrature remainders close the accounting without rescaling
raw expert values. The archive and receipt bind canonical nodes, tensor shapes,
endpoint KLD, model/window/teacher/provisional hashes, and fail on tampering.

The causal re-anchor currently evaluates the reconstructed tensors replayed in
the BF16 model path. Final KLD is separately captured by the pinned native BTX
student runtime after official checkpoint emission. Neither gate substitutes
for the other, and no native serving-quality claim is permitted until both are
sealed and pass their declared thresholds.

### Mandatory causal stages for every layer N

The runner schedules these four stages separately. Each request and each
result is bound to the same `predecessor_state_hash`.

1. `causal_fit_capture.layer_N`: reconstruct or reuse only a hash-exact
   installed predecessor checkpoint, replay the sealed routed windows, and
   capture layer N's gate/up and down inputs under that predecessor. Return an
   actual capture-file hash as `capture_sha256`.
2. `causal_fit.layer_N`: stream that new capture through `CalibrationFitter`;
   every saved statistic and the repeated transform-vector search must carry
   the request predecessor hash. Return the sealed fit/vector-search manifest
   hash as `fit_sha256`.
3. `causal_candidates.layer_N`: rebuild every expert candidate selected or
   needed for reallocation with the exact corrected codec, new fit, held-out
   routed scoring, and explicit K5 decisions. Return the exact ledger hash as
   `candidate_ledger_sha256`. A cached candidate is legal only if its capture,
   fit, predecessor, source tensor, codec, transform, and held-out identities
   all match byte-for-byte.
4. `causal_encode.layer_N`: choose the causally rescored candidate consistent
   with the allocation policy, install reconstructed gate/up/down tensors into
   the working model, and persist both the exact packed payload plus vectors
   and the reconstructed tensors needed for resume. The artifact includes its
   incoming predecessor hash, fit and ledger hashes, layer number, choice IDs,
   packed/reconstruction hashes and exact payload bytes. Return the complete
   installed-checkpoint manifest hash as `installed_checkpoint_sha256`.

On resume, the adapter starts from official BF16 and replays all completed
`causal_encode` artifacts in order, verifying their payload, reconstruction,
and predecessor bindings. In-memory state alone is never authoritative.
Every causal request now carries the complete ordered installed-layer prefix,
not merely the most recent encode dependency. Each prefix entry binds the
stage/layer, artifact, reconstructed checkpoint identity, incoming state, and
resulting state; the runner independently replays the entire predecessor hash
chain before dispatch.

### Re-anchor and final stages

- `kld_reanchor`: after no more than four accepted layers, run the installed
  state over the exact historical KLD window, compare against the BF16 teacher
  logits, save student logits and a KLD report, and return the strict gate
  object required by the runner. A failed gate causes only the declared
  `continue`, `rollback`, or `request_reallocation` action. A failed gate seals
  an immutable generation-supersession event, invalidates the active allocation
  and all downstream artifacts, and leaves their bytes/history intact. Resume
  begins a versioned allocation/replan generation from the state preceding the
  failed block; rollback is not a terminal campaign state.
- `checkpoint_emission`: emit the declared runtime checkpoint from the exact
  selected codec payloads; copy every non-quantized source tensor; write an
  index, config, per-layer manifests, exact payload and container byte counts,
  source identities, and a top-level seal. Re-encoding without reproducing the
  candidate packed and reconstruction hashes is forbidden.
- `checkpoint_audit`: independently enumerate the emitted checkpoint, verify
  the file set, hashes, tensor shapes/dtypes, payload bytes, reconstruction
  hashes, choices and predecessor chain, then load it through the target
  runtime reader. The audit must seal both the pinned reader identity and its
  structured result, and the adapter recomputes both seals and checks the
  emitted checkpoint identity. A provider `ok=true` or manifest-only check is
  insufficient.
- `student_capture`: load the emitted checkpoint—not the mutable in-memory
  model—and capture the same 2,047 positions and token identity as the teacher.
- `final_kld`: verify both capture receipts/files and independently recompute
  finite per-token, mean, standard deviation, p50/p95/p99, CVaR95, maximum and
  deterministic bootstrap confidence bounds from the bound logit bytes. Bind
  the report to the exact token/window artifact, teacher, student, checkpoint,
  installed prefix, request and journal-head hashes. Provider-reported KLD is
  not authoritative. Re-anchor KLD uses the same independent computation.

Routed-capture resume binds the full request: role, ordered layers, predecessor,
geometry, routing normalization, Fisher rank/seed, and every window's token,
document and offset identity. A chunk without its receipt (or a receipt without
its chunk) is moved into a hash-named quarantine and regenerated; a completed
capture for a different request is rejected.

## Frozen APIs used by the concrete module

These APIs now exist and are composed by
`quant_pipeline.campaign.qwen_services:build_qwen_campaign_services`:

1. A streaming Qwen route-capture reader/writer with document/offset identity,
   gate-up and down-input records, held-out route arrays and Fisher sketches.
2. The final scale-safe `CalibrationFitter` artifact schema and iterator/load
   bridge (`FittedExpertStatistics` remains the consumer boundary).
3. The final streaming `build_expert_candidate_input` bridge and
   `CandidateLedgerGenerator` API.
4. An exact-codec payload store that persists packed bytes, `suh`, `svh`, and
   reconstructed tensors for every admitted choice. A JSON ledger of hashes is
   not sufficient for causal installation or checkpoint emission.
5. A Qwen checkpoint emitter/reader for that exact payload format. The existing
   uniform reference checkpoint is test-only and cannot satisfy this stage.

Uniform quantization, synthetic production captures, hash-only payload records,
mutable in-memory checkpoints, and shell placeholders remain explicitly
prohibited. Their absence is a production preflight gate, not a documentation
convention.

The concrete adapter binding points are fixed as follows (names and return
types may be implemented in `qwen_adapter.py`; they are not shell commands):

```python
capture_moe(
    *, model, sealed_corpus, role, layers, predecessor_state_hash, output_dir
) -> RoutedCaptureManifest

fit_layer(
    *, capture: RoutedCaptureManifest, layer: int,
    predecessor_state_hash: str, output_dir
) -> LayerFitManifest

fit_vectors(
    *, layer: int, expert: int, source: ProjectionTensors,
    gate_up_statistics: FittedExpertStatistics,
    down_statistics: FittedExpertStatistics, heldout_batches,
    k5_screen, route_power: int, accounting: str,
    searched_transform: MCGTransformArtifact,
    transform_seed_sha256: str, codebook_scale: float
) -> ExpertCandidateInput

generate_layer_ledger(
    *, experts, codec: Exl3MCGCodec, attestation: BackendAttestation,
    journal: CandidateJournal, output_path
) -> CandidateLedger

install_layer(
    *, predecessor_checkpoint, layer_ledger, selected_choices, output_dir
) -> InstalledLayerManifest

emit_checkpoint(
    *, source_checkpoint, installed_layers, format_version, output_dir
) -> CheckpointManifest
```

`fit_layer` is implemented using the frozen `CalibrationFitter` call surface:
one `projection="gate_up"`, `hidden_size=2048` fitter and one
`projection="down"`, `hidden_size=768` fitter per layer, both full covariance
and bound to the predecessor SHA. `fit_vectors` is the frozen
`build_expert_candidate_input`/`CandidateLedgerGenerator.prepare_expert_input`
bridge and must receive searched transform evidence in competitive mode.
`generate_layer_ledger` is the frozen streaming
`CandidateLedgerGenerator.generate` call with an identity-bound
`CandidateJournal`. At the time of this runner handoff, `capture_moe`, the
payload-bearing `install_layer`, and the runtime-format `emit_checkpoint`
remained missing; the integration agent must implement those three rather than
replace them with name-only stage adapters.

## Required acceptance tests

- Deterministic tiny-MoE full campaign, including real numeric fitting,
  candidate scoring, causal predecessor recapture, encoding, checkpoint load,
  student logits and final KLD.
- Kill/restart at every stage boundary and after result-file creation but before
  journal completion; resume must be idempotent and byte-identical.
- Kill after a sealed re-anchor completion but before its gate event; resume
  must reconstruct the decision from sealed artifacts without model execution.
- Exercise a five-layer campaign and prove every causal request receives the
  complete, ordered and hash-linked installed-layer prefix.
- Input, spec, code, adapter, payload, reconstructed tensor, predecessor, KLD
  token and artifact drift each fail closed.
- A layer-N capture obtained before installing layer N-1 is rejected.
- Re-anchor spacing of five is rejected; failed gates exercise all three
  declared policies and never silently advance.
- Plan/status prove zero preflight/model/GPU calls. Execute rejects resource
  insufficiency and all remote endpoints before stage invocation.
- The final checkpoint is loaded by the target reader and its logits are
  captured from that load, not from a proxy model.
