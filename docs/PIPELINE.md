# Calibration to runtime pipeline

This document is the stage-by-stage method contract. The executable clean-node
sequence is in `docs/B200_RUNBOOK.md`; primary scientific and software lineage
is mapped to individual components in `docs/REFERENCES.md`. Passing unit tests
does not substitute for the final exact-codec, final-logit-KLD, checkpoint, and
runtime gates.

## Stage 0: freeze identity and repair the instrument

Resolve the model revision, hash `config.json`, index and shards, seal the four
document-disjoint corpus roles, and capture new BF16 teacher logits. Separately,
seal the historical GLM-style WikiText control for the target model. Calibration
data used for scales cannot appear in confirmation or KLD. Bind software
image/driver/tool versions in the run receipt.

### Historical KLD control

The cross-model control reproduces the *procedure*, not GLM token IDs or GLM
logits. Pin `Salesforce/wikitext` at an immutable revision, load
`wikitext-2-raw-v1` test, discard rows whose stripped text is empty, join the
remaining original text values with two newlines, take the first
`context_length * 5` characters, then tokenize with the target checkpoint's
tokenizer using no special tokens and truncation to 2,048. For Qwen, that yields
2,048 Qwen tokens and 2,047 next-token BF16 reference-logit rows. The seal binds
the source prefix, Qwen token IDs, dataset revision, tokenizer files, model
revision, and construction rules. It is not a calibration corpus.

## Stage 1: corrected actual-codec re-probe

**Status: implemented locally; real-Qwen gate unvalidated.** The full-second-
moment fitter, canonical absolute-v31/GSS artifact, corrected-codec adapter,
routed scorer, and causal runner surfaces exist and have tiny/local tests. No
clean-node real-Qwen probe result has yet passed the production gates.

Start with representative early/middle/late layers. Reconstruct candidates
through the exact deployed codec—not an MSE-only fake quant—and compare:

- incumbent unweighted Hessian plus router-mass allocation;
- router-weighted covariance arms (`g` and `g^2`);
- routed full-expert output residual;
- Aumann-Shapley layer damage; and
- the Fisher/Jacobian expert decomposition.

This is the branch point. If corrected probe rankings mostly agree with the old
map, continue to objective research. If they do not, re-allocation at the same
rate is the cheapest likely win.

## Stage 2: build Pareto frontiers

**Status: implemented locally; campaign evidence pending.** The candidate
generator persists exact trellis/vector/reconstruction objects, all eight
K3/K4 choices, versioned K5 decisions, routed metrics, and byte accounting.
Production remains blocked on sealed real captures, the corrected SM100 encoder
extension, and end-to-end Qwen validation.

For every sampled expert, encode and reconstruct gate/up/down candidates at the
allowed bit rates. Score complete expert-function outputs. Jointly evaluate the
8 or 27 most relevant triplets where the additivity diagnostic finds coupling.
Record exact codec payload bytes for every candidate. Container bytes become
exact only after candidates are grouped into physical shards and are audited then.

## Stage 3: global allocation

**Status: allocator and causal orchestration implemented locally.** The
multiple-choice knapsack charges expert-private choices plus layer-fixed shared
bytes. The runner schedules causal recapture/refit/re-encode and KLD re-anchors,
with immutable rollback/reallocation generations. This behavior is locally
tested but not yet validated on the complete 48-layer Qwen checkpoint.

Collapse dominated candidates into per-expert or per-layer Pareto frontiers,
then solve one multiple-choice knapsack over the whole model's exact codec
payload-byte budget.
Do not impose equal per-layer budgets. Re-encode causally so successor-layer
statistics observe accepted quantized predecessors, and re-anchor exact KLD
every configured layer interval.

The Qwen pilot preserves an uncalibrated proxy-control allocation. Its research
arm binds the sealed attribution to the exact candidate ledger and anchors each
expert's candidate proxy ratios at the provisional candidate's signed
Aumann-Shapley/Fisher share. Signed scores receive one constant offset per
expert unit so the DP sees finite non-negative values; because every candidate
for that unit receives the same offset, within-unit ordering and the global
optimizer's selected solution are unchanged. The receipt publishes every
scale, offset, provisional anchor, the shifted DP objective, its selected-unit
offset total, the reconstructed unshifted objective, and both allocation arms.
Every consumer revalidates the sealed ledger before allocation or causal
installation, and the official-BTX filter must retain at least one legal
candidate for every sealed expert unit. Exact KLD
re-anchors—not the proxy calibration—remain the acceptance authority.

## Stage 4: pack, reconstruct and audit

The internal Qwen carrier and pinned upstream `btx-atoms-v1` writer/auditor are
implemented with semantic/physical byte reconciliation. The included uniform
codec remains a deterministic reference only; competitive runs require the
pinned EXL3/MCG adapter and a real target-runtime reader. The pinned B12X
checkpoint closure does not itself prove an SM100 serving kernel.

Exact-codec choices use `quant-pipeline.exact-codec-choice.v2`. Its
`packed_sha256` is calculated over canonically labeled, uint64-length-framed
trellis, `suh`, and `svh` byte strings. Legacy v1 choices used an unframed
concatenation and are rejected rather than silently accepted or migrated;
regenerate them from their sealed source objects. Only `gate_proj.suh`,
`up_proj.suh`, and `down_proj.svh` may be declared layer-shared.

The production accounting handoff is
`allocate_validated_records -> selected_allocation_cost ->
reconcile_installed_allocation`. The reconciled total must be supplied to both
emitters as `expected_allocated_payload_bytes`, followed by the corresponding
checkpoint audit. The campaign composition root must use this handoff; raw
`Candidate` JSON is intentionally excluded from competitive use.

## Stage 5: final quality and runtime qualification

Run final-window KLD first. Only a frozen winner advances to task evaluations,
runtime conversion, CUDA-graph parity, throughput/KV tests, and five-run
stability. Report graph topology, cache format, model bytes and image digest.

## Required publication bundle

Publish a result only with the Git commit; immutable model and dataset
revisions; resolved artifact/environment locks; sealed corpus and KLD-window
receipts; BF16 and student-logit receipts; candidate ledger; exact selected-cost
record; campaign journal/audit; internal and official checkpoint audits; final
per-window/per-token KLD; runtime topology; and a SHA256 manifest covering the
bundle. Report proxy predictions and explicit closure remainders separately
from measured KLD. A failed or unexecuted gate stays visible rather than being
converted into a success claim.

The Qwen fixed-Hadamard control uses
`scripts/upload_qwen_incremental_hf.py` for fitted Hessian layers,
`scripts/upload_qwen_candidates_hf.py` for exact K3/K4 candidates, and
`scripts/seal_qwen_control_bundle.py` for the final closure. Candidate deletion
is rejected until a zero-valued exact-KLD exit receipt exists. The final sealer
rejects any missing layer, upload receipt, allocation seal, or KLD seal.

## Command skeleton

The commands below demonstrate individual primitives. For the integrated
production adapter, use `docs/B200_RUNBOOK.md`; it deliberately stops after
read-only preflight and contains an owner-approval barrier before execution.

```bash
python3 -m pip install -e '.[hf,test]'
quant-pipeline inspect examples/qwen3-30b-a3b.toml
quant-pipeline seal-kld-window --model models/Qwen3-30B-A3B-Base --model-revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 --context-length 2048 --output-dir artifacts/qwen/kld-window --execute
quant-pipeline capture --model models/Qwen3-30B-A3B-Base --model-revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 --sealed-corpus artifacts/qwen/kld-window/kld-window.json --role kld --output-dir artifacts/qwen/teacher-kld --execute
quant-pipeline seal examples/qwen3-30b-a3b.toml --output artifacts/qwen/sealed-corpus.json
quant-pipeline inventory --family qwen3_moe --config models/Qwen3-30B-A3B-Base/config.json --output artifacts/qwen/inventory.json
quant-pipeline capture --model models/Qwen3-30B-A3B-Base --model-revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 --sealed-corpus artifacts/qwen/sealed-corpus.json --role final --output-dir artifacts/qwen/teacher-final --execute
quant-pipeline allocate --candidates artifacts/qwen/reference-candidates.json --byte-budget PAYLOAD_BYTES --quantum 1 --output artifacts/qwen/reference-allocation.json --non-competitive-reference
quant-pipeline encode-reference --model-path models/Qwen3-30B-A3B-Base --family qwen3_moe --allocation artifacts/qwen/allocation.json --output-dir artifacts/qwen/reference-packed --execute
quant-pipeline audit --packed-dir artifacts/qwen/reference-packed
quant-pipeline kld --teacher-dir artifacts/qwen/teacher-final --student-dir artifacts/qwen/candidate-final --output artifacts/qwen/final-kld.json
```

Commands that load a model or write checkpoint bytes require `--execute`.
