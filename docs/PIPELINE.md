# Calibration to runtime pipeline

## Stage 0: freeze identity and repair the instrument

Resolve the model revision, hash `config.json`, index and shards, seal the four
document-disjoint corpus roles, and capture new BF16 teacher logits. Calibration
data used for scales cannot appear in confirmation or final KLD. Bind software
image/driver/tool versions in the run receipt.

## Stage 1: corrected actual-codec re-probe

**Status in v0.1: planned/manual.** The codec adapter and attribution primitives
exist, but the covariance/vector fitter and campaign runner do not yet generate
this stage end to end.

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

**Status in v0.1: planned/manual.** `candidates.json` is an external research
handoff. The CLI validates and allocates it but does not yet fit or emit it.

For every sampled expert, encode and reconstruct gate/up/down candidates at the
allowed bit rates. Score complete expert-function outputs. Jointly evaluate the
8 or 27 most relevant triplets where the additivity diagnostic finds coupling.
Record exact codec payload bytes for every candidate. Container bytes become
exact only after candidates are grouped into physical shards and are audited then.

## Stage 3: global allocation

**Status in v0.1: allocator implemented; causal orchestration manual.** The
multiple-choice knapsack is exact for declared payload bytes. Causal re-encode
and configured KLD re-anchors are not yet wired into a campaign runner.

Collapse dominated candidates into per-expert or per-layer Pareto frontiers,
then solve one multiple-choice knapsack over the whole model's exact codec
payload-byte budget.
Do not impose equal per-layer budgets. Re-encode causally so successor-layer
statistics observe accepted quantized predecessors, and re-anchor exact KLD
every configured layer interval.

## Stage 4: pack, reconstruct and audit

Write immutable packed shards, a tensor-to-shard manifest, actual file bytes,
and SHA256 hashes. Decode a sample from each rate and layer, verify candidate
identity, and bind the allocation ledger to the emitted bytes. The included
uniform codec is a deterministic reference for these invariants; competitive
runs must use the pinned EXL3/MCG adapter.

## Stage 5: final quality and runtime qualification

Run final-window KLD first. Only a frozen winner advances to task evaluations,
runtime conversion, CUDA-graph parity, throughput/KV tests, and five-run
stability. Report graph topology, cache format, model bytes and image digest.

## Command skeleton

The commands below demonstrate implemented primitives. The `candidates.json`
step between capture and allocation is intentionally not shown as a command in
v0.1 because its competitive fitter/scorer runner is not implemented yet.

```bash
python3 -m pip install -e '.[hf,test]'
quant-pipeline inspect examples/qwen3-30b-a3b.toml
quant-pipeline seal examples/qwen3-30b-a3b.toml --output artifacts/qwen/sealed-corpus.json
quant-pipeline inventory --family qwen3_moe --config models/Qwen3-30B-A3B-Base/config.json --output artifacts/qwen/inventory.json
quant-pipeline capture --model models/Qwen3-30B-A3B-Base --model-revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 --sealed-corpus artifacts/qwen/sealed-corpus.json --role final --output-dir artifacts/qwen/teacher-final --execute
quant-pipeline allocate --candidates artifacts/qwen/candidates.json --byte-budget PAYLOAD_BYTES --quantum 1 --output artifacts/qwen/allocation.json
quant-pipeline encode-reference --model-path models/Qwen3-30B-A3B-Base --family qwen3_moe --allocation artifacts/qwen/allocation.json --output-dir artifacts/qwen/reference-packed --execute
quant-pipeline audit --packed-dir artifacts/qwen/reference-packed
quant-pipeline kld --teacher-dir artifacts/qwen/teacher-final --student-dir artifacts/qwen/candidate-final --output artifacts/qwen/final-kld.json
```

Commands that load a model or write checkpoint bytes require `--execute`.
