# TurboDerp and Shapley-paper superiority protocol

## Claim discipline

Two different claims require two different controls.

1. **Allocator superiority over TurboDerp v0.0.1.** Compare allocation rules on
   identical K3/K4 reconstructed candidate bytes, at identical realized expert
   BPW, with the parent, non-expert body, teacher logits, token panel, attention
   backend, and KLD arithmetic fixed.
2. **Pipeline superiority over a native TurboDerp conversion.** Run the pinned
   v0.0.1 converter at 3.5 BPW, then compare whole checkpoints at matched
   realized stored bytes. This is practically useful, but it is not an
   allocator-only ablation because a native mixed-bit conversion generates its
   candidates under its own progressive state.

The full ShapleyMCG allocation first beat the predecessor allocation by
3.322359% inside the reconstructed TurboDerp K3/K4 pool. The subsequent strict
v0.0.1-rule comparison below now establishes the allocator claim as well.

## TurboDerp v0.0.1 calibration and conversion audit

Pinned upstream revision:
`ae04741f22324cc746ab78c27365e53e3f9f1cf4` (`v0.0.1`). The published Qwen
K3 and K4 checkpoints declare 100 rows by 2,048 tokens, K6 head, and automatic
output scales.

The default 100-row calibration corpus is deterministic. Integer truncation of
the source weights produces this exact row allocation:

| source | configured weight | rows | tokens |
| --- | ---: | ---: | ---: |
| C4 | 10 | 7 | 14,336 |
| code | 15 | 11 | 22,528 |
| multilingual | 15 | 11 | 22,528 |
| technical | 10 | 7 | 14,336 |
| Wikipedia | 48 | 37 | 75,776 |
| TinyStories | 10 | 7 | 14,336 |
| uniform random token IDs | remainder | 20 | 40,960 |
| **total** | 128 | **100** | **204,800** |

C4 and multilingual lines are shuffled with Python seed 0. Random IDs use
Torch seed 0. Article-oriented sources alternate BOS/EOS insertion on completed
rows. Code and technical text are contiguous raw token streams.

The more consequential behavior is progressive state propagation. For each
module the converter captures Hessian statistics on the current activation
state, encodes the module, reloads the quantized result, and forwards the same
calibration rows through it. Every later module is therefore calibrated in the
context of already quantized predecessors. The earlier ShapleyMCG candidate
run captured expert Hessians from the BF16 source state and is not an encoding-
pipeline match to this behavior.

TurboDerp's routed-expert allocation is coarse. Every expert in a projection
shares one format. At 3.5 expert BPW on Qwen3-30B-A3B, its original carried-
surplus arithmetic alternates:

- layer 0: gate K3, up K3, down K4;
- layer 1: gate K3, up K4, down K4;
- repeat for 48 layers.

That produces exactly 9,216 K3 and 9,216 K4 expert matrices, the same logical
3.5 expert BPW as the ShapleyMCG allocation. The sealed reproducer is
`scripts/allocate_turboderp_v001_expert_exact_3p5.py`.

## Experiments

### A. Strict TurboDerp allocator isolation

Use the published TurboDerp K3 and K4 checkpoint reconstructions as the common
candidate pool. Fix TurboDerp K4 attention/dense weights and its K6 head. Score:

- v0.0.1 carried-surplus expert allocation;
- full ShapleyMCG causal expert allocation.

Both have exactly half K3 and half K4 matrices. The primary endpoint is mean
`KL(BF16 || quantized)` over the sealed 10 x 2,048 panel; secondary endpoints
are top-1 agreement, per-row KLD, bootstrap confidence interval for the paired
tokenwise KLD difference, and the count of rows favoring each arm. Superiority
requires lower mean KLD and a paired interval excluding zero. This is the
cleanest answer to whether the allocation method beats TurboDerp's allocation.

**Measured result.** TurboDerp's v0.0.1 rule measured mean KLD
`0.033941535180377104` and top-1 agreement `0.931201171875`. Full ShapleyMCG
measured `0.029269076647285147` and `0.93720703125`, respectively. The KLD
reduction is **13.766197%** and the top-1 gain is **0.600586 percentage
points**. ShapleyMCG won all 10 rows. A 200,000-draw row-block bootstrap with
seed 20260824 gave a 95% interval of
`[0.0035278601966, 0.0059290927123]` for the absolute mean-KLD reduction, wholly
above zero. The sealed comparison SHA-256 is
`85376441fc6279a3d17549921f310d1247ee39810bcbf7fe8e3e8c35f602bc65`.

This passes the allocator-superiority gate over common candidate bytes. It
does not satisfy the separate native-mixed-checkpoint claim in experiment C.

### B. Calibration-data versus state-propagation factorial

Generate the same K3/K4 candidate ladder in four arms:

| arm | corpus | predecessor state |
| --- | --- | --- |
| B0 | current four-axis REAP/recall fit split | independent BF16 |
| B1 | TurboDerp standard mixture | independent BF16 |
| B2 | current four-axis REAP/recall fit split | progressive quantized |
| B3 | TurboDerp standard mixture | progressive quantized |

Keep codec revision, random seeds, output-scale policy, candidate rates,
allocator, teacher, and evaluation panel fixed. The B1-B0 contrast estimates
the corpus effect; B2-B0 estimates progressive-state effect under our corpus;
and B3-B1 estimates it under TurboDerp's corpus. An interaction term reveals
whether a corpus helps only when propagated through quantized predecessors.

The current corpus has stronger experimental hygiene—document-disjoint fit,
selection, confirmation, and final roles—but its 65,536-token fit role is only
about one third of TurboDerp's 204,800-token calibration state and emphasizes
general, legal, code-agentic, and reasoning/termination axes rather than web,
Wikipedia, multilingual, technical, story, and random-token coverage. A larger
hybrid corpus is a justified candidate, not yet a proven improvement.

### C. Native stock conversion

Run ExLlamaV3 v0.0.1 at target 3.5 decoder BPW, 6-bit head, 100 x 2,048 default
calibration, automatic output scales, and its native progressive converter.
Measure the resulting checkpoint with the same BF16 teacher and panels used in
experiment A. Report actual stored bytes and both routed-expert and whole-model
effective BPW. This is a pipeline result and must remain labeled as such.

### D. Shapley-paper protocol

The paper's Qwen3-30B allocation table is not a WikiText EXL3 comparison. It
uses W4A4 NVFP4 with the `{16, 8, 4}` format alphabet, matched whole-model
effective bits, and a task-aligned BFCL-v3 plus RULER corpus tokenized into 16
sequences of 2,048 tokens. KL is measured on a disjoint half. Its 4.2-effective-
bit row reports AS-additive `0.0353` versus ModelOpt `0.0429`.

To claim a paper win, reproduce that format alphabet, W4A4 arithmetic, whole-
model effective-bit accounting, 16-sequence split, and evaluation KLD. The
existing reconstructed BFCL/RULER result is supporting evidence, not a strict
paper reproduction, because EXL3 K3/K4 weight-only candidates and routed-expert
BPW are a different quantizer and rate definition.

## Promotion gate

No model card should say “beats TurboDerp” or “beats the Shapley paper” until:

- all compared arms share immutable source, candidate, panel, teacher, and
  evaluator hashes appropriate to the claim;
- realized rates are equal in the stated scope;
- an independent float64 replay reproduces tokenwise KLD;
- paired uncertainty and every per-row result are published; and
- the result is labeled allocator-only or pipeline-level without mixing them.
