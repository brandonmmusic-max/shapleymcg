# Qwen3-30B-A3B post-trained predecessor-pipeline matched TurboDerp comparison

> **Allocator identity:** every ShapleyMCG-labeled number in this report uses
> corrected MCG candidates selected by the historical minimum diagonal
> routed-p2 Hessian/router allocator. It does **not** use the full-model
> Aumann–Shapley/Fisher causal allocator. The latter is a separate experiment
> and is only labeled full ShapleyMCG after executing the complete method in
> [SHAPLEYMCG_METHOD.md](SHAPLEYMCG_METHOD.md).

## Outcome

This experiment uses the post-trained `Qwen/Qwen3-30B-A3B` parent, the same
sealed 10 x 2,048-token WikiText-2 panel and BF16 teacher logits, and a common
BF16 Transformers replay for every arm. EXL3 runtime kernels, packing, CUDA
graphs, and KV-cache formats are not executed during KLD measurement. Any gap
therefore comes from the reconstructed weights rather than runtime fusion.

| Expert rate and body scope | ShapleyMCG mean KLD | TurboDerp mean KLD | ShapleyMCG top-1 | TurboDerp top-1 |
| --- | ---: | ---: | ---: | ---: |
| Exact 3.5 expert BPW; K4 q/k/v/o; K6 head | 0.046834114392727964 | 0.030274917976982833 | 0.916259765625 | 0.93447265625 |
| Uniform K4 experts; K4 q/k/v/o; K6 head | 0.03776677825098351 | 0.02137210911467856 | 0.924755859375 | 0.94345703125 |

At exact 3.5 expert BPW, the current full-body ShapleyMCG reconstruction has
54.6961% higher KLD than the matched TurboDerp-codec arm. At uniform K4 it has
76.7106% higher KLD. These are negative results and are reported as such.

## What the ablation establishes

Before attention quantization, the ShapleyMCG selected 3.5-BPW experts with
source-BF16 attention measured `0.04112263218133531`. Replacing q/k/v/o with
ShapleyMCG K4 reconstructions increased KLD by 13.8889% to
`0.046834114392727964`. The uniform-K4 expert arm similarly increased from
`0.030848146342326514` to `0.03776677825098351`, or 22.4280%.

This falsifies the narrow hypothesis that keeping the quantized body in one
MCG-based encoding system would itself close the TurboDerp gap. The leading
differences that remain are:

1. TurboDerp uses EXL3's default trellis codebook; the corrected-R10 candidate
   path used here forces MCG.
2. TurboDerp calibrates all experts with a 204,800-token corpus, whereas this
   run used 69,632 source tokens followed by each expert's routed subset.
3. TurboDerp advances calibration state through already-quantized predecessor
   layers; the ShapleyMCG expert Hessians were captured independently from the
   BF16 source state.
4. TurboDerp's automatic output-scale policy can decline a scale when Hessian
   skew makes it unsafe; this encoder always applies its richer scale scheme.

The next causal experiment should retain the selected Shapley allocation while
swapping these factors one at a time, starting with trellis versus MCG under
identical Hessians and scaling. The already measured hybrid supports that
priority: the selected allocation encoded with TurboDerp's codec measured
`0.030274917976982833`.

## Positive allocation claim

At exactly 3.5 routed-expert logical BPW, the selected predecessor allocation
measured `0.04112263218133531` against a five-seed score-blind mean of
`0.06302444048293131`, a 34.7513% KLD reduction. None of the five blind
allocations beat it. This supports the predecessor allocation; it does not
establish that the current encoder beats TurboDerp or that the full causal
allocator has been tested on this post-trained parent.

## Subsequent full causal allocation result

The complete Aumann–Shapley/Fisher allocator was subsequently executed on the
same post-trained parent and retained MCG candidate inventory. On this same
20,480-position eager panel with source-BF16 non-expert weights, exact 3.5
expert BPW, teacher, tokens, and evaluator fixed, it measured KLD
`0.040368551745534186` and top-1 `0.92568359375`. The predecessor row measured
`0.04112263218133531` and `0.924267578125`, respectively. The full allocator
therefore reduced KLD by 1.833736% and gained 0.141602 percentage points of
top-1 agreement. Independent float64 replay reproduced the tokenwise KLD with
zero maximum difference.

This establishes a matched improvement from the full allocator on the
post-trained parent. It does not retroactively turn the earlier rows into
full-method results, and it does not establish that MCG encoding beats the
TurboDerp trellis codec.

## Full causal allocation through both codecs

The causal allocation was then installed into both codecs while fixing the
post-trained parent, sealed 20,480-position panel, TurboDerp K4 body/K6 head,
and the exact 9,216 K3 plus 9,216 K4 expert-matrix count.

| Expert codec | Predecessor allocation KLD | Causal allocation KLD | Causal top-1 |
|---|---:|---:|---:|
| TurboDerp trellis | 0.0302749179770 | **0.0292690766473** | **0.937207031250** |
| ShapleyMCG MCG | 0.0455629356710 | 0.0457931025429 | 0.918847656250 |

The causal choices lower TurboDerp-codec KLD by **3.322359%** and improve
top-1 agreement by **0.273438 percentage points**, proving that the allocation
signal transfers beyond the MCG encoder. At the identical causal choices and
body scope, however, MCG KLD is **56.455576% higher** than TurboDerp KLD. The
MCG hybrid also regresses 0.505163% versus its predecessor-allocation hybrid.
That context reversal shows that expert allocation interacts with the
quantized non-expert body: a source-BF16 allocation must be re-anchored before
claiming the same ordering in a fully quantized body.

The logical result is therefore two-part: the full causal allocator is a real
improvement, while the current MCG codec/calibration path—not allocation—is the
larger remaining quality bottleneck.

## Seals and lineage

- Source revision: `4c446470ba0aec43e22ac1128f9ffd915f338ba3`
- TurboDerp K4 revision: `0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef`
- Evaluation panel SHA-256: `f67b22af3930184864d12576796807b29552bda3f55dcaa7d6af1d8b262ef9fb`
- Full-scope summary SHA-256: `3ea7ccd00d252a01ae9f7362286e2d9a31156d09d9b3058c08ef619ed59d7bf9`
- Publication manifest SHA-256: `d8cf343c1712923ddebdb239ab8dcbd8913c23abd9d9fa8b40e921e1fc06a8d1`
- Implementing Git revision: `0071eb1400c2f6ff6acbe0862d6854a45c5e3957`

The Hugging Face reproducibility dataset stores the two attention-Hessian
shards, 48 attention reconstruction receipts, raw student logits, tokenwise
KLD reports, logs, `MANIFEST.json`, and `SHA256SUMS` under
`fullscope-extension-v1/`.
