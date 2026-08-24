# Qwen3-30B-A3B complete result ledger

This is the human-readable index of every headline KLD result produced by the
Qwen/B200 campaign. The machine ledger in
`results/qwen-complete-results-ledger.json` is authoritative for exact seals.
Lower KLD is better. “Expert BPW” covers routed-expert weight elements only;
dense attention, embedding, router, normalization, and head scope are stated
separately.

## Naming correction

The first Base and post-trained experiments used corrected MCG candidates but
selected them with the historical diagonal routed-p2 Hessian/router objective.
They are **predecessor-pipeline** results. They did not execute the full
Aumann–Shapley/Fisher causal allocator. The full method is defined in
[SHAPLEYMCG_METHOD.md](SHAPLEYMCG_METHOD.md).

## Base parent: controlled allocation results

Parent revision: `Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
All rows below use expanded-BF16 reconstruction and BF16 non-expert source
weights; they are not compact runtime measurements.

| Panel and arm | Allocator | Expert rate | Attention | Positions | Mean KLD | Top-1 |
|---|---|---:|---|---:|---:|---:|
| GLM-lineage WikiText; matched control | predecessor routed-p2 | 3.5 | source BF16, SDPA | 2,047 | 0.0624957262898 | 0.901319003420 |
| GLM-lineage WikiText; matched causal | **full ShapleyMCG** | 3.5 | source BF16, SDPA | 2,047 | **0.0550610707425** | **0.912066438691** |
| 10 x 2,048 WikiText; matched control | predecessor routed-p2 | 3.5 | source BF16, SDPA | 20,480 | 0.0490888864730 | 0.908203125000 |
| 10 x 2,048 WikiText; matched causal | **full ShapleyMCG** | 3.5 | source BF16, SDPA | 20,480 | **0.0452937027269** | **0.910888671875** |

The full allocator lowered KLD by 11.896262% on the 2,047-position window and
7.731248% on the 20,480-position panel. It changed 9,392 of 18,432 matrix
choices while retaining exactly 9,216 K3 and 9,216 K4 choices. Independent
float64 replay reproduced the 20k tokenwise vectors with zero maximum
difference.

## Base parent: predecessor controls and reconstructions

| Panel and arm | Allocator / codec | Expert rate | Positions | Mean KLD | Top-1 |
|---|---|---:|---:|---:|---:|
| Original GLM-lineage control | predecessor / MCG | 3.5 | 2,047 | 0.0633594932131 | — |
| WikiText 10 x 2,048 uniform K3 | uniform / MCG | 3.0 | 20,480 | 0.0994321777898 | — |
| WikiText 10 x 2,048 selected | predecessor / MCG | 3.5 | 20,480 | 0.0500558179565 | 0.908447265625 |
| WikiText 10 x 2,048 uniform K4 | uniform / MCG | 4.0 | 20,480 | 0.0339915489141 | — |
| WikiText score-blind 3.5, five-seed mean | score-blind / MCG | 3.5 | 20,480 each | 0.0699741946501 | — |
| Reconstructed Hill BFCL/RULER panel | predecessor / MCG | 3.5 | 32,752 | 0.0182608619700 | — |

The score-blind sample SD was `0.0107375700273`, with range
`0.0559568927638`–`0.0805794371846`; none of five beat the selected predecessor
allocation. The Hill row is a reconstruction, not a strict paper reproduction:
the author token panel was unavailable, the parent/scope differ, and this run
uses BF16 replay rather than the paper's W4A4 NVFP4 execution.

## Post-trained parent: predecessor allocation and codec controls

Parent revision: `Qwen/Qwen3-30B-A3B@4c446470ba0aec43e22ac1128f9ffd915f338ba3`.
Panel: the same sealed 10 x 2,048 WikiText tokens and post-trained BF16 teacher.
These published rows use eager Transformers BF16 replay with KV cache disabled.

| Arm | Allocator | Expert codec/rate | Non-expert scope | Mean KLD | Top-1 |
|---|---|---|---|---:|---:|
| Selected experts | predecessor routed-p2 | MCG, exact 3.5 | source BF16 | 0.0411226321813 | 0.924267578125 |
| Uniform expert K4 | uniform | MCG, 4.0 | source BF16 | 0.0308481463423 | 0.935205078125 |
| Score-blind 3.5, five-seed mean | score-blind | MCG, exact 3.5 | source BF16 | 0.0630244404829 | — |
| Full-body MCG exact 3.5 | predecessor routed-p2 | MCG, exact 3.5 | MCG K4 q/k/v/o; TurboDerp K6 head | 0.0468341143927 | 0.916259765625 |
| Full-body MCG uniform K4 | uniform | MCG, 4.0 | MCG K4 q/k/v/o; TurboDerp K6 head | 0.0377667782510 | 0.924755859375 |
| Matched EXL3 exact 3.5 | same predecessor choices | TurboDerp trellis, exact 3.5 | TurboDerp K4 body/K6 head | 0.0302749179770 | 0.934472656250 |
| TurboDerp full K4 | uniform | TurboDerp trellis, 4.0 | TurboDerp K4 body/K6 head | 0.0213721091147 | 0.943457031250 |
| TurboDerp dense + MCG K4 experts | uniform | MCG, 4.0 | TurboDerp K4 body/K6 head | 0.0356300732056 | 0.926611328125 |
| TurboDerp dense + MCG exact-3.5 experts | predecessor routed-p2 | MCG, exact 3.5 | TurboDerp K4 body/K6 head | 0.0455629356710 | — |

The predecessor allocation was 34.7513% below the five score-blind mean, but
the matched codec swap favored TurboDerp: at the identical expert choices and
dense scope, MCG KLD was 50.4973% higher. Adding MCG K4 attention also worsened
the MCG predecessor arm. These results support a useful predecessor allocation
signal while falsifying codec superiority.

## Post-trained parent: full causal arm

The full post-trained Aumann–Shapley/Fisher attribution and exact-rate
allocation uses the retained, identical MCG candidate inventory. The uniform-K4
path endpoint measured `0.0242691167199`; raw five-node attribution left an
explicit `0.00722590999461` remainder (29.7741% of the endpoint). The causal
allocation changed 9,418 of 18,432 matrix choices without changing the exact
9,216 K3 / 9,216 K4 rate or stored payload bytes.

| Panel and arm | Allocator | Expert rate | Attention | Positions | Mean KLD | Top-1 |
|---|---|---:|---|---:|---:|---:|
| WikiText 10 x 2,048; matched control | predecessor routed-p2 | 3.5 | source BF16, eager | 20,480 | 0.0411226321813 | 0.924267578125 |
| WikiText 10 x 2,048; matched causal | **full ShapleyMCG** | 3.5 | source BF16, eager | 20,480 | **0.0403685517455** | **0.925683593750** |
| GLM-lineage WikiText gate | **full ShapleyMCG** | 3.5 | source BF16, SDPA | 2,047 | 0.0314525639171 | 0.931118710308 |
| Reconstructed Hill BFCL/RULER panel | **full ShapleyMCG** | 3.5 | source BF16, eager | 32,752 | 0.0188884578892 | 0.973803126527 |

On the decisive matched eager panel, the full allocator reduced KLD by
**1.833736%** (`0.000754080435801` absolute) and gained **0.141602 percentage
points** of top-1 agreement. The independent float64 replay reproduced the
20,480-position tokenwise vector with zero maximum difference. The separate
2,047-position SDPA gate reproduced its mean within `6.9e-16` and had maximum
per-token replay difference `7.39e-13`.

The reconstructed Hill-panel value was independently replayed in float64 with
zero mean and maximum tokenwise difference; verification seal
`1b22ff77e43e35ed6d94ff7bdf02e7100460696b0cbf19703608ec9ec388d5c5`.

Primary Hugging Face publication commit:
`15e3c0f810c364bccb0770e39006482d70b61cb1`; 48 manifest entries,
15,011,224,030 bytes, manifest seal
`9abb3061ad744fb1826f2bb503d900c42c8a2ff896b48653a5345b8325c99ca5`,
verification receipt
`95f701dd4bea586e97293dd26bfd158a6aef38f127664569a1830d640bb016d3`.

### Causal-allocation codec transfer control

The causal allocation was also replayed through both expert codecs with the
same post-trained parent, sealed 20,480-position panel, TurboDerp K4 body/K6
head, and exact 9,216 K3 / 9,216 K4 matrix count.

| Arm | Allocation | Expert codec | Mean KLD | Top-1 |
|---|---|---|---:|---:|
| Prior matched EXL3 control | predecessor routed-p2 | TurboDerp trellis | 0.0302749179770 | 0.934472656250 |
| Causal matched EXL3 control | **full ShapleyMCG** | TurboDerp trellis | **0.0292690766473** | **0.937207031250** |
| Prior matched MCG hybrid | predecessor routed-p2 | MCG | 0.0455629356710 | — |
| Causal matched MCG hybrid | **full ShapleyMCG** | MCG | 0.0457931025429 | 0.918847656250 |

On TurboDerp's unchanged codec, the causal allocation lowers KLD by
**3.322359%** and improves top-1 agreement by **0.273438 percentage points**.
This demonstrates that the allocation improvement transfers to a second
encoder and is not merely fitted to MCG reconstruction noise. With allocation
and non-expert scope fixed, however, MCG KLD is **56.455576% higher** than
TurboDerp KLD (equivalently, TurboDerp is 36.084093% lower). The MCG hybrid is
also 0.505163% worse than its predecessor-allocation hybrid. The latter does
not contradict the source-BF16 matched allocation win: it shows that quantizing
the surrounding body changes expert/non-expert interactions enough to require
allocation or re-anchoring in the final body context.

Independent float32-formula replays reproduced both 20,480-position tokenwise
vectors with zero maximum difference. Verification seals are
`7f999884d53c90208e96c95f0870026ea3d77206f557793a153ba27bdb1c8888`
(TurboDerp) and
`a801e6d5cf750429a2aa1dca9861d937cd973bd785751e116fa8c0657123bf02`
(MCG).

## External paper context

Hill et al. report `0.0353` for their additive method and `0.0429` for their
ModelOpt baseline in the closest cited paper table. The full-method
post-trained reconstruction scores `0.0188884578892`, numerically 46.49% and
55.97% lower, respectively. The Base predecessor reconstruction scores
`0.0182608619700`, numerically 48.3% and 57.4% lower. Neither comparison proves
method superiority because the checkpoint, exact tokens, quantized scope, and
execution arithmetic are not strictly matched. They are cross-system context
only.

## What is and is not established

- Established: on Base, the full causal allocation beats the predecessor at
  fixed candidates, bytes, parent, teacher, tokens, backend, and non-expert
  scope on both measured panels.
- Established: on the post-trained parent, the predecessor allocation beats
  five direct score-blind allocations, but the present MCG codec/calibration
  stack loses to matched TurboDerp trellis reconstructions.
- Established: the full causal allocation improves TurboDerp's unchanged
  exact-3.5 codec arm by 3.322359%, so the allocation signal transfers across
  codecs. At the same causal choices and body scope, the current MCG expert
  reconstruction remains the dominant measured quality deficit.
- Established: the raw five-node attribution leaves a stable material
  remainder (29.77% post-trained; 30.42% Base), so exact end-to-end anchors are
  necessary and reconciliation must remain labeled accounting rather than raw
  additivity.
- Not established: strict superiority to Hill et al. or packed-runtime
  quality/parity. The matched post-trained allocation improvement is now
  established, but it is modest and does not by itself establish codec
  superiority.
- Never measured: an interpolated K3/K4 midpoint is not a naive 3.5 result.

Full logits, tokenwise KLD, candidate receipts, and hashes are stored in the two
Hugging Face reproducibility datasets linked from the repository README.
