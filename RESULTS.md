# ShapleyMCG results

This page is the human-readable index of measured results. Lower KLD is better.
Unless a row says otherwise, “BPW” means logical bits per routed-expert weight,
not whole-model stored BPW. Expanded-BF16 reconstruction means codec-produced
weights were installed into a BF16 Transformers model for controlled quality
measurement; it is not a compact-runtime benchmark.

The normalized machine-readable index is
[`results/RESULTS_LEDGER.json`](results/RESULTS_LEDGER.json). The cited sealed
reports, independent replays, comparison bundles, and immutable Hub revisions
are the evidence authority for each claim; the older
[`results/qwen-complete-results-ledger.json`](results/qwen-complete-results-ledger.json)
is retained as the historical source ledger. The longer human evidence map is
[here](docs/QWEN_COMPLETE_RESULTS_LEDGER.md).

## Headline conclusions

- On Qwen3-30B-A3B-Base at an identical exact 3.5 expert BPW, the full
  Aumann–Shapley/Fisher allocation reduced 20,480-position KLD by **7.731248%**
  relative to the predecessor Hessian/router allocation.
- On the post-trained Qwen parent, the full allocation reduced KLD by
  **13.766197%** relative to TurboDerp v0.0.1's carried-surplus allocation when
  both rules selected from the **same published K3/K4 reconstructions**. It won
  all 10 rows, and the paired interval excluded zero. This is an allocator-only
  result, not a claim that an independently encoded ShapleyMCG checkpoint beats
  every part of TurboDerp's conversion pipeline. The checked-in bundle does not
  yet attach the separate logit-level replay receipt required by this repo's
  strict promotion protocol.
- The full allocation also improved the predecessor allocation inside both the
  native MCG pool and the published TurboDerp pool. The separately calibrated
  R10/MCG candidate pipeline did not beat TurboDerp's published reconstructions
  in the matched post-trained full-body comparison, which motivated the
  additive multi-factory and progressive-state work.
- On the Base parent, a frozen-rate native/progressive whole-layer union looked
  better on its selection row but was **0.388% worse** on the nine untouched
  rows. The paired interval crossed zero. This is a useful negative diagnostic,
  not a joint factory/rate result.
- Cross-system comparisons with Hill's paper are context only. They do not
  reproduce the same parent, token panel, quantized scope, or W4A4 arithmetic.

## Qwen3-30B-A3B-Base

Parent:
`Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
The matched allocation rows keep all non-expert weights in source BF16.

### Full allocation versus predecessor allocation

| Panel | Allocation | Expert BPW | Positions | Mean KLD | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| GLM-lineage WikiText | predecessor routed-p2 | 3.5 | 2,047 | 0.062495726290 | 0.901319003420 |
| GLM-lineage WikiText | **full ShapleyMCG** | 3.5 | 2,047 | **0.055061070742** | **0.912066438691** |
| 10 x 2,048 WikiText, SDPA | predecessor routed-p2 | 3.5 | 20,480 | 0.049088886473 | 0.908203125000 |
| 10 x 2,048 WikiText, SDPA | **full ShapleyMCG** | 3.5 | 20,480 | **0.045293702727** | **0.910888671875** |

The full allocation lowered KLD by 11.896262% on the short gate and 7.731248%
on the main panel. It changed 9,392 of 18,432 matrix choices while preserving
exactly 9,216 K3 and 9,216 K4 choices. Independent float64 replay reproduced
both main-panel per-token vectors with zero maximum difference.

Detailed evidence:
[causal allocation result](docs/QWEN_CAUSAL_ALLOCATION_RESULT.md) and
[checked-in comparison](results/qwen3-30b-a3b-base/causal-vs-historical-sdpa-20k.json).

### Rate and score-blind controls

| Arm | Allocator / candidates | Expert BPW | Positions | Mean KLD |
| --- | --- | ---: | ---: | ---: |
| Original GLM-lineage control | predecessor / MCG | 3.5 | 2,047 | 0.063359493213 |
| Uniform K3 | uniform / MCG | 3.0 | 20,480 | 0.099432177790 |
| Earlier fixed-Hadamard selected K3/K4 control | predecessor / MCG | 3.5 | 20,480 | 0.050055817956 |
| Uniform K4 | uniform / MCG | 4.0 | 20,480 | 0.033991548914 |
| Five score-blind exact-3.5 allocations | score-blind / MCG | 3.5 | 20,480 each | 0.069974194650 mean |
| Reconstructed Hill BFCL/RULER panel | predecessor / MCG | 3.5 | 32,752 | 0.018260861970 |

The score-blind sample standard deviation was `0.010737570027`; its range was
`0.055956892764` to `0.080579437185`. The selected predecessor allocation was
28.4653% below that measured five-seed mean and beat all five seeds. The Hill
row is a reconstructed cross-system panel, not a paper reproduction.

The `0.050055817956` fixed-Hadamard row is an earlier sealed predecessor
allocation, not the matched `0.049088886473` control in the causal comparison
above; their coarse labels should not be read as duplicate reruns.

Detailed evidence:
[fixed-Hadamard control](docs/QWEN_B200_CONTROL_RESULT.md) and
[five-seed summary](results/qwen3-30b-a3b-base/naive-3p5-controls-summary.json).

### Progressive-state native/factory union diagnostic

This experiment kept every K3/K4 matrix rate from the validated causal
allocation fixed. Row 0 selected whole progressive-state layers; rows 1–9 were
excluded from every choice and then evaluated once.

| Arm | Selection row 0 mean KLD | Untouched rows 1–9 mean KLD | Rows better |
| --- | ---: | ---: | ---: |
| Native source-state MCG | 0.056077588647 | **0.044095493180** | 4/9 |
| Selected native/progressive union | **0.047448903735** | 0.044266736703 | 5/9 |

Selection admitted progressive layers 20, 34, 23, 18, 25, 37, 13, 14, 24,
and 39 and improved the selection-row mean by 15.387%. On untouched rows, the
union was worse by `0.000171243523` (0.388%). The paired interval for native
minus union was `[-0.001035685136, 0.000737749010]`, which crosses zero.

The correct conclusion is that this frozen-rate whole-layer selector did not
generalize a mean improvement. It says nothing decisive about a future
matrix-level joint factory/rate allocator, which was not run.

Report seal:
`54d3ecb81e350158e809db427b9a189cc56486a1da69ee27823ffe26978c28ba`.
Independent float64 replay seal:
`400bc0961b04906084ef3cbdc5be8f181b7942d58eb91dd03d514d527a9a3a44`.
The 26,158,691,040-byte sealed tree is remotely verified at
[Hugging Face revision `a7714174`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility/tree/a77141740749a53ede41d96115ba911f5b569f76/results/qwen3-30b-a3b-base/progressive-candidate-v1),
with manifest seal
`51793e22a35261ebddc1f9fb300c900947c12d07f64b01906abf8f82f1d34b96`.
That same immutable revision contains the
[`GLM-to-Qwen lineage receipt`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility/blob/a77141740749a53ede41d96115ba911f5b569f76/results/qwen3-30b-a3b-base/progressive-candidate-v1/glm-lineage/lineage.json),
whose downloaded SHA-256 is
`4e9ac680d13750ec2c5e1e1744701b788663db6d03323df0f2d59397b0909066`
and whose internal lineage seal is
`eadb5a8f579e7f40cd25ead031348c526cb6b6da0870e5a19fcb5eaf0831bee5`.

## Post-trained Qwen3-30B-A3B

Parent:
`Qwen/Qwen3-30B-A3B@4c446470ba0aec43e22ac1128f9ffd915f338ba3`.
These published comparisons use the same sealed 10 x 2,048 WikiText panel and
post-trained BF16 teacher. The non-expert scope is stated explicitly.

### Source-BF16-body allocation results

| Arm | Allocation | Expert candidates / BPW | Mean KLD | Top-1 agreement |
| --- | --- | --- | ---: | ---: |
| Predecessor control | predecessor routed-p2 | MCG / 3.5 | 0.041122632181 | 0.924267578125 |
| **Full causal arm** | **full ShapleyMCG** | MCG / 3.5 | **0.040368551746** | **0.925683593750** |
| Uniform expert K4 | uniform | MCG / 4.0 | 0.030848146342 | 0.935205078125 |
| Five score-blind exact-3.5 allocations | score-blind | MCG / 3.5 | 0.063024440483 mean | — |

At fixed candidates, parent, panel, and source-BF16 body, the full allocation
reduced KLD by 1.833736% and improved top-1 agreement by 0.141602 percentage
points. It changed 9,418 of 18,432 choices without changing the 9,216 K3 plus
9,216 K4 count.

Additional full-causal panels:

| Panel | Positions | Mean KLD | Top-1 agreement |
| --- | ---: | ---: | ---: |
| GLM-lineage WikiText, SDPA | 2,047 | 0.031452563917 | 0.931118710308 |
| Reconstructed Hill BFCL/RULER | 32,752 | 0.018888457889 | 0.973803126527 |

### Full-body and candidate-pool controls

| Arm | Allocation | Expert source / BPW | Non-expert scope | Mean KLD | Top-1 |
| --- | --- | --- | --- | ---: | ---: |
| Full-body MCG exact 3.5 | predecessor | MCG / 3.5 | MCG K4 attention, TurboDerp K6 head | 0.046834114393 | 0.916259765625 |
| Full-body MCG uniform K4 | uniform | MCG / 4.0 | MCG K4 attention, TurboDerp K6 head | 0.037766778251 | 0.924755859375 |
| Matched EXL3 exact 3.5 | predecessor choices | published TurboDerp K3/K4 / 3.5 | TurboDerp K4 body/K6 head | 0.030274917977 | 0.934472656250 |
| TurboDerp full K4 | uniform | published TurboDerp K4 / 4.0 | TurboDerp K4 body/K6 head | 0.021372109115 | 0.943457031250 |
| TurboDerp dense + MCG K4 experts | uniform | MCG / 4.0 | TurboDerp K4 body/K6 head | 0.035630073206 | 0.926611328125 |
| TurboDerp dense + MCG exact-3.5 experts | predecessor | MCG / 3.5 | TurboDerp K4 body/K6 head | 0.045562935671 | — |

These rows compare complete reconstruction pipelines. Calibration corpus,
progressive state, Hessian state, transforms, scales, and numeric encoding may
all differ; they do not isolate MCG or any other single codec component.

### Does the allocation transfer to another candidate pool?

| Allocation | Expert reconstruction pool | Mean KLD | Top-1 agreement |
| --- | --- | ---: | ---: |
| predecessor routed-p2 | published TurboDerp K3/K4 | 0.030274917977 | 0.934472656250 |
| **full ShapleyMCG** | published TurboDerp K3/K4 | **0.029269076647** | **0.937207031250** |
| predecessor routed-p2 | independently encoded R10/MCG | 0.045562935671 | — |
| full ShapleyMCG | independently encoded R10/MCG | 0.045793102543 | 0.918847656250 |

The causal allocation improved the unchanged TurboDerp candidate pool by
3.322359%, which shows that its signal transfers beyond the native MCG pool.
It did not improve the R10/MCG hybrid after the surrounding body changed; that
interaction is why final-body re-anchoring remains part of the method.

### Strict allocator comparison with TurboDerp v0.0.1

| Allocation rule | Identical candidate pool | Exact expert rate | Mean KLD | Top-1 agreement |
| --- | --- | ---: | ---: | ---: |
| TurboDerp v0.0.1 carried surplus | published TurboDerp K3/K4 | 3.5 | 0.033941535180 | 0.931201171875 |
| **Full ShapleyMCG causal** | published TurboDerp K3/K4 | 3.5 | **0.029269076647** | **0.937207031250** |

ShapleyMCG reduced KLD by **13.766197%**, improved top-1 agreement by
**0.600586 percentage points**, and won all 10 rows. A 200,000-draw seeded
row-block bootstrap gave `[0.0035278602, 0.0059290927]` for TurboDerp minus
ShapleyMCG KLD. Because the candidate bytes are identical, this is strong
matched evidence for an allocator-only advantage. TurboDerp did not publish a
native 3.5 checkpoint for this model, so it is not a whole-pipeline checkpoint
claim. The stored proof bundle seals both reports and token-KLD vectors, but a
separate logit-level replay receipt has not been attached; under the strict
promotion protocol, that attachment remains outstanding.

Checked-in proof:
[`results/qwen3-30b-a3b-posttrained/turboderp-v001-allocation-proof`](results/qwen3-30b-a3b-posttrained/turboderp-v001-allocation-proof/comparison.json).

### Frozen-rate candidate-factory union

At the full causal per-matrix K3/K4 choices, selection row 0 chose MCG
reconstructions for layers 5, 32, 34, 40, and 46 and retained published
TurboDerp reconstructions elsewhere. On that selection-only row, mean KLD
fell from `0.020135947869` to `0.016742801007` (16.851190%). That value is a
search diagnostic, not an endpoint.

| Untouched rows 1–9 | Mean KLD | Rows better |
| --- | ---: | ---: |
| Published TurboDerp candidates | 0.030283841053 | 4/9 |
| **Selected TurboDerp + MCG union** | **0.029368144806** | **5/9** |

The held-out mean fell by 3.023712%, but the paired 64-token block interval
`[-0.0001170474, 0.0019815816]` crosses zero. This is directional evidence that
the candidate pools are complementary, not a statistically decisive claim
that one factory is superior.

Full remotely verified tree:
[Hugging Face candidate-factory union](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility/tree/2de2d5e87e3493739784eec74b1446991b910614/results/qwen3-30b-a3b-posttrained-v1/candidate-factory-union-v1).

## External paper context

Hill et al. report `0.0353` for the additive method and `0.0429` for the
ModelOpt baseline in the nearest cited table. The reconstructed ShapleyMCG
panels are numerically lower, but the experiments do not share exact tokens,
parent, quantized scope, format alphabet, or W4A4 arithmetic. They therefore do
not establish superiority to the paper. The exact reproduction requirements
are in
[the superiority protocol](docs/TURBODERP_SHAPLEY_SUPERIORITY_PROTOCOL.md).

## What the evidence establishes

Established:

- the full causal allocator beats the predecessor allocator on the Base model
  with candidates, bytes, parent, teacher, tokens, backend, and body fixed;
- the allocation signal transfers to the published TurboDerp reconstruction
  pool;
- at equal expert rate and identical TurboDerp candidate bytes, the measured
  full-allocator arm is lower than TurboDerp v0.0.1's carried-surplus rule on
  all 10 rows, with the independent logit-replay attachment still outstanding;
  and
- raw five-node attribution leaves a material remainder, validating the
  method's explicit end-to-end anchor and remainder accounting.

Not established:

- strict superiority to Hill et al.;
- whole-pipeline superiority to a native TurboDerp 3.5 checkpoint;
- codec-only superiority of the independently calibrated R10/MCG path;
- compact BTX/EXL3 runtime quality, graph parity, or throughput; or
- a validated matrix-level joint factory/rate result.

## Published artifacts

- Base reproducibility dataset:
  [`brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility)
- Base expanded validation model:
  [`brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction`](https://huggingface.co/brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction)
- Post-trained reproducibility dataset:
  [`brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-posttrained-reproducibility)

See the [Base reproducibility audit](docs/QWEN_BASE_REPRODUCIBILITY_AUDIT.md)
for requirement-to-evidence mapping and remote verification receipts.
