---
license: other
pretty_name: ShapleyMCG Qwen3-30B-A3B reproducibility artifacts
---

# ShapleyMCG Qwen3-30B-A3B reproducibility artifacts

This dataset preserves the calibration statistics, exact corrected-R10
EXL3/MCG candidates, source and corpus identities, BF16 teacher/student logits,
tokenwise KLD, allocations, attribution ledgers, hashes, and publication
receipts for the Qwen3-30B-A3B experiments in
[`brandonmmusic-max/shapleymcg`](https://github.com/brandonmmusic-max/shapleymcg).

## Causal allocation result

The complete two-level Aumann–Shapley/Fisher allocation improves on the
historical Hessian/router allocation at an identical 3.5 logical bits per
routed-expert weight. The primary comparison holds the Base revision, exact MCG
candidates, BF16 teacher, 20,480 token positions, SDPA attention, every
non-expert weight, and 9,216 K3 plus 9,216 K4 choices fixed.

| Exact-3.5 routed-expert allocation | Mean KLD | Top-1 agreement |
|---|---:|---:|
| Historical Hessian/router | 0.04908888647295481 | 0.908203125 |
| **Aumann–Shapley/Fisher causal** | **0.04529370272688347** | **0.910888671875** |

This is a **7.731248% KLD reduction** and a **0.268555 percentage-point**
top-1 gain. Independent NumPy float64 replay reproduced both stored per-token
vectors with zero maximum difference. A separate 2,047-position SDPA comparison
also favors the causal allocation by 11.896262%.

- Parent: `Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`
- Candidate inventory:
  `205cdd1e0b36dcaa545e551df9ca3eb88a6ebb385276236fc45a3f105c25b4e9`
- Causal allocation:
  `10bb3b71a2258500b281e3b37d57657e5e0ecf9318bb9c1a3fc6dbd2326e63c4`
- 20k panel:
  `6c509f3496d5a1fe6739c58ffa152c3d4202932f9d8a6514e55d17f8a4cfa741`
- 20k comparison:
  `8d8438df03d0f9a34a4a5a79912dd232baa401f24d8e6e14bb942bf00062a608`
- Causal/historical verifier seals:
  `6e4215f276d69ca53ce1ed29c2a900f5e040d41594a9e2ab24cb247ddec90265` /
  `752e83112522f1f8bfe2cbf7797196cc7961164b7617ccb22d92f7e60c8a1911`

The raw five-node layer attribution summed to `0.0306227734756427` against a
measured uniform-K4 endpoint of `0.04401261771637709`. The unresolved
`0.013389844240734392` remainder (30.42%) is published explicitly. Reconciled
expert scores close to measured layer attribution and reconciled layer scores
close to endpoint KLD, but exact bookkeeping closure is not represented as raw
proxy additivity.

The full causal evidence is under `causal-arm-v3/`. Teacher and both allocation
arms' panel logits, reports, per-token vectors, independent verifiers,
attribution/reconciliation artifacts, and retention receipts are preserved.

## Verified validation model publication

The exact expanded validation reconstruction is published at
[`brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction`](https://huggingface.co/brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction).

- Manifest-verified model revision:
  `c3447f3f8e231ae83afa10ba263346d1ceb98c11`
- Publication-receipt revision:
  `fbdd05904fd4018dea9fd0f25ed4800c15df1493`
- Model manifest seal:
  `32fec13a96c6c85eb394939a6eb511c821bd05e8ac0960268852d33150d0cde1`
- Publication receipt seal:
  `e7b3b344321c11e872fc006cbd009c935b9fdbf0fdb8b2b0a99197cce14bffe8`
- Verified content inventory: 31 files and 61,079,798,645 bytes; every file
  matched its remote size and either Hub LFS SHA-256 or Git-blob SHA-1.
- Post-assembly verification: all 18,432 selected expert-matrix
  reconstructions were installed, and the reloaded checkpoint reproduced the
  sealed 2,047-position student logit tensor exactly (`max_abs_delta = 0`).

The receipt describes and verifies the immutable model revision above. The
repository head may advance for documentation without changing that sealed
content revision.

## Historical fixed-Hadamard control

The first 48-layer corrected-R10 K3/K4 control remains preserved as a separate
historical experiment:

- exact routed-expert rate: 9,216 K3 plus 9,216 K4 matrix choices;
- 2,047-position GLM-style mean KLD: `0.06335949321311507`;
- historical allocation:
  `5edda4ab29f9961f6f2ab5be3aab6cc0aaa3817afa325b40fd5063fef05de034`;
- KLD report:
  `3685fa5eac6064aa6bd5e51ddc53fb10d791c0744c955f551283fdcad62469ff`.

Uniform K3/K4 endpoints and five score-blind exact-3.5 controls are retained
under `results/qwen3-30b-a3b-v1`. Those earlier panels used their documented
attention backend and must not be silently mixed with the matched SDPA causal
comparison above.

## Reconstructed paper panel

The reconstructed Hill-style BFCL/RULER panel measured
`0.018260861970005038` over 32,752 positions with the predecessor pipeline. It
is 48.3% below the paper's reported `0.0353` additive row and 57.4% below its
`0.0429` ModelOpt row, but it is cross-system evidence, not a strict allocator
head-to-head: exact author tokens were unavailable, the parent and arithmetic
differ, and the new causal allocation has not been measured on that panel.

## Format and runtime boundary

The exact validation model installs selected codec-produced reconstructions
into a Transformers BF16 checkpoint. It is intentionally an expanded
validation reconstruction, not a compact 3.5-bpw checkpoint. Current official
BTX couples gate/up rates, while this allocation selects those matrices
independently. CUDA-graph, packed-runtime, and throughput qualification are not
claimed by these KLD artifacts.

## Attribution

- Base model: the Qwen team.
- EXL3/TRELLIS and MCG codec lineage: turboderp and ExLlamaV3 contributors.
- Aumann–Shapley quantization precedent: Joshua Hill,
  [*Saturation Makes Quantization Error Additive*](https://arxiv.org/abs/2607.12266),
  and [NVIDIA Model Optimizer PR #2183](https://github.com/NVIDIA/Model-Optimizer/pull/2183).
- Routed Fisher/Jacobian attribution, explicit residual reconciliation, MCG
  integration, experiments, and publication: Brandon Music / ShapleyMCG.

See the repository's `docs/REFERENCES.md`, `THIRD_PARTY_NOTICES.md`, and
`THIRD_PARTY_LICENSES` for complete citations and license boundaries.
