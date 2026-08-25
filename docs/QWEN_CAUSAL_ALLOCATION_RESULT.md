# Qwen3-30B-A3B-Base causal allocation result

## Outcome

The complete two-level Aumann–Shapley/Fisher allocation arm improved on the
historical Hessian/router allocation at an identical 3.5 logical bits per
routed-expert weight. The primary replication uses 20,480 next-token positions
and holds the parent revision, BF16 teacher logits, token IDs, SDPA attention,
candidate reconstructions, K3/K4 counts, and every non-expert weight fixed.
Only placement of the K3 and K4 expert-matrix reconstructions changes.

| Matched SDPA panel | Historical allocation | Causal allocation | Relative KLD reduction | Top-1 change |
|---|---:|---:|---:|---:|
| 2,047-position sealed window | 0.06249572628978888 | 0.05506107074246945 | 11.896262% | +1.074744 pp |
| **20,480-position WikiText panel** | **0.04908888647295481** | **0.04529370272688347** | **7.731248%** | **+0.268555 pp** |

Both 20k arms were independently replayed with NumPy float64 log-softmax and
`KL(BF16 teacher || quantized student)`. Each replay reproduced its stored
per-token vector with a maximum difference of exactly zero. The causal and
historical top-1 agreements are `0.910888671875` and `0.908203125`.

The checked-in sealed comparisons are:

- [`causal-vs-historical-sdpa-single-window.json`](../results/qwen3-30b-a3b-base/causal-vs-historical-sdpa-single-window.json), seal
  `7d96fd707a8b195f12bbac28801b77957e2b3cfb49cd3c449ce394737b37b481`;
- [`causal-vs-historical-sdpa-20k.json`](../results/qwen3-30b-a3b-base/causal-vs-historical-sdpa-20k.json), seal
  `8d8438df03d0f9a34a4a5a79912dd232baa401f24d8e6e14bb942bf00062a608`.

## Attribution and allocation that actually ran

The experiment did not relabel the historical allocation as Shapley-based. It
executed the full attribution path and bound the resulting scores to the exact
candidate inventory:

1. Measure the full-model uniform-K4 endpoint KLD, `0.04401261771637709`.
2. Integrate layer effects over five Aumann–Shapley path nodes.
3. Split each layer attribution among routed experts with actual expert-output
   residuals, a downstream Fisher/Jacobian sketch, and cross-expert terms.
4. Preserve the unexplained remainder, then reconcile expert shares to their
   measured layer attribution and layer shares to the measured endpoint.
5. Solve the exact-rate global allocation over the existing decoded MCG K3/K4
   candidates and measure the resulting model end to end.

The raw five-node layer attribution summed to `0.0306227734756427`; the endpoint
was `0.04401261771637709`. The unresolved `0.013389844240734392` remainder is
30.42% of the endpoint. The published reconciled ledger closes exactly, but the
raw quadrature is not represented as naturally additive. This distinction is
important: exact bookkeeping closure is not evidence that the proxy explained
all nonlinear and routing interactions.

The final allocation contains exactly 9,216 K3 and 9,216 K4 expert-matrix
choices. It changes 9,392 of 18,432 choices relative to the historical
allocation without changing the logical rate or stored candidate payload
budget. The allocation seal is
`10bb3b71a2258500b281e3b37d57657e5e0ecf9318bb9c1a3fc6dbd2326e63c4`.

## What this says about the method

The matched experiments support a causal claim about allocation: within this
Qwen Base/MCG candidate system, the new two-level attribution and exact-rate
allocation choose better placements than the prior Hessian/router method. The
20k result is the primary estimate because it is broader; the same-direction
2,047-position result is a replication on a separate sealed window.

The earlier reconstructed Hill-style BFCL/RULER panel remains useful only as
numerical context. The historical ShapleyMCG predecessor pipeline measured
`0.018260861970005038` over 32,752 positions, while the paper reports `0.0353`
for its additive row and `0.0429` for its ModelOpt row. No superiority inference
follows: the exact author tokens were not published, the parent checkpoint,
quantized scope, and activation/runtime arithmetic differ, and routed-expert
BPW is not the paper's whole-model effective-bit measure. The new causal
allocation has also not been measured on that reconstructed panel.

Accordingly, the supported claim is:

> At fixed Qwen3-30B-A3B-Base parent, MCG candidates, SDPA arithmetic, and exact
> 3.5 routed-expert BPW, the proposed two-level Aumann–Shapley/Fisher allocation
> reduced mean next-token KLD by 7.73% over the historical Hessian/router
> allocation across 20,480 positions, with an independently verified 0.269
> percentage-point increase in top-1 agreement.

It is not yet appropriate to say that the allocator itself beats Hill by 48.3%.
A strict paper-method comparison requires both allocation methods to consume
the same candidates, tokens, parent, quantized scope, arithmetic, and bit-cost
accounting.

## Reproducibility and publication

- Parent: `Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`
- Candidate inventory seal:
  `205cdd1e0b36dcaa545e551df9ca3eb88a6ebb385276236fc45a3f105c25b4e9`
- 20k panel seal:
  `6c509f3496d5a1fe6739c58ffa152c3d4202932f9d8a6514e55d17f8a4cfa741`
- Causal panel report / verification seals:
  `e80ea25fd29f9adad8dd043698d912ead9c0d3124758a6666dacc7a9338d3912` /
  `6e4215f276d69ca53ce1ed29c2a900f5e040d41594a9e2ab24cb247ddec90265`
- Historical panel report / verification seals:
  `01a3dfef796de6e35c087b133d655534fb00badcfd0374ac695e84cef588a7ea` /
  `752e83112522f1f8bfe2cbf7797196cc7961164b7617ccb22d92f7e60c8a1911`
- Dataset:
  [`brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility)
- Manifest-verified validation model revision:
  `c3447f3f8e231ae83afa10ba263346d1ceb98c11`
- Validation-model publication-receipt revision:
  `fbdd05904fd4018dea9fd0f25ed4800c15df1493`
- Dataset receipt mirror:
  `causal-arm-v3/validation-model-hf-publication-receipt.json`
- Model manifest / publication receipt seals:
  `32fec13a96c6c85eb394939a6eb511c821bd05e8ac0960268852d33150d0cde1` /
  `e7b3b344321c11e872fc006cbd009c935b9fdbf0fdb8b2b0a99197cce14bffe8`

The validation model is an expanded BF16 reconstruction of the selected MCG
weights, not a compact 3.5-bpw runtime checkpoint. Packed-runtime and throughput
qualification remain separate from the allocation-quality result.

`scripts/assemble_qwen_validation_model.py` supports clean-node reconstruction
after incremental candidate reclamation. When a local `layer-NNN` candidate is
absent, `--candidate-hf-repo`, `--candidate-hf-revision`, and
`--candidate-download-root` restore only the layers needed by the current model
shard. Every restored file must match the candidate-file SHA256 bound by the
measured KLD report; every selected tensor and persisted model tensor is checked
again, and only the temporary restored copy is removed after that shard is
sealed. The final checkpoint is reloaded with SDPA and must reproduce the
measured student logit tensor exactly.

The completed assembly installed all 18,432 selected expert-matrix
reconstructions. Reloading the published checkpoint reproduced the sealed
2,047-position student logits bit-for-bit: both raw tensor SHA-256 values are
`548e4f67db7a9e1b085db655cdfa280ba0fee98b6821187e3b6ddfd23e1f19b0`
and `max_abs_delta` is exactly zero. The manifest-verified remote inventory is
31 files totaling 61,079,798,645 bytes.

## Attribution

Joshua Hill's *Saturation Makes Quantization Error Additive* and NVIDIA Model
Optimizer PR #2183 are the direct modern Aumann–Shapley quantization precedent.
EXL3/TRELLIS and MCG codec lineage belongs to turboderp and ExLlamaV3
contributors. The base model belongs to the Qwen team. The routed Fisher/
Jacobian decomposition, explicit residual reconciliation, MCG integration,
experiment, and publication are the ShapleyMCG work described in this
repository. Full citations and license boundaries are in
[`REFERENCES.md`](REFERENCES.md) and the third-party notices.
