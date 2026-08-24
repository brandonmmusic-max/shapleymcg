# Scientific lineage and acknowledgements

ShapleyMCG is an independent implementation and synthesis. It does not claim
authorship of the underlying model, trellis format, Aumann-Shapley method, or
the research ideas cited below. No third-party source is vendored unless a
file-level license and notice explicitly say otherwise.

## Methods implemented in this repository

| Repository component | Scientific or software lineage | How it is used here |
| --- | --- | --- |
| Aumann-Shapley path attribution | Robert J. Aumann and Lloyd S. Shapley, *Values of Non-Atomic Games* (Princeton University Press, 1974); Joshua Hill, [*Saturation Makes Quantization Error Additive: A Coverage Model with a Certificate*](https://arxiv.org/abs/2607.12266); Joshua Hill and NVIDIA, [Model Optimizer PR #2183](https://github.com/NVIDIA/Model-Optimizer/pull/2183) | Full-model next-token KL is differentiated at canonical Gauss-Legendre points along a simultaneous source-to-decoded path. The implementation is original code, but Hill's paper and ModelOpt work are the direct modern quantization precedent. |
| Trellis quantization and incoherence processing | Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa, [*QTIP: Quantization with Trellises and Incoherence Processing*](https://arxiv.org/abs/2406.11235); Albert Tseng et al., [*QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*](https://arxiv.org/abs/2402.04396) | The corrected exact-codec path uses the EXL3/MCG trellis family and Hadamard-style incoherence processing rather than claiming a new decode format. |
| EXL3 encoder/runtime ecosystem | [turboderp and ExLlamaV3 contributors](https://github.com/turboderp-org/exllamav3); [EXL3 format notes](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md) | Exact K3/K4/K5 encoding, reconstruction, and packed-byte accounting are bound to a pinned corrected EXL3 numeric core and an extension rebuilt from the exact v0.0.43 commit. EXL3 describes itself as a QTIP variant. |
| Global expert-level mixed precision | Wei Huang et al., [*MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More*](https://arxiv.org/abs/2410.06270); V. Malinovskii et al., [*Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*](https://arxiv.org/abs/2411.17525) | Exact payload-byte Pareto frontiers feed one model-wide multiple-choice knapsack. The implementation retains measured end-to-end gates because low-bit additivity is an experimental question. |
| Quantized-point and end-loss guidance | Y. Hu et al., [*Identifying Sensitive Weights via Post-quantization Integral*](https://arxiv.org/abs/2503.01901); J. Kim et al., [*GuidedQuant*](https://arxiv.org/abs/2505.07004) | These works motivate measuring damage at the quantized point and model output. In ShapleyMCG, proxy scores propose candidates; sealed final-logit KL adjudicates them. |
| Router-aware MoE calibration and route-shift diagnostics | X. Hu et al., [*MoEQuant*](https://arxiv.org/abs/2505.03804); Y. Chen et al., [*EAC-MoE*](https://arxiv.org/abs/2508.01625); H. Park et al., [*VSRAQ*](https://arxiv.org/abs/2606.05688) | Router-mass covariance arms, route-set/mass agreement, and an explicit routing/backend residual are measured rather than folded invisibly into expert damage. |
| Base control model | [Qwen Team, Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) and the [Qwen3 technical report](https://arxiv.org/abs/2505.09388) | The first controlled pilot is the 48-layer, 128-expert base MoE. Qwen model weights and code remain under their own license. |
| BTX checkpoint and runtime ecosystem | [Luke Alonso](https://github.com/lukealonso), [Local Inference Lab B12X](https://github.com/local-inference-lab/b12x), and its contributors | The official checkpoint writer/auditor ports atom assembly from the pinned upstream `btx_synth.py` and validates against the pinned `btx_schema.py`/`btx.py` closure. B12X is Apache-2.0; the derived module is SPDX-marked and the complete pinned upstream license is preserved under `THIRD_PARTY_LICENSES`. Runtime qualification remains a separate gate. |

## Original control carried forward

The absolute-v31 normalization, per-matrix/per-selected-bit GSS, five
permutation controls, three scale families, reconstructed gate/up conditional
down-Hessian re-encode, exact codec payload accounting, and five-run final-KLD
discipline are carried forward from Brandon M. Music's prior GLM-5.2 3.5-bpw
work. The public reference model is
[`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78).
The reproducibility bundle downloads the exact prior encoder closure from that
repository at a pinned commit and verifies every file; its accompanying license
remains controlling for those downloaded files.

The Shapley/Fisher/global-allocation components are additive experimental arms
around that control. A control component is replaced only after a matched
model, corpus, byte-budget, exact-codec, and final-logit-KL ablation wins.

## Related work evaluated but not represented as implemented

The research process also evaluated SpinQuant, FlatQuant, AQLM, VPTQ, QuIP#
as an alternative format, BCJR-QAT, PV-Tuning, AlphaQ, GEMQ, ScaleBITS, and
other recent work. Those ideas are not automatically part of this codebase.
They should be credited in any future commit that implements them; discussion
alone is not described as incorporation.

## Citation policy

When publishing results produced by this repository, cite ShapleyMCG and the
upstream method or software that materially produced the result. At minimum,
an Aumann-Shapley/EXL3 pilot should cite Hill's paper and ModelOpt PR, QTIP,
ExLlamaV3/EXL3, Qwen3, and the ShapleyMCG repository. See `CITATION.cff` for a
machine-readable citation of this software.
