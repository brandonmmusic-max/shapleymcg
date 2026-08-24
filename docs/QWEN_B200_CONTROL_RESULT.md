# Qwen3-30B-A3B fixed-Hadamard K3/K4 B200 control

## Result

The first full 48-layer Qwen control completed on two NVIDIA B200 GPUs. It
measures an exact reconstructed-BF16 replay of corrected-R10 EXL3/MCG K3/K4
candidates selected at an exact 3.5-bit MoE expert-weight rate.

| Metric | Value |
|---|---:|
| Prediction positions | 2,047 |
| Mean KLD | 0.06335949321311507 |
| Standard deviation | 0.1711659971317654 |
| Median / P50 | 0.014639648031201399 |
| P95 | 0.2958056229323483 |
| P99 | 0.8551368681377688 |
| CVaR95 | 0.6597548748723323 |
| Maximum | 2.6841288415761797 |

This is a measured control result, not a prediction from the routed-damage
proxy. The primary KLD process exited zero. A separate verifier checked the raw
BF16 payload SHA256 of every selected matrix and recomputed token KLD with an
independent `logaddexp.reduce` implementation.

## Exact experimental identity

- Base model: `Qwen/Qwen3-30B-A3B-Base`
- Base revision: `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`
- Model architecture: 48 MoE layers, 128 experts per layer, three expert
  matrices per expert
- Candidate codec: hash-pinned corrected-R10 EXL3/MCG K3 and K4
- Candidate policy: `energy_balanced`
- Scale family: `per128-grid`
- Transform signs: deterministic, seed sealed by every encode plan
- SM100 extension SHA256:
  `0e98de75bf3cdf1f5e87394e5cc0eefcb24a66deaf1b4f60d3e0ed7acb837c3d`
- Allocation objective: minimum diagonal routed-p2 damage at an exact half-K4
  matrix-choice rate
- Selected choices: 9,216 K3 and 9,216 K4
- Logical rate: 3.5 bits per MoE expert-weight element
- Allocation SHA256:
  `5edda4ab29f9961f6f2ab5be3aab6cc0aaa3817afa325b40fd5063fef05de034`
- KLD report seal:
  `3685fa5eac6064aa6bd5e51ddc53fb10d791c0744c955f551283fdcad62469ff`
- Student-logit file SHA256:
  `15218b9109f0802b107c19bf4a37e668976662f41006d782e5ceb321a1dbbfdc`
- Token-KLD file SHA256:
  `705a8492a5483dfb905a5035037de90635fd14f6223d969fe431e82cf3a7f4d4`
- Independent token-KLD SHA256:
  `8798b283ee44edd833e3975da7ce6bb9f5ab877c8d429f99325b1508ac4e61c1`
- Independent verification seal:
  `83389bb000d93421daba4c9ed96201363898e63885bf7929d84a870a9506cac9`
- Maximum absolute primary/independent token-KLD delta:
  `8.278655538873636e-13`

## KLD control window

The comparison uses the same procedure class as the historical GLM control:
the first 2,048 target-tokenized tokens from the WikiText-2 raw test split and
2,047 next-token prediction positions. It is model-specific: the source text
is shared, while token IDs and BF16 logits are generated with the Qwen model
and tokenizer.

- Dataset: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, test split
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Procedure: `glm-wikitext-2-raw-test-prefix-v1`
- KLD-window file SHA256:
  `fa370b884ec7e9dab4d53e13fcd4dac3ee0e6ae27b54ebd662100a0f621ce73b`
- Source-prefix SHA256:
  `293ff4a5d4d8e4e5a1b875d2f786895e491180523af99a8eea89321ae22e68ce`
- Token SHA256:
  `551b98fd34866582068d77bf0875557bafbfe5cb1b1fa94459b4e5cc38d9073b`
- Teacher-logit file SHA256:
  `ae11557e20e0705a20fa24ec5def667403ca0f1771d64c638e4349ffb6ce0bb9`

## Calibration identity

- Original `reap_recall_calib.jsonl` SHA256:
  `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`
- Role-safe packed JSONL SHA256:
  `5d47324b5d8aa97240104a811764811c673a1e71876158f0dd2799da5369eab5`
- Sealed-corpus file SHA256:
  `0dc4007b6a8fe614e4f37d24d7e83546148af08b3b570a3476f6db112be73349`
- Sealed-corpus internal seal:
  `147682143ded101abbe48e159481a651a3c1b3a7c5c5ddcb73e7698faafd8e59`
- Roles: 32 fit, 16 selection, 16 confirmation, and 25 final windows;
  four packed documents per role, document-disjoint

The token IDs, token hashes, capture receipts, fit manifests, and per-layer
remote publication receipts are in the sealed HF bundle and its `SHA256SUMS`.

Historical capture manifests wrote `start_token: 0` for every calibration
chunk because the capture-time code preceded the offset-metadata fix. This did
not change the tensors: exact token IDs and hashes were sealed, and fitted
sample identity included `document_id@token_sha256`. Current code records the
true sealed offset.

## Additional KLD panels

### Reconstructed Hill-paper BFCL/RULER panel

The paper-source categories and dimensions were reconstructed as 16 disjoint
2,048-token evaluation sequences: eight BFCL-v3 sequences and eight RULER
sequences. The paper does not publish its exact row IDs or token IDs, so this is
explicitly labeled a reconstruction rather than an exact author-panel replay.

| Metric | Value |
|---|---:|
| Prediction positions | 32,752 |
| Mean KLD | 0.018260861970005038 |
| Sample SD of sequence means | 0.010442577577935023 |
| Standard error of sequence means | 0.0026106443944837556 |
| P95 | 0.050585024611849064 |
| P99 | 0.27827231287309023 |
| CVaR95 | 0.2612709356943377 |

The nearest same-corpus result in Hill's Qwen3-30B Table 2 is 0.0353 for the
additive method at 4.2 effective bits. This control has 3.8787878788 logical
bits in the paper's 240-linear allocation scope. The lower measured value is
encouraging, but it is **not** a strict claim of beating that paper: this work
uses `Qwen3-30B-A3B-Base`, mixed K3/K4 expert-weight reconstruction with BF16
activations, and reconstructed prompts; the paper appears to use the
post-trained Qwen checkpoint and W4A4 NVFP4.

- Panel seal: `51f6f9d9f6acde8c2fd92c929981aa72ff22b74e38f1c93aebd4914132b8f848`
- Report seal: `c2c750197107fa308befa502b2e58facc4e12d6de77a7a35fb5fe41460ae8179`
- Token-KLD SHA256: `032f94568db7c01d4367c282591cacaef9960b2ed180704e9c9d217ad12f4e9b`

### TurboDerp/ExLlamaV3 WikiText-2 20k panel

This control mirrors `eval/model_diff.py`: raw WikiText-2 test text is joined
with double newlines, tokenized without added special tokens, and divided into
ten consecutive non-overlapping 2,048-token rows. All 20,480 logits positions
are scored with float32 `KL(reference || student)`, and KV caching is disabled.

| Metric | Value |
|---|---:|
| Logit positions | 20,480 |
| Mean KLD | 0.05005581795647327 |
| Top-1 agreement | 0.908447265625 |
| Sample SD of row means | 0.018849720769000416 |
| P95 | 0.19061099812388418 |
| P99 | 0.5020050489902488 |
| CVaR95 | 0.4215475404780591 |

The published TurboDerp card reports 0.0688 KLD / 89.44% top-1 agreement at
3.0 bpw and 0.0215 / 94.33% at 4.0 bpw. Inspection of those exact branches
shows that the card number is both the routed-expert K and the attention-
projection K: all 18,432 expert matrices and all 192 attention projections are
K3 on the 3.0 branch (K4 on the 4.0 branch), while `lm_head` has its separate
rate and the router remains unquantized. EXL3 also records `bits` as the body-
linear bitrate. The two descriptions are numerically identical for those
uniform branches.

This control should therefore be reported first as **3.5 routed-expert logical
bpw** under the expert-K convention. Because its attention and routers remain
BF16, its secondary body-linear equivalent is 3.8838872528 logical bpw and
3.9116533446 payload bpw; routed experts alone are 3.5 logical and
3.5286458333 payload bpw. Its 0.0500558 KLD and 90.8447% top-1 agreement fall
between TurboDerp's measured K3 and K4 endpoints. There is no published K3.5
endpoint on this card, and the post-trained TurboDerp parent differs from this
Base control, so no strict same-rate winner is claimed.

- Upstream comparison: [`turboderp/Qwen3-30B-A3B-exl3`](https://huggingface.co/turboderp/Qwen3-30B-A3B-exl3)
- Evaluator source: [`turboderp-org/exllamav3/eval/model_diff.py`](https://github.com/turboderp-org/exllamav3/blob/master/eval/model_diff.py)
- Panel seal: `6c509f3496d5a1fe6739c58ffa152c3d4202932f9d8a6514e55d17f8a4cfa741`
- Report seal: `8e9a55f56051ee62a6fd3299ae4344403073864f58d9553cf5bb7dd961d15426`
- Token-KLD SHA256: `63402eb16197d64fc4d78d6f5110a6fbb1b35a2748875511ace5d93e3605284a`

## What this result establishes

It establishes that the fixed-Hadamard, source-derived absolute-v31,
per128-grid, corrected-R10 K3/K4 control can be calibrated and encoded across
all Qwen MoE layers and measured end to end at an exact 3.5-bit expert-weight
rate. It also provides a reproducible baseline against which the full
Aumann-Shapley/Fisher allocation arm and future rotation proposals can be
compared.

It does **not** establish packed-runtime throughput, CUDA-graph compatibility,
or an official-BTX checkpoint. The exact KLD gate installs codec-produced BF16
reconstructions into Transformers and captures logits with eager attention.
Official BTX currently couples gate/up rates; this control selects gate and up
independently. The published validation model is therefore an expanded BF16
checkpoint and is labeled accordingly, not advertised as compact 3.5 bpw.

## Published artifacts

- Reproducibility dataset:
  [`brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility)
- Exact expanded validation model:
  [`brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction`](https://huggingface.co/brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction)

The final publication receipts record immutable verified revisions, total
bytes, and size plus LFS-SHA256/Git-blob verification for every uploaded file.
