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
