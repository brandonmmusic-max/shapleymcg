# Qwen3-30B-A3B-Base reproducibility audit

Audit date: 2026-08-24

This ledger maps the completed fixed-Hadamard Qwen Base control and subsequent
causal-allocation experiment to current remote evidence. It does not promote
the expanded validation reconstruction to a compact runtime checkpoint.

## Requirement-to-evidence map

| Requirement | Current evidence | Result |
| --- | --- | --- |
| Executable calibration-to-encoding implementation | GitHub commit `2fadc8f7e322376cf0b0f9e69ac301f59d74b580` contains the fixed-Hadamard and causal paths; its full suite passes 246 tests with 19 explicit skips | Proven for the fixed-Hadamard and causal Base experiments |
| Immutable Base parent | `Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`; source receipt is in the control bundle | Proven |
| Historical GLM KLD procedure lineage | Pinned `Salesforce/wikitext` revision, source prefix, Qwen token IDs, construction rules, KLD window, and Qwen BF16 teacher logits are all sealed in the bundle | Proven as a target-tokenized reproduction of the procedure, not reused GLM token IDs/logits |
| Calibration identity and role separation | Original and role-safe JSONL, sealed corpus, fit/selection/confirmation capture manifests, exact token hashes, and receipts are in the bundle | Proven |
| Complete fitter and candidate coverage | Hub contains `fits/layer-000` through `layer-047` and `candidates/layer-000` through `layer-047` | Proven |
| Per-layer remote provenance | 48 fit plus 48 candidate publication receipts reconcile exactly with the sealed bundle manifest | Proven |
| Selected quantization validation | Exact 3.5 logical expert-BPW allocation, selected tensor hashes, student logits, tokenwise KLD, and independent replay are sealed | Proven |
| Published reproducibility artifacts | Base dataset, validation reconstruction model, manifests, receipts, hashes, environment lock, and software extension are remotely available at immutable revisions | Proven |
| Packed runtime qualification | The published model is an expanded BF16 reconstruction; official BTX cannot express its independent gate/up choices | Not claimed and outside this completed control |
| Full Aumann-Shapley/causal research arm | 48-layer path attribution, routed Fisher/Jacobian split, explicit remainder reconciliation, exact-rate allocation, two matched SDPA panels, and independent replays are sealed | Proven for allocation-quality KLD; packed runtime remains separate |

## Immutable publication pins

### Reproducibility dataset

- Repository: `brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility`
- Current audited repository head: `18dec5c769b4e39ee641067d0aea462e9c9406e6`
- Fixed control folder revision: `c9c2b001dd943b8251fc0102ec76ab1b8d572219`
- All 37,301 current files under `fits/`, `candidates/`, and
  `controls/fixed-hadamard-k34-v1/` are metadata-identical to that immutable
  control revision (size plus LFS SHA-256 or Git blob identity; zero missing,
  changed, or extra files).
- Remote control folder: 53 files, 2,715,887,831 bytes
- Manifest inventory: 52 files; every row rechecked against remote size and
  either Hub LFS SHA-256 or a fresh-download SHA-256
- Bundle manifest seal:
  `5df0be67d3eac61902cb265bb3a9c5a57e889c7ebcaaf0924fa77c77517b106f`
- Result publication receipt: 154 artifacts; all 154 rows rechecked against
  remote size and SHA-256 metadata/content
- Result publication receipt seal:
  `b0af2ce0c0af3bafd1b8e5a101269a1c91fdf2f9d2cce105321b63b3ee130895`
- Per-layer receipts: 48 fit plus 48 candidate receipts; all internal seals,
  layer identities, repository paths, byte counts, and bundle references match
- The 96 per-layer manifests cover 37,152 payload files and
  259,295,735,932 bytes. Every fit and candidate internal manifest/receipt seal
  revalidated; every candidate layer covers experts 0 through 127, 384 matrices,
  and 768 K3/K4 candidates.
- Causal evidence content revision: `a1f669e5b42f041fa28017670fd86b183f5103d0`
- Previously audited causal/card revision:
  `89b989fc45cba9e841c5083d38c16cd8734041bb`
- All 94 files under `causal-arm-v3/` at the current head are
  metadata-identical to that audited revision (zero missing, changed, or extra
  files).
- Current dataset-card revision: `18dec5c769b4e39ee641067d0aea462e9c9406e6`
- Validation-model receipt mirror:
  `causal-arm-v3/validation-model-hf-publication-receipt.json` (remote file
  SHA-256 `778670c94a6db7d8147d3ee4757064fe3dcf586dfee22765917074be9f8c462a`)
- Matched causal/control 20k comparison seal:
  `8d8438df03d0f9a34a4a5a79912dd232baa401f24d8e6e14bb942bf00062a608`

### Validation reconstruction model

- Repository:
  `brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction`
- Manifest-verified content revision:
  `c3447f3f8e231ae83afa10ba263346d1ceb98c11`
- Publication-receipt revision:
  `fbdd05904fd4018dea9fd0f25ed4800c15df1493`
- Current model-card revision:
  `785cb47dbd900ed4a3f045b5f9f7a638db72841e`
- Verified content inventory: 31 files, 61,079,798,645 bytes
- Model manifest inventory: 30 payload files plus the manifest; every row was
  rechecked against remote size and either Hub LFS SHA-256 or Git-blob SHA-1
- Model manifest seal:
  `32fec13a96c6c85eb394939a6eb511c821bd05e8ac0960268852d33150d0cde1`
- Publication receipt seal:
  `e7b3b344321c11e872fc006cbd009c935b9fdbf0fdb8b2b0a99197cce14bffe8`
- Publication receipt file SHA-256:
  `778670c94a6db7d8147d3ee4757064fe3dcf586dfee22765917074be9f8c462a`
- Post-assembly model-logit verification seal:
  `247179326b2dd6b0f904d1ca976e4086920c0b3ce4a3b2421e481b51e822e717`
- The reloaded checkpoint's 2,047-position logit tensor matched the measured
  causal student bit-for-bit (`max_abs_delta = 0`).
- Every non-README payload file at the current model-card revision remains
  byte-identical to the publication-receipt revision. The README is deliberately
  outside that payload-identity assertion because the model card was updated
  after the sealed checkpoint publication.

## Corpus and evaluation seals

- Original `reap_recall_calib.jsonl`:
  `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`
- Role-safe packed JSONL:
  `5d47324b5d8aa97240104a811764811c673a1e71876158f0dd2799da5369eab5`
- Sealed corpus file:
  `0dc4007b6a8fe614e4f37d24d7e83546148af08b3b570a3476f6db112be73349`
- GLM-style Qwen KLD window:
  `fa370b884ec7e9dab4d53e13fcd4dac3ee0e6ae27b54ebd662100a0f621ce73b`
- Source prefix:
  `293ff4a5d4d8e4e5a1b875d2f786895e491180523af99a8eea89321ae22e68ce`
- Qwen token IDs:
  `551b98fd34866582068d77bf0875557bafbfe5cb1b1fa94459b4e5cc38d9073b`
- BF16 teacher logits:
  `ae11557e20e0705a20fa24ec5def667403ca0f1771d64c638e4349ffb6ce0bb9`

The historical control takes the first 2,048 Qwen-tokenized tokens from the
pinned WikiText-2 raw test prefix and scores 2,047 next-token positions. The
source text and construction procedure are inherited from the GLM evaluation;
the token IDs and logits are necessarily regenerated with Qwen.

The sealed corpus contains 32 fit, 16 selection, 16 confirmation, and 25 final
windows. Every stored token hash revalidated, and document-ID overlap is zero
for every pair of roles. The BF16 teacher-logit LFS object is 1,294,388,616
bytes and its remote SHA-256 exactly matches the capture receipt.

## Measured control

- Allocation seal:
  `5edda4ab29f9961f6f2ab5be3aab6cc0aaa3817afa325b40fd5063fef05de034`
- Logical routed-expert rate: exactly 3.5 BPW
- GLM-style 2,047-position mean KLD: `0.06335949321311507`
- KLD report seal:
  `3685fa5eac6064aa6bd5e51ddc53fb10d791c0744c955f551283fdcad62469ff`
- Independent maximum absolute token-KLD delta:
  `8.278655538873636e-13`
- Same-parent 20,480-position WikiText panel mean KLD:
  `0.05005581795647327`
- Same-parent panel top-1 agreement: `0.908447265625`
- Same-parent report seal:
  `8e9a55f56051ee62a6fd3299ae4344403073864f58d9553cf5bb7dd961d15426`

## Measured causal allocation

- Allocation seal:
  `10bb3b71a2258500b281e3b37d57657e5e0ecf9318bb9c1a3fc6dbd2326e63c4`
- Exact rate: 9,216 K3 plus 9,216 K4 routed-expert matrix choices
- Changed placements versus historical allocation: 9,392 / 18,432
- Matched 2,047-position historical / causal mean KLD:
  `0.06249572628978888` / `0.05506107074246945`
- Relative KLD reduction on that matched window: `0.11896262334556107`
  (11.896262%)
- Historical / causal top-1 agreement on that window:
  `0.9013190034196384` / `0.912066438690767`
- Matched 20,480-position historical / causal mean KLD:
  `0.04908888647295481` / `0.04529370272688347`
- Relative KLD reduction: `0.0773124839195991` (7.731248%)
- Historical / causal top-1 agreement: `0.908203125` / `0.910888671875`
- Causal / historical independent verification seals:
  `6e4215f276d69ca53ce1ed29c2a900f5e040d41594a9e2ab24cb247ddec90265` /
  `752e83112522f1f8bfe2cbf7797196cc7961164b7617ccb22d92f7e60c8a1911`
- A fresh cross-binding audit passed 25 / 25 checks: allocation to reconciled
  attribution and candidate inventory, both reports to their allocations,
  comparison records to report hashes and file hashes, and both independent
  20,480-position replays. Both 20,480-position replays have
  `max_absolute_delta = 0`.

## Reproduction entry points

- `docs/B200_RUNBOOK.md` is the clean-node command sequence.
- `docs/PIPELINE.md` is the stage and scientific-gate contract.
- `docs/QWEN_B200_CONTROL_RESULT.md` is the complete result interpretation.
- `configs/qwen3-30b-a3b-b200/` is intentionally fail-closed and portable;
  machine-specific paths remain placeholders. The executed hashes and software
  identities are preserved in
  `controls/fixed-hadamard-k34-v1/resolved-artifact-lock.json` and the adjacent
  environment files in the immutable dataset revision above.
