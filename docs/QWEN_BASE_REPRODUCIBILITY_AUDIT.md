# Qwen3-30B-A3B-Base reproducibility audit

Audit date: 2026-08-24

This ledger maps the completed fixed-Hadamard Qwen Base control to current,
remote evidence. It does not promote the control to a compact checkpoint or
claim that the separate Aumann-Shapley/causal research arm has been executed.

## Requirement-to-evidence map

| Requirement | Current evidence | Result |
| --- | --- | --- |
| Executable calibration-to-encoding implementation | GitHub commit `e40190a0e1bb0416cdf4e9fe65b91358fb4d676b`, retained in current `main`; current suite is 254/254 passing | Proven for the fixed-Hadamard Base control |
| Immutable Base parent | `Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`; source receipt is in the control bundle | Proven |
| Historical GLM KLD procedure lineage | Pinned `Salesforce/wikitext` revision, source prefix, Qwen token IDs, construction rules, KLD window, and Qwen BF16 teacher logits are all sealed in the bundle | Proven as a target-tokenized reproduction of the procedure, not reused GLM token IDs/logits |
| Calibration identity and role separation | Original and role-safe JSONL, sealed corpus, fit/selection/confirmation capture manifests, exact token hashes, and receipts are in the bundle | Proven |
| Complete fitter and candidate coverage | Hub contains `fits/layer-000` through `layer-047` and `candidates/layer-000` through `layer-047` | Proven |
| Per-layer remote provenance | 48 fit plus 48 candidate publication receipts reconcile exactly with the sealed bundle manifest | Proven |
| Selected quantization validation | Exact 3.5 logical expert-BPW allocation, selected tensor hashes, student logits, tokenwise KLD, and independent replay are sealed | Proven |
| Published reproducibility artifacts | Base dataset, validation reconstruction model, manifests, receipts, hashes, environment lock, and software extension are remotely available at immutable revisions | Proven |
| Packed runtime qualification | The published model is an expanded BF16 reconstruction; official BTX cannot express its independent gate/up choices | Not claimed and outside this completed control |
| Full Aumann-Shapley/causal research arm | Pipeline code and contracts exist, but the fixed-Hadamard control is not evidence that this separate arm ran | Not claimed |

## Immutable publication pins

### Reproducibility dataset

- Repository: `brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility`
- Audited repository head: `e4d8a67ddb1f0b4c7605c5efcdc3c54e87e22b9f`
- Fixed control folder revision: `c9c2b001dd943b8251fc0102ec76ab1b8d572219`
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

### Validation reconstruction model

- Repository:
  `brandonmusic/Qwen3-30B-A3B-ShapleyMCG-K34-Validation-Reconstruction`
- Verified revision: `cfb34123f77edda0a1daa3881c48992c5b2db7ac`
- Remote inventory: 31 files, 61,079,795,251 bytes
- Model manifest inventory: 30 files; all rows rechecked against remote size
  and either Hub LFS SHA-256 or a fresh-download SHA-256
- Model manifest seal:
  `9f454affeae42392a928e50c1bdf704c3db871012733f2310548d6c02ffc2134`

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

## Reproduction entry points

- `docs/B200_RUNBOOK.md` is the clean-node command sequence.
- `docs/PIPELINE.md` is the stage and scientific-gate contract.
- `docs/QWEN_B200_CONTROL_RESULT.md` is the complete result interpretation.
- `configs/qwen3-30b-a3b-b200/` is intentionally fail-closed and portable;
  machine-specific paths remain placeholders. The executed hashes and software
  identities are preserved in
  `controls/fixed-hadamard-k34-v1/resolved-artifact-lock.json` and the adjacent
  environment files in the immutable dataset revision above.
