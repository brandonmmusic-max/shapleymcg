# Release review and completion audit

This page separates completed scientific evidence from local code and
documentation that still requires owner approval before publication. It is a
release checklist, not an additional result claim.

## Current state

- Review branch: `publish/v0.2`.
- The committed precursor branch is already mirrored as
  `github/publish/v0.2`; there is no open pull request.
- The final method, result-ledger, claim-boundary, and fail-closed runtime
  rewrite remains uncommitted and has not been pushed.
- The rented B200 instance was deleted after the completed experiment. This
  release does not require or authorize another model run.

## Evidence already preserved

| Evidence | Immutable revision | What it supports |
|---|---|---|
| Base reproducibility dataset | `a77141740749a53ede41d96115ba911f5b569f76` | Base calibration/candidate lineage, causal allocation evidence, progressive frozen-rate diagnostic, reports, hashes, and replay receipts |
| Post-trained reproducibility dataset | `68d3be1b1e64f8bf947e734d4656e3cf13a19469` | Post-trained controls, matched TurboDerp allocation evidence, and frozen-rate factory-union evidence |
| Normalized local result ledger | `974dfe1adc7397d16418da3c13a2fb35b34a09f2f669c345f363971290ca8aa5` | Nineteen sealed result rows with explicit design, parent, panel, rate, backend, evidence, and status boundaries |

The evidence ledger is [`../results/RESULTS_LEDGER.json`](../results/RESULTS_LEDGER.json).
Its canonical supplemental source is
[`../results/canonical-supplemental-results.json`](../results/canonical-supplemental-results.json).

The Base dataset's
[`GLM-to-Qwen lineage receipt`](https://huggingface.co/datasets/brandonmusic/shapleymcg-qwen3-30b-a3b-reproducibility/blob/a77141740749a53ede41d96115ba911f5b569f76/results/qwen3-30b-a3b-base/progressive-candidate-v1/glm-lineage/lineage.json)
has downloaded SHA-256
`4e9ac680d13750ec2c5e1e1744701b788663db6d03323df0f2d59397b0909066`
and internal lineage seal
`eadb5a8f579e7f40cd25ead031348c526cb6b6da0870e5a19fcb5eaf0831bee5`.
It binds the original GLM model revision
`7c73450f05a151439d0f184f216b1eefcc394a31` to the Qwen fixed-control
dataset revision `c9c2b001dd943b8251fc0102ec76ab1b8d572219` without claiming
cross-model reuse of token IDs or logits.

## Local review surface

The uncommitted tree is intentionally grouped below so the owner can review
method changes separately from result claims and release plumbing.

| Review group | Primary files | Purpose and claim boundary |
|---|---|---|
| Human method, evidence, and attribution | [`../README.md`](../README.md), [`../RESULTS.md`](../RESULTS.md), [`SHAPLEYMCG_METHOD.md`](SHAPLEYMCG_METHOD.md), [`QWEN_COMPLETE_RESULTS_LEDGER.md`](QWEN_COMPLETE_RESULTS_LEDGER.md), [`QWEN_BASE_REPRODUCIBILITY_AUDIT.md`](QWEN_BASE_REPRODUCIBILITY_AUDIT.md), [`REFERENCES.md`](REFERENCES.md), and [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) | Defines the current end-to-end method, separates measured results from proposals, pins the evidence revisions and seals, and carries exact upstream credit from the named GLM/MTP-78 predecessor without implying that its runtime is part of the Qwen experiment. |
| Five-role corpus and resource accounting | `src/quant_pipeline/calibration/windows.py`, `src/quant_pipeline/spec.py`, `scripts/prepare_reap_recall_corpus.py`, `scripts/run_qwen_calibration_capture.py`, `scripts/estimate_qwen_b200_resources.py`, the Qwen B200 configs, and [`B200_RUNBOOK.md`](B200_RUNBOOK.md) | Keeps `fit`, `conditional_fit`, `selection`, `confirmation`, and `final` distinct; budgets routed capture separately from retained teacher/student logits. |
| Concrete Qwen campaign path | `src/quant_pipeline/campaign/qwen_adapter.py`, `qwen_attribution.py`, `qwen_services.py`, `qwen_work_units.py`, and `runner.py` | Connects fitting, encoding, attribution, exact allocation, and validation; confirmation installs the requested allocation exactly and fails closed instead of silently substituting a fallback. |
| Candidate factory and exact-rate infrastructure | `src/quant_pipeline/candidates/factory_union.py`, `factory_calibration.py`, `factory_allocation.py`, `ledger.py`, and `src/quant_pipeline/cli.py` | Provides model-neutral source identity, score calibration, signed damage, and exact-budget allocation primitives. These are reusable infrastructure, not evidence that a Qwen matrix-level joint factory/rate experiment was run. |
| Normalized result and publication machinery | `scripts/build_results_ledger.py`, `src/quant_pipeline/results/`, `results/RESULTS_LEDGER.json`, `results/canonical-supplemental-results.json`, `scripts/upload_sealed_artifact_tree_hf.py`, and `scripts/verify_hf_artifact_tree.py` | Migrates the preserved results without strengthening their evidence status and requires immutable identities for publication/verification. |
| Offline regression coverage | `.github/workflows/tests.yml` and the modified/new files under `tests/` | Exercises corpus roles, orchestration, candidate identity/allocation, ledger determinism, and fail-closed Hub publication using synthetic artifacts and mocked Hub calls. It is software evidence only, not renewed model-quality evidence. |

The untracked `uv.lock` is user-owned and is excluded from this proposed
release. No file in this review map has been staged, committed, or pushed.

## Claims approved by the evidence structure

1. On the Base matched 20,480-position panel, the complete causal allocation
   lowered mean KLD by 7.731248% relative to the predecessor allocation while
   holding the candidate pool and expert rate fixed.
2. On the unchanged published TurboDerp K3/K4 reconstruction pool, the complete
   causal allocator measured 13.766197% lower mean KLD than the reproduced
   v0.0.1 carried-surplus rule. This is an allocator-only comparison, not a
   whole-pipeline or codec-superiority claim.
3. The Base native/progressive factory selector was a completed negative
   frozen-rate diagnostic. The post-trained TurboDerp/MCG selector was a
   positive but confidence-interval-crossing-zero frozen-rate diagnostic.
4. No sealed result supports a matrix-level joint factory/rate claim or a
   superiority claim over the Hill paper.

The full explanation is in [`../RESULTS.md`](../RESULTS.md), and the normative
method is in [`SHAPLEYMCG_METHOD.md`](SHAPLEYMCG_METHOD.md).

## Local integrity evidence

The current review tree has passed these non-model checks:

- deterministic regeneration and validation of the 19-row result ledger;
- exact equality between direct builder output and the checked-in ledger;
- local Markdown-link resolution;
- JSON and TOML parsing;
- Python AST parsing and shell syntax checks;
- `git diff --check`;
- independent read-only audit with no remaining P0/P1 finding;
- reproduction-command help surfaces for every documented calibration,
  fitting, encoding, inventory, attribution, allocation, KLD, sealing, upload,
  and verification stage;
- explicit GLM-to-Qwen lineage closure: byte-identical calibration JSONL,
  pinned WikiText procedure/source prefix, and separately regenerated and
  sealed Qwen token IDs and BF16 teacher logits;
- the bundled B12X Apache-2.0 license is byte-identical to `LICENSE` at pinned
  upstream commit `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` (SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`),
  and the derived writer carries an Apache-2.0 SPDX identifier;
- five-role resource accounting: `conditional_fit` is budgeted as routed
  capture, while prospective `confirmation` is excluded from fitting and
  budgeted as retained teacher/student logits.
- the owner-authorized final offline suite passed 310 tests in 30.59 seconds
  with Hugging Face Hub, Datasets, and Transformers all forced offline.

The proposed GitHub Actions unit job is explicitly offline for Hugging Face
Hub, Datasets, and Transformers. Its new tests use synthetic temporary
artifacts and mocked Hub interfaces; CI cannot silently download a checkpoint
or calibration dataset while validating this release.

No new model evaluation or KLD experiment was run during the closeout. The
310-test software suite is implementation evidence, not renewed model-quality
evidence.

## Owner decision checklist

The proposed release currently contains 67 changed paths: 55 tracked
modifications and 12 new paths. The separate untracked `uv.lock` and the
local-only PR body are not part of
that count or this release. Approval can be given by item rather than as an
all-or-nothing instruction.

- [ ] **Claims:** approve the four bounded claims above and the explicit
  non-claims in [`../RESULTS.md`](../RESULTS.md).
- [ ] **Method:** approve the human pipeline and normative method in
  [`../README.md`](../README.md) and
  [`SHAPLEYMCG_METHOD.md`](SHAPLEYMCG_METHOD.md).
- [ ] **Attribution and licensing:** approve the upstream credits, the existing
  source-available project license, and the disclosed Apache-2.0 B12X-derived
  file boundary. This item does not amend the project license.
- [ ] **Implementation:** approve the five-role corpus changes, concrete Qwen
  campaign wiring, fail-closed allocation installation, generic candidate
  infrastructure, and normalized result-ledger machinery summarized above.
- [ ] **Software validation:** authorize the offline repository unit/integrity
  suite. This does not authorize a checkpoint download, GPU run, KLD rerun, or
  cloud instance.
- [ ] **GitHub publication:** after validation passes, authorize committing the
  reviewed paths, pushing `publish/v0.2`, and opening a pull request to `main`.
- [ ] **Hugging Face documentation publication:** after GitHub review,
  authorize publishing only the normalized ledger and updated documentation to
  the named dataset repositories. Immutable evidence revisions remain
  untouched.

A limited approval such as “approve the first five items; do not publish” is
sufficient to authorize software validation without authorizing GitHub or Hub
changes. Publication requires the corresponding explicit checklist items.

## Approval gates still open

The project is not release-complete until all of these occur:

1. The owner reviews the uncommitted diff and approves its claim language,
   implementation changes, and attribution.
2. The agreed local test scope is executed. If model experiments remain frozen,
   this should at minimum include the repository unit/integrity suite; it must
   not be represented as renewed model-quality evidence.
3. The approved changes are committed and pushed to `publish/v0.2`.
4. A pull request to `main` is opened and its rendered Markdown and CI results
   are reviewed.
5. The normalized result ledger and updated dataset-card documentation are
   published to the intended Hugging Face repositories without modifying the
   immutable evidence revisions cited above.
6. The new GitHub and Hugging Face revisions are recorded, and every published
   link, manifest, and receipt is verified at those revisions.

Until those gates close, the correct status is **scientific artifacts preserved;
final repository release awaiting owner review and publication**.
