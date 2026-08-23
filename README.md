# Quant Pipeline

Auditable foundations for a calibration, attribution, allocation, encoding and
validation pipeline for low-bit dense and MoE language models.

## Status: v0.1 research foundation

This repository currently supplies tested primitives and receipt formats, not
yet a one-command competitive quantization campaign. The calibration
covariance/vector fitter, the exact-codec candidate-ledger generator, and the
runner that causally connects fitting, routed scoring, re-anchoring, global
allocation, checkpoint emission, and student-logit capture are still to be
implemented. `ObjectiveSpec` records the intended experiment settings but is
not yet consumed by a campaign runner. Until those pieces land, Stages 1-3 in
the pipeline document are a manual research protocol and `candidates.json` is
an externally produced, schema-checked handoff—not an artifact this CLI can
create. No quality improvement is claimed without sealed end-to-end KLD.

The primary pilot is `Qwen/Qwen3-30B-A3B-Base`. It tests model-level
Aumann-Shapley path attribution and route-aware expert attribution in one MoE,
then compares the predicted choices with exact next-token KLD. Gemma 4 26B-A4B
is the portability model after the Qwen control is established.

The central rule is simple: proxy scores may propose candidates, but sealed
end-to-end KLD decides. The package preserves raw proxy predictions and their
closure residual instead of normalizing expert scores to look exact.

Implemented now:

- immutable experiment specs and canonical SHA256 receipts;
- document-disjoint calibration/selection/confirmation/final corpus sealing;
- Qwen3-MoE and Gemma 4 expert inventory adapters;
- arbitrary 2-8 bit deterministic reference packing and reconstruction;
- a hash-bound adapter to the pinned corrected EXL3/MCG K3/K4/K5 encoder;
- multi-window KLD with tail metrics and paired bootstrap support;
- Aumann-Shapley quadrature and module-blend execution helpers;
- exact routed Qwen expert residuals with downstream Fisher/Jacobian sketches;
- cross-expert quadratic attribution with exact surrogate closure;
- explicit unresolved-remainder accounting;
- exact codec-payload-byte Pareto pruning and global multiple-choice knapsack allocation;
- packed-shard manifests, hashes, reconstruction and audits; and
- guarded GPU/model commands requiring explicit `--execute`.

The reference uniform codec proves the pipeline mechanics. It is not presented
as a quality competitor. Competitive experiments use `Exl3MCGCodec`, which
loads the explicitly supplied, hash-bound corrected encoder/numeric core and
SM-specific extension. It requires matrix dimensions divisible by 128, which
Qwen3-30B-A3B satisfies and Gemma 4's 704-wide experts do not. Native runtime
qualification remains a separate gate.

See [the full pipeline](docs/PIPELINE.md), [scientific gates](docs/SCIENTIFIC_METHOD.md),
[model decision](docs/MODEL_SELECTION.md), and [two-B200 runbook](docs/B200_RUNBOOK.md).

## Development

```bash
python3 -m pip install -e '.[hf,test]'
pytest -q
```

Licensed under Apache-2.0. See `THIRD_PARTY_NOTICES.md` for acknowledgements.
