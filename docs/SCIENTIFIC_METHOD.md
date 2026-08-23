# Scientific method and acceptance gates

## The estimand

The primary outcome is teacher-to-candidate next-token KL divergence on sealed,
document-disjoint final windows. Layer/expert SSE, Hessian loss, Fisher damage,
perplexity, and task scores are diagnostics; none substitutes for final KLD.

Report at least mean, standard deviation, p50, p95, p99, CVaR95, maximum, and
per-window values. Compare candidate arms with paired windows and bootstrap
confidence intervals. A five-boot serving test measures runtime stability, not
the document-level statistical uncertainty of KLD.

## Four independent corpus roles

- `fit`: covariance, Hessian, scale, rotation and sketch fitting.
- `selection`: codec candidate ranking and allocation.
- `confirmation`: prospective ranking/sign/regret and proxy-closure checks.
- `final`: untouched KLD and task-quality verdict.

Documents—not token windows—are assigned to roles. The seal records exact token
IDs, source hash, tokenizer revision and role. At least 25 final windows spanning
four or more domains are required for a headline result.

## Hierarchical attribution

At path node `t`, blend each selected MoE unit between source and actual-codec
candidate output and differentiate next-token KL with respect to the blend
coefficient. Gauss-Legendre integration yields signed model-level Aumann-Shapley
layer contributions.

Inside a layer, compute the exact routed candidate-minus-source residual for
each expert's full gate -> SiLU -> up -> down function. Project these residuals
through downstream score-function Fisher/Jacobian sketches. For projected
expert residuals `z_e`, assign

```
psi_e = 0.5 * mean(z_e * sum_j(z_j))
```

This shares cross-expert terms symmetrically and closes exactly to the quadratic
surrogate. Keep signed contributions: cancellation is information, not noise.

Three quantities must always remain separate:

1. direct expert-codec damage;
2. routing/state-shift damage; and
3. unresolved nonlinear remainder.

The ledger stores raw proxy total, exact measured KLD, and closure residual.
The residual may be appended as an explicit accounting component, but raw
expert values are never silently scaled to manufacture closure.

## Prospective validation

On confirmation windows, test:

- sign accuracy for candidate improvements/regressions;
- Spearman ranking and top-choice regret;
- raw proxy versus exact KLD closure;
- p95/p99/CVaR tail behavior;
- route-set and route-mass drift; and
- additive predictions against joint gate/up/down candidate tuples.

Re-anchor with exact end-to-end KLD at least every four accepted layers. If
coupling is material, allocation consumes per-expert triplet Pareto frontiers
instead of independent tensor scores.

## Acceptance gates

1. Instrument gate: immutable model/config/index hashes, sealed windows,
   regenerated teacher logits, no scale/eval leakage, same software image.
2. Re-probe gate: actual corrected codec bytes; prospective ranking improves
   over the incumbent mass-weighted expert-SSE objective.
3. Closure gate: raw proxy residual and interaction share remain within limits
   chosen before final evaluation.
4. Allocation gate: exact codec-payload-byte budget followed by an exact packed
   checkpoint file-size gate; no nominal-bpw substitution.
5. Quality gate: improvement on untouched final KLD with no material tail loss.
6. Runtime gate: eager/graph logit parity, runtime format/ABI validation, and
   repeatable serving results. A good Python fake-quant result is not release.
