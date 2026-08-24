from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def gauss_legendre_nodes(count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("path node count must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (nodes + 1.0) / 2.0, weights / 2.0


def aumann_shapley(
    deltas: Sequence[np.ndarray],
    gradient_at: Callable[[float], Sequence[np.ndarray]],
    path_nodes: int = 5,
) -> np.ndarray:
    """Integrate <gradient_i(t), delta_i> along a common quantization path."""
    nodes, weights = gauss_legendre_nodes(path_nodes)
    result = np.zeros(len(deltas), dtype=np.float64)
    for node, quadrature_weight in zip(nodes, weights, strict=True):
        gradients = gradient_at(float(node))
        if len(gradients) != len(deltas):
            raise ValueError("gradient_at returned the wrong number of units")
        for index, (gradient, delta) in enumerate(zip(gradients, deltas, strict=True)):
            result[index] += quadrature_weight * float(np.vdot(gradient, delta).real)
    return result


def quadratic_expert_attribution(projected_residuals: np.ndarray) -> np.ndarray:
    """Signed expert shares that close exactly to 0.5*||sum(z_e)||^2.

    Each z_e may already include the downstream Fisher/Jacobian square-root
    projection. Cross-expert terms are shared symmetrically by the identity
    psi_e = 0.5 <z_e, sum_j z_j>.
    """
    z = np.asarray(projected_residuals, dtype=np.float64)
    if z.ndim < 2:
        raise ValueError("expected [experts, observations...] residuals")
    total = np.sum(z, axis=0)
    axes = tuple(range(1, z.ndim))
    return 0.5 * np.mean(z * total, axis=axes)


def conditional_quadratic_damage(current: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Forecast 0.5||R+d||^2 - 0.5||R||^2 for projected candidates."""
    residual = np.asarray(current, dtype=np.float64)
    deltas = np.asarray(candidates, dtype=np.float64)
    if deltas.ndim < residual.ndim + 1 or deltas.shape[1:] != residual.shape:
        raise ValueError("candidates must have shape [candidate, *current.shape]")
    axes = tuple(range(1, deltas.ndim))
    return np.mean(deltas * residual, axis=axes) + 0.5 * np.mean(deltas * deltas, axis=axes)


@dataclass(frozen=True)
class Reconciliation:
    raw: np.ndarray
    reconciled: np.ndarray
    measured_total: float
    raw_total: float
    closure_residual: float
    method: str


def reconcile_explicit_remainder(raw: Sequence[float], measured_total: float) -> Reconciliation:
    """Keep proxy values untouched and expose non-closure as its own component."""
    values = np.asarray(raw, dtype=np.float64)
    residual = float(measured_total - np.sum(values))
    reconciled = np.concatenate([values, np.asarray([residual])])
    return Reconciliation(
        raw=values,
        reconciled=reconciled,
        measured_total=float(measured_total),
        raw_total=float(np.sum(values)),
        closure_residual=residual,
        method="explicit-unresolved-remainder",
    )


def split_layer_damage(
    measured_layer_damage: float,
    projected_expert_residuals: np.ndarray,
    routing_state_shift: float = 0.0,
    projected_routing_residual: np.ndarray | None = None,
) -> dict:
    expert = np.asarray(projected_expert_residuals, dtype=np.float64)
    if projected_routing_residual is not None:
        routing = np.asarray(projected_routing_residual, dtype=np.float64)
        if routing.shape != expert.shape[1:]:
            raise ValueError("projected routing residual must match one expert residual observation shape")
        joint = np.concatenate([expert, routing[None]], axis=0)
        shares = quadratic_expert_attribution(joint)
        direct = shares[:-1]
        routing_state_shift = float(shares[-1])
    else:
        direct = quadratic_expert_attribution(expert)
    raw = np.concatenate([direct, np.asarray([routing_state_shift], dtype=np.float64)])
    accounting = reconcile_explicit_remainder(raw, measured_layer_damage)
    return {
        "expert_direct": direct.tolist(),
        "routing_state_shift": float(routing_state_shift),
        "unresolved_nonlinear_remainder": accounting.closure_residual,
        "raw_total": accounting.raw_total,
        "measured_layer_damage": accounting.measured_total,
        "closed_total": float(np.sum(accounting.reconciled)),
    }
