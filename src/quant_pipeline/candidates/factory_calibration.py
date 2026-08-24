"""Cross-fitted calibration of factory-local proxy deltas to causal KLD deltas."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.artifacts import canonical_json, sha256_bytes


SCHEMA_FACTORY_DELTA_CALIBRATION = "quant-pipeline.candidate-factory-delta-calibration.v1"


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class FactoryDeltaAnchor:
    """One selection-only direct swap and its factory-blind local proxy."""

    group_id: str
    reference_factory: str
    challenger_factory: str
    proxy_delta: float
    measured_delta_kld: float
    weight: float = 1.0
    evidence_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        if not self.group_id or not self.reference_factory or not self.challenger_factory:
            raise ValueError("factory calibration identities must be non-empty")
        if self.reference_factory == self.challenger_factory:
            raise ValueError("factory calibration requires distinct factories")
        weight = _finite(self.weight, "anchor weight")
        if weight <= 0.0:
            raise ValueError("anchor weight must be positive")
        if self.evidence_sha256 and (
            len(self.evidence_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.evidence_sha256)
        ):
            raise ValueError("anchor evidence identity must be a SHA-256")
        return {
            "group_id": self.group_id,
            "reference_factory": self.reference_factory,
            "challenger_factory": self.challenger_factory,
            "proxy_delta": _finite(self.proxy_delta, "proxy delta"),
            "measured_delta_kld": _finite(self.measured_delta_kld, "measured KLD delta"),
            "weight": weight,
            "evidence_sha256": self.evidence_sha256 or None,
        }


def _fit_identity_residual(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    ridge: float,
) -> tuple[float, float]:
    """Fit y = x + intercept + slope_adjustment*x, shrunk to y=x."""

    design = np.stack((np.ones_like(x), x), axis=1)
    gram = design.T @ (weight[:, None] * design)
    rhs = design.T @ (weight * (y - x))
    penalty = np.eye(2, dtype=np.float64) * ridge
    # The intercept is expressed in the observed KLD scale. Penalize it after
    # normalizing by the proxy RMS so ridge strength is unit-stable.
    scale = max(float(np.sqrt(np.average(x * x, weights=weight))), 1e-12)
    penalty[0, 0] /= scale * scale
    theta = np.linalg.solve(gram + penalty, rhs)
    return float(theta[0]), float(1.0 + theta[1])


def _predict(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * x


def fit_factory_delta_calibration(
    anchors: Iterable[FactoryDeltaAnchor],
    *,
    ridge_grid: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> dict[str, Any]:
    """Fit one cross-validated shrinkage map for each factory comparison.

    The target is a direct selection-only end-to-end KLD swap.  The predictor
    is the difference between factory-blind common local scores.  Ridge shrinks
    toward the identity map rather than toward either factory.  Leave-one-group
    out predictions select ridge strength and are retained as evidence; final
    evaluation rows must remain outside this function.
    """

    rows = [anchor.as_dict() for anchor in anchors]
    if not rows:
        raise ValueError("factory calibration requires anchors")
    ridge_values = sorted({_finite(value, "ridge value") for value in ridge_grid})
    if not ridge_values or ridge_values[0] < 0.0:
        raise ValueError("ridge grid must contain nonnegative values")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["reference_factory"], row["challenger_factory"]), []
        ).append(row)

    fits = []
    for (reference, challenger), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["group_id"])
        if len(values) < 4 or len({row["group_id"] for row in values}) != len(values):
            raise ValueError("each factory comparison needs at least four unique groups")
        x = np.asarray([row["proxy_delta"] for row in values], dtype=np.float64)
        y = np.asarray([row["measured_delta_kld"] for row in values], dtype=np.float64)
        weight = np.asarray([row["weight"] for row in values], dtype=np.float64)
        candidates = []
        for ridge in ridge_values:
            predictions = np.empty_like(y)
            for index in range(len(values)):
                keep = np.arange(len(values)) != index
                intercept, slope = _fit_identity_residual(
                    x[keep], y[keep], weight[keep], ridge
                )
                predictions[index] = intercept + slope * x[index]
            mse = float(np.average((predictions - y) ** 2, weights=weight))
            candidates.append({
                "ridge": ridge,
                "weighted_oof_mse": mse,
                "oof_predictions": predictions.tolist(),
            })
        # Prefer stronger shrinkage on an exact CV tie.
        chosen = min(candidates, key=lambda row: (row["weighted_oof_mse"], -row["ridge"]))
        intercept, slope = _fit_identity_residual(
            x, y, weight, float(chosen["ridge"])
        )
        oof = np.asarray(chosen["oof_predictions"], dtype=np.float64)
        effective = max(float(weight.sum()) - 2.0, 1.0)
        residual_sigma = float(np.sqrt(np.sum(weight * (oof - y) ** 2) / effective))
        fits.append({
            "reference_factory": reference,
            "challenger_factory": challenger,
            "anchor_count": len(values),
            "anchor_group_ids": [row["group_id"] for row in values],
            "selected_ridge": float(chosen["ridge"]),
            "intercept": intercept,
            "slope": slope,
            "oof_weighted_mse": float(chosen["weighted_oof_mse"]),
            "oof_residual_sigma": residual_sigma,
            "identity_oof_weighted_mse": float(np.average((x - y) ** 2, weights=weight)),
            "ridge_search": candidates,
            "anchors": values,
        })
    body = {
        "schema": SCHEMA_FACTORY_DELTA_CALIBRATION,
        "method": "leave-one-group-out-ridge-to-identity-v1",
        "selection_only": True,
        "final_evaluation_rows_consumed": False,
        "fits": fits,
    }
    body["calibration_sha256"] = _hash_json(body)
    return body


def apply_factory_delta_calibration(
    calibration: Mapping[str, Any],
    *,
    reference_factory: str,
    challenger_factory: str,
    proxy_delta: float,
) -> tuple[float, float]:
    """Return calibrated challenger-minus-reference damage and uncertainty."""

    body = dict(calibration)
    seal = body.pop("calibration_sha256", None)
    if seal != _hash_json(body):
        raise ValueError("factory delta calibration seal mismatch")
    matches = [
        row
        for row in calibration["fits"]
        if row["reference_factory"] == reference_factory
        and row["challenger_factory"] == challenger_factory
    ]
    if len(matches) != 1:
        raise ValueError("factory comparison calibration is absent or ambiguous")
    row = matches[0]
    value = float(row["intercept"] + row["slope"] * _finite(proxy_delta, "proxy delta"))
    return value, float(row["oof_residual_sigma"])
