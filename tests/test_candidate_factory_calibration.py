from __future__ import annotations

import pytest

from quant_pipeline.candidates.factory_calibration import (
    FactoryDeltaAnchor,
    apply_factory_delta_calibration,
    fit_factory_delta_calibration,
)


def anchors(values):
    return [
        FactoryDeltaAnchor(
            group_id=f"layer-{index:03d}",
            reference_factory="upstream",
            challenger_factory="mcg",
            proxy_delta=x,
            measured_delta_kld=y,
            weight=weight,
            evidence_sha256=(f"{index + 1:x}" * 64)[:64],
        )
        for index, (x, y, weight) in enumerate(values)
    ]


def test_cross_fitted_calibration_learns_systematic_factory_delta_mapping():
    values = [
        (-0.004, -0.0079, 1.0),
        (-0.002, -0.0039, 1.0),
        (0.001, 0.0021, 1.0),
        (0.003, 0.0062, 1.0),
        (0.005, 0.0101, 1.0),
        (0.008, 0.0160, 1.0),
    ]
    calibration = fit_factory_delta_calibration(anchors(values))
    fit = calibration["fits"][0]
    assert fit["slope"] == pytest.approx(2.0, rel=0.08)
    assert fit["oof_weighted_mse"] < fit["identity_oof_weighted_mse"]
    value, uncertainty = apply_factory_delta_calibration(
        calibration,
        reference_factory="upstream",
        challenger_factory="mcg",
        proxy_delta=0.004,
    )
    assert value == pytest.approx(0.008, abs=4e-4)
    assert uncertainty >= 0.0


def test_calibration_is_sealed_and_requires_enough_unique_groups():
    with pytest.raises(ValueError, match="at least four"):
        fit_factory_delta_calibration(anchors([(0.1, 0.1, 1.0)] * 3))

    calibration = fit_factory_delta_calibration(
        anchors([(0.1, 0.1, 1.0), (0.2, 0.2, 1.0), (0.3, 0.3, 1.0), (0.4, 0.4, 1.0)])
    )
    calibration["fits"][0]["slope"] += 1.0
    with pytest.raises(ValueError, match="seal mismatch"):
        apply_factory_delta_calibration(
            calibration,
            reference_factory="upstream",
            challenger_factory="mcg",
            proxy_delta=0.2,
        )


def test_factory_comparison_cannot_compare_a_factory_to_itself():
    with pytest.raises(ValueError, match="distinct factories"):
        FactoryDeltaAnchor(
            group_id="layer-0",
            reference_factory="mcg",
            challenger_factory="mcg",
            proxy_delta=0.0,
            measured_delta_kld=0.0,
        ).as_dict()
