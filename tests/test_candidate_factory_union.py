from __future__ import annotations

from dataclasses import dataclass

import pytest

from quant_pipeline.allocation.global_dp import allocate
from quant_pipeline.candidates.factory_union import (
    CandidateUnit,
    CommonCandidateScore,
    FactoryIdentity,
    FactoryProposal,
    build_factory_union,
)
from quant_pipeline.candidates.factory_allocation import allocate_factory_union


HASHES = {
    name: (format(index, "x") * 64)
    for index, name in enumerate(
        ("source", "calibration", "native-code", "upstream-code", "packed", "reconstruction", "instrument", "fit", "runtime", "other-runtime"),
        start=1,
    )
}


@dataclass
class Factory:
    identity: FactoryIdentity
    damages: dict[tuple[str, int], float]

    def propose(self, unit):
        identity = self.identity.as_dict()["identity_sha256"]
        for rate in unit.requested_rates:
            key = (unit.unit_id, rate)
            if key not in self.damages:
                continue
            suffix = f"{self.identity.name}-{unit.unit_id}-{rate}"
            yield FactoryProposal(
                unit_id=unit.unit_id,
                rate=rate,
                factory_name=self.identity.name,
                factory_identity_sha256=identity,
                packed_sha256=HASHES["packed"],
                reconstruction_sha256=("a" if self.identity.name == "native-mcg" else "b") * 64,
                exact_stored_bytes=rate * 100,
                payload_ref=f"payloads/{suffix}",
                reconstruction_ref=f"reconstructions/{suffix}",
                compatibility_domain_sha256=(
                    "c" if self.identity.name == "native-mcg" else "d"
                ) * 64,
                fixed_shared_bytes=0,
                # Deliberately adversarial: this value must never reach the allocator.
                metadata={"factory_reported_damage": -1e30},
            )


@dataclass
class Scorer:
    damages: dict[tuple[str, str, int], float]

    def score(self, unit, proposal):
        value = self.damages[(proposal.factory_name, unit.unit_id, proposal.rate)]
        return CommonCandidateScore(
            raw_damage=value,
            calibrated_damage=value,
            uncertainty=value / 10,
            instrument_sha256=HASHES["instrument"],
            calibration_fit_sha256=HASHES["fit"],
            metadata={"basis": "common-test-instrument"},
        )


def identity(name, implementation):
    return FactoryIdentity(
        name,
        "1",
        implementation,
        "exl3-mcg",
        HASHES["runtime"],
        {"test": True},
    )


def unit(name="unit-0"):
    return CandidateUnit(
        unit_id=name,
        coupling_group_id="layer-0",
        source_sha256=HASHES["source"],
        source_shape=(8, 16),
        requested_rates=(3, 4),
        calibration_identity_sha256=HASHES["calibration"],
        metadata={"projection": "gate_proj"},
    )


def test_optional_upstream_candidates_join_native_mcg_and_common_scorer_decides():
    native = Factory(identity("native-mcg", HASHES["native-code"]), {("unit-0", 3): 9, ("unit-0", 4): 4})
    upstream = Factory(identity("upstream", HASHES["upstream-code"]), {("unit-0", 3): 6, ("unit-0", 4): 5})
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.9,
        ("native-mcg", "unit-0", 4): 0.4,
        ("upstream", "unit-0", 3): 0.6,
        ("upstream", "unit-0", 4): 0.5,
    })
    union = build_factory_union(
        [unit()], [native, upstream], scorer, required_factory_names=["native-mcg"]
    )
    assert union.ledger["coupled_allocation_required"] is True

    assert len(union.ledger["records"]) == 4
    assert {row["proposal"]["factory_name"] for row in union.ledger["records"]} == {
        "native-mcg",
        "upstream",
    }
    selected = allocate(union.allocator_candidates, byte_budget=400).choices
    assert len(selected) == 1
    assert selected[0].metadata["factory_name"] == "native-mcg"
    assert selected[0].metadata["rate"] == 4
    assert selected[0].predicted_damage == 0.4


def test_coupled_allocator_selects_one_complete_domain_per_layer():
    native = Factory(
        identity("native-mcg", HASHES["native-code"]),
        {
            ("unit-0", 3): 9,
            ("unit-0", 4): 4,
            ("unit-1", 3): 9,
            ("unit-1", 4): 4,
        },
    )
    upstream = Factory(
        identity("upstream", HASHES["upstream-code"]),
        {
            ("unit-0", 3): 6,
            ("unit-0", 4): 5,
            ("unit-1", 3): 6,
            ("unit-1", 4): 5,
        },
    )
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.1,
        ("native-mcg", "unit-0", 4): 0.09,
        ("native-mcg", "unit-1", 3): 0.8,
        ("native-mcg", "unit-1", 4): 0.7,
        ("upstream", "unit-0", 3): 0.8,
        ("upstream", "unit-0", 4): 0.7,
        ("upstream", "unit-1", 3): 0.1,
        ("upstream", "unit-1", 4): 0.09,
    })
    union = build_factory_union(
        [unit("unit-0"), unit("unit-1")],
        [native, upstream],
        scorer,
        required_factory_names=["native-mcg"],
    )

    # A flat allocator would illegally take one matrix from each transform
    # domain.  The compatibility-safe allocator must choose one whole domain.
    flat = allocate(union.allocator_candidates, byte_budget=600).choices
    assert {row.metadata["factory_name"] for row in flat} == {"native-mcg", "upstream"}
    coupled = allocate_factory_union(union, byte_budget=600)
    assert len({row.metadata["compatibility_domain_sha256"] for row in coupled.selected_candidates}) == 1
    assert len(coupled.selected_candidates) == 2
    assert coupled.manifest["stored_bytes"] == 600


def test_coupled_allocator_ignores_incomplete_optional_domain():
    native = Factory(
        identity("native-mcg", HASHES["native-code"]),
        {
            ("unit-0", 3): 9,
            ("unit-0", 4): 4,
            ("unit-1", 3): 9,
            ("unit-1", 4): 4,
        },
    )
    upstream = Factory(
        identity("upstream", HASHES["upstream-code"]),
        {("unit-0", 3): 6, ("unit-0", 4): 5},
    )
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.4,
        ("native-mcg", "unit-0", 4): 0.3,
        ("native-mcg", "unit-1", 3): 0.4,
        ("native-mcg", "unit-1", 4): 0.3,
        ("upstream", "unit-0", 3): 0.01,
        ("upstream", "unit-0", 4): 0.005,
    })
    union = build_factory_union(
        [unit("unit-0"), unit("unit-1")],
        [native, upstream],
        scorer,
        required_factory_names=["native-mcg"],
    )
    coupled = allocate_factory_union(union, byte_budget=800)
    assert {row.metadata["factory_name"] for row in coupled.selected_candidates} == {"native-mcg"}


def test_coupled_allocator_can_mix_factories_inside_identical_shared_domain():
    class SameDomainFactory(Factory):
        def propose(self, candidate_unit):
            for proposal in super().propose(candidate_unit):
                yield FactoryProposal(
                    **(
                        proposal.__dict__
                        | {
                            "compatibility_domain_sha256": "c" * 64,
                            "fixed_shared_bytes": 50,
                        }
                    )
                )

    native = SameDomainFactory(
        identity("native-mcg", HASHES["native-code"]),
        {
            ("unit-0", 3): 9,
            ("unit-0", 4): 4,
            ("unit-1", 3): 9,
            ("unit-1", 4): 4,
        },
    )
    upstream = SameDomainFactory(
        identity("upstream", HASHES["upstream-code"]),
        {
            ("unit-0", 3): 6,
            ("unit-0", 4): 5,
            ("unit-1", 3): 6,
            ("unit-1", 4): 5,
        },
    )
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.1,
        ("native-mcg", "unit-0", 4): 0.09,
        ("native-mcg", "unit-1", 3): 0.8,
        ("native-mcg", "unit-1", 4): 0.7,
        ("upstream", "unit-0", 3): 0.8,
        ("upstream", "unit-0", 4): 0.7,
        ("upstream", "unit-1", 3): 0.1,
        ("upstream", "unit-1", 4): 0.09,
    })
    union = build_factory_union(
        [unit("unit-0"), unit("unit-1")],
        [native, upstream],
        scorer,
        required_factory_names=["native-mcg"],
    )
    coupled = allocate_factory_union(union, byte_budget=650)
    assert union.ledger["coupled_allocation_required"] is False
    assert {row.metadata["factory_name"] for row in coupled.selected_candidates} == {
        "native-mcg",
        "upstream",
    }
    assert coupled.manifest["stored_bytes"] == 650
    assert coupled.manifest["selected_groups"][0]["fixed_shared_bytes"] == 50


def test_new_model_needs_no_optional_upstream_factory():
    native = Factory(identity("native-mcg", HASHES["native-code"]), {("unit-0", 3): 9, ("unit-0", 4): 4})
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.9,
        ("native-mcg", "unit-0", 4): 0.4,
    })
    union = build_factory_union(
        [unit()], [native], scorer, required_factory_names=["native-mcg"]
    )
    assert len(union.allocator_candidates) == 2
    assert union.ledger["coverage"]["native-mcg"] == [
        {"unit_id": "unit-0", "rate": 3},
        {"unit_id": "unit-0", "rate": 4},
    ]


def test_required_native_factory_must_cover_every_unit_and_rate():
    native = Factory(identity("native-mcg", HASHES["native-code"]), {("unit-0", 3): 9})
    scorer = Scorer({("native-mcg", "unit-0", 3): 0.9})
    with pytest.raises(ValueError, match="lacks 1 unit/rate proposals"):
        build_factory_union(
            [unit()], [native], scorer, required_factory_names=["native-mcg"]
        )


def test_factory_identity_and_proposal_identity_must_match():
    class BrokenFactory(Factory):
        def propose(self, candidate_unit):
            for proposal in super().propose(candidate_unit):
                yield FactoryProposal(
                    **(
                        proposal.__dict__
                        | {"factory_identity_sha256": HASHES["upstream-code"]}
                    )
                )

    native = BrokenFactory(
        identity("native-mcg", HASHES["native-code"]),
        {("unit-0", 3): 9, ("unit-0", 4): 4},
    )
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.9,
        ("native-mcg", "unit-0", 4): 0.4,
    })
    with pytest.raises(ValueError, match="identity differs"):
        build_factory_union(
            [unit()], [native], scorer, required_factory_names=["native-mcg"]
        )


def test_incompatible_runtime_payload_factory_cannot_enter_allocation_union():
    native = Factory(identity("native-mcg", HASHES["native-code"]), {("unit-0", 3): 9, ("unit-0", 4): 4})
    foreign_identity = FactoryIdentity(
        "foreign",
        "1",
        HASHES["upstream-code"],
        "foreign-codec",
        HASHES["other-runtime"],
        {"test": True},
    )
    foreign = Factory(foreign_identity, {("unit-0", 3): 2, ("unit-0", 4): 1})
    scorer = Scorer({
        ("native-mcg", "unit-0", 3): 0.9,
        ("native-mcg", "unit-0", 4): 0.4,
        ("foreign", "unit-0", 3): 0.2,
        ("foreign", "unit-0", 4): 0.1,
    })
    with pytest.raises(ValueError, match="not directly co-emittable"):
        build_factory_union(
            [unit()], [native, foreign], scorer, required_factory_names=["native-mcg"]
        )
