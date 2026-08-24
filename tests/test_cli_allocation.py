import json
from types import SimpleNamespace

import pytest

from quant_pipeline.cli import command_allocate


def test_allocation_metadata_cannot_override_reserved_fields(tmp_path, capsys):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "unit_id": "unit",
                        "choice_id": "real",
                        "stored_bytes": 10,
                        "predicted_damage": 1.0,
                        "metadata": {"bits": 3, "choice_id": "forged", "stored_bytes": 999, "predicted_damage": -99},
                    }
                ]
            }
        )
    )
    output = tmp_path / "allocation.json"
    args = SimpleNamespace(
        candidates=str(candidates),
        byte_budget=10,
        quantum=1,
        output=str(output),
        non_competitive_reference=True,
    )
    command_allocate(args)
    document = json.loads(output.read_text())
    assert document["competitive"] is False
    assert document["eligibility"] == "reference-only-not-admissible-for-production-or-quality-claims"
    choice = document["choices"]["unit"]
    assert choice["choice_id"] == "real"
    assert choice["stored_bytes"] == 10
    assert choice["predicted_damage"] == 1.0
    assert choice["bits"] == 3


def test_raw_candidate_allocation_fails_closed_without_noncompetitive_acknowledgement(tmp_path):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidates": []}))
    output = tmp_path / "allocation.json"
    with pytest.raises(ValueError, match="not a validated ledger"):
        command_allocate(
            SimpleNamespace(
                candidates=str(candidates),
                byte_budget=0,
                quantum=1,
                output=str(output),
                non_competitive_reference=False,
            )
        )
    assert not output.exists()
