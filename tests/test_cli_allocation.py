import json
from types import SimpleNamespace

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
    command_allocate(SimpleNamespace(candidates=str(candidates), byte_budget=10, quantum=1, output=str(output)))
    choice = json.loads(output.read_text())["choices"]["unit"]
    assert choice["choice_id"] == "real"
    assert choice["stored_bytes"] == 10
    assert choice["predicted_damage"] == 1.0
    assert choice["bits"] == 3

