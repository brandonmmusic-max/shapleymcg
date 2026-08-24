from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_reap_recall_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_reap_recall_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packing_preserves_every_source_row_in_one_axis_bucket(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"axis": f"axis-{axis}", "text": f"text-{axis}-{index}"}
        for axis in range(4)
        for index in range(64)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "packed.jsonl"
    receipt = tmp_path / "receipt.json"

    document = MODULE.prepare(source, output, receipt, expected, 17)

    packed = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(packed) == 16
    assert {row["domain"] for row in packed} == {f"axis-{axis}" for axis in range(4)}
    assert sum(row["source_rows"] for row in document["aggregates"]) == len(rows)
    assert document["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(receipt.read_text()) == document


def test_packing_rejects_wrong_source_hash(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"axis":"axis-0","text":"x"}\n')
    try:
        MODULE.prepare(source, tmp_path / "out", tmp_path / "receipt", "0" * 64, 17)
    except ValueError as error:
        assert "SHA256 mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("wrong source hash was accepted")
