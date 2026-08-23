from pathlib import Path

import pytest

from quant_pipeline.core.artifacts import prepare_empty_destination


def test_prepare_destination_creates_or_accepts_only_empty_directory(tmp_path):
    new = tmp_path / "new"
    assert prepare_empty_destination(new) == new
    assert prepare_empty_destination(new) == new
    (new / "partial.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_empty_destination(new)


def test_prepare_destination_rejects_file(tmp_path):
    target = tmp_path / "file"
    target.write_text("data")
    with pytest.raises(ValueError, match="not a directory"):
        prepare_empty_destination(target)
