from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError


SCRIPT = Path(__file__).parents[1] / "scripts" / "upload_qwen_bulk_remaining_hf.py"
SPEC = importlib.util.spec_from_file_location("upload_qwen_bulk_remaining_hf", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_remote_layer_requires_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args, **_kwargs) -> None:
        response = httpx.Response(
            404,
            request=httpx.Request("GET", "https://huggingface.co/api/datasets/owner/dataset/tree/revision/fits"),
        )
        raise RemoteEntryNotFoundError("layer prefix does not exist", response=response)

    monkeypatch.setattr(MODULE, "_verify_remote_layer", missing)

    assert not MODULE._batch_is_remote(
        object(), "owner/dataset", "revision", "fits", [{"layer": 0}]
    )


def test_other_hub_failures_remain_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def unauthorized(*_args, **_kwargs) -> None:
        response = httpx.Response(
            401,
            request=httpx.Request("GET", "https://huggingface.co/api/datasets/owner/dataset"),
        )
        raise HfHubHTTPError("authentication failed", response=response)

    monkeypatch.setattr(MODULE, "_verify_remote_layer", unauthorized)

    with pytest.raises(HfHubHTTPError, match="authentication failed"):
        MODULE._batch_is_remote(
            object(), "owner/dataset", "revision", "fits", [{"layer": 0}]
        )
