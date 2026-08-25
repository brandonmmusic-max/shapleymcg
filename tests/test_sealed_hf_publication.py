from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPLOAD = _load("upload_sealed_artifact_tree_hf")
VERIFY = _load("verify_hf_artifact_tree")


def _tree(root: Path) -> dict:
    (root / "nested").mkdir(parents=True)
    data = root / "nested" / "result.json"
    data.write_bytes(b"sealed\n")
    rows = [{"path": "nested/result.json", "bytes": data.stat().st_size, "sha256": sha256_file(data)}]
    manifest = {
        "schema": "quant-pipeline.artifact-tree-manifest.v1",
        "label": "fixture",
        "file_count": 1,
        "total_bytes": data.stat().st_size,
        "files": rows,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    (root / "MANIFEST.json").write_bytes(canonical_json(manifest))
    (root / "SHA256SUMS").write_text(f"{rows[0]['sha256']}  nested/result.json\n")
    return manifest


def test_upload_set_excludes_unsealed_and_excluded_files(tmp_path: Path) -> None:
    manifest = _tree(tmp_path)
    (tmp_path / "scratch.log").write_text("must not upload")
    (tmp_path / "PUBLICATION_RECEIPT.json").write_text("must not upload")
    paths = UPLOAD._sealed_upload_paths(tmp_path, manifest)
    assert [relative for relative, _ in paths] == [
        "nested/result.json",
        "MANIFEST.json",
        "SHA256SUMS",
    ]


def test_upload_set_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid sealed artifact path"):
        UPLOAD._sealed_upload_paths(tmp_path, {"files": [{"path": "../escape"}]})


def test_remote_inventory_distinguishes_files_and_folders() -> None:
    class Api:
        def list_repo_tree(self, **_kwargs):
            return [
                SimpleNamespace(path="prefix/folder", size=None),
                SimpleNamespace(path="prefix/file", size=0),
            ]

    assert VERIFY._remote_file_inventory(
        Api(), repo="owner/repo", repo_type="dataset", revision="a" * 40, prefix="prefix"
    ) == {"prefix/file"}


def test_remote_inventory_unavailable_for_old_hub_api() -> None:
    assert VERIFY._remote_file_inventory(
        object(), repo="owner/repo", repo_type="dataset", revision="a" * 40, prefix="prefix"
    ) is None


def test_remote_namespace_rejects_unsealed_extra_file() -> None:
    with pytest.raises(ValueError, match="unexpected=.*scratch.log"):
        VERIFY._require_closed_namespace(
            {"prefix/MANIFEST.json", "prefix/SHA256SUMS", "prefix/scratch.log"},
            {"prefix/MANIFEST.json", "prefix/SHA256SUMS"},
        )
