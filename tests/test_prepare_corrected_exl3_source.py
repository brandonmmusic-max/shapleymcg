import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_corrected_exl3_source.py"
MANIFEST = ROOT / "configs" / "qwen3-30b-a3b-b200" / "corrected-exl3-source-manifest.json"


def test_corrected_exl3_source_preparer_is_pinned_and_dry_run(tmp_path):
    destination = tmp_path / "source"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    manifest = json.loads(MANIFEST.read_text())
    assert plan["dry_run"] is True
    assert plan["revision"] == manifest["revision"]
    assert plan["file_count"] == len(manifest["files"])
    assert plan["source_root"].endswith("reproducibility/r10")
    assert not destination.exists()


def test_corrected_exl3_source_manifest_rejects_unsafe_paths(tmp_path):
    document = json.loads(MANIFEST.read_text())
    document["files"]["../escape.py"] = "0" * 64
    broken = tmp_path / "manifest.json"
    broken.write_text(json.dumps(document))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(tmp_path / "source"), "--manifest", str(broken)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unsafe entry" in result.stderr


def test_corrected_exl3_source_preparer_refuses_existing_destination(tmp_path):
    destination = tmp_path / "source"
    destination.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(destination)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "destination must not exist" in result.stderr
