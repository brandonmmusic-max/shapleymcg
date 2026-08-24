import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_qwen_checkpoint.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_qwen_checkpoint", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen_checkpoint_preparer_is_pinned_and_dry_run(tmp_path):
    destination = tmp_path / "qwen"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["revision"] == "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
    assert plan["shard_count"] == 16
    assert not destination.exists()


def test_qwen_checkpoint_receipt_hashes_required_files(tmp_path):
    module = load_script()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for index, relative in enumerate(module.REQUIRED_FILES):
        (checkpoint / relative).write_bytes(f"payload-{index}".encode())
    receipt = module.verify_checkpoint(checkpoint)
    assert receipt["schema"] == "quant-pipeline.qwen-source-receipt.v1"
    assert receipt["config_sha256"] == receipt["files"]["config.json"]["sha256"]
    assert receipt["index_sha256"] == receipt["files"]["model.safetensors.index.json"]["sha256"]
    assert len(receipt["files"]) == len(module.REQUIRED_FILES)
    assert len(receipt["receipt_sha256"]) == 64
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    canonical = (
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    assert receipt["receipt_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_qwen_checkpoint_preparer_refuses_existing_destination(tmp_path):
    destination = tmp_path / "qwen"
    destination.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(destination)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "destination must not exist" in result.stderr


def test_qwen_checkpoint_preparer_accepts_an_explicit_immutable_parent(tmp_path):
    destination = tmp_path / "posttrained"
    revision = "4c446470ba0aec43e22ac1128f9ffd915f338ba3"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--destination",
            str(destination),
            "--repository",
            "Qwen/Qwen3-30B-A3B",
            "--revision",
            revision,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["repository"] == "Qwen/Qwen3-30B-A3B"
    assert plan["revision"] == revision
    assert plan["dry_run"] is True
    assert not destination.exists()
