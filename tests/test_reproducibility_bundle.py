from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-30b-a3b-b200"
SCRIPTS = ROOT / "scripts"
PLACEHOLDER = re.compile(r"^__REQUIRED_[A-Z0-9_]+__$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
    )


def test_qwen_example_configs_parse_and_bind_exact_geometry():
    artifact = json.loads((CONFIG / "artifact-lock.json").read_text())
    adapter = json.loads((CONFIG / "adapter-config.json").read_text())
    campaign = json.loads((CONFIG / "campaign.json").read_text())
    experiment = tomllib.loads((CONFIG / "experiment.toml").read_text())
    assert artifact["model"]["revision"] == "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
    assert artifact["corpus"]["dataset_revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert artifact["b12x"]["commit"] == "36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8"
    assert all(SHA256.fullmatch(value) for value in artifact["b12x"]["closure"].values())
    assert adapter["schema"] == "quant-pipeline.qwen-production-adapter.v1"
    assert (adapter["num_hidden_layers"], adapter["num_experts"], adapter["hidden_size"], adapter["moe_intermediate_size"]) == (48, 128, 2048, 768)
    assert adapter["scientific_contract"]["normalization"] == "source-derived-absolute-v31"
    assert campaign["layers"] == list(range(48))
    assert campaign["retention_mode"] == "capture-plus-ledger"
    assert experiment["codec"]["bits"] == [3, 4, 5]


def test_hash_fields_are_exact_or_explicit_required_placeholders():
    artifact = json.loads((CONFIG / "artifact-lock.json").read_text())
    values = [
        artifact["model"]["config_sha256"],
        artifact["model"]["index_sha256"],
        artifact["model"]["shard_manifest_sha256"],
        artifact["corpus"]["calibration_jsonl_sha256"],
        artifact["corpus"]["sealed_corpus_sha256"],
        artifact["corpus"]["kld_window_sha256"],
        artifact["exl3"]["corrected_source_manifest_sha256"],
        artifact["exl3"]["source_closure_sha256"],
        artifact["exl3"]["numeric_core_sha256"],
        artifact["exl3"]["r10_codec_sha256"],
        artifact["exl3"]["sm100_extension_sha256"],
    ]
    assert all(SHA256.fullmatch(value) or PLACEHOLDER.fullmatch(value) for value in values)
    assert any(PLACEHOLDER.fullmatch(value) for value in values)


def test_environment_manifest_is_exactly_versioned_and_requires_machine_seals():
    lock = json.loads((ROOT / "environments/b200-cu132.lock.json").read_text())
    assert lock["python"] == "3.12.3"
    assert lock["cuda_toolkit"] == "13.2"
    assert lock["compute_capability"] == "10.0"
    assert lock["packages"]["torch"] == "2.12.1+cu132"
    assert lock["packages"]["transformers"] == "5.12.1"
    assert PLACEHOLDER.fullmatch(lock["nvidia_driver"])
    requirements = (ROOT / "environments/requirements-b200-cu132.txt").read_text().splitlines()
    assert "torch==2.12.1+cu132" in requirements
    assert "datasets==4.7.0" in requirements
    assert all("==" in row or row.startswith("--") for row in requirements if row)


def test_placeholder_validator_is_reviewable_but_fails_production_use():
    review = run("validate_repro_config.py", "--config-dir", str(CONFIG), "--allow-placeholders")
    assert review.returncode == 0, review.stderr
    assert json.loads(review.stdout)["ok"] is False
    production = run("validate_repro_config.py", "--config-dir", str(CONFIG))
    assert production.returncode != 0
    assert "refusing production use" in production.stderr


def test_mutating_scripts_default_to_dry_run(tmp_path):
    qwen = run("prepare_qwen_checkpoint.py", "--destination", str(tmp_path / "qwen"))
    assert qwen.returncode == 0, qwen.stderr
    assert json.loads(qwen.stdout)["dry_run"] is True
    assert json.loads(qwen.stdout)["shard_count"] == 16
    assert not (tmp_path / "qwen").exists()

    corrected = run("prepare_corrected_exl3_source.py", "--destination", str(tmp_path / "corrected-r10"))
    assert corrected.returncode == 0, corrected.stderr
    assert json.loads(corrected.stdout)["dry_run"] is True
    assert not (tmp_path / "corrected-r10").exists()

    exllama = run("prepare_exllamav3_checkout.py", "--destination", str(tmp_path / "exllamav3"))
    assert exllama.returncode == 0, exllama.stderr
    assert json.loads(exllama.stdout)["commit"] == "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
    assert json.loads(exllama.stdout)["dry_run"] is True
    assert not (tmp_path / "exllamav3").exists()

    destination = tmp_path / "b12x"
    checkout = run("prepare_b12x_checkout.py", "--destination", str(destination))
    assert checkout.returncode == 0, checkout.stderr
    assert json.loads(checkout.stdout)["dry_run"] is True
    assert not destination.exists()

    source = tmp_path / "exl3"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
    build = run("bootstrap_sm100_exl3.py", "--source", str(source))
    assert build.returncode == 0, build.stderr
    assert json.loads(build.stdout)["dry_run"] is True

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "plan.json").write_text("{}")
    preflight = run("preflight_qwen_campaign.py", "--campaign-dir", str(campaign))
    assert preflight.returncode == 0, preflight.stderr
    assert json.loads(preflight.stdout)["dry_run"] is True


def test_resource_estimator_uses_qwen_geometry_and_retention():
    result = run("estimate_qwen_b200_resources.py", "--retention", "capture-plus-ledger")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["geometry"] == {
        "experts": 128,
        "hidden_size": 2048,
        "intermediate_size": 768,
        "layers": 48,
        "parameters": 30_500_000_000,
        "top_k": 8,
    }
    assert report["recommended"]["estimated_peak_disk_bytes"] > 0
    assert report["recommended"]["minimum_free_vram_bytes_per_gpu"] > 61_000_000_000
    adapter = json.loads((CONFIG / "adapter-config.json").read_text())
    assert adapter["estimated_peak_bytes"] >= report["recommended"]["estimated_peak_disk_bytes"]
    assert adapter["safety_margin_bytes"] >= report["recommended"]["safety_margin_bytes"]
    assert adapter["min_free_gpu_bytes_per_device"] >= report["recommended"]["minimum_free_vram_bytes_per_gpu"]
    assert adapter["min_available_ram_bytes"] >= report["recommended"]["minimum_available_host_ram_bytes"]


def test_scripts_and_runbook_expose_fail_closed_boundaries():
    required = {
        "prepare_qwen_checkpoint.py",
        "prepare_corrected_exl3_source.py",
        "prepare_exllamav3_checkout.py",
        "verify_exllamav3_checkout.py",
        "prepare_b12x_checkout.py",
        "verify_b12x_checkout.py",
        "bootstrap_sm100_exl3.py",
        "smoke_sm100_stack.py",
        "verify_b200_environment.py",
        "estimate_qwen_b200_resources.py",
        "validate_repro_config.py",
        "preflight_qwen_campaign.py",
    }
    assert required <= {path.name for path in SCRIPTS.glob("*.py")}
    for name in required:
        help_result = run(name, "--help")
        assert help_result.returncode == 0, f"{name}: {help_result.stderr}"
    runbook = (ROOT / "docs/B200_RUNBOOK.md").read_text()
    assert "## OWNER APPROVAL BARRIER" in runbook
    assert "Stop here" in runbook
    assert "campaign execute ... --execute" in runbook


def test_distribution_configuration_includes_reproducibility_assets():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    included = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    for name in ("configs", "docs", "environments", "examples", "scripts"):
        assert name in included
