from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from quant_pipeline.results.ledger import render_markdown, seal_row, validate_ledger


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_results_ledger", ROOT / "scripts" / "build_results_ledger.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_checked_in_results_ledger_is_sealed_reproducible_and_renderable() -> None:
    path = ROOT / "results" / "RESULTS_LEDGER.json"
    observed = json.loads(path.read_text())
    validate_ledger(observed)
    reproduced = BUILDER.build(ROOT / "results" / "qwen-complete-results-ledger.json")
    assert reproduced == observed
    rendered = render_markdown(observed)
    assert "| Result | Design | Parent revision |" in rendered
    assert len(rendered.splitlines()) == len(observed["rows"]) + 2


def test_result_row_requires_real_verification_hashes_for_verified_status() -> None:
    body = {
        "schema": "shapleymcg.result-ledger-row.v1",
        "id": "fixture",
        "method_profile": {
            "allocation": "fixture",
            "candidate_factory": None,
            "calibration": None,
            "codec": "fixture",
            "encoding": None,
        },
        "design": {
            "kind": "legacy-reported-endpoint",
            "rate_selection": "not-recorded",
            "factory_selection": "not-recorded",
            "bindings": {
                "allocation_plan_sha256": None,
                "candidate_inventories": [],
                "score_calibration_sha256": None,
                "budget_closure_sha256": None,
                "untouched_endpoint_sha256": None,
                "replay_receipt_sha256": None,
            },
        },
        "parent": {"repository": "owner/model", "revision": "a" * 40},
        "panel": {"id": "fixture", "sha256": None, "positions": 64},
        "backend": {"attention": "sdpa", "kv_cache": "not-used"},
        "evaluator": {"metric": "mean-kld", "implementation": "fixture"},
        "scope": {"weights": "experts", "nonexpert": "bf16", "comparison": "fixture"},
        "rate": {"value": 3.5, "unit": "bpw", "scope": "experts"},
        "value": {"metric": "mean_kld", "value": 0.1},
        "evidence": {
            "path": "report.json",
            "evidence_sha256": "b" * 64,
            "report_sha256": "c" * 64,
            "verification_sha256": None,
        },
        "status": {"state": "independently-verified", "detail": "fixture"},
    }
    with pytest.raises(ValueError, match="require evidence, report, and verification"):
        seal_row(body)


def test_results_ledger_tamper_is_rejected() -> None:
    ledger = json.loads((ROOT / "results" / "RESULTS_LEDGER.json").read_text())
    ledger["rows"][0]["value"]["value"] += 0.001
    with pytest.raises(ValueError, match="row seal mismatch"):
        validate_ledger(ledger)


def test_completed_progressive_union_can_be_appended_without_invented_hashes(
    tmp_path: Path,
) -> None:
    inventories = {}
    for role, marker in (("baseline", "c"), ("challenger", "d")):
        inventory = {
            "repo_id": f"owner/{role}",
            "revision": marker * 40,
            "layers": [],
        }
        inventory["inventory_sha256"] = BUILDER._hash_json(inventory)
        path = tmp_path / f"{role}-inventory.json"
        path.write_bytes(BUILDER.canonical_json(inventory))
        inventories[role] = (inventory, path)
    allocation = {
        "average_weight_bits": 3.5,
        "k3_count": 2,
        "k4_count": 2,
        "baseline_candidate_inventory_sha256": inventories["baseline"][0]["inventory_sha256"],
        "challenger_candidate_inventory_sha256": inventories["challenger"][0]["inventory_sha256"],
        "choices": [{"bits": bit} for bit in (3, 4, 3, 4)],
    }
    allocation["allocation_sha256"] = BUILDER._hash_json(allocation)
    report = {
        "factory_allocation_sha256": allocation["allocation_sha256"],
        "panel_sha256": "a" * 64,
        "fixed_nonexpert_scope": "source-BF16",
        "selection": {"row": 0, "greedy_path": []},
        "untouched_validation": {
            "union_records": [{"positions": 2048} for _ in range(9)],
            "union_summary": {"mean": 0.029},
            "baseline_summary": {"mean": 0.031},
            "absolute_reduction": 0.002,
            "relative_reduction": 0.002 / 0.031,
            "rows_union_better": 7,
            "row_count": 9,
        },
    }
    report["report_sha256"] = BUILDER._hash_json(report)
    verification = {
        "report_sha256": report["report_sha256"],
        "implementation": "torch.float64 log_softmax independent recomputation",
    }
    verification["verification_sha256"] = BUILDER._hash_json(verification)
    (tmp_path / "factory-allocation.json").write_bytes(BUILDER.canonical_json(allocation))
    (tmp_path / "report.json").write_bytes(BUILDER.canonical_json(report))
    (tmp_path / "independent-verification.json").write_bytes(BUILDER.canonical_json(verification))
    (tmp_path / "plan.json").write_bytes(BUILDER.canonical_json({
        "attention_backend": "sdpa",
        "baseline_inventory": str(inventories["baseline"][1]),
        "challenger_inventory": str(inventories["challenger"][1]),
    }))
    row = BUILDER._progressive_row(
        tmp_path,
        parent_repository="Qwen/Qwen3-30B-A3B-Base",
        parent_revision="b" * 40,
    )
    assert row["value"]["value"] == 0.029
    assert row["panel"]["positions"] == 9 * 2048
    assert row["evidence"]["report_sha256"] == report["report_sha256"]
    assert row["evidence"]["verification_sha256"] == verification["verification_sha256"]
    assert row["status"]["state"] == "independently-verified"
    assert row["design"]["kind"] == "frozen-rate-factory-diagnostic"
    assert len(row["design"]["bindings"]["candidate_inventories"]) == 2


def test_joint_factory_rate_claim_cannot_reuse_a_frozen_or_incomplete_design() -> None:
    sealed = json.loads((ROOT / "results" / "RESULTS_LEDGER.json").read_text())["rows"][0]
    body = {key: value for key, value in sealed.items() if key != "row_sha256"}
    body["id"] = "invalid-joint-claim"
    body["design"] = {
        "kind": "joint-factory-rate-allocation",
        "rate_selection": "frozen",
        "factory_selection": "selected",
        "bindings": {
            "allocation_plan_sha256": None,
            "candidate_inventories": [],
            "score_calibration_sha256": None,
            "budget_closure_sha256": None,
            "untouched_endpoint_sha256": None,
            "replay_receipt_sha256": None,
        },
    }
    with pytest.raises(ValueError, match="identify both selections as joint"):
        seal_row(body)
