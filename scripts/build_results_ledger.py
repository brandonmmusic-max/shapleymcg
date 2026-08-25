#!/usr/bin/env python3
"""Build or validate the canonical, sealed experiment-results ledger.

Historical rows are migrated without inventing missing provenance.  A completed
progressive factory-union result may be appended only when its producer report
and independent float64 verification are both present and mutually bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from quant_pipeline.results.ledger import (
    ROW_SCHEMA,
    render_markdown,
    seal_ledger,
    seal_row,
    validate_ledger,
)


PANEL_BY_ID = {
    "base-glm-predecessor": ("glm-lineage-wikitext-2047", "sdpa"),
    "base-glm-full-causal": ("glm-lineage-wikitext-2047", "sdpa"),
    "base-20k-predecessor": ("wikitext-10x2048", "sdpa"),
    "base-20k-full-causal": ("wikitext-10x2048", "sdpa"),
    "base-20k-uniform-k3": ("wikitext-10x2048", "sdpa"),
    "base-20k-uniform-k4": ("wikitext-10x2048", "sdpa"),
    "base-20k-score-blind-five-seed-mean": ("wikitext-10x2048-five-seed-aggregate", "sdpa"),
    "base-hill-reconstruction-predecessor": ("reconstructed-hill-bfcl-ruler-32752", "legacy-unspecified"),
    "posttrained-20k-predecessor-source-bf16-body": ("wikitext-10x2048", "eager"),
    "posttrained-20k-full-causal-source-bf16-body": ("wikitext-10x2048", "eager"),
    "posttrained-glm-full-causal": ("glm-lineage-wikitext-2047", "sdpa"),
    "posttrained-hill-reconstruction-full-causal": ("reconstructed-hill-bfcl-ruler-32752", "eager"),
    "posttrained-turbo-codec-predecessor-allocation": ("wikitext-10x2048", "eager"),
    "posttrained-turbo-codec-full-causal-allocation": ("wikitext-10x2048", "eager"),
    "posttrained-mcg-codec-full-causal-allocation-turbo-body": ("wikitext-10x2048", "eager"),
    "posttrained-turbo-uniform-k4": ("wikitext-10x2048", "eager"),
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SUPPLEMENTAL_ROWS = REPOSITORY_ROOT / "results/canonical-supplemental-results.json"


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _verify_seal(document: dict[str, Any], field: str, label: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if document.get(field) != _hash_json(body):
        raise ValueError(f"{label} seal mismatch")


def _split_parent(parent: str) -> tuple[str, str]:
    repository, separator, revision = parent.rpartition("@")
    if not separator or not repository or not revision:
        raise ValueError(f"parent is not repository@revision: {parent}")
    return repository, revision


def _legacy_row(row: dict[str, Any], source: Path, source_sha256: str) -> dict[str, Any]:
    identifier = str(row["id"])
    repository, revision = _split_parent(str(row["parent"]))
    panel_id, default_attention = PANEL_BY_ID.get(
        identifier, ("legacy-unspecified", "legacy-unspecified")
    )
    attention = str(row.get("attention", default_attention)).lower()
    positions = row.get("positions") or row.get("positions_per_seed")
    report_sha256 = row.get("report_sha256")
    verification_sha256 = row.get("independent_verification_sha256")
    reconstructed = row.get("comparison_status") == "reconstructed-not-strict-paper-reproduction"
    if reconstructed:
        state = "reconstructed"
        detail = "Cross-system reconstruction; not a strict paper reproduction."
    else:
        state = "source-sealed"
        detail = (
            "Claim is sealed to the legacy ledger. Recorded report/replay identities are retained, "
            "but this migration does not claim to have re-opened their remote bytes."
        )
    nonexpert = str(row.get("nonexpert_scope", "source-BF16-or-legacy-unspecified"))
    return seal_row(
        {
            "schema": ROW_SCHEMA,
            "id": identifier,
            "method_profile": {
                "allocation": str(row.get("allocator", "legacy-unspecified")),
                "candidate_factory": str(row.get("codec", "legacy-unspecified")),
                "calibration": None,
                "codec": str(row.get("codec", "legacy-unspecified")),
                "encoding": None,
            },
            "design": {
                "kind": "legacy-reported-endpoint",
                "rate_selection": "not-recorded",
                "factory_selection": "not-recorded",
                "bindings": {
                    "allocation_plan_sha256": row.get("allocation_sha256"),
                    "candidate_inventories": [],
                    "score_calibration_sha256": None,
                    "budget_closure_sha256": None,
                    "untouched_endpoint_sha256": None,
                    "replay_receipt_sha256": verification_sha256,
                },
            },
            "parent": {"repository": repository, "revision": revision},
            "panel": {"id": panel_id, "sha256": None, "positions": int(positions) if positions else None},
            "backend": {
                "attention": attention,
                "kv_cache": "not-used-offline-full-forward",
            },
            "evaluator": {
                "metric": "mean-tokenwise-kl-bf16-teacher-to-reconstructed-student",
                "implementation": "legacy-saved-logit-evaluator",
            },
            "scope": {
                "weights": "routed-expert-weight-elements",
                "nonexpert": nonexpert,
                "comparison": str(row.get("comparison_status", "same-row-reported-endpoint")),
            },
            "rate": {
                "value": float(row["expert_bpw"]),
                "unit": "logical-bpw",
                "scope": "routed-expert-weight-elements",
            },
            "value": {"metric": "mean_kld", "value": float(row["mean_kld"])},
            "secondary_metrics": {
                "top1_agreement": row.get("top1_agreement"),
                "sample_sd": row.get("sample_sd"),
            },
            "evidence": {
                "path": _portable_path(source),
                "evidence_sha256": source_sha256,
                "report_sha256": report_sha256,
                "verification_sha256": verification_sha256,
            },
            "status": {"state": state, "detail": detail},
        }
    )


def _progressive_row(
    root: Path,
    *,
    parent_repository: str,
    parent_revision: str,
    baseline_inventory_path: Path | None = None,
    challenger_inventory_path: Path | None = None,
    evidence_uri: str | None = None,
) -> dict[str, Any]:
    report_path = root / "report.json"
    verification_path = root / "independent-verification.json"
    plan_path = root / "plan.json"
    allocation_path = root / "factory-allocation.json"
    for path in (report_path, verification_path, plan_path, allocation_path):
        if not path.is_file():
            raise ValueError(f"progressive union evidence is missing {path.name}")
    report = json.loads(report_path.read_text())
    verification = json.loads(verification_path.read_text())
    plan = json.loads(plan_path.read_text())
    allocation = json.loads(allocation_path.read_text())
    _verify_seal(report, "report_sha256", "progressive union report")
    _verify_seal(verification, "verification_sha256", "progressive union verification")
    _verify_seal(allocation, "allocation_sha256", "progressive union allocation")
    if verification.get("report_sha256") != report["report_sha256"]:
        raise ValueError("progressive union verification belongs to another report")
    if report.get("factory_allocation_sha256") != allocation["allocation_sha256"]:
        raise ValueError("progressive union report belongs to another factory allocation")
    baseline_inventory_path = (
        baseline_inventory_path or Path(str(plan.get("baseline_inventory", "")))
    )
    challenger_inventory_path = (
        challenger_inventory_path or Path(str(plan.get("challenger_inventory", "")))
    )
    inventory_bindings = []
    for role, path, allocation_key in (
        ("baseline", baseline_inventory_path, "baseline_candidate_inventory_sha256"),
        ("challenger", challenger_inventory_path, "challenger_candidate_inventory_sha256"),
    ):
        if not path.is_file():
            raise ValueError(f"progressive union {role} inventory is unavailable: {path}")
        inventory = json.loads(path.read_text())
        _verify_seal(inventory, "inventory_sha256", f"progressive union {role} inventory")
        if allocation.get(allocation_key) != inventory["inventory_sha256"]:
            raise ValueError(f"progressive union allocation belongs to another {role} inventory")
        inventory_bindings.append(
            {
                "role": role,
                "repository": str(inventory["repo_id"]),
                "revision": str(inventory["revision"]),
                "inventory_sha256": inventory["inventory_sha256"],
                "file_sha256": sha256_file(path),
            }
        )
    choices = allocation.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("progressive union allocation lacks exact matrix choices")
    observed_k3 = sum(int(choice["bits"]) == 3 for choice in choices)
    observed_k4 = sum(int(choice["bits"]) == 4 for choice in choices)
    if observed_k3 + observed_k4 != len(choices):
        raise ValueError("progressive union allocation contains a non-K3/K4 choice")
    if (
        observed_k3 != int(allocation["k3_count"])
        or observed_k4 != int(allocation["k4_count"])
        or float(allocation["average_weight_bits"])
        != (3.0 * observed_k3 + 4.0 * observed_k4) / len(choices)
    ):
        raise ValueError("progressive union allocation rate budget does not close")
    budget_closure = {
        "choice_count": len(choices),
        "k3_count": observed_k3,
        "k4_count": observed_k4,
        "average_weight_bits": float(allocation["average_weight_bits"]),
    }
    endpoint = report["untouched_validation"]
    positions = sum(
        int(record.get("positions", 2048)) for record in endpoint["union_records"]
    )
    return seal_row(
        {
            "schema": ROW_SCHEMA,
            "id": "base-20k-progressive-native-factory-union",
            "method_profile": {
                "allocation": "full-aumann-shapley-fisher-frozen-exact-3p5",
                "candidate_factory": "native-source-state-mcg-plus-progressive-state-mcg-whole-layer-union",
                "calibration": "progressive-quantized-predecessor-state",
                "codec": "exl3-mcg-native-and-progressive-state-candidate-families",
                "encoding": "exact-candidate-bytes-selected-on-row-0",
            },
            "design": {
                "kind": "frozen-rate-factory-diagnostic",
                "rate_selection": "frozen",
                "factory_selection": "selected",
                "bindings": {
                    "allocation_plan_sha256": sha256_file(plan_path),
                    "candidate_inventories": inventory_bindings,
                    "score_calibration_sha256": _hash_json(report["selection"]),
                    "budget_closure_sha256": _hash_json(budget_closure),
                    "untouched_endpoint_sha256": _hash_json(endpoint),
                    "replay_receipt_sha256": verification["verification_sha256"],
                },
            },
            "parent": {"repository": parent_repository, "revision": parent_revision},
            "panel": {
                "id": "wikitext-10x2048-untouched-rows-1-through-9",
                "sha256": report["panel_sha256"],
                "positions": positions,
            },
            "backend": {
                "attention": str(plan["attention_backend"]),
                "kv_cache": "not-used-offline-full-forward",
            },
            "evaluator": {
                "metric": "mean-tokenwise-kl-bf16-teacher-to-reconstructed-student",
                "implementation": str(verification["implementation"]),
            },
            "scope": {
                "weights": "routed-expert-weight-elements",
                "nonexpert": str(report.get("fixed_nonexpert_scope", "source-BF16-fixed-nonexpert")),
                "comparison": "factory-union-vs-native-baseline-at-identical-frozen-bit-rates",
            },
            "rate": {
                "value": float(allocation["average_weight_bits"]),
                "unit": "logical-bpw",
                "scope": "routed-expert-weight-elements",
            },
            "value": {
                "metric": "mean_kld",
                "value": float(endpoint["union_summary"]["mean"]),
            },
            "secondary_metrics": {
                "baseline_mean_kld": float(endpoint["baseline_summary"]["mean"]),
                "absolute_reduction": float(endpoint["absolute_reduction"]),
                "relative_reduction": float(endpoint["relative_reduction"]),
                "rows_union_better": int(endpoint["rows_union_better"]),
                "row_count": int(endpoint["row_count"]),
            },
            "evidence": {
                "path": evidence_uri or report_path.as_posix(),
                "evidence_sha256": sha256_file(report_path),
                "report_sha256": report["report_sha256"],
                "verification_sha256": verification["verification_sha256"],
            },
            "status": {
                "state": "independently-verified",
                "detail": "Factory selection used row 0; the endpoint uses untouched rows 1 through 9 and was replayed independently in float64.",
            },
        }
    )


def build(
    legacy_path: Path,
    *,
    additional_row_paths: tuple[Path, ...] = (),
    include_canonical_supplemental: bool = True,
    progressive_union_root: Path | None = None,
    progressive_parent_repository: str | None = None,
    progressive_parent_revision: str | None = None,
    progressive_baseline_inventory: Path | None = None,
    progressive_challenger_inventory: Path | None = None,
    progressive_evidence_uri: str | None = None,
) -> dict[str, Any]:
    legacy = json.loads(legacy_path.read_text())
    if legacy.get("schema") != "shapleymcg.qwen-complete-results-ledger.v1":
        raise ValueError("unsupported legacy results ledger")
    source_sha256 = sha256_file(legacy_path)
    rows = [_legacy_row(row, legacy_path, source_sha256) for row in legacy["results"]]
    supplemental_paths: list[Path] = []
    if include_canonical_supplemental:
        supplemental_paths.append(CANONICAL_SUPPLEMENTAL_ROWS)
    supplemental_paths.extend(additional_row_paths)
    unique_supplemental_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in supplemental_paths:
        resolved = path.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_supplemental_paths.append(path)
    for path in unique_supplemental_paths:
        document = json.loads(path.read_text())
        supplied = document.get("rows") if isinstance(document, dict) else document
        if not isinstance(supplied, list):
            raise ValueError(f"additional-row document must be a list or contain rows: {path}")
        for value in supplied:
            if not isinstance(value, dict):
                raise ValueError(f"additional result row is not an object: {path}")
            if "row_sha256" in value:
                # validate_ledger below will independently check its row seal.
                rows.append(dict(value))
            else:
                rows.append(seal_row(value))
    if progressive_union_root is not None:
        if not progressive_parent_repository or not progressive_parent_revision:
            raise ValueError("progressive union requires explicit parent repository and revision")
        parsed_evidence_uri = urlsplit(progressive_evidence_uri or "")
        if (
            not progressive_evidence_uri
            or parsed_evidence_uri.scheme != "https"
            or not parsed_evidence_uri.hostname
            or re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", progressive_evidence_uri)
            is None
        ):
            raise ValueError(
                "progressive union regeneration requires a remote HTTPS evidence URI "
                "containing an immutable 40-hex revision; "
                "local producer paths are not publishable provenance"
            )
        progressive_row = _progressive_row(
            progressive_union_root.resolve(),
            parent_repository=progressive_parent_repository,
            parent_revision=progressive_parent_revision,
            baseline_inventory_path=progressive_baseline_inventory,
            challenger_inventory_path=progressive_challenger_inventory,
            evidence_uri=progressive_evidence_uri,
        )
        # A freshly verified producer report supersedes the portable canonical
        # row for this one result. Other duplicate identifiers still fail in
        # validate_ledger rather than being silently overwritten.
        rows = [row for row in rows if row["id"] != progressive_row["id"]]
        rows.append(progressive_row)

    def source_record(role: str, path: Path) -> dict[str, str]:
        return {"role": role, "path": _portable_path(path), "sha256": sha256_file(path)}

    return seal_ledger(
        rows,
        generated_from={
            "sources": [
                source_record("legacy-results-ledger", legacy_path),
                *(
                    source_record("supplemental-result-rows", path)
                    for path in unique_supplemental_paths
                ),
            ],
            "migration": "lossless-claim-migration-with-null-for-unavailable-evidence",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-ledger", type=Path, default=Path("results/qwen-complete-results-ledger.json"))
    parser.add_argument("--progressive-union-root", type=Path)
    parser.add_argument(
        "--progressive-evidence-uri",
        help="Immutable URI for the progressive report; required with --progressive-union-root.",
    )
    parser.add_argument(
        "--additional-rows",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional JSON result rows; may be repeated. Canonical supplemental rows "
            "remain included unless --no-canonical-supplemental is set."
        ),
    )
    parser.add_argument("--no-canonical-supplemental", action="store_true")
    parser.add_argument("--progressive-parent-repository")
    parser.add_argument("--progressive-parent-revision")
    parser.add_argument("--progressive-baseline-inventory", type=Path)
    parser.add_argument("--progressive-challenger-inventory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/RESULTS_LEDGER.json"))
    parser.add_argument("--render", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate:
        ledger = json.loads(args.validate.read_text())
        validate_ledger(ledger)
        if args.render:
            args.render.write_text(render_markdown(ledger))
        print(json.dumps({"ok": True, "rows": len(ledger["rows"]), "ledger_sha256": ledger["ledger_sha256"]}, sort_keys=True))
        return 0
    ledger = build(
        args.legacy_ledger,
        additional_row_paths=tuple(args.additional_rows),
        include_canonical_supplemental=not args.no_canonical_supplemental,
        progressive_union_root=args.progressive_union_root,
        progressive_parent_repository=args.progressive_parent_repository,
        progressive_parent_revision=args.progressive_parent_revision,
        progressive_baseline_inventory=args.progressive_baseline_inventory,
        progressive_challenger_inventory=args.progressive_challenger_inventory,
        progressive_evidence_uri=args.progressive_evidence_uri,
    )
    if args.execute:
        write_json(args.output, ledger)
        if args.render:
            args.render.write_text(render_markdown(ledger))
    print(json.dumps({"dry_run": not args.execute, "rows": len(ledger["rows"]), "ledger_sha256": ledger["ledger_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
