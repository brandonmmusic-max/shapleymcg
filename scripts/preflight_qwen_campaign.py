#!/usr/bin/env python3
"""Run only the read-only production preflight; never dispatch a campaign stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--execute-preflight", action="store_true", help="perform read-only host/GPU/import checks")
    args = parser.parse_args()
    plan_path = args.campaign_dir.resolve() / "plan.json"
    if not plan_path.is_file():
        raise SystemExit("sealed campaign plan.json is required")
    print(json.dumps({"dry_run": not args.execute_preflight, "plan": str(plan_path)}, sort_keys=True))
    if not args.execute_preflight:
        return 0
    from quant_pipeline.campaign.qwen_adapter import QwenCampaignAdapter

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = dict(QwenCampaignAdapter(production=True).preflight(plan))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("ok") is not True:
        raise SystemExit("production preflight failed; no stage was dispatched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
