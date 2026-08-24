#!/usr/bin/env python3
"""Prepare a new detached pinned B12X checkout; dry-run unless --execute."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMMIT = "36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8"
REPOSITORY = "https://github.com/local-inference-lab/b12x.git"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="perform network clone and detached checkout")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("destination must not exist; refusing to alter an existing checkout")
    commands = [
        ["git", "clone", "--filter=blob:none", REPOSITORY, str(destination)],
        ["git", "-C", str(destination), "checkout", "--detach", COMMIT],
        [sys.executable, str(Path(__file__).with_name("verify_b12x_checkout.py")), "--source", str(destination), "--require-clean"],
    ]
    print(json.dumps({"dry_run": not args.execute, "commands": commands}, sort_keys=True))
    if not args.execute:
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
