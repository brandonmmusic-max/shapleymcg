#!/usr/bin/env python3
"""Prepare the pinned ExLlamaV3 v0.0.43 checkout; dry-run by default."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
REPOSITORY = "https://github.com/turboderp-org/exllamav3.git"


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
        [
            sys.executable,
            str(Path(__file__).with_name("verify_exllamav3_checkout.py")),
            "--source",
            str(destination),
            "--require-clean",
        ],
    ]
    print(
        json.dumps(
            {
                "dry_run": not args.execute,
                "repository": REPOSITORY,
                "tag": "v0.0.43",
                "commit": COMMIT,
                "commands": commands,
            },
            sort_keys=True,
        )
    )
    if not args.execute:
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
