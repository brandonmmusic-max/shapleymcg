#!/usr/bin/env python3
"""Verify the exact clean ExLlamaV3 source used for the SM100 extension."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    # The pinned v0.0.43 commit predates the project's pyproject.toml
    # migration and is built by setup.py.  Requiring pyproject.toml here made
    # the exact checkout prepared by the companion script unverifiable.
    if not (source / ".git").exists() or not (source / "setup.py").is_file():
        raise SystemExit("ExLlamaV3 v0.0.43 source must be a Git checkout with setup.py")
    head = git(source, "rev-parse", "HEAD")
    if head != COMMIT:
        raise SystemExit(f"ExLlamaV3 checkout drift: expected {COMMIT}, observed {head}")
    status = git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if args.require_clean and status:
        raise SystemExit("ExLlamaV3 checkout is not clean")
    print(json.dumps({"ok": True, "source": str(source), "commit": head, "clean": not bool(status)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
