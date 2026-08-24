#!/usr/bin/env python3
"""Verify the exact upstream BTX writer/reader closure; never fetch or checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


COMMIT = "36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8"
CLOSURE = {
    "docs/btx-checkpoint-format.md": "62ed1996ba54d4f2ab63ccb14ba9dc7e22e15d4443bec330226696600368aebd",
    "b12x/moe/_shared/btx_schema.py": "282190602b38c70a2085b40a9a2ef895ba6925c38725671cc99b0204d8d1bbdd",
    "b12x/moe/_shared/kernels/w4a16/btx.py": "d6b241b59b29265235914e1be955607b3e9b0cd83a40f72a40350455adf927bd",
    "b12x/moe/_shared/kernels/w4a16/btx_synth.py": "f94b5e50c02551a041660d194266849f656e8a08da179d1e36fa64119d04f54e",
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD", help="must resolve exactly to the pinned commit")
    parser.add_argument("--require-clean", action="store_true", default=False)
    args = parser.parse_args()
    root = args.source.resolve()
    observed_commit = git(root, "rev-parse", args.ref)
    if observed_commit != COMMIT:
        raise SystemExit(f"B12X ref mismatch: {observed_commit} != {COMMIT}")
    if args.require_clean and git(root, "status", "--porcelain"):
        raise SystemExit("B12X checkout is dirty")
    observed = {}
    for relative, expected in CLOSURE.items():
        raw = subprocess.run(
            ["git", "-C", str(root), "show", f"{args.ref}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise SystemExit(f"B12X closure drift: {relative}: {digest} != {expected}")
        observed[relative] = digest
    print(json.dumps({"ok": True, "commit": observed_commit, "closure": observed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
