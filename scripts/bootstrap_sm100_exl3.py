#!/usr/bin/env python3
"""Build the corrected EXL3 encoding extension for SM100; dry-run by default."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="prepared exllamav3 v0.0.43 source tree")
    parser.add_argument("--max-jobs", type=int, default=32)
    parser.add_argument("--execute", action="store_true", help="actually compile/install into the active environment")
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "pyproject.toml").is_file():
        raise SystemExit("EXL3 source must be an existing prepared source tree with pyproject.toml")
    if args.max_jobs < 1:
        raise SystemExit("max-jobs must be positive")
    command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."]
    plan = {
        "dry_run": not args.execute,
        "source": str(source),
        "required_source_commit": "c5d9c657966ffeeaa9353f0cc899f18629da4a13",
        "environment": {"TORCH_CUDA_ARCH_LIST": "10.0", "MAX_JOBS": str(args.max_jobs)},
        "command": command,
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("verify_exllamav3_checkout.py")),
            "--source",
            str(source),
            "--require-clean",
        ],
        check=True,
    )
    import torch

    if torch.__version__ != "2.12.1+cu132" or torch.version.cuda != "13.2":
        raise SystemExit("SM100 build requires the sealed torch 2.12.1+cu132 / CUDA 13.2 environment")
    if not torch.cuda.is_available() or any(torch.cuda.get_device_capability(i) != (10, 0) for i in range(torch.cuda.device_count())):
        raise SystemExit("every visible build-validation GPU must be SM100")
    environment = os.environ.copy()
    environment.update(plan["environment"])
    subprocess.run(command, cwd=source, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
