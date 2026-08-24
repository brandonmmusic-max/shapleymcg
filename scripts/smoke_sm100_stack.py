#!/usr/bin/env python3
"""Hash-check and optionally smoke the SM100 EXL3 encoder plus pinned BTX closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


NUMERIC_CORE_SHA256 = "e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--numeric-core", type=Path, required=True)
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--execute", action="store_true", help="import the extension and run on-device smoke gates")
    args = parser.parse_args()
    extension = args.extension.resolve()
    core = args.numeric_core.resolve()
    b12x = args.b12x_source.resolve()
    if not extension.is_file() or extension.suffix != ".so":
        raise SystemExit("SM100 extension must be an existing .so")
    if not core.is_file() or digest(core) != NUMERIC_CORE_SHA256:
        raise SystemExit("numeric-core identity mismatch")
    verifier = Path(__file__).with_name("verify_b12x_checkout.py")
    subprocess.run([sys.executable, str(verifier), "--source", str(b12x)], check=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or any(not value.isdecimal() for value in devices):
        raise SystemExit("devices must be a comma-separated integer list")
    plan = {
        "dry_run": not args.execute,
        "extension_sha256": digest(extension),
        "numeric_core_sha256": digest(core),
        "b12x_commit": "36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8",
        "devices": devices,
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return 0
    for device in devices:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["TORCH_CUDA_ARCH_LIST"] = "10.0"
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util,torch;"
                    "assert torch.cuda.get_device_capability(0)==(10,0);"
                    f"s=importlib.util.spec_from_file_location('exllamav3_ext',{str(extension)!r});"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print('extension-smoke-ok')"
                ),
            ],
            env=environment,
            check=True,
        )
        subprocess.run([sys.executable, str(core), "--smoke"], env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
