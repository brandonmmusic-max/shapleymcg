#!/usr/bin/env python3
"""Read-only verification of the pinned B200 numeric environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    unresolved = [key for key in lock.get("required_placeholders", []) if "__REQUIRED_" in json.dumps(lock)]
    if unresolved and not args.allow_placeholders:
        raise SystemExit("environment lock still contains required placeholders")
    observed = {
        "platform": "linux-x86_64" if platform.system() == "Linux" and platform.machine() == "x86_64" else f"{platform.system()}-{platform.machine()}",
        "python": platform.python_version(),
        "packages": {},
        "required_environment": {key: os.environ.get(key) for key in lock["required_environment"]},
    }
    failures = []
    if observed["platform"] != lock["platform"] or observed["python"] != lock["python"]:
        failures.append("platform-or-python")
    for package, expected in lock["packages"].items():
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            value = None
        observed["packages"][package] = value
        if value != expected:
            failures.append(f"package:{package}")
    if observed["required_environment"] != lock["required_environment"]:
        failures.append("numeric-environment")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()[0].strip()
    except Exception:
        driver = None
    observed["nvidia_driver"] = driver
    if not unresolved and driver != lock["nvidia_driver"]:
        failures.append("nvidia-driver")
    print(json.dumps({"ok": not failures, "failures": failures, "observed": observed}, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
