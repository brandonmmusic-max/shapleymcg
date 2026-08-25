#!/usr/bin/env python3
"""Create a role-safe fixed-window view of the sealed reap/recall corpus.

The source corpus contains many records shorter than the experiment's 2,048
token window.  Packing is performed before target tokenization, within each
original axis and deterministic bucket.  Every source row is assigned to one
and only one aggregate document, so the downstream document splitter cannot
leak source text between fit, conditional-fit, selection, confirmation, and
final roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "quant-pipeline.reap-recall-packed-corpus.v2"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: Path, output: Path, receipt: Path, expected_sha256: str, seed: int) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise FileExistsError("output and receipt destinations must not already exist")
    observed_source_sha256 = sha256_file(source)
    if observed_source_sha256 != expected_sha256:
        raise ValueError(
            f"source corpus SHA256 mismatch: expected {expected_sha256}, observed {observed_source_sha256}"
        )

    buckets: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    source_rows = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            axis = str(row["axis"])
            text = str(row["text"])
            key = hashlib.sha256(f"{seed}\0{axis}\0{line_number}".encode()).digest()
            buckets[(axis, int.from_bytes(key[:8], "big") % 5)].append((line_number, text))
            source_rows += 1

    domains = sorted({axis for axis, _ in buckets})
    if len(domains) != 4 or any((axis, bucket) not in buckets for axis in domains for bucket in range(5)):
        raise ValueError("expected five populated deterministic buckets in each of four corpus axes")

    output.parent.mkdir(parents=True, exist_ok=True)
    aggregates = []
    with output.open("xb") as stream:
        for axis in domains:
            for bucket in range(5):
                rows = buckets[(axis, bucket)]
                source_line_numbers = [line_number for line_number, _ in rows]
                text = "\n\n".join(value for _, value in rows)
                aggregate = {
                    "id": f"reap-recall-packed-{axis}-{bucket}",
                    "domain": axis,
                    "text": text,
                }
                stream.write(canonical_json(aggregate) + b"\n")
                aggregates.append(
                    {
                        "id": aggregate["id"],
                        "domain": axis,
                        "bucket": bucket,
                        "source_rows": len(rows),
                        "source_line_numbers_sha256": hashlib.sha256(
                            canonical_json(source_line_numbers)
                        ).hexdigest(),
                        "text_characters": len(text),
                        "text_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                )

    document = {
        "schema": SCHEMA,
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": observed_source_sha256,
            "records": source_rows,
        },
        "method": {
            "name": "axis-preserving-five-role-hash-buckets-v2",
            "seed": seed,
            "buckets_per_axis": 5,
            "join_separator": "\\n\\n",
            "assignment_key": "sha256(seed\\0axis\\0one_based_source_line) mod 5",
        },
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "documents": len(aggregates),
        },
        "aggregates": aggregates,
    }
    document["receipt_sha256"] = hashlib.sha256(canonical_json(document)).hexdigest()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(canonical_json(document) + b"\n")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    document = prepare(
        args.source.resolve(),
        args.output.resolve(),
        args.receipt.resolve(),
        args.expected_sha256,
        args.seed,
    )
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
