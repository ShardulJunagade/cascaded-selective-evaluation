#!/usr/bin/env python3
"""Remove only an incomplete final JSONL record left by scheduler termination."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
args = parser.parse_args()
if not args.path.exists():
    raise SystemExit(0)

with args.path.open("rb") as handle:
    lines = handle.readlines()
for index, line in enumerate(lines):
    try:
        json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if index != len(lines) - 1:
            raise SystemExit(f"Invalid non-final record in {args.path}: line {index + 1}: {exc}")
        with args.path.open("wb") as handle:
            handle.writelines(lines[:index])
        print(f"Removed incomplete final record from {args.path}")
