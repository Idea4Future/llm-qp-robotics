#!/usr/bin/env python3
"""Validate the downloaded robot source/assets against the pinned manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = json.loads((root / "configs/isaac/robot_assets.json").read_text())
base = root / "third_party/rby1_isaac"
failures = []
for name, expected in spec["files"].items():
    path = base / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit("Missing or modified upstream files: " + ", ".join(failures))
print(f"Verified {len(spec['files'])} robot source/asset files")
