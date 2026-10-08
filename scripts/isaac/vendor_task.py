"""Load the pinned vendor physics task without its unused SDK wire codec.

The public task eagerly imports rby1_udp_bridge, whose udp_protocol binary is
distributed in the vendor Docker image, not the public source checkout.
For our local control demo udp_bridge is always None. Remove only that import
in memory; do not alter vendor files, PD equations, articulation or physics.
This adapter does not implement or claim support for the vendor SDK protocol.
"""
from __future__ import annotations
import ast
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any


def load_standalone_task(root: Path):
    source_path = root / "third_party/rby1_isaac/src/rby1_task.py"
    source = source_path.read_bytes()
    manifest = json.loads((root / "configs/isaac/robot_assets.json").read_text())
    expected = manifest["files"]["src/rby1_task.py"]
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError("Vendor task differs from the reviewed, pinned source")
    tree = ast.parse(source, filename=str(source_path))
    excluded = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "rby1_udp_bridge"]
    if len(excluded) != 1 or [a.name for a in excluded[0].names] != ["RBY1UdpBridge"]:
        raise RuntimeError("Unexpected vendor SDK import; adapter needs review")
    tree.body.remove(excluded[0])
    module = types.ModuleType("opti_vendor_rby1_task")
    module.__file__ = str(source_path)
    module.RBY1UdpBridge = Any  # used solely by postponed type annotations
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    return module.RBY1Task
