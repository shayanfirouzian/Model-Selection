#!/usr/bin/env python3
"""Write MANIFEST.sha256 for the current repository tree.

The manifest omits itself, Git internals, interpreter caches, virtual environments,
OS metadata, runtime logs, and compressed bundles. Paths are relative to the repository root, making
checksums portable across machines.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.sha256"
EXCLUDED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules", "logs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_NAMES = {"MANIFEST.sha256", ".DS_Store", "Thumbs.db"}

def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative.as_posix()}")
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.name}: {len(rows)} files hashed")

if __name__ == "__main__":
    main()
