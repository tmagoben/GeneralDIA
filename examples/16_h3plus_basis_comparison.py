"""Emit strict JSON for the identical-geometry STO-3G / 6-31G H3+ CI gate."""

from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

from generaldia.electronic_structure.h3plus_comparison import certify_h3plus_basis_comparison


def main() -> None:
    revision = os.environ.get("GITHUB_SHA")
    if revision is None:
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unrecorded-local"
    report = certify_h3plus_basis_comparison(revision=revision)
    root = Path(__file__).resolve().parents[1]
    # Content hashes identify the numerical implementation even in a local
    # checkout with uncommitted changes; a revision alone would be incomplete.
    report["source_files_sha256"] = {
        name: sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "generaldia/electronic_structure/h3plus_ci.py",
            "generaldia/electronic_structure/h3plus_comparison.py",
            "generaldia/electronic_structure/fci_dipoles.py",
            "generaldia/loop_convergence.py",
            "generaldia/topology.py",
            "examples/16_h3plus_basis_comparison.py",
        )
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
