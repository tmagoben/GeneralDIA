"""Physical H2 FCI cross-geometry Berry-loop convergence; optional PySCF gate.

This is a *trivial* one-dimensional bond-stretch loop with expected Berry zero,
not a molecular conical-intersection model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend
from generaldia.loop_convergence import certify_abelian_loop_convergence


def _closed_h2_scan(count):
    angle = 2 * math.pi * np.arange(count) / count
    bond = 0.75 + 0.035 * np.sin(angle) + 0.015 * np.cos(angle)
    geometries = np.zeros((count, 2, 3), dtype=np.float64)
    geometries[:, 1, 2] = bond
    return geometries


def main():
    import pyscf

    levels = (4, 8, 12)
    backend = FCIDipoleBackend(n_states=1, basis="sto-3g", conv_tol=1e-11)
    loops = []
    provenance = {}
    for n in levels:
        geometry = _closed_h2_scan(n)
        result = backend.calculate_loop([1, 1], geometry, path_id=f"h2-loop-{n}")
        loops.append((n, result.cyclic_overlaps))
        provenance[str(n)] = hashlib.sha256(geometry.tobytes()).hexdigest()

    phase_tolerance = 1e-5
    certified = certify_abelian_loop_convergence(
        loops,
        phase_tolerance=phase_tolerance,
        reference_phase=0.0,
    )
    record = {
        "stage": "v3.4-slice-4-physical-closure",
        "revision": os.getenv("GITHUB_SHA", "unrecorded-local"),
        "pyscf_version": pyscf.__version__,
        "molecule": "neutral singlet H2, all-electron FCI/STO-3G",
        "geometry": "closed smooth bond-length scan; one dimensional, no CI",
        "geometry_sha256_by_points": provenance,
        "overlap_method": "physical cross-AO determinant overlaps, including last-to-first",
        "point_counts": certified.point_counts,
        "berry_phases_rad": certified.phases,
        "maximum_coarse_to_finest_error_rad": certified.coarse_to_finest_phase_error,
        "reference_phase_rad": certified.reference_phase,
        "reference_phase_error_rad": certified.reference_phase_error,
        "minimum_link_singular_value": certified.minimum_link_singular_value,
        "phase_tolerance_rad": phase_tolerance,
        "pass": certified.passed,
        "limits": "no molecular CI, physical NAC, SOC, or state-completeness certification",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
