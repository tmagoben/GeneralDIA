"""Independent finite-basis H3+ CI evidence on identical geometry paths.

This compares physical FCI references, not learned Hamiltonians. No states or
overlaps are transported between orbital bases. Energies are in hartree and
geometry amplitudes in angstrom.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .h3plus_ci import certify_h3plus_molecular_ci

COMPARISON_KIND = "molecular-h3plus-basis-comparison"
THRESHOLDS = {
    "central_gap_max_hartree": 1e-6,
    "branching_ratio_min_exclusive": 1.3,
    "branching_ratio_max_exclusive": 2.7,
    "root_isolation_min_exclusive_hartree": 1e-3,
    "excited_gap_min_exclusive_hartree": 1e-4,
    "phase_tolerance_radians": 0.06,
    "minimum_link_singular_value": 0.65,
    "gauge_error_max_exclusive_radians": 1e-9,
    "energy_only_zero_max_exclusive_radians": 1e-12,
    "energy_only_pi_distance_min_exclusive_radians": 3.0,
}


def certify_h3plus_basis_comparison(
    *,
    bond_angstrom: float = 0.875,
    radius_angstrom: float = 0.06,
    refinement_points: tuple[int, ...] = (8, 16, 32),
    revision: str = "unrecorded-local",
) -> dict[str, Any]:
    """Run separate STO-3G and 6-31G four-root FCI certifications.

    The geometry and numerical thresholds are identical in both calculations.
    Each basis has its own SCF, FCI roots, cross-AO overlaps, numerical results
    and pass status. Solver failures propagate instead of substituting the
    other basis's data. A false scientific check produces a failing report.
    Invalid geometry/mesh inputs raise ValueError before electronic evaluation.
    Requires the optional PySCF backend.
    """
    import pyscf

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")
    settings = {
        "bond_angstrom": bond_angstrom,
        "radius_angstrom": radius_angstrom,
        "refinement_points": list(refinement_points),
        "charge": 1,
        "spin": 0,
        "n_states": 4,
        "conv_tol": 1e-11,
        "max_orbitals": 8,
        "thresholds": dict(THRESHOLDS),
    }
    results = {}
    for basis in ("sto-3g", "6-31g"):
        evidence = certify_h3plus_molecular_ci(
            basis=basis,
            bond_angstrom=bond_angstrom,
            radius_angstrom=radius_angstrom,
            refinement_points=refinement_points,
        )
        results[basis] = {
            "settings": {**settings, "basis": basis},
            "result": asdict(evidence),
            "pass": evidence.passed,
        }
    checks = {
        "identical_geometry_paths": (
            results["sto-3g"]["result"]["geometry_sha256"]
            == results["6-31g"]["result"]["geometry_sha256"]
        ),
        "all_basis_certificates_pass": all(item["pass"] for item in results.values()),
    }
    return {
        "kind": COMPARISON_KIND,
        "schema_version": 1,
        "stage": "v3.4.0.dev5",
        "revision": revision,
        "backend": f"PySCF {pyscf.__version__}",
        "method": "independent charged singlet full CI; four lowest singlet roots per basis",
        "geometry_units": "angstrom",
        "energies_and_gaps_units": "hartree",
        "branching_gap_order": ["x_half", "x_full", "y_half", "y_full"],
        "phase_convention": "minus argument of the cyclic physical overlap product",
        "settings": settings,
        "results": results,
        "checks": checks,
        "pass": all(checks.values()),
        "limits": (
            "two finite-basis physical references; no complete-basis convergence, "
            "learned-Hamiltonian certification, derivative-coupling accuracy, SOC "
            "or nuclear trajectories"
        ),
    }
