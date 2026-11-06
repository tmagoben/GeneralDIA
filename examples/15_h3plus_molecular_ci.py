"""Reproducible excited-singlet H3+ *molecular* conical-intersection certificate.

Requires the optional PySCF backend. This is an FCI/STO-3G finite-basis
calculation and not a nonadiabatic dynamics or chemical-accuracy claim.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import pyscf

from generaldia.electronic_structure.h3plus_ci import certify_h3plus_molecular_ci


def main() -> None:
    record = certify_h3plus_molecular_ci(
        bond_angstrom=0.875, radius_angstrom=0.06, refinement_points=(8, 16, 32)
    )
    output = {
        "kind": "molecular-h3plus-excited-singlet-ci-certification",
        "stage": "v3.4.0.dev3",
        "revision": os.environ.get("GITHUB_SHA", "unrecorded-local"),
        "backend": f"PySCF {pyscf.__version__}",
        "method": "charged singlet full CI, STO-3G; four lowest singlet roots",
        "reference_papers": [
            "https://doi.org/10.1039/D4CP00391H",
            "https://doi.org/10.1063/1.2193516",
        ],
        "phase_convention": "minus argument of the cyclic physical overlap product",
        "geometry_units": "angstrom",
        "energies_and_gaps_units": "hartree",
        "result": asdict(record),
        "pass": record.passed,
        "limits": (
            "finite-basis restricted H3+ CI evidence only; no global state "
            "completeness, derivative-coupling accuracy, SOC or nuclear trajectories"
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
