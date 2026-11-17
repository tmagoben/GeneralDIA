"""Emit physical H3+ optical-strength evidence; exit nonzero if any fixed gate fails."""

import json
from importlib.metadata import version

from generaldia.electronic_structure.h3plus_band import GATES, certify_transition_band


def main():
    results = [certify_transition_band(basis) for basis in ("sto-3g", "6-31g")]
    same_geometries = len({item["geometry_sha256"] for item in results}) == 1
    passed = same_geometries and all(item["pass"] for item in results)
    report = {
        "kind": "h3plus-ground-to-excited-doublet-length-gauge",
        "schema_version": 1,
        "pyscf": version("pyscf"),
        "charge": 1,
        "spin": 0,
        "gates": GATES,
        "observable": "unweighted Cartesian line tensor and energy-weighted oscillator sum (a.u.)",
        "comparison": "PySCF RHF/singlet FCI RDMs vs dense symmetric products in Lowdin AOs",
        "scope": "four static geometries; no learned dipole, statewise gauge or nuclear dynamics",
        "same_geometries": same_geometries,
        "results": results,
        "pass": passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
