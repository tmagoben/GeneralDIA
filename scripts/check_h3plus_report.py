"""Check H3+ evidence without importing GeneralDIA or scientific dependencies.

This independently checks the reported numbers, schema and fixed acceptance
thresholds. It cannot authenticate a report or replace recomputing physical FCI
wavefunctions/overlaps. Run with ``python -I -S`` to exclude site packages.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from itertools import pairwise
from pathlib import Path

SINGLE_KIND = "molecular-h3plus-excited-singlet-ci-certification"
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


class InvalidReport(ValueError):
    """Malformed or numerically failing certification evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidReport(message)


def mapping(value: object, label: str) -> dict:
    require(isinstance(value, dict), f"{label}: expected an object")
    return value


def number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label}: expected a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise InvalidReport(f"{label}: number is not finite") from error
    require(math.isfinite(converted), f"{label}: number is not finite")
    return converted


def numbers(value: object, size: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == size, f"{label}: expected {size} numbers")
    return [number(item, f"{label}[{i}]") for i, item in enumerate(value)]


def close(value: object, expected: float, label: str) -> None:
    actual = number(value, label)
    require(
        math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
        f"{label}: inconsistent with reported measurements",
    )


def distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def check_flags(value: object, expected: dict[str, bool], label: str) -> None:
    flags = mapping(value, label)
    require(set(flags) == set(expected), f"{label}: missing or unknown required checks")
    for key, computed in expected.items():
        require(type(flags[key]) is bool, f"{label}.{key}: expected a JSON boolean")
        require(flags[key] == computed, f"{label}.{key}: inconsistent with numerical evidence")
        require(computed, f"{label}.{key}: failed required check")


def check_loop(value: object, counts: list[int], label: str) -> tuple[float, float]:
    loop = mapping(value, label)
    require(loop.get("point_counts") == counts, f"{label}.point_counts: mesh mismatch")
    phases = numbers(loop.get("phases"), len(counts), f"{label}.phases")
    errors = [distance(a, b) for a, b in pairwise(phases)]
    supplied = numbers(
        loop.get("successive_phase_errors"), len(errors), f"{label}.successive_phase_errors"
    )
    for actual, expected in zip(supplied, errors, strict=True):
        close(actual, expected, f"{label}.successive_phase_errors")
    coarse = max(distance(phase, phases[-1]) for phase in phases[:-1])
    reference_error = distance(phases[-1], math.pi)
    close(loop.get("coarse_to_finest_phase_error"), coarse, f"{label}.coarse_to_finest_phase_error")
    close(loop.get("reference_phase_error"), reference_error, f"{label}.reference_phase_error")
    require(loop.get("phase_tolerance") == 0.06, f"{label}.phase_tolerance: changed threshold")
    close(loop.get("reference_phase"), math.pi, f"{label}.reference_phase")
    strength = number(
        loop.get("minimum_link_singular_value"), f"{label}.minimum_link_singular_value"
    )
    require(0.65 <= strength <= 1 + 1e-8, f"{label}: unresolved or invalid physical overlap link")
    computed = max(errors) <= 0.06 and coarse <= 0.06 and reference_error <= 0.06
    require(type(loop.get("passed")) is bool, f"{label}.passed: expected a JSON boolean")
    require(loop["passed"] == computed, f"{label}.passed: inconsistent with phase evidence")
    require(computed, f"{label}: failed pi-phase or mesh-convergence check")
    return phases[-1], strength


def check_result(value: object, basis: str, *, branching_required: bool) -> dict:
    result = mapping(value, basis)
    require(result.get("basis") == basis, f"{basis}: basis label mismatch")
    require(
        type(result.get("charge")) is int and result["charge"] == 1, f"{basis}: charge must be +1"
    )
    for key in ("bond_angstrom", "radius_angstrom"):
        require(number(result.get(key), f"{basis}.{key}") > 0, f"{basis}.{key}: must be positive")
    counts = result.get("refinement_points")
    require(
        isinstance(counts, list)
        and len(counts) >= 3
        and all(type(n) is int and n >= 6 for n in counts)
        and counts == sorted(set(counts)),
        f"{basis}.refinement_points: require >=3 increasing integer meshes",
    )
    energies = numbers(result.get("center_energies_hartree"), 4, f"{basis}.center_energies_hartree")
    require(energies == sorted(energies), f"{basis}: electronic energies must be ordered")
    central = number(
        result.get("central_excited_gap_hartree"), f"{basis}.central_excited_gap_hartree"
    )
    close(central, energies[2] - energies[1], f"{basis}.central_excited_gap_hartree")
    ratios = numbers(result.get("branching_gap_ratios"), 2, f"{basis}.branching_gap_ratios")
    if branching_required or "branching_gaps_hartree" in result:
        gaps = numbers(result.get("branching_gaps_hartree"), 4, f"{basis}.branching_gaps_hartree")
        require(all(gap > 0 for gap in gaps), f"{basis}.branching_gaps_hartree: must be positive")
        close(ratios[0], gaps[1] / gaps[0], f"{basis}.branching_gap_ratios.x")
        close(ratios[1], gaps[3] / gaps[2], f"{basis}.branching_gap_ratios.y")
    lower, lower_strength = check_loop(
        result.get("lower_excited_loop"), counts, f"{basis}.lower_excited_loop"
    )
    _, upper_strength = check_loop(
        result.get("upper_excited_loop"), counts, f"{basis}.upper_excited_loop"
    )
    scalar_keys = (
        "min_ground_isolation_hartree",
        "min_upper_isolation_hartree",
        "min_excited_gap_on_loop_hartree",
        "nonenclosing_phase_radians",
        "reverse_orientation_phase_radians",
        "gauge_phase_error_radians",
        "isospectral_energy_only_phase_radians",
        "minimum_link_strength",
    )
    scalars = {key: number(result.get(key), f"{basis}.{key}") for key in scalar_keys}
    require(
        0.65 <= scalars["minimum_link_strength"] <= min(lower_strength, upper_strength) + 1e-12,
        f"{basis}.minimum_link_strength: unresolved or inconsistent overlap link",
    )
    expected = {
        "equilateral_excited_degeneracy": abs(central) <= 1e-6,
        "two_independent_linear_branching_directions": all(1.3 < ratio < 2.7 for ratio in ratios),
        "ground_and_higher_root_isolation": (
            scalars["min_ground_isolation_hartree"] > 1e-3
            and scalars["min_upper_isolation_hartree"] > 1e-3
        ),
        "positive_excited_gap_on_loops": scalars["min_excited_gap_on_loop_hartree"] > 1e-4,
        "lower_excited_pi_and_mesh": True,
        "upper_excited_pi_and_mesh": True,
        "physical_nonencircling_zero": distance(scalars["nonenclosing_phase_radians"], 0) < 0.06,
        "reversal_covariance": distance(scalars["reverse_orientation_phase_radians"], -lower)
        < 0.06,
        "local_complex_gauge_covariance": 0 <= scalars["gauge_phase_error_radians"] < 1e-9,
        "energy_only_control_rejected": (
            distance(scalars["isospectral_energy_only_phase_radians"], 0) < 1e-12
            and distance(scalars["isospectral_energy_only_phase_radians"], math.pi) > 3.0
        ),
    }
    check_flags(result.get("checks"), expected, f"{basis}.checks")
    digests = mapping(result.get("geometry_sha256"), f"{basis}.geometry_sha256")
    required_digests = {"symmetry_and_branching", "nonenclosing", "reversed"} | {
        f"encircling_{n}" for n in counts
    }
    require(set(digests) == required_digests, f"{basis}.geometry_sha256: missing or unknown paths")
    require(
        all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in digests.values()),
        f"{basis}.geometry_sha256: malformed SHA256 digest",
    )
    return result


def validate_report(value: object) -> None:
    """Raise InvalidReport for any malformed, inconsistent or failing report."""
    report = mapping(value, "report")
    for key in ("stage", "revision", "backend", "method"):
        require(
            isinstance(report.get(key), str) and bool(report[key].strip()),
            f"report.{key}: missing metadata",
        )
    require(report.get("geometry_units") == "angstrom", "geometry_units: expected angstrom")
    require(
        report.get("energies_and_gaps_units") == "hartree",
        "energies_and_gaps_units: expected hartree",
    )
    require(
        report.get("phase_convention") == "minus argument of the cyclic physical overlap product",
        "phase_convention: unsupported convention",
    )
    kind = report.get("kind")
    if kind == SINGLE_KIND:
        check_result(report.get("result"), "sto-3g", branching_required=False)
    elif kind == COMPARISON_KIND:
        require(
            type(report.get("schema_version")) is int and report["schema_version"] == 1,
            "schema_version: unsupported",
        )
        require(
            report.get("branching_gap_order") == ["x_half", "x_full", "y_half", "y_full"],
            "branching_gap_order: unsupported ordering",
        )
        settings = mapping(report.get("settings"), "settings")
        require(
            settings.get("thresholds") == THRESHOLDS,
            "settings.thresholds: changed or missing thresholds",
        )
        for key, expected in {"charge": 1, "spin": 0, "n_states": 4, "max_orbitals": 8}.items():
            require(
                type(settings.get(key)) is int and settings[key] == expected,
                f"settings.{key}: unsupported setting",
            )
        require(settings.get("conv_tol") == 1e-11, "settings.conv_tol: changed tolerance")
        results = mapping(report.get("results"), "results")
        require(set(results) == {"sto-3g", "6-31g"}, "results: require STO-3G and 6-31G")
        checked = {}
        for basis, item in results.items():
            item = mapping(item, basis)
            require(
                item.get("settings") == {**settings, "basis": basis}, f"{basis}.settings: mismatch"
            )
            checked[basis] = check_result(item.get("result"), basis, branching_required=True)
            for key in ("bond_angstrom", "radius_angstrom", "refinement_points"):
                require(
                    checked[basis][key] == settings.get(key),
                    f"{basis}.{key}: shared geometry mismatch",
                )
            require(item.get("pass") is True, f"{basis}.pass: expected true")
        check_flags(
            report.get("checks"),
            {
                "identical_geometry_paths": checked["sto-3g"]["geometry_sha256"]
                == checked["6-31g"]["geometry_sha256"],
                "all_basis_certificates_pass": True,
            },
            "comparison.checks",
        )
    else:
        raise InvalidReport("kind: unrecognized report kind")
    require(report.get("pass") is True, "report.pass: expected true")


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise InvalidReport(f"nonfinite JSON constant: {value}")


def _finite_tree(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif type(value) in (int, float):
        number(value, "JSON number")


def load_report(path: Path) -> object:
    """Parse strict JSON, rejecting duplicate keys and all nonfinite numbers."""
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=_constant
    )
    _finite_tree(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.reports:
        try:
            validate_report(load_report(path))
        except (OSError, UnicodeError, ValueError, RecursionError) as error:
            print(f"FAIL {path}: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"PASS {path}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
