"""Fit and validate local H3+ Hamiltonians against held-out physical FCI paths.

Run with OMP_NUM_THREADS=1. Outputs strict JSON; exits nonzero on any gate
failure. The symmetry prior supplies the intersection structure; spectra
fit its coefficients. No raw cross-geometry CI coefficients are compared.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
from hashlib import sha256
from importlib.metadata import version
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend
from generaldia.electronic_structure.h3plus_ci import h3plus_ci_loop, h3plus_triangle
from generaldia.h3plus_model import fit_h3plus_spectra, h3plus_bonds
from generaldia.topology import closed_loop_overlaps, wilson_loop

# Fixed before evaluating held-out data; not fitted or relaxed per seed/basis.
THRESHOLDS = {
    "energy_mae_hartree": 0.002,
    "energy_max_error_hartree": 0.005,
    "branching_relative_error": 0.10,
    "central_gap_hartree": 1e-8,
    "phase_error_radians": 0.06,
    "minimum_link_strength": 0.65,
    "minimum_isolation_hartree": 1e-3,
    "minimum_loop_gap_hartree": 1e-4,
}


def geometry_keys(positions: np.ndarray) -> set[tuple[float, ...]]:
    """Rigid-motion/proton-permutation invariant duplicate keys (1e-9 A)."""
    bonds = h3plus_bonds(torch.as_tensor(positions)).numpy()
    return {tuple(row) for row in np.round(np.sort(bonds, axis=1), 9)}


def split_geometries() -> tuple[np.ndarray, dict[str, tuple[np.ndarray, float]]]:
    """Predeclared local train/test paths; all held-out mean bonds are unseen."""
    training = np.concatenate(
        [
            np.stack(
                [
                    h3plus_triangle(bond, radius * math.cos(a), radius * math.sin(a))
                    for a in 0.17 + np.arange(12) * 2 * math.pi / 12
                ]
            )
            for bond in (0.85, 0.875, 0.90)
            for radius in (0.03, 0.06, 0.09)
        ]
    )
    paths = {
        f"heldout-R{bond}-r{radius}": (h3plus_ci_loop(16, radius, bond_angstrom=bond), math.pi)
        for bond in (0.8625, 0.8875)
        for radius in (0.045, 0.075)
    }
    paths.update(
        {
            f"refinement-{n}": (h3plus_ci_loop(n, 0.055, bond_angstrom=0.8625), math.pi)
            for n in (8, 16, 32)
        }
    )
    paths["reverse"] = (h3plus_ci_loop(32, 0.055, bond_angstrom=0.8625, orientation=-1), math.pi)
    paths["nonenclosing"] = (
        h3plus_ci_loop(24, 0.015, bond_angstrom=0.8875, center_x_angstrom=0.06),
        0.0,
    )
    paths["ellipse"] = (
        np.stack(
            [
                h3plus_triangle(0.8875, 0.055 * math.cos(a), 0.044 * math.sin(a))
                for a in np.arange(24) * 2 * math.pi / 24
            ]
        ),
        math.pi,
    )
    train_keys = geometry_keys(training)
    if any(train_keys & geometry_keys(geometry) for geometry, _ in paths.values()):
        raise ValueError("training/test geometry leakage, including proton permutations")
    return training, paths


def phase_distance(a: float, b: float) -> float:
    """Wrapped phase distance in radians."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def evaluate_candidate(model, references: list[dict]) -> dict:
    """Apply identical energy, gap and Berry gates to any candidate Hamiltonian.

    Reference links come from physical cross-AO contractions; model frames
    share a fixed latent basis. Only gauge-invariant phases are compared.
    """
    paths = []
    errors = []
    for reference in references:
        with torch.no_grad():
            h = model(h3plus_bonds(torch.as_tensor(reference["geometry"])))
            energies, frames = torch.linalg.eigh(h)
        errors.append((energies - reference["energies"][:, :3]).abs())
        links = closed_loop_overlaps(frames)
        phases = []
        strengths = []
        physical = []
        physical_strengths = []
        # Return failed checks for unresolved nonzero links instead of
        # accidentally passing a phase from a poorly resolved discretization.
        for state in (1, 2):
            result = wilson_loop(
                links[:, state : state + 1, state : state + 1], minimum_link_singular_value=1e-12
            )
            truth = wilson_loop(
                reference["links"][:, state : state + 1, state : state + 1],
                minimum_link_singular_value=1e-12,
            )
            phases.append(result.berry_phase)
            physical.append(truth.berry_phase)
            strengths.append(result.minimum_link_singular_value)
            physical_strengths.append(truth.minimum_link_singular_value)
        paths.append(
            {
                "name": reference["name"],
                "points": len(h),
                "expected_phase_radians": reference["expected"],
                "model_phases_radians": phases,
                "physical_phases_radians": physical,
                "minimum_model_link": min(strengths),
                "minimum_physical_link": min(physical_strengths),
                "phase_error_radians": max(
                    phase_distance(a, b) for a, b in zip(phases, physical, strict=True)
                ),
                "physical_target_error_radians": max(
                    phase_distance(p, reference["expected"]) for p in physical
                ),
                "model_ground_isolation_hartree": float((energies[:, 1] - energies[:, 0]).min()),
                "model_excited_gap_hartree": float((energies[:, 2] - energies[:, 1]).min()),
                "physical_excited_gap_hartree": float(
                    (reference["energies"][:, 2] - reference["energies"][:, 1]).min()
                ),
                "physical_minimum_isolation_hartree": float(
                    torch.diff(reference["energies"], dim=-1)[:, [0, 2]].min()
                ),
            }
        )
    error = torch.cat(errors)
    mae, maximum = float(error.mean()), float(error.max())
    checks = {
        "energy_mae": mae <= THRESHOLDS["energy_mae_hartree"],
        "energy_max_error": maximum <= THRESHOLDS["energy_max_error_hartree"],
        "physical_reference_topology": all(
            p["physical_target_error_radians"] <= THRESHOLDS["phase_error_radians"] for p in paths
        ),
        "physical_and_model_links_resolved": all(
            min(p["minimum_model_link"], p["minimum_physical_link"])
            >= THRESHOLDS["minimum_link_strength"]
            for p in paths
        ),
        "heldout_topology": all(
            p["phase_error_radians"] <= THRESHOLDS["phase_error_radians"] for p in paths
        ),
        "isolated_states": all(
            min(p["model_ground_isolation_hartree"], p["physical_minimum_isolation_hartree"])
            >= THRESHOLDS["minimum_isolation_hartree"]
            and min(p["model_excited_gap_hartree"], p["physical_excited_gap_hartree"])
            >= THRESHOLDS["minimum_loop_gap_hartree"]
            for p in paths
        ),
    }
    return {
        "energy_mae_hartree": mae,
        "energy_max_error_hartree": maximum,
        "paths": paths,
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_benchmark(basis: str, seeds: tuple[int, ...] = (0, 7, 23)) -> dict:
    """Fit separate seed initializations to one finite-basis physical dataset."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    training, paths = split_geometries()
    backend = FCIDipoleBackend(basis=basis, n_states=4, charge=1, conv_tol=1e-11, max_orbitals=8)
    training_reference = backend.calculate_path((1, 1, 1), training, path_id="training")
    references = []
    for name, (geometry, expected) in paths.items():
        loop = backend.calculate_loop((1, 1, 1), geometry, path_id=name)
        references.append(
            {
                "name": name,
                "geometry": geometry,
                "expected": expected,
                "energies": loop.path.energies,
                "links": loop.cyclic_overlaps,
            }
        )
    axis = np.stack(
        [
            h3plus_triangle(0.8625, x, y)
            for x, y in ((0, 0), (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005))
        ]
    )
    if geometry_keys(axis) & geometry_keys(training):
        raise ValueError("branching evaluation leaks training geometries")
    axis_reference = backend.calculate_path((1, 1, 1), axis, path_id="heldout-branching")
    physical_gaps = torch.diff(axis_reference.energies[:, 1:3], dim=-1)[:, 0]
    runs = []
    for seed in seeds:
        fitted = fit_h3plus_spectra(
            h3plus_bonds(torch.as_tensor(training)), training_reference.energies[:, :3], seed=seed
        )
        model = fitted.model
        result = evaluate_candidate(model, references)

        def energy_only(bonds, model=model):
            return torch.diag_embed(torch.linalg.eigvalsh(model(bonds)))

        negative = evaluate_candidate(energy_only, references)
        with torch.no_grad():
            model_gaps = torch.diff(
                torch.linalg.eigvalsh(model.from_positions(torch.as_tensor(axis)))[:, 1:3], dim=-1
            )[:, 0]
            relative = ((model_gaps[1:] - physical_gaps[1:]) / physical_gaps[1:]).abs()
            bonds = h3plus_bonds(torch.as_tensor(paths["ellipse"][0]))
            h = model(bonds)
            covariance_error = max(
                float((model(bonds[:, list(p)]) - h[:, list(p)][:, :, list(p)]).abs().max())
                for p in permutations(range(3))
            )
            _, frames = torch.linalg.eigh(h)
            angle = torch.arange(len(frames), dtype=torch.float64)[:, None] * torch.tensor(
                [0.2, 0.3, 0.7]
            )
            gauged = frames.to(torch.complex128) * torch.exp(1j * angle)[:, None, :]
            gauge_error = max(
                phase_distance(
                    wilson_loop(closed_loop_overlaps(frames[:, :, s : s + 1])).berry_phase,
                    wilson_loop(closed_loop_overlaps(gauged[:, :, s : s + 1])).berry_phase,
                )
                for s in (1, 2)
            )
        checks = result["checks"] | {
            "equilateral_degeneracy": max(abs(float(model_gaps[0])), abs(float(physical_gaps[0])))
            <= THRESHOLDS["central_gap_hartree"],
            "independent_branching_axes": float(relative.max())
            <= THRESHOLDS["branching_relative_error"],
            "permutation_covariance": covariance_error < 1e-10,
            "complex_gauge_invariance": gauge_error < 1e-10,
            "isospectral_negative_control_rejected": not negative["pass"]
            and negative["checks"]["energy_mae"]
            and negative["checks"]["energy_max_error"]
            and not negative["checks"]["heldout_topology"],
        }
        runs.append(
            {
                "seed": seed,
                "training_mae_hartree": fitted.training_mae_hartree,
                "training_max_error_hartree": fitted.training_max_error_hartree,
                "scalar_coefficients": model.scalar_coefficients.detach().tolist(),
                "splitting_coefficients": model.splitting_coefficients.detach().tolist(),
                "heldout": result,
                "isospectral_negative_control": negative,
                "physical_branching_gaps_hartree": physical_gaps.tolist(),
                "model_branching_gaps_hartree": model_gaps.tolist(),
                "maximum_branching_relative_error": float(relative.max()),
                "permutation_covariance_error": covariance_error,
                "gauge_phase_error_radians": gauge_error,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return {
        "basis": basis,
        "charge": 1,
        "reference_states": 4,
        "model_states": 3,
        "training_geometries": len(training),
        "training_unique_permutation_invariant_geometries": len(geometry_keys(training)),
        "training_geometry_sha256": sha256(training.tobytes()).hexdigest(),
        "heldout_geometry_sha256": {
            name: sha256(geo.tobytes()).hexdigest() for name, (geo, _) in paths.items()
        },
        "branching_geometry_sha256": sha256(axis.tobytes()).hexdigest(),
        "train_test_disjoint_under_permutations": True,
        "conv_tol": 1e-11,
        "max_orbitals": 8,
        "runs": runs,
        "pass": all(r["pass"] for r in runs),
    }


def main() -> None:
    """Write the two-basis, three-seed experiment and fail on any failed gate."""
    root = Path(__file__).resolve().parents[1]
    revision = (
        os.environ.get("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )
    report = {
        "schema": "generaldia.learned-h3plus.v1",
        "revision": revision,
        "scope": "Local spectral fit with an imposed A1+E symmetry prior; single-molecule holdout",
        "thresholds": THRESHOLDS,
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "torch", "pyscf", "generaldia")},
        },
        "source_files_sha256": {
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "generaldia/h3plus_model.py",
                "generaldia/topology.py",
                "generaldia/electronic_structure/fci_dipoles.py",
                "generaldia/electronic_structure/h3plus_ci.py",
                "examples/17_learned_h3plus_topology.py",
            )
        },
        "bases": [run_benchmark(basis) for basis in ("sto-3g", "6-31g")],
    }
    report["pass"] = all(b["pass"] for b in report["bases"])
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
