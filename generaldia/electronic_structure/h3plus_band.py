"""H3+ ground-to-excited-doublet optical benchmark; no fit or dynamics claim."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from ..transition_band import transition_band
from .fci_dipoles import FCIDipoleBackend
from .h3plus_ci import ATOMIC_NUMBERS, h3plus_triangle
from .singlet_oracle import singlet_spectrum

GATES = {
    "energy_abs_hartree": 1e-8,
    "tensor_abs_e2_bohr2": 2e-6,
    "oscillator_abs": 2e-6,
    "min_external_gap_hartree": 1e-3,
    "center_doublet_gap_hartree": 1e-6,
    "min_line_strength_e2_bohr2": 1e-5,
}


def ao_reference(xyz, basis):
    from pyscf import gto

    mol = gto.M(
        atom=[("H", p) for p in xyz], basis=basis, unit="Angstrom", charge=1, spin=0, verbose=0
    )
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        position = mol.intor("int1e_r", comp=3)
    return singlet_spectrum(
        mol.intor_symmetric("int1e_ovlp"),
        mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc"),
        mol.intor("int2e", aosym="s1"),
        position,
        mol.energy_nuc(),
    )


def certify_transition_band(basis: str) -> dict:
    """Check two independent singlet calculations at named static geometries in one basis."""
    if basis not in ("sto-3g", "6-31g"):
        raise ValueError("benchmark supports only sto-3g and 6-31g")
    bond = 0.88125
    center = h3plus_triangle(bond)
    x = h3plus_triangle(bond, 0.035, 0)
    y = h3plus_triangle(bond, 0, -0.035)
    # A proper rotation that moves the molecular plane, followed by an origin shift.
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    xyz = np.stack([center, x, y, x @ rotation.T + np.array([0.20, -0.13, 0.16])])
    names = ["center", "x-distorted", "y-distorted", "x-rotated-translated"]
    backend = FCIDipoleBackend(basis=basis, charge=1, n_states=4, conv_tol=1e-11, max_orbitals=8)
    path = backend.calculate_path(ATOMIC_NUMBERS, xyz, path_id=f"h3plus-optical-{basis}")
    rows = []
    targets = []
    for label, pos, sample in zip(names, xyz, path, strict=True):
        energies = sample.energies.numpy()
        dipoles = sample.dipole_matrix_elements.numpy()
        obs = transition_band(dipoles[:, 0, 1:3], np.diag(energies[1:3] - energies[0]))
        oracle_e, oracle_mu = ao_reference(pos, basis)
        # Independent sum over excited eigenstates (not the production tensor formula).
        vectors = oracle_mu[:, 0, 1:3].T
        oracle_q = sum(np.outer(vec, vec) for vec in vectors)
        oracle_f = (2 / 3) * sum(
            (oracle_e[j + 1] - oracle_e[0]) * float(vec @ vec) for j, vec in enumerate(vectors)
        )
        de = float(np.max(np.abs(energies - oracle_e[:4])))
        dq = float(np.max(np.abs(obs.tensor - oracle_q)))
        df = float(abs(obs.oscillator - oracle_f))
        isolation = float(min(energies[1] - energies[0], energies[3] - energies[2]))
        # Same spectrum, deliberately zero transition moments. Energy agreement alone accepts it.
        zero_error = float(np.max(np.abs(oracle_q)))
        row_checks = {
            "independent_singlet_energies": de <= GATES["energy_abs_hartree"],
            "independent_line_tensor": dq <= GATES["tensor_abs_e2_bohr2"],
            "independent_oscillator_sum": df <= GATES["oscillator_abs"],
            "band_isolated": isolation > GATES["min_external_gap_hartree"],
            "bright_band": obs.line_strength > GATES["min_line_strength_e2_bohr2"],
            "energy_matched_zero_dipoles_rejected": zero_error > GATES["tensor_abs_e2_bohr2"],
        }
        targets.append(obs)
        rows.append(
            {
                "geometry": label,
                "positions_angstrom": pos.tolist(),
                "energies_hartree": energies.tolist(),
                "line_tensor_e2_bohr2": obs.tensor.real.tolist(),
                "line_strength_e2_bohr2": obs.line_strength,
                "oscillator_sum": obs.oscillator,
                "oracle_oscillator_sum": float(oracle_f),
                "oracle_line_tensor_e2_bohr2": oracle_q.tolist(),
                "oracle_energy_error_hartree": de,
                "oracle_tensor_max_error_e2_bohr2": dq,
                "oracle_oscillator_error": df,
                "min_external_gap_hartree": isolation,
                "zero_dipole_tensor_error_e2_bohr2": zero_error,
                "checks": row_checks,
            }
        )
    equilateral, original, _, moved = targets
    e = rows[0]["energies_hartree"]
    planar = np.diag([equilateral.line_strength / 2] * 2 + [0])
    center_error = float(np.max(np.abs(equilateral.tensor - planar)))
    rotated_error = float(np.max(np.abs(moved.tensor - rotation @ original.tensor @ rotation.T)))
    moved_f_error = abs(moved.oscillator - original.oscillator)
    checks = {
        "all_geometry_checks": all(all(row["checks"].values()) for row in rows),
        "center_doublet_degenerate": abs(e[2] - e[1]) <= GATES["center_doublet_gap_hartree"],
        "center_planar_isotropy": center_error <= GATES["tensor_abs_e2_bohr2"],
        "rotation_and_translation_tensor_covariance": rotated_error <= GATES["tensor_abs_e2_bohr2"],
        "rotation_and_translation_oscillator_invariant": moved_f_error <= GATES["oscillator_abs"],
    }
    return {
        "basis": basis,
        "geometry_sha256": sha256(xyz.astype("<f8").tobytes()).hexdigest(),
        "backend": path.metadata,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "center_tensor_symmetry_error_e2_bohr2": center_error,
        "rigid_tensor_error_e2_bohr2": rotated_error,
        "rigid_oscillator_error": moved_f_error,
        "geometries": rows,
    }
