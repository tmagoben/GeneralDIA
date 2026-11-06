"""Charged singlet FCI reference tests (no CI topology claim yet)."""

import numpy as np
import pytest
import torch

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def pyscf_single_thread():
    pyscf = pytest.importorskip("pyscf")
    previous = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    yield
    pyscf.lib.num_threads(previous)


def _geometries():
    # Intentionally asymmetric triangles: avoid state degeneracies in charge tests.
    points = np.array([[0.0, 0.0, 0.0], [0.91, 0.0, 0.0], [0.41, 0.74, 0.0]])
    next_points = points.copy()
    next_points[2, 1] += 0.025
    return np.stack((points, next_points))


def test_charged_singlet_h3plus_reference_and_physical_overlaps():
    path = FCIDipoleBackend(charge=1, n_states=3, conv_tol=1e-11).calculate_path(
        [1, 1, 1], _geometries(), path_id="h3plus-charge"
    )
    assert path.metadata["charge"] == 1
    assert path.metadata["spin"] == 0
    assert path.metadata["n_states"] == 3
    assert path.energies.shape == (2, 3)
    assert torch.isfinite(path.energies).all()
    assert torch.all(torch.diff(path.energies, dim=1) >= -1e-10)
    assert path.adjacent_overlaps.shape == (1, 3, 3)
    assert torch.linalg.svdvals(path.adjacent_overlaps).max() <= 1 + 1e-8


def test_total_dipole_of_ion_obeys_charge_translation_law():
    coords = _geometries()
    move_angstrom = np.array([0.21, -0.13, 0.16])
    backend = FCIDipoleBackend(charge=1, n_states=2, conv_tol=1e-11)
    original = backend.calculate_path([1, 1, 1], coords, path_id="h3plus-original")
    moved = backend.calculate_path([1, 1, 1], coords + move_angstrom, path_id="h3plus-moved")
    # 1 bohr = 0.529177210903 angstrom; a +1 ion shifts by +delta_bohr.
    displacement_bohr = torch.from_numpy(move_angstrom / 0.529177210903)
    for before, after in zip(original, moved, strict=True):
        torch.testing.assert_close(before.energies, after.energies, atol=2e-8, rtol=1e-8)
        before_diag = torch.diagonal(before.dipole_matrix_elements, dim1=-2, dim2=-1)
        after_diag = torch.diagonal(after.dipole_matrix_elements, dim1=-2, dim2=-1)
        torch.testing.assert_close(
            after_diag - before_diag,
            displacement_bohr[:, None].expand_as(after_diag),
            atol=3e-7,
            rtol=3e-7,
        )
    assert original.metadata["dipole_translation_law"].startswith("mu(R+delta)")


def test_charged_overlap_matches_explicit_two_electron_determinant_products():
    from pyscf import ao2mo, fci, gto, scf

    coords = _geometries()
    backend = FCIDipoleBackend(charge=1, n_states=2, conv_tol=1e-11)
    path = backend.calculate_path([1, 1, 1], coords, path_id="h3plus-oracle")
    references = []
    for positions in coords:
        mol = gto.M(
            atom=list(zip((1, 1, 1), positions, strict=True)),
            basis="sto-3g",
            unit="Angstrom",
            charge=1,
            spin=0,
            verbose=0,
        )
        mf = scf.RHF(mol).run(conv_tol=1e-11)
        solver = fci.direct_spin0.FCI(mol)
        solver.nroots = 2
        solver.conv_tol = 1e-11
        mo = mf.mo_coeff
        _, roots = solver.kernel(
            mo.T @ mf.get_hcore() @ mo,
            ao2mo.kernel(mol, mo),
            mo.shape[1],
            mol.nelec,
            ecore=mol.energy_nuc(),
        )
        references.append((mol, mo, roots))
    left, right = references
    orbital_overlap = left[1].T @ gto.intor_cross("int1e_ovlp", left[0], right[0]) @ right[1]
    explicit = np.array(
        [
            [
                np.einsum("ab,ac,bd,cd->", bra.conj(), orbital_overlap, orbital_overlap, ket)
                for ket in right[2]
            ]
            for bra in left[2]
        ]
    )
    np.testing.assert_allclose(
        path.adjacent_overlaps[0].abs().numpy(),
        np.abs(explicit),
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize("charge", [True, 1.0, "1"])
def test_bad_charges_rejected(charge):
    with pytest.raises(ValueError, match="charge must be an integer"):
        FCIDipoleBackend(charge=charge)


def test_even_positive_electron_count_required():
    coords = _geometries()
    with pytest.raises(ValueError, match="positive even electron count"):
        FCIDipoleBackend().calculate_path([1, 1, 1], coords, path_id="odd-neutral-h3")
    with pytest.raises(ValueError, match="positive even electron count"):
        FCIDipoleBackend(charge=2).calculate_path([1, 1, 1], coords, path_id="odd-ion")
    with pytest.raises(ValueError, match="positive even electron count"):
        FCIDipoleBackend(charge=3).calculate_path([1, 1, 1], coords, path_id="no-electrons")
