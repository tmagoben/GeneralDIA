"""Physical reference checks independent of the adapter's dipole/overlap formulas."""

import numpy as np
import pytest
import torch

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend
from generaldia.path_losses import _dipole_descriptors

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def single_threaded_pyscf():
    pyscf = pytest.importorskip("pyscf")
    previous = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    yield
    pyscf.lib.num_threads(previous)


def reference_calculation(numbers, coordinates):
    from pyscf import ao2mo, fci, gto, scf

    mol = gto.M(
        atom=list(zip(numbers, coordinates, strict=True)),
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mo = mf.mo_coeff
    h1 = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol, mo)
    solver = fci.direct_spin0.FCI(mol)
    solver.conv_tol = 1e-12
    solver.nroots = 2
    e, roots = solver.kernel(h1, eri, mol.nao_nr(), mol.nelec, ecore=mol.energy_nuc())
    return mol, mo, h1, eri, solver, np.asarray(e), roots


def test_lih_dipoles_match_finite_field_energies_and_wavefunction_response():
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    path = FCIDipoleBackend(conv_tol=1e-12).calculate_path(
        [3, 1], [coordinates, coordinates * 1.02], path_id="lih-field", family_id="LiH"
    )
    mol, mo, h1, eri, solver, energies, roots = reference_calculation([3, 1], coordinates)
    np.testing.assert_allclose(path[0].energies, energies, atol=1e-10)
    rz = mo.T @ mol.intor("int1e_r", comp=3)[2] @ mo
    nuclear_z = np.dot(mol.atom_charges(), mol.atom_coords()[:, 2])
    step = 2e-5
    perturbed = []
    for field in (step, -step):
        e, c = solver.kernel(
            h1 + field * rz,
            eri,
            mol.nao_nr(),
            mol.nelec,
            ecore=mol.energy_nuc() - field * nuclear_z,
        )
        # Align the perturbed ket to the zero-field ket before differentiating it.
        ket = c[1] * np.sign(np.vdot(roots[1], c[1]).real)
        perturbed.append((np.asarray(e), ket))
    permanent = -(perturbed[0][0] - perturbed[1][0]) / (2 * step)
    response = (perturbed[0][1] - perturbed[1][1]) / (2 * step)
    transition = -(energies[1] - energies[0]) * np.vdot(roots[0], response)
    dipoles = path[0].dipole_matrix_elements[2].numpy()
    np.testing.assert_allclose(dipoles.diagonal(), permanent, atol=2e-6)
    np.testing.assert_allclose(abs(dipoles[0, 1]), abs(transition), atol=2e-6)
    assert abs(transition) > 0.1
    assert path.tracked().ambiguous_steps == ()
    assert path.metadata["dipole_unit"] == "e*bohr"
    assert path.metadata["molecular_family"] == "LiH"


def test_two_electron_overlaps_match_explicit_determinant_products():
    from pyscf import gto

    coordinates = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, length]] for length in (0.7, 1.05)])
    path = FCIDipoleBackend(conv_tol=1e-12).calculate_path(
        [1, 1], coordinates, path_id="h2-overlap"
    )
    left = reference_calculation([1, 1], coordinates[0])
    right = reference_calculation([1, 1], coordinates[1])
    s = left[1].T @ gto.intor_cross("int1e_ovlp", left[0], right[0]) @ right[1]
    # One alpha and one beta electron: each determinant overlap is S_ac * S_bd.
    expected = np.array(
        [
            [np.einsum("ab,ac,bd,cd->", bra.conj(), s, s, ket) for ket in right[-1]]
            for bra in left[-1]
        ]
    )
    np.testing.assert_allclose(path.adjacent_overlaps[0], expected, atol=1e-9)
    naive = np.array([[np.vdot(bra, ket) for ket in right[-1]] for bra in left[-1]])
    assert np.max(np.abs(expected - naive)) > 0.01


def test_neutral_fci_dipoles_rotate_and_are_origin_independent():
    coordinates = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, length]] for length in (1.5, 1.55)])
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    backend = FCIDipoleBackend(conv_tol=1e-12)
    original = backend.calculate_path([3, 1], coordinates, path_id="original")
    moved = backend.calculate_path([3, 1], coordinates @ q.T + [0.7, -0.5, 0.3], path_id="moved")
    for before, after in zip(original, moved, strict=True):
        torch.testing.assert_close(before.energies, after.energies, atol=1e-9, rtol=1e-9)
        expected = torch.einsum("ab,bij->aij", torch.from_numpy(q), before.dipole_matrix_elements)
        # Compare gauge invariants because independent FCI roots have arbitrary signs.
        torch.testing.assert_close(
            _dipole_descriptors(expected, ((0,), (1,))),
            _dipole_descriptors(after.dipole_matrix_elements, ((0,), (1,))),
            atol=1e-7,
            rtol=1e-7,
        )


def test_fci_adapter_rejects_invalid_inputs_and_oversized_spaces():
    for kwargs in ({"n_states": 0}, {"conv_tol": float("nan")}, {"max_orbitals": 0}):
        with pytest.raises(ValueError):
            FCIDipoleBackend(**kwargs)
    coordinates = [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]] * 2
    with pytest.raises(ValueError, match="even electron"):
        FCIDipoleBackend().calculate_path([2, 1], coordinates, path_id="odd")
    with pytest.raises(ValueError, match="max_orbitals"):
        FCIDipoleBackend(max_orbitals=1).calculate_path([1, 1], coordinates, path_id="large")
    with pytest.raises(ValueError, match="positive integers"):
        FCIDipoleBackend().calculate_path([0, 2], coordinates, path_id="invalid")
