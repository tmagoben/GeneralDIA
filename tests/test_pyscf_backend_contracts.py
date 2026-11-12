"""PySCF wrapper boundary tests and independent molecular gradient checks."""

import builtins

import numpy as np
import pytest

from generaldia.dataset import ANGSTROM_TO_BOHR
from generaldia.electronic_structure.pyscf_backend import RHFBackend, SACASSCFBackend, _atoms


@pytest.mark.parametrize(
    "numbers", [[1, 1.5], [True, True], [1 + 0j, 1], [1, np.nan], [0, 1], [21, 1], []]
)
def test_atom_labels_are_not_silently_coerced(numbers):
    with pytest.raises(ValueError):
        _atoms(numbers, np.zeros((len(numbers), 3)))


@pytest.mark.parametrize(
    "positions", [np.zeros((2, 3), dtype=complex), np.full((2, 3), np.inf), np.zeros((2, 2))]
)
def test_coordinates_reject_complex_nonfinite_and_wrong_shape(positions):
    with pytest.raises(ValueError):
        _atoms([1, 1], positions)


@pytest.mark.parametrize("factory", [RHFBackend, lambda **kw: SACASSCFBackend(2, 2, **kw)])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"basis": " "},
        {"basis": None},
        {"conv_tol": np.nan},
        {"conv_tol": np.inf},
        {"conv_tol": 0},
        {"charge": 0.5},
        {"charge": True},
        {"spin": 0.5},
        {"spin": -1},
    ],
)
def test_solver_settings_fail_before_any_calculation(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ncas": 1.5},
        {"ncas": True},
        {"ncas": 0},
        {"n_states": 2.5},
        {"n_states": True},
        {"n_states": 1},
        {"nelecas": 1.5},
        {"nelecas": (1, -1)},
        {"nelecas": 5},
        {"use_etfs": "false"},
    ],
)
def test_casscf_rejects_invalid_active_space_and_controls(kwargs):
    with pytest.raises(ValueError):
        SACASSCFBackend(**{"ncas": 2, "nelecas": 2, **kwargs})


def test_missing_pyscf_has_actionable_error(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyscf":
            raise ImportError("deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="'pyscf' extra"):
        RHFBackend().calculate([1, 1], np.zeros((2, 3)))


@pytest.fixture
def serial_pyscf():
    pyscf = pytest.importorskip("pyscf")
    old = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    yield pyscf
    pyscf.lib.num_threads(old)


@pytest.mark.optional
@pytest.mark.parametrize("method", ["rhf", "casscf"])
def test_real_wrapper_gradients_match_energy_finite_differences(serial_pyscf, method):
    numbers = [3, 1]
    geometry = np.array([[0, 0, 0], [0, 0, 1.5]], dtype=float)
    backend = RHFBackend() if method == "rhf" else SACASSCFBackend(2, (1, 1))
    center = backend.calculate(numbers, geometry)

    # State-specific energies are not stationary under SA orbital rotations.
    # Default sqrt(conv_tol)=1e-5 orbital residuals can amplify into O(1e-4)
    # finite-difference noise even when the average energy has converged.
    # Keep the default wrapper under test. The independent energy oracle
    # uses LiH's exact axial symmetry and converges orbital/energy residuals.
    # Symmetry-unrestricted orbital rotations left individual root energies
    # noisy despite a converged state average. Check the same central roots
    # explicitly so symmetry cannot silently substitute a different manifold.
    def reference_energy(positions):
        from pyscf import gto, mcscf, scf

        molecule = gto.M(
            atom=[("Li", positions[0]), ("H", positions[1])],
            basis="sto-3g",
            unit="Angstrom",
            symmetry=True,
            verbose=0,
        )
        mean_field = scf.RHF(molecule)
        mean_field.conv_tol = 1e-13
        mean_field.conv_tol_grad = 1e-10
        mean_field.kernel()
        assert mean_field.converged
        if method == "rhf":
            return np.array([mean_field.e_tot])
        reference = mcscf.CASSCF(mean_field, 2, (1, 1)).state_average([0.5, 0.5])
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-9
        reference.fcisolver.conv_tol = 1e-13
        reference.max_cycle_macro = 100
        reference.kernel()
        assert reference.converged
        return np.asarray(reference.e_states)

    np.testing.assert_allclose(
        reference_energy(geometry), center.energies_hartree, atol=1e-6, rtol=0
    )
    derivatives = []
    # The step study resolves truncation and solver noise at 0.001--0.002 A.
    # Smaller displacements amplify tiny state-specific energy residuals.
    for step in (2e-3, 1e-3):
        plus, minus = geometry.copy(), geometry.copy()
        plus[1, 2] += step
        minus[1, 2] -= step
        derivative = (reference_energy(plus) - reference_energy(minus)) / (
            2 * step * ANGSTROM_TO_BOHR
        )
        derivatives.append(derivative)
        np.testing.assert_allclose(
            center.gradients_hartree_per_bohr[:, 1, 2], derivative, atol=2e-5, rtol=0
        )
    np.testing.assert_allclose(derivatives[0], derivatives[1], atol=2e-6, rtol=0)
    np.testing.assert_allclose(center.gradients_hartree_per_bohr.sum(axis=1), 0, atol=2e-7)
    assert center.metadata["converged"] is True
    assert center.metadata["gradient_unit"] == "hartree/bohr"
    if method == "casscf":
        assert center.n_states == 2
        assert center.metadata["weights"] == [0.5, 0.5]
        assert set(center.scaled_nac_pyscf) == {(0, 1)}
        assert np.isfinite(center.scaled_nac_pyscf[(0, 1)]).all()
        assert center.metadata["scaled_nac_convention"] == "PySCF mult_ediff=True, key=(ket, bra)"


@pytest.mark.optional
@pytest.mark.parametrize("method", ["rhf", "casscf_reference", "casscf"])
def test_nonconverged_wrapper_never_returns_success(serial_pyscf, monkeypatch, method):
    from pyscf import mcscf, scf

    cls = mcscf.mc1step.CASSCF if method == "casscf" else scf.hf.RHF
    original = cls.kernel

    def failed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.converged = False
        return result

    monkeypatch.setattr(cls, "kernel", failed)
    backend = RHFBackend() if method == "rhf" else SACASSCFBackend(2, 2)
    with pytest.raises(RuntimeError, match="did not converge"):
        backend.calculate([3, 1], [[0, 0, 0], [0, 0, 1.5]])
