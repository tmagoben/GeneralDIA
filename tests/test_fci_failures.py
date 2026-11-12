"""Inject faults into real PySCF solves and require the adapter to stop."""

import numpy as np
import pytest

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def serial_pyscf():
    pyscf = pytest.importorskip("pyscf")
    old = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    yield
    pyscf.lib.num_threads(old)


@pytest.fixture
def scan():
    return np.array([[[0, 0, 0], [0, 0, r]] for r in (0.74, 0.76)])


def test_rhf_nonconvergence_stops_before_fci(monkeypatch, scan):
    from pyscf import fci, scf

    original = scf.hf.RHF.kernel

    def unconverged(self, *args, **kwargs):
        energy = original(self, *args, **kwargs)
        self.converged = False
        return energy

    def forbidden(*args, **kwargs):
        pytest.fail("FCI must not run after an unconverged RHF reference")

    monkeypatch.setattr(scf.hf.RHF, "kernel", unconverged)
    monkeypatch.setattr(fci.direct_spin0.FCI, "kernel", forbidden)
    with pytest.raises(RuntimeError, match="RHF reference did not converge"):
        FCIDipoleBackend().calculate_path([1, 1], scan, path_id="bad-rhf")


@pytest.mark.parametrize("fault", ["nonconvergence", "missing_energy", "missing_root"])
def test_fci_faults_stop_before_observables(monkeypatch, scan, fault):
    from pyscf import fci

    original = fci.direct_spin0.FCI.kernel

    def corrupted(self, *args, **kwargs):
        energies, roots = original(self, *args, **kwargs)
        if fault == "nonconvergence":
            self.converged = [True, False]
        elif fault == "missing_energy":
            energies = energies[:-1]
        else:
            roots = roots[:-1]
        return energies, roots

    def forbidden(*args, **kwargs):
        pytest.fail("Transition densities must not use invalid FCI roots")

    monkeypatch.setattr(fci.direct_spin0.FCI, "kernel", corrupted)
    monkeypatch.setattr(fci.direct_spin0.FCI, "trans_rdm1", forbidden)
    error = RuntimeError if fault == "nonconvergence" else ValueError
    message = (
        "FCI calculation did not converge"
        if fault == "nonconvergence"
        else "available singlet roots"
    )
    with pytest.raises(error, match=message):
        FCIDipoleBackend().calculate_path([1, 1], scan, path_id="bad-fci")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_physical_closing_overlap_is_rejected(monkeypatch, scan, value):
    from pyscf import fci

    geometry = np.concatenate((scan, scan[:1] + np.array([0, 0, 0.01])))
    original = fci.addons.overlap
    calls = []

    def corrupted(*args, **kwargs):
        answer = original(*args, **kwargs)
        calls.append(answer)
        # Two open-chain edges, four bra/ket links each; corrupt closure only.
        return value if len(calls) == 9 else answer

    monkeypatch.setattr(fci.addons, "overlap", corrupted)
    with pytest.raises(ValueError, match="physical closing overlap contains nonfinite"):
        FCIDipoleBackend().calculate_loop([1, 1], geometry, path_id="bad-closure")
    assert len(calls) == 12
    assert np.isfinite(calls).all()


@pytest.mark.parametrize("bad", ["complex_positions", "boolean_atoms", "complex_atoms"])
def test_fci_rejects_silent_input_coercions(scan, bad):
    numbers = [1, 1]
    if bad == "complex_positions":
        scan = scan.astype(complex) + 0.01j
    elif bad == "boolean_atoms":
        numbers = [True, True]
    else:
        numbers = [1 + 0j, 1 + 0j]
    with pytest.raises(ValueError, match=r"real numbers|positive integers"):
        FCIDipoleBackend().calculate_path(numbers, scan, path_id="invalid-input")
