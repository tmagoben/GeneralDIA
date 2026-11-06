"""Physical FCI adapter guardrails, including errors before costly SCF work."""

import numpy as np
import pytest

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        ({"basis": ""}, "basis must be a non-empty"),
        ({"basis": None}, "basis must be a non-empty"),
        ({"basis": "  "}, "basis must be a non-empty"),
        ({"n_states": True}, "positive integer"),
        ({"n_states": 0}, "positive integer"),
        ({"conv_tol": float("nan")}, "finite and positive"),
        ({"max_orbitals": 0}, "positive integer"),
        ({"max_orbitals": True}, "positive integer"),
    ],
)
def test_fci_adapter_rejects_unsupported_configuration(kwargs, pattern):
    with pytest.raises(ValueError, match=pattern):
        FCIDipoleBackend(**kwargs)


@pytest.fixture
def h2_scan():
    geometry = np.array([[0, 0, -0.37], [0, 0, 0.37]], dtype=np.float64)
    return np.stack((geometry, geometry + np.array([[0, 0, 0], [0, 0, 0.02]])))


@pytest.mark.optional
@pytest.mark.parametrize(
    "numbers, expected",
    [
        ([1, 1.5], "positive integers"),
        ([1, 0], "positive integers"),
        ([float("nan"), 1.0], "positive integers"),
    ],
)
def test_fci_adapter_rejects_invalid_charge_and_atom_labels_before_scf(numbers, expected, h2_scan):
    pytest.importorskip("pyscf")
    with pytest.raises(ValueError, match=expected):
        FCIDipoleBackend().calculate_path(numbers, h2_scan, path_id="invalid-atoms")


@pytest.mark.optional
def test_fci_adapter_rejects_invalid_path_shape_and_empty_family_id(h2_scan):
    pytest.importorskip("pyscf")
    backend = FCIDipoleBackend()
    with pytest.raises(ValueError, match="shape"):
        backend.calculate_path([1, 1], h2_scan[:1], path_id="too-short")
    with pytest.raises(ValueError, match="family_id"):
        backend.calculate_path([1, 1], h2_scan, path_id="family", family_id=" ")


@pytest.mark.optional
def test_fci_adapter_refuses_orbitals_beyond_declared_exact_fci_budget(h2_scan):
    pytest.importorskip("pyscf")
    # H2/STO-3G has two spatial MOs; enforce the budget before SCF.
    backend = FCIDipoleBackend(max_orbitals=1)
    with pytest.raises(ValueError, match="max_orbitals"):
        backend.calculate_path([1, 1], h2_scan, path_id="too-many-orbitals")
