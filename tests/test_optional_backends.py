import numpy as np
import pytest

from generaldia.quantum.pauli import exact_ground_energy, matrix_to_pauli

HAMILTONIAN = np.array([[0.2, 0.03 - 0.04j], [0.03 + 0.04j, -0.1]])


@pytest.mark.optional
def test_pennylane_vqe_if_installed() -> None:
    pytest.importorskip("pennylane")
    from generaldia.quantum.pennylane_backend import ground_state_vqe

    result = ground_state_vqe(matrix_to_pauli(HAMILTONIAN), maxiter=500)
    assert abs(result["energy"] - exact_ground_energy(HAMILTONIAN)) < 1e-5


@pytest.mark.optional
def test_qiskit_vqe_if_installed() -> None:
    pytest.importorskip("qiskit")
    from generaldia.quantum.qiskit_backend import ground_state_vqe

    result = ground_state_vqe(matrix_to_pauli(HAMILTONIAN), maxiter=500)
    assert abs(result["energy"] - exact_ground_energy(HAMILTONIAN)) < 1e-5
