import numpy as np
import pytest

from generaldia.quantum.ansatz import (
    deterministic_initial_parameters,
    parameter_count,
    validate_vqe_inputs,
)
from generaldia.quantum.pauli import (
    exact_ground_energy,
    matrix_to_pauli,
    pauli_matrix,
    pauli_to_matrix,
)


def random_hermitian(dimension: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(dimension, dimension)) + 1j * generator.normal(
        size=(dimension, dimension)
    )
    return (matrix + matrix.conj().T) / 2


@pytest.mark.parametrize("dimension", [2, 4, 8])
def test_pauli_roundtrip(dimension: int) -> None:
    hamiltonian = random_hermitian(dimension, dimension)
    terms = matrix_to_pauli(hamiltonian, tol=1e-14)
    assert np.allclose(pauli_to_matrix(terms), hamiltonian, atol=1e-12)


def test_known_pauli_matrix_order() -> None:
    assert np.array_equal(pauli_matrix("XI"), np.kron(pauli_matrix("X"), pauli_matrix("I")))


def test_exact_ground_energy() -> None:
    hamiltonian = np.diag([0.4, -0.2])
    assert exact_ground_energy(hamiltonian) == pytest.approx(-0.2)


def test_shared_vqe_configuration_validation() -> None:
    terms = matrix_to_pauli(np.diag([0.4, -0.2]))
    real_terms, n_qubits, exact = validate_vqe_inputs(terms, layers=2, maxiter=10)
    assert n_qubits == 1
    assert exact == pytest.approx(-0.2)
    assert all(isinstance(value, float) for value in real_terms.values())
    assert deterministic_initial_parameters(1, 2).shape == (parameter_count(1, 2),)
    with pytest.raises(ValueError):
        validate_vqe_inputs(terms, layers=0, maxiter=10)
    with pytest.raises(ValueError):
        validate_vqe_inputs(terms, layers=2, maxiter=0)
    with pytest.raises(ValueError):
        validate_vqe_inputs({"X": 1j}, layers=2, maxiter=10)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: pauli_matrix("A"),
        lambda: matrix_to_pauli(np.eye(3)),
        lambda: matrix_to_pauli(np.array([[0, 1], [0, 0]])),
        lambda: pauli_to_matrix({}),
        lambda: pauli_to_matrix({"X": 1, "II": 2}),
    ],
)
def test_pauli_validation(operation) -> None:
    with pytest.raises(ValueError):
        operation()
