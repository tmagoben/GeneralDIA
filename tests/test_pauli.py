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


def test_one_qubit_coefficients_match_closed_form() -> None:
    diagonal_0 = 0.7
    diagonal_1 = -0.1
    real_coupling = 0.23
    y_coefficient = 0.31
    hamiltonian = np.array(
        [
            [diagonal_0, real_coupling - 1j * y_coefficient],
            [real_coupling + 1j * y_coefficient, diagonal_1],
        ]
    )

    terms = matrix_to_pauli(hamiltonian, tol=1e-14)

    expected = {
        "I": (diagonal_0 + diagonal_1) / 2,
        "X": real_coupling,
        "Y": y_coefficient,
        "Z": (diagonal_0 - diagonal_1) / 2,
    }
    assert terms == pytest.approx(expected, abs=1e-14)
    assert terms["Y"] == pytest.approx(-hamiltonian[0, 1].imag)


def test_two_qubit_coefficients_match_independent_tensor_construction() -> None:
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    expected = {"II": 0.4, "XI": 0.2, "YZ": -0.15, "ZZ": 0.3}
    hamiltonian = (
        expected["II"] * np.kron(identity, identity)
        + expected["XI"] * np.kron(pauli_x, identity)
        + expected["YZ"] * np.kron(pauli_y, pauli_z)
        + expected["ZZ"] * np.kron(pauli_z, pauli_z)
    )

    terms = matrix_to_pauli(hamiltonian, tol=1e-14)

    assert terms == pytest.approx(expected, abs=1e-14)


def test_two_qubit_labels_lock_left_to_right_kronecker_order() -> None:
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    high_order_factor = matrix_to_pauli(np.kron(pauli_x, identity), tol=1e-14)
    low_order_factor = matrix_to_pauli(np.kron(identity, pauli_x), tol=1e-14)

    assert high_order_factor == pytest.approx({"XI": 1.0}, abs=1e-14)
    assert low_order_factor == pytest.approx({"IX": 1.0}, abs=1e-14)
    assert not np.array_equal(pauli_matrix("XI"), pauli_matrix("IX"))


def test_tolerance_prunes_small_coefficients_with_a_bounded_residual() -> None:
    small_coefficient = 5e-13
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    hamiltonian = np.eye(2, dtype=np.complex128) + small_coefficient * pauli_x

    terms = matrix_to_pauli(hamiltonian, tol=1e-12)

    assert terms == pytest.approx({"I": 1.0}, abs=1e-14)
    assert np.linalg.norm(pauli_to_matrix(terms) - hamiltonian) == pytest.approx(
        np.sqrt(2) * small_coefficient
    )


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
