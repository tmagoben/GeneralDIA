import numpy as np
import pytest

from generaldia.quantum.ansatz import validate_vqe_inputs
from generaldia.quantum.measurement import measurement_plan
from generaldia.quantum.pauli import exact_ground_energy, matrix_to_pauli

HAMILTONIAN = np.array([[0.2, 0.03 - 0.04j], [0.03 + 0.04j, -0.1]])


def two_qubit_hamiltonian() -> np.ndarray:
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    return (
        0.4 * np.kron(identity, identity)
        + 0.2 * np.kron(pauli_x, identity)
        - 0.15 * np.kron(pauli_y, pauli_z)
        + 0.3 * np.kron(pauli_z, pauli_z)
    )


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


@pytest.mark.optional
def test_pennylane_two_qubit_operator_matches_dense_matrix() -> None:
    qml = pytest.importorskip("pennylane")
    from generaldia.quantum.pennylane_backend import _build_hamiltonian

    expected = two_qubit_hamiltonian()
    real_terms, n_qubits, _ = validate_vqe_inputs(
        matrix_to_pauli(expected, tol=1e-14), layers=1, maxiter=1
    )
    operator = _build_hamiltonian(qml, real_terms)

    assert n_qubits == 2
    assert np.allclose(qml.matrix(operator, wire_order=range(n_qubits)), expected, atol=1e-12)


@pytest.mark.optional
def test_qiskit_two_qubit_operator_matches_dense_matrix() -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp

    from generaldia.quantum.qiskit_backend import _build_operator

    expected = two_qubit_hamiltonian()
    real_terms, n_qubits, _ = validate_vqe_inputs(
        matrix_to_pauli(expected, tol=1e-14), layers=1, maxiter=1
    )
    operator = _build_operator(SparsePauliOp, real_terms)

    assert n_qubits == 2
    assert np.allclose(operator.to_matrix(), expected, atol=1e-12)


@pytest.mark.optional
def test_qiskit_grouped_shots_preserve_label_and_bitstring_order() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    from generaldia.quantum.qiskit_backend import grouped_shot_energy

    terms = {"II": 0.1, "YI": 0.2, "IZ": 0.3, "YZ": 0.4}
    plan = measurement_plan(terms, method="exact")
    state = QuantumCircuit(2)
    state.h(1)
    state.s(1)

    result = grouped_shot_energy(state, plan, shots=128, seed=19)

    assert plan.n_measurement_settings == 1
    assert result["counts"] == ({"00": 128},)
    assert result["expectations"] == pytest.approx({"II": 1.0, "YI": 1.0, "IZ": 1.0, "YZ": 1.0})
    assert result["energy"] == pytest.approx(1.0)


@pytest.mark.optional
def test_pennylane_grouped_shots_reconstruct_shared_expectations() -> None:
    qml = pytest.importorskip("pennylane")

    from generaldia.quantum.pennylane_backend import grouped_shot_energy

    terms = {"II": 0.1, "YI": 0.2, "IZ": 0.3, "YZ": 0.4}
    plan = measurement_plan(terms, method="exact")

    def prepare_state() -> None:
        qml.Hadamard(wires=0)
        qml.S(wires=0)

    result = grouped_shot_energy(prepare_state, plan, shots=128, seed=19)

    assert plan.n_measurement_settings == 1
    assert result["counts"] == ({"00": 128},)
    assert result["expectations"] == pytest.approx({"II": 1.0, "YI": 1.0, "IZ": 1.0, "YZ": 1.0})
    assert result["energy"] == pytest.approx(1.0)


@pytest.mark.optional
def test_qiskit_grouped_shot_vqe_smoke() -> None:
    pytest.importorskip("qiskit")
    from generaldia.quantum.qiskit_backend import ground_state_vqe_shots

    result = ground_state_vqe_shots({"I": 0.1, "Z": 0.4}, layers=1, maxiter=4, shots=32)
    assert np.isfinite(result["energy"])
    assert result["measurement_settings"] == 1
    assert result["shots"] == 32


@pytest.mark.optional
def test_pennylane_grouped_shot_vqe_smoke() -> None:
    pytest.importorskip("pennylane")
    from generaldia.quantum.pennylane_backend import ground_state_vqe_shots

    result = ground_state_vqe_shots({"I": 0.1, "Z": 0.4}, layers=1, maxiter=4, shots=32)
    assert np.isfinite(result["energy"])
    assert result["measurement_settings"] == 1
    assert result["shots"] == 32
