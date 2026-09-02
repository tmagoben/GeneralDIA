import numpy as np

from generaldia.quantum.pauli import matrix_to_pauli, pauli_to_matrix


def verify_case(name: str, hamiltonian: np.ndarray, expected_terms: dict[str, float]) -> None:
    terms = matrix_to_pauli(hamiltonian, tol=1e-14)
    assert set(terms) == set(expected_terms)
    for label, expected in expected_terms.items():
        assert np.isclose(terms[label], expected, atol=1e-14)
    residual = np.linalg.norm(pauli_to_matrix(terms) - hamiltonian)
    assert residual < 1e-12
    print(name, terms)
    print(f"{name} reconstruction residual: {residual:.3e}")


two_state = np.array([[0.7, 0.23 - 0.31j], [0.23 + 0.31j, -0.1]])
verify_case("two-state", two_state, {"I": 0.3, "X": 0.23, "Y": 0.31, "Z": 0.4})

identity = np.eye(2, dtype=np.complex128)
pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_z = np.diag([1, -1]).astype(np.complex128)
four_state = (
    0.4 * np.kron(identity, identity)
    + 0.2 * np.kron(pauli_x, identity)
    - 0.15 * np.kron(pauli_y, pauli_z)
    + 0.3 * np.kron(pauli_z, pauli_z)
)
verify_case("four-state", four_state, {"II": 0.4, "XI": 0.2, "YZ": -0.15, "ZZ": 0.3})
