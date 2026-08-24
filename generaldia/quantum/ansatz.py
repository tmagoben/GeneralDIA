"""Shared validation and initialization for optional VQE backends."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .pauli import exact_ground_energy, pauli_to_matrix


def parameter_count(n_qubits: int, layers: int = 2) -> int:
    """Return the number of RY/RZ angles in the shared ansatz."""

    if int(n_qubits) < 1 or int(layers) < 1:
        raise ValueError("n_qubits and layers must be positive integers")
    return 2 * int(n_qubits) * int(layers)


def deterministic_initial_parameters(n_qubits: int, layers: int = 2) -> NDArray[np.float64]:
    """Return reproducible nonzero initial angles for backend comparisons."""

    return np.linspace(0.07, 0.31, parameter_count(n_qubits, layers))


def validate_vqe_inputs(
    terms: Mapping[str, complex], layers: int, maxiter: int
) -> tuple[dict[str, float], int, float]:
    """Validate a Hermitian Pauli mapping and VQE iteration settings."""

    if int(maxiter) < 1:
        raise ValueError("maxiter must be a positive integer")
    matrix = pauli_to_matrix(terms)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-8):
        raise ValueError("Pauli terms must define a Hermitian Hamiltonian")
    real_terms = {}
    for label, coefficient in terms.items():
        coefficient = complex(coefficient)
        if abs(coefficient.imag) > 1e-10:
            raise ValueError("Hermitian Pauli coefficients must be real")
        real_terms[label] = float(coefficient.real)
    n_qubits = len(next(iter(real_terms)))
    parameter_count(n_qubits, layers)
    return real_terms, n_qubits, exact_ground_energy(matrix)
