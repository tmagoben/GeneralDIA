"""Exact Pauli decompositions for small finite-state Hamiltonians."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

IDENTITY = np.array([[1, 0], [0, 1]], dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
OPS = {"I": IDENTITY, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}


def pauli_matrix(label: str) -> NDArray[np.complex128]:
    """Return the Kronecker-order matrix represented by a Pauli label."""

    if not label or any(character not in OPS for character in label):
        raise ValueError("label must contain one or more characters from I, X, Y, Z")
    result = np.array([[1.0 + 0j]])
    for character in label:
        result = np.kron(result, OPS[character])
    return result


def _n_qubits(dimension: int) -> int:
    if dimension < 2:
        raise ValueError("matrix dimension must be at least two")
    n_qubits = round(math.log2(dimension))
    if 2**n_qubits != dimension:
        raise ValueError("matrix dimension must be a power of two")
    return n_qubits


def matrix_to_pauli(hamiltonian: ArrayLike, tol: float = 1e-12) -> dict[str, complex]:
    """Expand a power-of-two Hermitian matrix in the complete Pauli basis."""

    if tol < 0:
        raise ValueError("tol cannot be negative")
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.isfinite(hamiltonian).all():
        raise ValueError("hamiltonian must contain finite values")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-10, rtol=1e-8):
        raise ValueError("hamiltonian must be Hermitian")

    n_qubits = _n_qubits(hamiltonian.shape[0])
    terms: dict[str, complex] = {}
    for characters in itertools.product("IXYZ", repeat=n_qubits):
        label = "".join(characters)
        operator = pauli_matrix(label)
        coefficient = np.trace(operator.conj().T @ hamiltonian) / (2**n_qubits)
        if abs(coefficient) > tol:
            terms[label] = complex(coefficient)
    return terms


def pauli_to_matrix(terms: Mapping[str, complex]) -> NDArray[np.complex128]:
    """Reconstruct a dense matrix from a nonempty Pauli coefficient mapping."""

    if not terms:
        raise ValueError("terms cannot be empty")
    n_qubits = len(next(iter(terms)))
    if n_qubits < 1:
        raise ValueError("Pauli labels cannot be empty")
    hamiltonian = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for label, coefficient in terms.items():
        if len(label) != n_qubits:
            raise ValueError("Pauli labels must have a consistent length")
        if not np.isfinite(complex(coefficient)):
            raise ValueError("Pauli coefficients must be finite")
        hamiltonian += complex(coefficient) * pauli_matrix(label)
    return hamiltonian


def exact_ground_energy(hamiltonian: ArrayLike) -> float:
    """Return the lowest eigenvalue of a finite Hermitian matrix."""

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-8):
        raise ValueError("hamiltonian must be Hermitian")
    return float(np.linalg.eigvalsh(matrix)[0].real)
