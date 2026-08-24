"""PennyLane state-space ground-state VQE adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .ansatz import deterministic_initial_parameters, validate_vqe_inputs


def _require() -> tuple[Any, Any]:
    try:
        import pennylane as qml
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError("install GeneralDIA with the 'quantum' extra") from error
    return qml, minimize


def _operator(qml: Any, label: str) -> Any:
    factors = []
    for wire, character in enumerate(label):
        if character == "X":
            factors.append(qml.PauliX(wire))
        elif character == "Y":
            factors.append(qml.PauliY(wire))
        elif character == "Z":
            factors.append(qml.PauliZ(wire))
    if not factors:
        return qml.Identity(0)
    operator = factors[0]
    for factor in factors[1:]:
        operator = operator @ factor
    return operator


def ground_state_vqe(
    terms: Mapping[str, complex], layers: int = 2, maxiter: int = 300
) -> dict[str, Any]:
    """Approximate the lowest eigenvalue with a small hardware-efficient ansatz."""

    real_terms, n_qubits, exact_energy = validate_vqe_inputs(terms, layers, maxiter)
    qml, minimize = _require()
    device = qml.device("default.qubit", wires=n_qubits, shots=None)
    hamiltonian = qml.Hamiltonian(
        list(real_terms.values()), [_operator(qml, label) for label in real_terms]
    )

    @qml.qnode(device)
    def energy(parameters: np.ndarray) -> Any:
        angles = np.asarray(parameters).reshape(layers, n_qubits, 2)
        for layer in range(layers):
            for qubit in range(n_qubits):
                qml.RY(angles[layer, qubit, 0], wires=qubit)
                qml.RZ(angles[layer, qubit, 1], wires=qubit)
            if n_qubits > 1:
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.expval(hamiltonian)

    initial = deterministic_initial_parameters(n_qubits, layers)
    result = minimize(
        lambda values: float(energy(values)),
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "tol": 1e-10},
    )
    value = float(result.fun)
    return {
        "energy": value,
        "exact_energy": exact_energy,
        "absolute_error": abs(value - exact_energy),
        "parameters": np.asarray(result.x),
        "objective_evaluations": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
    }
