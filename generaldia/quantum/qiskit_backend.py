"""Qiskit state-space ground-state VQE adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .ansatz import deterministic_initial_parameters, validate_vqe_inputs


def _require() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError("install GeneralDIA with the 'quantum' extra") from error
    return QuantumCircuit, ParameterVector, SparsePauliOp, Statevector, minimize


def ground_state_vqe(
    terms: Mapping[str, complex], layers: int = 2, maxiter: int = 300
) -> dict[str, Any]:
    """Approximate the lowest eigenvalue with a small hardware-efficient ansatz."""

    real_terms, n_qubits, exact_energy = validate_vqe_inputs(terms, layers, maxiter)
    QuantumCircuit, ParameterVector, SparsePauliOp, Statevector, minimize = _require()
    parameters = ParameterVector("theta", 2 * n_qubits * layers)
    circuit = QuantumCircuit(n_qubits)
    parameter_index = 0
    for _layer in range(layers):
        for qubit in range(n_qubits):
            circuit.ry(parameters[parameter_index], qubit)
            circuit.rz(parameters[parameter_index + 1], qubit)
            parameter_index += 2
        if n_qubits > 1:
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            if n_qubits > 2:
                circuit.cx(n_qubits - 1, 0)

    # SparsePauliOp strings use left-to-right Kronecker matrix order, matching
    # GeneralDIA labels despite Qiskit's little-endian physical qubit numbering.
    operator = SparsePauliOp.from_list(list(real_terms.items()))

    def energy(values: np.ndarray) -> float:
        assignments = dict(zip(parameters, values, strict=True))
        bound = circuit.assign_parameters(assignments, inplace=False)
        state = Statevector.from_instruction(bound)
        return float(np.real(state.expectation_value(operator)))

    initial = deterministic_initial_parameters(n_qubits, layers)
    result = minimize(
        energy,
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
