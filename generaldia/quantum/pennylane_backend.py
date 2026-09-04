"""PennyLane state-space VQE and grouped-shot measurement adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .ansatz import deterministic_initial_parameters, validate_vqe_inputs
from .measurement import MeasurementPlan, measurement_plan
from .shots import grouped_energy_from_counts


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


def _build_hamiltonian(qml: Any, terms: Mapping[str, float]) -> Any:
    """Build the backend Hamiltonian while preserving GeneralDIA label order."""

    return qml.Hamiltonian(list(terms.values()), [_operator(qml, label) for label in terms])


def _apply_ansatz(qml: Any, parameters: np.ndarray, n_qubits: int, layers: int) -> None:
    angles = np.asarray(parameters, dtype=float).reshape(layers, n_qubits, 2)
    for layer in range(layers):
        for qubit in range(n_qubits):
            qml.RY(angles[layer, qubit, 0], wires=qubit)
            qml.RZ(angles[layer, qubit, 1], wires=qubit)
        if n_qubits > 1:
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])


def _validate_shots(shots: int) -> int:
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer")
    return int(shots)


def _apply_measurement_basis(qml: Any, group: Any) -> None:
    for wire, gates in group.basis_changes:
        for gate in gates:
            if gate == "Sdg":
                qml.adjoint(qml.S)(wires=wire)
            elif gate == "H":
                qml.Hadamard(wires=wire)
            else:
                raise ValueError(f"unsupported measurement basis-change gate: {gate}")


def build_grouped_measurement_qnodes(
    prepare_state: Callable[[], None],
    plan: MeasurementPlan,
    *,
    shots: int = 2048,
    seed: int | None = 7,
) -> tuple[Any, ...]:
    """Build one finite-shot PennyLane QNode for each QWC measurement group."""

    shots = _validate_shots(shots)
    qml, _minimize = _require()
    qnodes = []
    for group_index, group in enumerate(plan.groups):
        device_seed = None if seed is None else int(seed) + group_index
        device = qml.device(
            "default.qubit",
            wires=plan.n_qubits,
            shots=shots,
            seed=device_seed,
        )

        def make_circuit(group: Any) -> Callable[[], Any]:
            def circuit() -> Any:
                prepare_state()
                _apply_measurement_basis(qml, group)
                return qml.counts(wires=range(plan.n_qubits))

            return circuit

        qnodes.append(qml.qnode(device)(make_circuit(group)))
    return tuple(qnodes)


def run_grouped_measurements(
    prepare_state: Callable[[], None],
    plan: MeasurementPlan,
    *,
    shots: int = 2048,
    seed: int | None = 7,
) -> tuple[dict[str, int], ...]:
    """Execute all QWC PennyLane QNodes and return one count mapping per group."""

    qnodes = build_grouped_measurement_qnodes(
        prepare_state,
        plan,
        shots=shots,
        seed=seed,
    )
    return tuple({str(key): int(value) for key, value in qnode().items()} for qnode in qnodes)


def grouped_shot_energy(
    prepare_state: Callable[[], None],
    plan: MeasurementPlan,
    *,
    shots: int = 2048,
    seed: int | None = 7,
) -> dict[str, Any]:
    """Estimate every grouped Pauli expectation and the total energy from PennyLane shots."""

    counts = run_grouped_measurements(prepare_state, plan, shots=shots, seed=seed)
    estimate = grouped_energy_from_counts(plan, counts)
    return {**estimate, "counts": counts, "measurement_plan": plan}


def ground_state_vqe(
    terms: Mapping[str, complex], layers: int = 2, maxiter: int = 300
) -> dict[str, Any]:
    """Approximate the lowest eigenvalue with an analytic expectation-value objective."""

    real_terms, n_qubits, exact_energy = validate_vqe_inputs(terms, layers, maxiter)
    qml, minimize = _require()
    device = qml.device("default.qubit", wires=n_qubits, shots=None)
    hamiltonian = _build_hamiltonian(qml, real_terms)

    @qml.qnode(device)
    def energy(parameters: np.ndarray) -> Any:
        _apply_ansatz(qml, parameters, n_qubits, layers)
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


def ground_state_vqe_shots(
    terms: Mapping[str, complex],
    layers: int = 2,
    maxiter: int = 100,
    *,
    shots: int = 2048,
    grouping_method: str = "largest_first",
    exact_max_terms: int = 20,
    seed: int = 7,
) -> dict[str, Any]:
    """Approximate the ground state using the shared ansatz and grouped finite-shot energy."""

    shots = _validate_shots(shots)
    real_terms, n_qubits, exact_energy = validate_vqe_inputs(terms, layers, maxiter)
    qml, minimize = _require()
    plan = measurement_plan(real_terms, method=grouping_method, exact_max_terms=exact_max_terms)
    evaluation = 0

    def energy(values: np.ndarray) -> float:
        nonlocal evaluation

        def prepare_state() -> None:
            _apply_ansatz(qml, values, n_qubits, layers)

        counts = run_grouped_measurements(
            prepare_state,
            plan,
            shots=shots,
            seed=seed + evaluation,
        )
        evaluation += 1
        return float(grouped_energy_from_counts(plan, counts)["energy"])

    initial = deterministic_initial_parameters(n_qubits, layers)
    result = minimize(
        energy,
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "tol": 1e-4},
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
        "shots": shots,
        "measurement_settings": plan.n_measurement_settings,
        "grouping_method": grouping_method,
    }
