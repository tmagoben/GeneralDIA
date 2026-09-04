"""Qiskit state-space VQE and grouped-shot measurement adapters."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

from .ansatz import deterministic_initial_parameters, parameter_count, validate_vqe_inputs
from .measurement import MeasurementPlan, measurement_plan
from .shots import grouped_energy_from_counts


def _require() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.providers.basic_provider import BasicSimulator
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError("install GeneralDIA with the 'quantum' extra") from error
    return QuantumCircuit, ParameterVector, SparsePauliOp, Statevector, BasicSimulator, minimize


def _build_operator(sparse_pauli_op: Any, terms: Mapping[str, float]) -> Any:
    """Build the backend operator while preserving GeneralDIA label order."""

    return sparse_pauli_op.from_list(list(terms.items()))


def _build_ansatz(
    quantum_circuit: Any, parameter_vector: Any, n_qubits: int, layers: int
) -> tuple[Any, Any]:
    parameters = parameter_vector("theta", parameter_count(n_qubits, layers))
    circuit = quantum_circuit(n_qubits)
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
    return circuit, parameters


def _validate_shots(shots: int) -> int:
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer")
    return int(shots)


def build_grouped_measurement_circuits(
    state_circuit: Any, plan: MeasurementPlan
) -> tuple[Any, ...]:
    """Append QWC basis changes and measurements to a Qiskit state-preparation circuit.

    GeneralDIA Pauli factor index ``i`` maps to Qiskit physical qubit ``n - 1 - i``.
    Measuring each physical qubit into the same-index classical bit makes Qiskit's
    displayed count strings use the same left-to-right order as GeneralDIA labels.
    """

    (
        QuantumCircuit,
        _ParameterVector,
        _SparsePauliOp,
        _Statevector,
        _BasicSimulator,
        _minimize,
    ) = _require()
    if state_circuit.num_qubits != plan.n_qubits:
        raise ValueError("state_circuit qubit count must match the measurement plan")
    if state_circuit.num_clbits != 0:
        raise ValueError("state_circuit must not contain classical bits or measurements")

    circuits = []
    for group in plan.groups:
        circuit = QuantumCircuit(plan.n_qubits, plan.n_qubits)
        circuit.compose(state_circuit, qubits=range(plan.n_qubits), inplace=True)
        for factor_index, gates in group.basis_changes:
            physical_qubit = plan.n_qubits - 1 - factor_index
            for gate in gates:
                if gate == "Sdg":
                    circuit.sdg(physical_qubit)
                elif gate == "H":
                    circuit.h(physical_qubit)
                else:
                    raise ValueError(f"unsupported measurement basis-change gate: {gate}")
        circuit.measure(range(plan.n_qubits), range(plan.n_qubits))
        circuits.append(circuit)
    return tuple(circuits)


def run_grouped_measurements(
    state_circuit: Any,
    plan: MeasurementPlan,
    *,
    shots: int = 2048,
    backend: Any | None = None,
    seed: int | None = 7,
    run_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, int], ...]:
    """Execute all QWC measurement circuits and return one count mapping per group.

    When ``backend`` is omitted, Qiskit's ``BasicSimulator`` is used and ``seed`` is
    forwarded as ``seed_simulator``. For an explicitly supplied backend, backend-specific
    options should be passed through ``run_options``; ``seed`` is not injected.
    """

    shots = _validate_shots(shots)
    circuits = build_grouped_measurement_circuits(state_circuit, plan)
    if not circuits:
        return ()

    (
        _QuantumCircuit,
        _ParameterVector,
        _SparsePauliOp,
        _Statevector,
        BasicSimulator,
        _minimize,
    ) = _require()
    options = dict(run_options or {})
    if "shots" in options:
        raise ValueError("run_options must not override shots")
    if backend is None:
        backend = BasicSimulator()
        if seed is not None:
            options.setdefault("seed_simulator", int(seed))

    result = backend.run(list(circuits), shots=shots, **options).result()
    return tuple(dict(result.get_counts(index)) for index in range(len(circuits)))


def grouped_shot_energy(
    state_circuit: Any,
    plan: MeasurementPlan,
    *,
    shots: int = 2048,
    backend: Any | None = None,
    seed: int | None = 7,
    run_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate every grouped Pauli expectation and the total energy from Qiskit shots."""

    counts = run_grouped_measurements(
        state_circuit,
        plan,
        shots=shots,
        backend=backend,
        seed=seed,
        run_options=run_options,
    )
    estimate = grouped_energy_from_counts(plan, counts)
    return {**estimate, "counts": counts, "measurement_plan": plan}


def ground_state_vqe(
    terms: Mapping[str, complex], layers: int = 2, maxiter: int = 300
) -> dict[str, Any]:
    """Approximate the lowest eigenvalue with an analytic statevector objective."""

    real_terms, n_qubits, exact_energy = validate_vqe_inputs(terms, layers, maxiter)
    (
        QuantumCircuit,
        ParameterVector,
        SparsePauliOp,
        Statevector,
        _BasicSimulator,
        minimize,
    ) = _require()
    circuit, parameters = _build_ansatz(QuantumCircuit, ParameterVector, n_qubits, layers)

    operator = _build_operator(SparsePauliOp, real_terms)

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
    (
        QuantumCircuit,
        ParameterVector,
        _SparsePauliOp,
        _Statevector,
        BasicSimulator,
        minimize,
    ) = _require()
    circuit, parameters = _build_ansatz(QuantumCircuit, ParameterVector, n_qubits, layers)
    plan = measurement_plan(real_terms, method=grouping_method, exact_max_terms=exact_max_terms)
    backend = BasicSimulator()
    evaluation = 0

    def energy(values: np.ndarray) -> float:
        nonlocal evaluation
        assignments = dict(zip(parameters, values, strict=True))
        bound = circuit.assign_parameters(assignments, inplace=False)
        counts = run_grouped_measurements(
            bound,
            plan,
            shots=shots,
            backend=backend,
            seed=None,
            run_options={"seed_simulator": seed + evaluation},
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
