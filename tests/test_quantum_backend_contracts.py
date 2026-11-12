"""Backend boundaries, independent three-qubit oracles and failure reporting."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from generaldia.quantum import pennylane_backend as pl
from generaldia.quantum import qiskit_backend as qk
from generaldia.quantum.measurement import measurement_plan


@pytest.mark.parametrize("module", [pl, qk])
@pytest.mark.parametrize("shots", [True, 0, -1, 1.2, float("nan")])
def test_invalid_shot_budgets_fail_before_backend_import(module, shots):
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        module.ground_state_vqe_shots({"Z": 1}, shots=shots)


@pytest.mark.parametrize("module,dependency", [(pl, "pennylane"), (qk, "qiskit")])
def test_missing_quantum_dependency_has_actionable_error(monkeypatch, module, dependency):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == dependency or name.startswith(dependency + "."):
            raise ImportError("deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="'quantum' extra"):
        module.ground_state_vqe({"Z": 1})


@pytest.mark.optional
@pytest.mark.parametrize("backend", ["qiskit", "pennylane"])
def test_three_qubit_ghz_multiple_groups_match_stabilizer_oracle(backend):
    terms = {"III": 0.2, "ZZI": 0.3, "XXX": 0.4, "YYX": 0.5}
    plan = measurement_plan(terms)
    assert plan.n_measurement_settings == 3
    if backend == "qiskit":
        qiskit = pytest.importorskip("qiskit")
        state = qiskit.QuantumCircuit(3)
        state.h(0)
        state.cx(0, 1)
        state.cx(1, 2)
        result = qk.grouped_shot_energy(state, plan, shots=np.int64(64), seed=None)
    else:
        qml = pytest.importorskip("pennylane")

        def prepare():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([1, 2])

        result = pl.grouped_shot_energy(prepare, plan, shots=np.int64(64), seed=None)
    # GHZ is an exact +1 eigenstate of ZZI and XXX and a -1 eigenstate of YYX.
    assert result["expectations"] == pytest.approx({"III": 1, "ZZI": 1, "XXX": 1, "YYX": -1})
    assert result["energy"] == pytest.approx(0.4)
    assert len(result["counts"]) == 3
    assert all(sum(counts.values()) == 64 for counts in result["counts"])


@pytest.mark.optional
def test_three_qubit_ansatz_agrees_across_backends():
    qiskit = pytest.importorskip("qiskit")
    qml = pytest.importorskip("pennylane")
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector

    angles = np.linspace(-0.8, 0.9, 12)
    circuit, parameters = qk._build_ansatz(qiskit.QuantumCircuit, ParameterVector, 3, 2)
    state = Statevector.from_instruction(
        circuit.assign_parameters(dict(zip(parameters, angles, strict=True)))
    )

    @qml.qnode(qml.device("default.qubit", wires=3))
    def reference():
        pl._apply_ansatz(qml, angles, 3, 2)
        return qml.state()

    # Reverse Qiskit's tensor axes, not the physical gates or parameter order.
    actual = state.data.reshape(2, 2, 2).transpose(2, 1, 0).reshape(-1)
    np.testing.assert_allclose(actual, reference(), atol=1e-12, rtol=0)


@pytest.mark.optional
def test_qiskit_rejects_incompatible_preparation_and_shot_override():
    qiskit = pytest.importorskip("qiskit")
    plan = measurement_plan({"Z": 1})
    with pytest.raises(ValueError, match="qubit count"):
        qk.build_grouped_measurement_circuits(qiskit.QuantumCircuit(2), plan)
    with pytest.raises(ValueError, match="classical bits"):
        qk.build_grouped_measurement_circuits(qiskit.QuantumCircuit(1, 1), plan)
    with pytest.raises(ValueError, match="override shots"):
        qk.run_grouped_measurements(qiskit.QuantumCircuit(1), plan, run_options={"shots": 5})


@pytest.mark.optional
def test_qiskit_custom_backend_preserves_options_and_propagates_failure():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.providers.basic_provider import BasicSimulator

    calls = []

    class Backend:
        def run(self, circuits, **kwargs):
            calls.append(kwargs)
            return BasicSimulator().run(circuits, **kwargs)

    state = qiskit.QuantumCircuit(1)
    result = qk.grouped_shot_energy(
        state,
        measurement_plan({"Z": 1}),
        backend=Backend(),
        shots=32,
        seed=999,
        run_options={"seed_simulator": 13},
    )
    assert calls == [{"shots": 32, "seed_simulator": 13}]
    assert result["energy"] == 1

    class BrokenBackend:
        def run(self, *args, **kwargs):
            raise RuntimeError("backend execution failed")

    with pytest.raises(RuntimeError, match="backend execution failed"):
        qk.run_grouped_measurements(state, measurement_plan({"Z": 1}), backend=BrokenBackend())
    # An identity-only plan needs no shots or backend execution.
    identity = qk.grouped_shot_energy(state, measurement_plan({"I": 0.7}), backend=BrokenBackend())
    assert identity["energy"] == 0.7
    assert identity["counts"] == ()


@pytest.mark.optional
def test_pennylane_legacy_shot_api_uses_real_device():
    qml = pytest.importorskip("pennylane")

    class LegacyAPI:
        def __getattr__(self, name):
            if name == "set_shots":
                raise AttributeError(name)
            return getattr(qml, name)

    def circuit():
        qml.PauliX(wires=0)
        return qml.counts(wires=range(1))

    node = pl._finite_shot_qnode(LegacyAPI(), circuit, n_qubits=1, shots=32, seed=7)
    assert node() == {"1": 32}


@pytest.mark.optional
@pytest.mark.parametrize("module", [pl, qk])
@pytest.mark.parametrize("sampled", [False, True])
def test_optimizer_failure_is_reported_not_relabelled_success(monkeypatch, module, sampled):
    pytest.importorskip("pennylane" if module is pl else "qiskit")
    required = module._require()

    def failed_optimizer(objective, initial, **kwargs):
        return SimpleNamespace(
            fun=objective(initial),
            x=initial,
            nfev=1,
            success=False,
            message="evaluation budget exhausted",
        )

    monkeypatch.setattr(module, "_require", lambda: (*required[:-1], failed_optimizer))
    function = module.ground_state_vqe_shots if sampled else module.ground_state_vqe
    result = function({"I": 0.2, "Z": 0.4}, layers=1, maxiter=1)
    assert result["success"] is False
    assert result["message"] == "evaluation budget exhausted"
    assert result["objective_evaluations"] == 1
    assert np.isfinite(result["energy"])
