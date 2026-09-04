import pennylane as qml
from qiskit import QuantumCircuit

from generaldia.quantum import measurement_plan
from generaldia.quantum.pennylane_backend import (
    grouped_shot_energy as pennylane_grouped_energy,
)
from generaldia.quantum.qiskit_backend import grouped_shot_energy as qiskit_grouped_energy

terms = {"II": 0.1, "YI": 0.2, "IZ": 0.3, "YZ": 0.4}
plan = measurement_plan(terms, method="exact")
assert plan.n_measurement_settings == 1

# Qiskit: GeneralDIA factor 0 maps to physical qubit 1 for a two-qubit circuit.
qiskit_state = QuantumCircuit(2)
qiskit_state.h(1)
qiskit_state.s(1)  # |+Y> on GeneralDIA's left Pauli factor
qiskit_result = qiskit_grouped_energy(qiskit_state, plan, shots=256, seed=19)

# PennyLane: GeneralDIA factor positions map directly to wire indices.
def prepare_pennylane_state() -> None:
    qml.Hadamard(wires=0)
    qml.S(wires=0)  # |+Y> on GeneralDIA's left Pauli factor


pennylane_result = pennylane_grouped_energy(
    prepare_pennylane_state,
    plan,
    shots=256,
    seed=19,
)

for name, result in (("Qiskit", qiskit_result), ("PennyLane", pennylane_result)):
    assert result["counts"] == ({"00": 256},)
    assert result["energy"] == 1.0
    print(name)
    print("  counts:", result["counts"])
    print("  expectations:", result["expectations"])
    print("  energy:", result["energy"])
