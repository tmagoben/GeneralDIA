# Finite-state quantum encoding

If the diabatic model contains $N_s = 2^n$ states, its matrix can be expanded exactly
in the $n$-qubit Pauli basis:

$$
H \;=\; \sum_P c_P\, P, \qquad
c_P = \frac{1}{2^n}\operatorname{Tr}\!\big(P^\dagger H\big).
$$

This is an encoding of the **selected electronic-state manifold**. It can be useful
for algorithm prototyping, controlled comparisons between exact diagonalization and
variational quantum solvers, and hybrid workflows.

It should not be confused with constructing a second-quantized electronic Hamiltonian
from one- and two-electron integrals. That separate problem requires a fermionic
encoding and generally a different number of qubits.

The Pauli expansion implemented here is exact but scales exponentially with the number
of encoded states; it is therefore intended for small state manifolds.

## One-qubit coefficient oracle

For a two-state Hermitian matrix written as

$$
H = \begin{pmatrix}
a & u - iv \\
u + iv & d
\end{pmatrix},
$$

the exact expansion is

$$
H = \frac{a+d}{2}I + uX + vY + \frac{a-d}{2}Z.
$$

Therefore $c_Y=v=-\operatorname{Im}(H_{01})$ for the standard
$Y=\begin{psmallmatrix}0&-i\\i&0\end{psmallmatrix}$ convention. This closed-form
identity is tested independently of the round-trip reconstruction routine.

## State and label ordering

For four selected states, GeneralDIA uses the computational-basis mapping

$$
|s_0\rangle\mapsto|00\rangle,\quad
|s_1\rangle\mapsto|01\rangle,\quad
|s_2\rangle\mapsto|10\rangle,\quad
|s_3\rangle\mapsto|11\rangle.
$$

Pauli labels are written in left-to-right matrix Kronecker order. Thus `XI` means
$X\otimes I$, while `IX` means $I\otimes X$; these are distinct operators. Qiskit's
`SparsePauliOp` strings use the same displayed matrix order, even though Qiskit numbers
physical qubits in little-endian order.

The runnable example `examples/04_pauli_state_encoding.py` verifies analytic two-state
and four-state decompositions, including a complex $Y\otimes Z$ contribution.

## Dimension and pruning rules

`matrix_to_pauli` accepts finite Hermitian matrices with dimension $N_s=2^n$ and
$N_s\geq 2$. A non-power-of-two selected-state manifold must use a separately
documented embedding or padding convention; GeneralDIA does not add one silently.

Terms satisfying $|c_P|\leq\mathtt{tol}$ are omitted. The default
`tol=1e-12` makes the returned representation sparse when coefficients are numerically
negligible, so reconstruction is accurate only up to the discarded terms. Set a
smaller tolerance when those coefficients are scientifically meaningful.

## Qubit-wise commuting measurement groups

A Pauli expansion can contain many terms even when several terms can be measured from
the same prepared quantum state. GeneralDIA groups terms using the qubit-wise
commutativity (QWC) criterion of Verteletskyi, Yen, and Izmaylov.

Two Pauli words $P$ and $Q$ are QWC when, at every tensor-product factor, their local
operators are equal or at least one is the identity. Equivalently, for every qubit
index $k$,

$$
P_k = Q_k
\quad\text{or}\quad
P_k = I
\quad\text{or}\quad
Q_k = I.
$$

QWC is stronger than ordinary operator commutativity. For example, $XX$ and $YY$
commute as two-qubit operators but are not QWC, because each qubit would require two
different local measurement bases.

GeneralDIA constructs the **conflict graph**, the complement of the QWC graph: each
non-identity Pauli word is a vertex and an edge joins two vertices that are not QWC.
A proper coloring of this graph gives simultaneous-measurement groups. Minimizing the
number of colors is equivalent to the minimum clique cover of the QWC graph.

Use

```python
from generaldia.quantum import measurement_plan

plan = measurement_plan(terms, method="largest_first")
```

for the paper's Largest-First greedy coloring heuristic. This is the default because
it is deterministic, polynomial-time, and was among the strongest inexpensive
heuristics in the paper's molecular benchmarks. The returned groups are guaranteed
to be mutually QWC, but the heuristic does not certify the globally minimum number of
groups.

For small Pauli sets,

```python
plan = measurement_plan(terms, method="exact")
```

uses branch-and-bound DSATUR coloring to certify a minimum clique cover. Exact graph
coloring is exponential in the worst case, so GeneralDIA limits exact mode to 20
non-identity terms by default. The guard can be changed explicitly with
`exact_max_terms`.

The all-identity Pauli term is returned separately as `plan.identity_term` because its
expectation value is exactly one and requires no quantum measurement.

Each `MeasurementGroup` contains a tensor-product `basis` and the corresponding local
`basis_changes`. The convention is

- `X`: apply `H`, then measure in the computational basis;
- `Y`: apply `Sdg`, then `H`, then measure in the computational basis;
- `Z`: measure directly;
- `I`: no measurement is required for that factor.

Factor indices in `basis_changes` follow GeneralDIA's left-to-right Pauli-label order.
A backend with different physical-qubit numbering, notably Qiskit's little-endian
qubit indices, must map those factor positions to physical qubits explicitly.

The runnable example `examples/07_qwc_measurement_grouping.py` reproduces the paper's
seven-term Eq. (7) grouping into two measurement settings using both Largest-First and
exact minimum coloring.

This implementation deliberately targets QWC/tensor-product-basis measurements. It
does not group all globally commuting Pauli words with entangling Clifford basis
changes; that is a broader measurement-optimization problem with different circuit
costs and noise trade-offs.

## Grouped finite-shot execution

A QWC group needs only one sampled bitstring distribution. If a bitstring $b$ occurs
$N_b$ times in a group with $M$ total shots, GeneralDIA reconstructs each Pauli word
$P$ in that group as

$$
\widehat{\langle P\rangle}
= \frac{1}{M}\sum_b N_b
(-1)^{\sum_{k:P_k\neq I} b_k}.
$$

`grouped_energy_from_counts` applies this reconstruction to every group in a
`MeasurementPlan`, adds the all-identity coefficient exactly, and returns the total
energy, each Pauli expectation value, and the number of shots used by each group.
This reconstruction layer is backend independent.

### Qiskit

`generaldia.quantum.qiskit_backend` provides
`build_grouped_measurement_circuits`, `run_grouped_measurements`, and
`grouped_shot_energy`. A state-preparation circuit passed to these helpers must have
the same number of qubits as the plan and must not already contain classical bits or
measurements.

For an $n$-qubit GeneralDIA Pauli label, factor index $i$ maps to Qiskit physical qubit
$n-1-i$. The helper measures physical qubit $j$ into classical bit $j$, so Qiskit's
displayed count strings line up with GeneralDIA's left-to-right Pauli-label order.
This mapping is regression-tested with a deterministic $Y\otimes Z$ measurement.

With no explicit backend, execution uses Qiskit's `BasicSimulator`. A compatible
backend can instead be supplied with `backend=...`, with backend-specific execution
arguments passed through `run_options`. The helper owns the `shots` argument so it
cannot be silently overridden through `run_options`.

### PennyLane

`generaldia.quantum.pennylane_backend` provides
`build_grouped_measurement_qnodes`, `run_grouped_measurements`, and
`grouped_shot_energy`. The caller supplies a zero-argument `prepare_state` function
that emits the state-preparation operations. GeneralDIA factor positions map directly
to PennyLane wire indices, and each QWC group is sampled by its own finite-shot
`default.qubit` QNode.

The runnable optional-backend example
`examples/quantum/03_grouped_shots.py` prepares the same deterministic state in both
frameworks and verifies identical histograms, Pauli expectations, and energy.

### Finite-shot VQE

Both optional backends also expose `ground_state_vqe_shots`. It uses the same RY/RZ
ansatz as the existing `ground_state_vqe`, but each objective evaluation is assembled
from grouped finite-shot measurements rather than an analytic expectation value. The
result reports the shot count, number of measurement settings, and grouping method in
addition to the usual VQE fields.

Finite-shot optimization is stochastic: the objective can fluctuate between
iterations, and convergence is not guaranteed to be monotonic. The analytic
`ground_state_vqe` remains the deterministic ideal-state reference. The present shot
path assigns the same requested number of shots to every QWC group; variance-aware
shot allocation, readout mitigation, and hardware-noise mitigation are outside the
current implementation.
