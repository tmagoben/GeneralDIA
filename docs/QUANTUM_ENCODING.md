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
