# Finite-state quantum encoding

If the diabatic model contains $N_s = 2^n$ states, its matrix can be expanded exactly
in the $n$-qubit Pauli basis:

$$
H \;=\; \sum_P c_P\, P, \qquad
c_P = \frac{1}{2^n}{Tr}\!\big(P^\dagger H\big).
$$

This is an encoding of the **selected electronic-state manifold**. It can be useful
for algorithm prototyping, controlled comparisons between exact diagonalization and
variational quantum solvers, and hybrid workflows.

It should not be confused with constructing a second-quantized electronic Hamiltonian
from one- and two-electron integrals. That separate problem requires a fermionic
encoding and generally a different number of qubits.

The Pauli expansion implemented here is exact but scales exponentially with the number
of encoded states; it is therefore intended for small state manifolds.
