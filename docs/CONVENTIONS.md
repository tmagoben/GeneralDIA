# Mathematical conventions

## Adiabatic eigensystem

$$
H(R)\,U(R) = U(R)\,E(R), \qquad U^\dagger U = I.
$$

Columns of $U$ are adiabatic eigenvectors.

## Hamiltonian derivative matrix element

For Cartesian component $R_{A\alpha}$,

$$
N_{ij}^{A\alpha} =
\left\langle \phi_i \left|\n\frac{\partial H}{\partial R_{A\alpha}}\n\right| \phi_j \right\rangle.
$$

For nondegenerate states and a differentiable gauge,

$$
\tau_{ij}^{A\alpha} = \frac{N_{ij}^{A\alpha}}{E_j - E_i}.
$$

The code exposes $N_{ij}$ directly and does not divide by a small gap unless the user
explicitly asks it to.

## PySCF scaled NAC convention

PySCF SA-CASSCF defines `state=(ket, bra)` and returns
$\langle\mathrm{bra}\,|\,\partial\;\mathrm{ket}\rangle$. With `mult_ediff=True`, PySCF
returns its energy-difference-scaled NAC quantity. GeneralDIA stores that quantity as
`scaled_nac_pyscf` and does **not** silently relabel it as $N_{ij}$, because sign and
index conventions must remain explicit at the data boundary.

## Pauli labels

Internal Pauli labels are written left-to-right in matrix Kronecker order (i.e. the
left-most character corresponds to the highest-order tensor factor). Qiskit numbers
physical qubits in little-endian order, but `SparsePauliOp` strings in Qiskit are
already written in left-to-right matrix/Kronecker order, so no additional re-ordering
is performed by the code.
