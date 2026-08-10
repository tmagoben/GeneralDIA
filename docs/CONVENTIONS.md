# Mathematical conventions

## Adiabatic eigensystem

$$
H(R) U(R)=U(R)E(R), \qquad U^\dagger U=I.
$$

Columns of $U$ are adiabatic eigenvectors.

## Hamiltonian derivative matrix element

For Cartesian component $R_{A\alpha}$,

$$
N_{ij}^{A\alpha}
=
\left\langle \phi_i\middle|
\frac{\partial H}{\partial R_{A\alpha}}
\middle|\phi_j\right\rangle.
$$

For nondegenerate states and a differentiable gauge,

$$
\tau_{ij}^{A\alpha}
=
\frac{N_{ij}^{A\alpha}}{E_j-E_i}.
$$

The code exposes $N_{ij}$ directly and does not divide by a small gap unless the user
explicitly asks it to.

## PySCF scaled NAC convention

PySCF SA-CASSCF defines `state=(ket, bra)` and returns
$\langle bra|\partial ket\rangle$. With `mult_ediff=True`, PySCF returns its
energy-difference-scaled NAC quantity. GeneralDIA stores that quantity as
`scaled_nac_pyscf` and does **not** silently relabel it as $N_{ij}$, because sign and
index conventions must remain explicit at the data boundary.

## Pauli labels

Internal Pauli labels are written left-to-right in matrix Kronecker order. Qiskit numbers physical qubits little-endian, but `SparsePauliOp` strings are already written in left-to-right matrix/Kronecker order, so GeneralDIA passes the Pauli string through unchanged. The leftmost matrix factor corresponds to Qiskit's highest-numbered qubit.
