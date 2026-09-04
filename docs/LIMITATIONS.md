# Limitations

## Molecular representation

The default model uses all pair distances and a sum aggregation. It has no angular,
many-body, periodic, charge-state, or spin-state features. Pair distances cannot
distinguish enantiomers. The implementation processes one molecule at a time and
scales quadratically with atom count.

## Diabatic identifiability

Adiabatic energies do not select a unique diabatic Hamiltonian. Gradient targets
constrain eigenvalue variation but do not remove all gauge freedom. Off-diagonal
state-sensitive targets need phase and subspace alignment across geometries.

The connected-path tracker can apply supplied overlap information and diagnose weak
or tied assignments. It does not generate electronic-structure overlaps or select a
unique global diabatic gauge.

`MolecularPathDataset` prevents complete paths from crossing dataset partitions, but
it does not determine whether two nominally different paths contain correlated or
duplicate geometries. Dataset construction must define that higher-level grouping.

## Degeneracies

Individual eigenvectors become gauge-sensitive near degeneracy. The derivative
coupling utility suppresses divisions below a user-set gap threshold and can return a
mask. It does not construct a smooth degenerate-subspace gauge.

The tracker aligns a degenerate block when equal-dimensional blocks are present at
both adjacent geometries. A block that splits or merges is reported as ambiguous; the
code does not continue with invented individual root identities.

The second-best assignment margin requires repeated assignment solves and is designed
for the small finite-state manifolds in GeneralDIA, not hundreds of electronic roots.

## Training scale

The reference trainer uses one geometry per optimizer step and does not provide graph
batches, distributed training, mixed precision, early stopping, schedulers, or data
streaming. It serves small experiments and reference implementations.

The reference loss compares ascending adiabatic energy ranks. State-character
energies produced by path tracking can become non-ascending through a crossing and
are not accepted as a drop-in replacement. Path-aware invariant loss integration is
not yet implemented.

## Visual reports

The HTML report communicates stored tracking evidence and contains no molecular
viewer or live electronic-structure backend. It visualizes the supplied path and
thresholds; it does not certify that the overlaps are physically correct.

## Electronic structure

The bundled PySCF symbol table supports elements H through Ca. The SA-CASSCF adapter
uses equal state weights and assumes users selected a valid active space. Production
datasets need restart handling, state tracking, and calculation-level failure logs.

## Quantum backends

Pauli expansion requires a state dimension equal to a power of two and costs
$O(4^n)$ Pauli terms for $n$ qubits. The PennyLane and Qiskit adapters target the
ground state with a small hardware-efficient ansatz. They do not implement excited
states, subspace-search VQE, noise models, error mitigation, or fermionic encodings.

## Scientific validation

The synthetic example verifies software integration. It does not establish chemical
accuracy or suitability for nonadiabatic dynamics. Each application needs external
reference data and tests designed for its geometry domain.
