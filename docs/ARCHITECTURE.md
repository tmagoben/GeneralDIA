# Architecture

GeneralDIA separates scientific data, invariant modeling, observable calculation, and
external backends. Each layer has a narrow tensor contract.

## Layer 1: validated data

`generaldia.dataset` defines `MolecularSample` and `MolecularDataset`. The constructors
check atomic numbers, shapes, finite values, state counts, and Hermitian derivative
targets. Dataset metadata carries unit and provenance information.

`generaldia.electronic_structure.data` defines the PySCF-facing result container. It
keeps PySCF units at the backend boundary so the conversion into model coordinates
occurs in one documented place.

## Layer 2: molecular representation

`SimpleMolecularHamiltonian` embeds atomic numbers and builds one feature vector for
each unordered atom pair. A pair contains

$$
e_i+e_j,\quad |e_i-e_j|,\quad \mathrm{RBF}(|R_i-R_j|).
$$

The network transforms each pair and sums all pair vectors. Distances enforce
translation and orthogonal-transformation invariance. Symmetric pair features plus
summation enforce atom-order invariance.

You can replace this module with a graph or equivariant model if its `forward` method
accepts `(atomic_numbers, positions)` and returns one `(S, S)` Hermitian matrix.

## Layer 3: exact matrix construction

`generaldia.matrix` maps independent neural-network outputs to real symmetric or
complex Hermitian matrices. The constructor enforces symmetry by construction rather
than a penalty term.

## Layer 4: observables

`generaldia.observables` diagonalizes the Hamiltonian and differentiates it with
respect to Cartesian coordinates. This layer returns:

- ascending adiabatic energies;
- column eigenvectors;
- energy gradients;
- the full Hamiltonian Jacobian;
- derivative matrix elements;
- gap masks and derivative couplings.

The coupling function returns a validity mask on request. Callers can then exclude
state pairs whose gap makes division unstable.

## Layer 5: losses and training

`generaldia.losses` combines available observables with explicit weights.
`generaldia.training` supplies a deterministic reference loop, held-out metrics, and
checkpoint persistence. The loop processes one geometry per optimizer step and favors
inspection over throughput.

## Layer 6: connected-path gauge handling

`generaldia.state_tracking` consumes adjacent state-overlap matrices, follows state
character with maximum-overlap assignment, and applies phase or degenerate-subspace
Procrustes alignment. It returns the complete raw-to-tracked transformation and
transition diagnostics. State-indexed matrices transform covariantly with the same
unitary. External electronic-structure backends remain responsible for calculating
physical overlaps when their orbital or configuration bases change with geometry.

## Layer 7: optional backends

`generaldia.electronic_structure.pyscf_backend` generates reference calculations.
`generaldia.quantum` converts a small state-space matrix into Pauli operators and
offers PennyLane and Qiskit ground-state VQE adapters.

Optional imports remain local to backend calls. Core users do not need PySCF,
PennyLane, Qiskit, or SciPy.
