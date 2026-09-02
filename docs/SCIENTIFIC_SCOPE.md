# Scientific scope and claim boundary

## Supported tasks

GeneralDIA supports method development for small finite electronic-state manifolds:

- symmetry-constrained real or complex Hamiltonian construction;
- invariant pair-distance molecular models;
- adiabatic energies and Cartesian derivatives;
- phase and subspace alignment utilities;
- overlap-based connected-path state tracking with explicit ambiguity diagnostics;
- PySCF reference-data adapters;
- exact Pauli expansion and small ground-state VQE comparisons.

## Meaning of a learned matrix

Suppose a model reproduces reference energies at each geometry. For any unitary
matrix $Q(R)$,

$$
H'(R) = Q^\dagger(R) H(R) Q(R)
$$

has the same eigenvalues. Energy fitting therefore identifies an equivalence class of
matrices. It does not select one global diabatic gauge.

Energy gradients add information about eigenvalue variation. Off-diagonal derivative
matrix elements or other state-sensitive observables can constrain the eigenvectors
and gauge. Degenerate subspaces retain unitary freedom, so projector quantities remain
more stable than individual eigenvectors there.

## Evidence levels

| Evidence | Supported conclusion |
|---|---|
| Matrix-construction tests | The code enforces the stated Hermitian or symmetric form. |
| Invariance tests | The pair-distance model returns the same matrix under the tested transformations. |
| Energy fit | The learned spectrum matches labels within the reported error. |
| Energy and gradient fit | The spectrum and its local geometry variation match labels. |
| Gauge-consistent off-diagonal fit | The supplied state-sensitive derivative elements match in that gauge. |
| Unambiguous overlap-tracked path | State character and covariantly transformed matrix elements are continuous over the tested path resolution. |
| Held-out trajectory tests | The model transfers to the excluded geometry paths covered by the test design. |
| Comparison with dynamics | The learned representation supports the tested dynamical observable. |

Each result should state which evidence level the experiment reached.

## Separate quantum-chemistry problems

GeneralDIA maps an $N_s=2^n$ state-space matrix to $n$ qubits. For two selected
electronic states, it uses one qubit. This encoding does not map molecular orbitals,
creation operators, or electron repulsion integrals to qubits. Jordan-Wigner and
Bravyi-Kitaev mappings solve that separate fermionic encoding problem.

## Intended maturity

Version 3 remains an alpha research package. The simple molecular representation and
one-geometry training loop favor inspection over scale. Use independent benchmarks
before applying the model to chemical prediction or dynamics.
