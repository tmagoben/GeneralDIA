# Changelog

## Unreleased

- Added backend-independent reconstruction of grouped Pauli expectation values and
  Hamiltonian energies from shared finite-shot bitstring histograms.
- Added Qiskit and PennyLane grouped-shot execution helpers, including explicit
  cross-backend Pauli/bitstring-ordering regression tests.
- Added finite-shot grouped VQE paths while preserving the existing analytic VQE
  objective as the deterministic ideal-state reference.
- Added qubit-wise commuting Pauli measurement grouping using the paper's
  Largest-First conflict-graph coloring heuristic plus guarded exact minimum coloring
  for small term sets.
- Added tensor-product measurement-basis construction, local `H`/`Sdg` basis-change
  schedules, and explicit no-shot handling for the all-identity term.
- Added a regression oracle reproducing the two-group optimum for Eq. (7) of
  Verteletskyi, Yen, and Izmaylov (2020).
- Added overlap-based connected-path state tracking with complex phase correction,
  numerically degenerate-subspace Procrustes alignment, and fail-closed ambiguity
  diagnostics.
- Added covariant transformations for path-indexed state matrices and an independent
  two-state crossing example.
- Added adversarial two-, three-, and four-state regression tests for permutations,
  phases, degenerate rotations, assignment ties, and low-overlap failures.

## 3.0.1

- Locked the one-qubit Pauli coefficient normalization and Pauli-$Y$ sign against an
  independent closed-form oracle.
- Locked two-qubit labels to left-to-right Kronecker order with analytic `XI`, `IX`,
  `YZ`, and `ZZ` checks.
- Added backend-native PennyLane and Qiskit matrix reconstruction tests that do not
  depend on variational-optimizer convergence.
- Expanded the finite-state encoding example and documentation with explicit two- and
  four-state mappings, dimension requirements, and coefficient-pruning semantics.

## 3.0.0

- Added validated molecular datasets and electronic-structure result checks.
- Added observable-weighted losses, deterministic training, evaluation, and checkpoints.
- Vectorized unordered-pair feature construction.
- Added input, dtype, device, Hermiticity, and degeneracy validation.
- Added an end-to-end synthetic training experiment.
- Rebuilt the documentation around data shapes, units, claim boundaries, and reproducibility.
- Expanded tests, Python-version CI, package metadata, coverage, Ruff, and pre-commit checks.
- Renamed the quantum scope in documentation to finite-state ground-state VQE.

Version 3 changes error handling and validation. Code that relied on silent coercion or
undefined phase alignment may need updates.
