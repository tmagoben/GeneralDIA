# Changelog

## Unreleased

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
