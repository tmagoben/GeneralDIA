# Changelog

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
