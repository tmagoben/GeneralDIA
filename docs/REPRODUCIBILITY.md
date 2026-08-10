# Reproducibility

- Core tests use float64.
- Examples set deterministic random seeds.
- Core functionality requires no network access or external data.
- Optional backends are isolated behind lazy imports.
- PySCF metadata records method, basis, active space, state weights, and units.
- Quantum examples compare variational energies against exact NumPy eigenvalues.
