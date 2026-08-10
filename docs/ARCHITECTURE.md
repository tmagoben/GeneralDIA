# Architecture

GeneralDIA deliberately separates five layers.

1. **Geometry**: atomic numbers $Z_A$ and Cartesian coordinates $R_A$.
2. **Representation**: invariant molecular features generated from pair distances.
3. **Latent Hamiltonian**: $H_\theta(R)$, constructed to satisfy the required matrix symmetry exactly.
4. **Observables**: eigenspectrum, gradients, and matrix derivative elements.
5. **Backends**: optional PySCF reference data and optional finite-state qubit solvers.

The simple molecular encoder is intentionally not advertised as a state-of-the-art
chemical representation. Its purpose is to make every invariance claim inspectable.
More expressive equivariant encoders can replace it without changing the downstream
Hamiltonian/observable APIs.
