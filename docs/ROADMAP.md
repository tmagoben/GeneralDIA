# GeneralDIA development roadmap

This roadmap sequences scientific contracts before architectural expansion. A version
is complete only when its implementation, negative controls, documentation, and
reproducible examples pass together. Later-version code must not be used to weaken or
bypass an earlier validation gate.

## v3.2: path-aware supervision

Goal: make connected geometries first-class training data without leaking related
structures across evaluation partitions or assigning unsupported state identities.

Planned scope:

- path-aware datasets and deterministic whole-path train, validation, and test splits;
- overlap-guided permutation, phase, and degenerate-subspace tracking;
- covariant transformation of every state-indexed target;
- tracked supervision with losses invariant to admitted permutation, phase, and
  subspace-gauge choices;
- leakage tests at both path-identifier and higher-level molecular-family boundaries;
- visual evidence for assignments, confidence thresholds, and ambiguity decisions.

The v3.2.0.dev2 milestone adds the remaining core supervision layer: independently
tracked predictions, block-invariant losses, a complete-path trainer, explicit
molecular-family splits, and duplicate-geometry checks. Adversarial real/complex
tests cover permutation, phase, degenerate rotations, finite-difference gradients,
and incorrect state continuation. See [the loss contract](PATH_INVARIANT_LOSS.md).

The v3.2 change was reviewed and merged in PR #6; its core and optional-backend
workflows passed. The v3.3 work rechecked the merged baseline locally and adds a
limited molecular-reference closeout using physical LiH energies, dipoles and
adjacent overlaps. That is integration evidence, not nonadiabatic certification.
Block split/merge transitions remain fail-closed.

## v3.3: minimal shared latent operators

Goal: learn the smallest inspectable model that emits a Hamiltonian and dipole
operators in one shared latent basis.

Planned scope:

- a common molecular representation with separate Hermitian Hamiltonian and dipole
  heads;
- real-symmetric and complex-Hermitian modes with identical public contracts;
- one declared rotation from the shared latent basis to reported adiabatic operators;
- energy, gradient, derivative-matrix, and dipole supervision under the v3.2 tracking
  and loss conventions;
- ablations against the existing Hamiltonian-only reference model.

The release gate requires exact Hermiticity, gauge-covariant operator transformations,
deterministic checkpoints, and held-out-path results. This milestone does not require a
large attention architecture.

The v3.3.0.dev1 implementation supplies these interfaces and local checks, including
spatially covariant neutral dipoles and cross-operator moment supervision. The
fixed-budget molecular comparison uses held-out geometries within LiH, not unseen
families. Derivative/dipole coupling is currently validated on synthetic references.
See [the operator contract](SHARED_OPERATORS.md) and
[the validation record](V33_VALIDATION.md). This remains a development milestone;
v3.4 certification and broader molecular validation are separate gates.

## v3.4: scientific-boundary certification

Goal: certify the finite-state model exactly where nonadiabatic claims are most likely
to fail. Here **CI means conical intersection**, not continuous integration.

Planned scope:

- conical-intersection tests for local gap topology and state/subspace behavior;
- nonadiabatic-coupling numerator checks before any energy-gap division;
- closed-loop Berry-phase or holonomy tests with gauge-equivalent positive controls
  and topology-breaking negative controls;
- finite-state-boundary tests that expose omitted-state sensitivity rather than
  silently treating a truncated manifold as complete;
- machine-readable certification records linking every supported claim to its test,
  tolerance, dataset provenance, and software revision.

Passing these tests establishes only the documented finite-state and geometry-domain
claims. It does not by itself establish trajectory or chemical accuracy.

## v3.5: learned Hamiltonians on quantum backends

Goal: connect a learned finite-state Hamiltonian to the existing convention-locked
Pauli and grouped-measurement layers.

Planned scope:

- a runnable learned-Hamiltonian-to-Pauli example with exact matrix reconstruction;
- analytic, exact-state, and finite-shot energy comparisons using one stored model
  checkpoint and one declared basis/qubit ordering;
- grouped-shot benchmarks reporting term count, measurement-setting count, shot
  budget, statistical error, and reconstruction residual;
- real and complex finite-state cases with explicit Pauli-$Y$ and endianness checks;
- reproducible backend metadata and fail-closed dimension validation.

This release benchmarks finite-state encoding and measurement. It does not claim
quantum advantage or replace the v3.4 scientific certification.

## Later work

After v3.5, evidence may justify either a fuller attention-based molecular architecture
or integration with nonadiabatic trajectories. Those are separate decisions: neither
is implied by completion of the finite-state learning and quantum-measurement pathway.

Trajectory integration would require additional conservation, timestep-convergence,
surface-transition, decoherence, and long-time stability evidence. A larger
architecture would require controlled scaling and accuracy comparisons against the
minimal v3.3 model.
