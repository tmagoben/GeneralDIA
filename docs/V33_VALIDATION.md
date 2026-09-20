# v3.3.0.dev1 validation record

Local validation on 2026-09-20 uses Python 3.12.14, Torch 2.14.0+cpu, NumPy 2.5.3,
and PySCF 2.14.0. Scientific calculations use float64 and one CPU thread.

## Prior-version check

The starting point is merged v3.2 commit
`08b55b08d2fad2c4c67d95822c18e422cb0437be` (PR #6). A separate clean worktree passes
152 tests with 8 optional quantum tests skipped. The additional pass compared with
the previous 151/9 record is the now-installed PySCF RHF check. No further baseline
runtime defect was found in these checks. The pre-v3.2 fixes remain documented in
[the prior audit](AUDIT_2026_09_19.md).

This iteration also documents an existing scope limitation: componentwise derivative
spectral losses do not generally preserve spatial rotation invariance. Their behavior
is unchanged; new dipole/cross-operator terms use full Cartesian tensors instead.

## Fixed molecular experiment

Run `python examples/pyscf/03_lih_shared_operators.py` with the PySCF extra installed.
The reference is neutral LiH, all-electron singlet FCI/STO-3G, with two states and
convergence tolerance $10^{-12}$. Energies include nuclear repulsion. Dipoles include
electronic and nuclear terms; adjacent overlaps include cross-geometry AO integrals.

| Partition/path | Bond lengths (angstrom) |
| --- | --- |
| Train, short | 1.20, 1.25, 1.30 |
| Train, long | 1.80, 1.90, 2.00 |
| Validation | 1.35, 1.40, 1.45 |
| Test | 1.55, 1.60, 1.65 |

All paths belong to **one LiH molecular family**. ID and duplicate-geometry guards
pass. This deliberate within-molecule split measures held-out bond-length
interpolation; it is not a family-disjoint or unseen-molecule experiment. Both train
segments contain three points, so equal path weighting also gives equal point weights.

All three runs start from identical Hamiltonian parameters and use 250 epochs,
Adam learning rate 0.005, seed 23, hidden width 8 and 6 radial basis functions spanning
0.5–3.0 angstrom. Hamiltonian diagonal initialization uses mean training energies
only. The final epoch is fixed before evaluation; neither validation nor test selects
a checkpoint. Dipole supervision uses weight 0.1 and scale 2 e*bohr. The two shared
runs also begin with identical dipole-head parameters. No reference derivative
targets are available, so the molecular cross-operator weight is zero.

| Model/objective | Energy MAE (hartree) | Permanent-dipole mean vector error (e*bohr) | Transition-strength MAE (e²*bohr²) |
| --- | ---: | ---: | ---: |
| v3.2 Hamiltonian only, energy | 0.00289034 | — | — |
| Shared model, energy only | 0.00289034 | 1.83643 | 0.535337 |
| Shared model, energy + dipole | 0.00126130 | 0.0736907 | 0.0543653 |

Energy MAE averages both states at the three test geometries. Permanent error is
the mean Euclidean norm of each state's vector error. Transition strength is
$\sum_c |\mu_{c,01}|^2$; it is invariant to individual state phases and spatial
rotation. It is not an oscillator strength, which also contains an energy factor.
The unsupervised dipole head is an ablation control, not a molecular prediction claim.
Identical Hamiltonian-only energy results verify that adding the unused head does
not change the baseline training calculation in this setup.

Every model checkpoint reload reproduces energies exactly; both shared checkpoints
also reproduce reported dipoles exactly in this CPU environment. The script saves
raw reference tensors, provenance, split membership, source and data SHA-256 hashes,
settings, histories, component losses, physical errors and checkpoint metadata.
Generated tensors/checkpoints stay outside Git. Results from other versions or
hardware can vary and should be recorded as separate runs.

## Numerical evidence

- Exact Hermiticity for real and complex heads; orthogonal rotations/reflections,
  translations and atom permutations preserve the declared model transformation laws.
- Dipole coordinate derivatives agree with central differences at step $10^{-6}$,
  using $10^{-9}$ absolute and $10^{-5}$ relative tolerances.
- Independent complex phases, state permutations, degenerate U(2) rotations and
  constant latent unitary changes preserve joint loss and parameter gradients.
  Exact-degeneracy gradient checks use central differences at step $10^{-6}$ with
  $2\times10^{-8}$ absolute and $2\times10^{-5}$ relative tolerances.
- Negative controls detect inconsistent relative dipole components and a dipole-only
  sign change relative to a fixed derivative operator. Missing targets and charge
  mismatches raise; the former ambiguity policy remains in force.
- FCI permanent and transition dipoles agree with independent finite-field energy
  and wavefunction response at field step $2\times10^{-5}$, within $2\times10^{-6}$
  e*bohr. H2 overlaps agree with explicit two-electron determinant products within
  $10^{-9}$; naive coefficient dot products differ by more than 0.01 in the control.
- Independently recomputed translated/rotated neutral LiH preserves energy within
  $10^{-9}$ hartree and dipole descriptors within $10^{-7}$ absolute/relative tolerance.
- The analytic complex example reduces validation total loss from 0.0989831 to
  0.00225704 and checks exact checkpoint restoration of energies, derivatives and
  dipoles. Units are synthetic; this is integration evidence, not chemical accuracy.

The local suite passes **173 tests**, with **8 optional quantum tests skipped** and
**89.57% coverage** in this environment. Hosted multi-platform and optional-backend
CI results are reported separately on the pull request.

Ruff lint/format checks and wheel/source-distribution builds pass. All eight core
workflow examples and the two configured PySCF workflow examples pass locally.

## Interpretation

One seed and twelve minimal-basis LiH geometries support an integration result and
one controlled ablation. They do not establish chemical accuracy, unseen-family
generalization, or reproducible superiority across seeds. Coupled derivative/dipole
validation is synthetic so far. The model's homonuclear/planar expressivity limits,
moment identifiability limits, and lack of SOC/Kramers constraints are explicit in
[the operator contract](SHARED_OPERATORS.md). Broader molecular references and the
roadmap's conical-intersection, Berry-phase and omitted-state tests remain necessary
before nonadiabatic application claims.
