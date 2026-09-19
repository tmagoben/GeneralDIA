# Invariant supervision on complete paths

The v3.2.0.dev2 checkpoint supplies `predict_path`, `path_observable_loss`,
`train_path_model`, `evaluate_path_model`, and `assert_disjoint_paths`. The existing
one-geometry API remains an ascending-energy-rank workflow.

## Ordering and continuation

The target path carries physical adjacent electronic-state overlaps. It is tracked
with the existing ambiguity policy, then ordered by its first-geometry spectrum.
Predictions are diagonalized in one fixed orthonormal latent basis; their adjacent
column-eigenvector overlaps supply a separate tracking calculation. Predicted state
character therefore follows the first predicted spectrum through later crossings.

No target-to-model eigenvector overlap is assumed, and no hidden gauge is fitted.
The target and predicted energy-block partitions must agree at every geometry.
Both trackers must accept every transition; low overlaps, tied assignments,
near-degeneracy review conditions, and block split/merge transitions fail closed.
The diagnostic `on_ambiguous="record"` mode cannot be used for supervision.

## Objective

Let $B$ and $C$ denote tracked energy-degenerate blocks, including singleton states.
For a Cartesian derivative component $A=U^\dagger(\partial H/\partial R)U$, define

$$
d_B(A)=\operatorname{eigvalsh}(A_{BB}),
\qquad
s_{BC}(A)=\operatorname{svdvals}(A_{BC}),\quad B<C.
$$

| Component | Compared descriptors |
| --- | --- |
| Energy | Ascending energies inside each tracked block |
| Energy gradient | $d_B(A)$ for each block and Cartesian component |
| Derivative matrix | All $d_B(A)$ and $s_{BC}(A)$, counting each off-block pair once |

Each component is the mean squared descriptor difference over all geometries and
descriptor entries in one path. `LossWeights` forms their weighted sum. Energies
use squared energy units; derivative descriptors use squared energy/coordinate
units. The trainer averages paths with equal weight, so longer paths do not receive
more optimizer steps. `evaluate_path_model()` reports these invariant MSEs rather
than elementwise matrix MAEs.

Under an admitted block gauge change,

$$
A_{BB}\longmapsto Q_B^\dagger A_{BB}Q_B,
\qquad
A_{BC}\longmapsto Q_B^\dagger A_{BC}Q_C.
$$

These changes preserve the descriptors. First-geometry sorting removes arbitrary
initial nondegenerate root permutations, and path tracking handles subsequent
permutations. For singleton states, diagonal descriptors are signed gradients and
off-block descriptors are coupling-numerator magnitudes.

Scalar gradient targets suffice only for nondegenerate singleton blocks. Degenerate
gradient supervision needs full derivative matrices: their diagonal alone cannot
determine the invariant directional-derivative eigenvalues. When both target forms
are supplied, their raw diagonal consistency is checked before tracking.

## Differentiation at degeneracy

Individual eigenvector derivatives contain inverse energy gaps, producing undefined
within-block terms at exact degeneracy. The path implementation uses a first-order
subspace derivative that retains only rotations between distinct energy blocks:

$$
(U^\dagger\,dU)_{ij}
=\frac{(U^\dagger\,dH\,U)_{ij}}{E_j-E_i},
\qquad i,j\text{ in distinct blocks}.
$$

Within-block gauge derivatives are set to zero because the objective is invariant
to those rotations. This rule is for invariant block objectives, not arbitrary
matrix-entry losses. The backward is explicitly first order; it does not support
Hessians. Tracking assignments and block decisions are detached discrete choices.
Derivatives are local to a fixed accepted choice, and small inter-block gaps remain
ill-conditioned. Use a physically appropriate near-degeneracy review threshold.

Tests compare parameter gradients against central differences with step $10^{-6}$
for real and complex nondegenerate and exactly degenerate models. Gradient comparison
tolerances are $2\times10^{-8}$ absolute and $2\times10^{-5}$ relative. Adversarial
target gauges preserve both loss and gradients. Exact-match degenerate controls
have finite gradients and loss below $10^{-25}$ in float64.

## Family boundaries and training

```python
from generaldia import (
    LossWeights,
    PathTrackingSettings,
    TrainingConfig,
    assert_disjoint_paths,
    train_path_model,
    evaluate_path_model,
)

train, validation, test = paths.split(seed=23, family_key="molecular_family")
assert_disjoint_paths(train, validation, test, family_key="molecular_family")
settings = PathTrackingSettings()
weights = LossWeights(energy=1.0, derivative_matrix=0.1)
history = train_path_model(
    model,
    train,
    validation_data=validation,
    weights=weights,
    settings=settings,
    config=TrainingConfig(epochs=50),
    family_key="molecular_family",
)
metrics = evaluate_path_model(model, test, weights=weights, settings=settings)
```

Every path needs a nonempty string family label when the key is supplied; at least
three families are required for a three-way split. Fractions count families, so
record actual path and sample counts. Without a family key, `split()` preserves the
earlier whole-path behavior. Families are never inferred from atomic composition.

The disjointness guard rejects shared IDs, supplied family labels, and duplicate
ordered-atom pair-distance matrices. The default absolute distance tolerance is
$10^{-8}$ in input coordinate units, with zero relative tolerance. Rotation,
translation, and renaming do not conceal a duplicate. Atom permutations and general
chemical identity are outside this screen. The reference implementation uses an
exhaustive comparison suitable for small datasets.

Before updates, the trainer checks training/validation leakage and initial path
validity. A later ambiguous optimization step raises; previously completed updates
are retained. Save checkpoints and split IDs using the existing checkpoint API.
The runnable example records tracking settings, weights, provenance groups, and all
three split memberships together.

## Evidence and claim boundary

Run `python examples/09_invariant_path_training.py`. The known synthetic crossing
model is fitted from complete paths with scrambled labels and phases. A second
model has identical pointwise spectra but the wrong state continuation and receives
a positive path loss. Repeated runs are deterministic in the tested CPU setup.

These per-component invariants discard some relative operator information. Zero
loss does not prove joint gauge equivalence, a unique diabatic Hamiltonian, chemical
accuracy, or correct nonadiabatic trajectories. Molecular-reference benchmarks,
the shared Hamiltonian/dipole architecture, and the roadmap's scientific-boundary
certification remain subsequent work.

## Local validation snapshot (2026-09-19)

- Python 3.12.14, Torch 2.14.0 CPU, float64 scientific tests.
- 151 tests passed; 9 optional-backend tests skipped because their packages were
  absent. Core coverage: 89.27%, above the repository's 80% threshold.
- Ruff lint/format, wheel/sdist builds, and all seven core workflow examples passed.
- Example held-out total loss: 0.1221697778 initially, $3.7983\times10^{-11}$ after
  training. The exact-match control scored zero; the incorrect-continuation control
  scored 1.3022222222 despite matching the pointwise energy spectra.
- Hosted multi-platform CI and optional-backend results are separate from these
  local checks. This checkpoint has not been certified on molecular reference data.
