# Gauge and connected-path state tracking

## Scope

Electronic eigensolvers may independently permute states and choose a real sign or
complex phase at every geometry. Inside an exactly degenerate subspace they may apply
an arbitrary unitary rotation. These choices do not change a single-geometry
spectrum, but they do change state-indexed matrix elements and can create artificial
discontinuities along a geometry path.

GeneralDIA tracks state **character**, not ascending energy rank. The implementation
does not claim to construct a unique global diabatic basis. It identifies a locally
continuous labeling when adjacent overlap information supports one and reports when
that labeling is not numerically justified.

## Overlap contract

For adjacent geometries $R_k$ and $R_{k+1}$, the input is

$$
S^{k,k+1}_{ij}
= \left\langle \phi_i(R_k) \middle| \phi_j(R_{k+1}) \right\rangle,
$$

stored with shape `(n_geometries - 1, n_states, n_states)`. The row and column orders
are the raw backend orders at their respective geometries.

`adjacent_state_overlaps(frames)` calculates these matrices when every frame is
expressed in one compatible ambient basis. GeneralDIA model eigenvectors meet that
condition because they are coordinates in one fixed finite-state basis.

Raw wavefunction coefficients from separate electronic-structure calculations do
not generally meet it. Atomic-orbital, molecular-orbital, determinant, and
configuration-interaction bases can all change with geometry. In that setting,
calculate physical cross-geometry overlaps in the electronic-structure layer and
pass those overlap matrices to `track_states`; do not take an unqualified dot product
of raw coefficient arrays.

## Tracked frame and assignment

Let $U_k$ contain the raw state columns and let $W_k$ be the transformation returned
by the tracker. The tracked frame is

$$
\widetilde U_k = U_k W_k, \qquad W_0=I.
$$

At transition $k$, the overlap from the previous tracked frame to the new raw frame
is

$$
A_k = W_k^\dagger S^{k,k+1}.
$$

For nondegenerate states, GeneralDIA finds the one-to-one assignment maximizing the
sum of absolute overlaps. It then multiplies each new state by the phase that makes
its matched overlap real and nonnegative. Assignment uses a deterministic
maximum-weight solver implemented in the core package; SciPy is not a runtime
dependency.

When `energies` identify equal-dimensional degenerate blocks at both endpoints, the
score between two blocks is the mean of their principal overlaps. Once two blocks
are matched, a unitary Procrustes rotation maximizes their frame agreement. Individual
columns within that block are gauge-dependent; the matched subspace and its projector
are the meaningful objects.

Supplying energies is required for automatic degenerate-block and near-degeneracy
detection. If energies are omitted, every state is treated as a singleton; callers
must not use that mode where unresolved degeneracies invalidate individual roots.

An energy block means that its full energy spread is no larger than the configured
`degeneracy_tolerance`. Choosing that tolerance declares the block numerically
degenerate for tracking. If the internal splitting matters at the intended physical
resolution, use a smaller degeneracy tolerance and an explicit
`near_degeneracy_threshold` instead of rotating the states as one block.

If a degenerate block splits or merges between adjacent samples, the default behavior
is to raise. GeneralDIA does not invent individual state identities at that boundary.

## Covariant state-indexed observables

Any operator or derivative matrix represented in the raw state basis must receive
the same transformation:

$$
\widetilde A_k = W_k^\dagger A_k W_k.
$$

`transform_state_matrices` applies this rule to tensors with shape
`(n_geometries, ..., n_states, n_states)`. Intermediate axes may represent atoms,
Cartesian components, or several observables. Applying a permutation or phase only
to the eigenvectors while leaving derivative matrices unchanged is inconsistent.

## Ambiguity policy

`track_states` raises `AmbiguousStateTrackingError` by default when any of these
conditions occurs:

- the smallest matched principal overlap is not above `overlap_floor` (default
  `0.5`);
- the normalized score difference between the best and next-best assignment is not
  above `assignment_margin_floor` (default `1e-6`);
- degenerate-subspace dimensions split or merge across a transition;
- a gap falls below an explicitly supplied `near_degeneracy_threshold` without being
  an admitted degenerate block.

Each exception contains its `StateTrackingStep` diagnostic. Advanced workflows may
set `on_ambiguous="record"`, but they must inspect `result.ambiguous_steps` before
using individual state labels. This opt-in records a numerical continuation, and all
later transitions inherit that choice; it does not make an ambiguous assignment
physically unique.

`degeneracy_tolerance` and `near_degeneracy_threshold` are absolute values in the same
units as the supplied energies. Report them with every result.

## Runnable crossing oracle

Run:

```bash
python examples/06_state_tracking.py
```

The example diagonalizes the exact two-state Hamiltonian

$$
H(x)=\begin{pmatrix}x&0\\0&-x\end{pmatrix}
$$

at points on both sides of the crossing. Ascending energy order swaps the raw states.
Injected complex phases add independent gauge changes. Exact overlaps recover the
constant state characters and produce machine-precision frame and energy residuals.
The exact degeneracy is deliberately not sampled; a sampled or poorly resolved
ambiguous transition must be reported rather than forced.

## Validation and claim boundary

The regression suite independently injects permutations, complex phases, and exact
$U(2)$ rotations into two-, three-, and four-state paths. It checks frame recovery,
operator covariance, principal overlaps, ambiguity failures, and forward/reverse
consistency away from ambiguous points.

This layer is sequential and local to adjacent path points. It does not calculate
electronic-structure overlaps, derivative couplings, Berry connections, a globally
optimal path assignment, or a dynamics-ready diabatic transformation. Those require
additional physical inputs and application-specific validation.

For reproducibility, record the ordered geometry identifier, overlap-generation
method, energy units, state manifold, degeneracy tolerance, overlap floor, assignment
margin floor, near-degeneracy threshold, and every recorded ambiguous transition.
