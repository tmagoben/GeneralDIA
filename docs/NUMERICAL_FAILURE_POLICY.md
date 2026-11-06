# Numerical failure and fallback policy

## Scope

This policy governs the `v3.4.0.dev4` development line's finite-state,
reference-geometry, matrix, training, and small-quantum validation routines.
Scientific quantities must not be fabricated when an electronic calculation
fails, a gauge cannot be identified, or data are underconstrained.

**Distinguish three outcomes:**

- **Calculated:** a finite result satisfying the stated physical and numerical
  preconditions.
- **Certified with limits:** a finite result whose accuracy or physical
  interpretation has been checked against an independent oracle or declared
  tolerance. An internally self-consistent result is *not* automatically a
  physically accurate Hamiltonian.
- **Unresolved:** raise an explicit error or return a documented failure flag;
  never substitute a guessed eigenstate, zero nonadiabatic coupling, identity
  overlap, or synthetic Berry phase.

## Deterministic, mathematically justified fallbacks

### Large relative split weights

For three **finite strictly positive** dataset split weights
`(f_train,f_val,f_test)`, use

```text
scale = max(f)
g_i = f_i / scale
p_i = g_i / sum(g)
```

rather than `p_i = f_i / sum(f)`. This is algebraically identical in exact
arithmetic, prevents intermediate overflow when finite user weights sum to
infinity in floating point, and does not alter the seeded permutation.
Explicit minimum one-item counts and the existing finite-size 3-way fallback
still apply to small datasets. NaN/Inf/zero/negative weights **fail closed**.
The regression suite checks both huge finite weights and invalid inputs.

### Analytic two-state reference energies

For the real symmetric avoided crossing `H=[[aR,c],[c,-aR]]`, use

```text
E_± = ±hypot(aR, c)
```

instead of `±sqrt((aR)^2+c^2)`. The hypot formulation avoids avoidable
overflow in intermediate squares. If even the product `aR` or the final
matrix/energy cannot fit into finite precision, fail with an explicit
overflow error rather than returning infinities. Complex or nonfinite
nuclear coordinates cannot be silently cast to real.

### Training defaults

Passing `gradient_clip_norm=None` explicitly disables gradient clipping;
positive finite clipping thresholds enable it. This is a documented
configuration choice, not an implicit response to numerical instability.
Epochs and log intervals are positive nonboolean integers. Seeds are normalized\nto Python integers in NumPy's supported interval [0, 2**32 - 1]; learning rate and
clipping controls must be finite and positive. Real/complex electronic frames\nare promoted to a common complex dtype before gauge alignment; no phase is\ndropped when a real reference is compared with a complex candidate.

## Scientific situations where fallback is **not** allowed

- **Undefined phase / small overlap:** do not align an electronic state or
  assign its Berry phase below the declared overlap floor. Reject NaN/Inf
  states and invalid floor parameters.
- **Invalid state dimension or Hermiticity:** do not truncate a fractional
  number of states to an integer, discard an imaginary part from a
  real-symmetric packing, or accept NaN/Inf matrix entries.
- **Failed SCF/FCI convergence or rank collapse:** do not replace the missing
  electronic reference with a model-predicted energy or synthetic identity
  overlaps. The optional PySCF suite and the independent molecular-loop gate
  enforce physical results.
- **Autograd versus finite differences:** the five-point stencil is an
  **independent consistency check**, not a silently substituted successful
  derivative when autograd is disconnected. The existing detached-gradient
  test requires failure even if the energy spectrum happens to match.
- **Unresolved energy gap:** a zeroed coupling used for computational
  regularization must be accompanied by the explicit invalid-pair mask;
  do not interpret it as a physical zero.
- **Reference missing / model gauge unknown:** indicate that the result
  is internally checked only. Do not claim molecular NAC accuracy without
  physical AO/MO basis-connection terms, nor learned molecular topology
  from an energy-only fit.

## Test and CI acceptance

- Core job: Python 3.10, 3.11, 3.12, 3.13 on Ubuntu and 3.11 on Windows.
  `pytest --cov=generaldia --cov-report=term-missing --cov-fail-under=87`
  with the preexisting source-omission configuration.
- Full PySCF scientific job: *entire* suite installed with
  `.[dev,pyscf]`, `--cov-fail-under=92`, and mandatory archived XML.
  This job must trigger on relevant source/tests/docs-workflow changes and
  master pushes.
- Separate molecular certification: real charged H3+ singlet FCI,
  physical cross-AO cyclic overlaps, symmetry degeneracy, independent
  branching slopes, enclosing and nonenclosing Berry phases, phase/atom
  permutation covariance, mesh convergence and pinned golden evidence.
- Regression tests: numeric limits (NaN/Inf/complex), safe stable formulas,
  explicit error types/messages, dataset seed reproducibility, physical
  Hermiticity and no silent fallback for detached derivatives.

**Coverage is not proof of absence of bugs.** This milestone accepts a
scope-bounded, evidence-backed release only after every required CI job
passes at the exact final commit. Additional large-basis molecular benchmarks,
learned-Hamiltonian topology, SOC and nonadiabatic trajectories are
separate physical certification targets.
