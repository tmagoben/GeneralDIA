# v3.4 slice 3 — omitted-state sensitivity

**Contract:** Given a finite Hermitian reference Hamiltonian H and a retained
orthonormal N-by-k frame Q expressed in the **same Hilbert space**, determine
whether its k-state Rayleigh–Ritz truncation is consistent with a selected
k-state reference eigenspace. This does not identify any states **missing from
the supplied reference** and does not establish molecular completeness.

## What is checked

`certify_state_truncation(H, Q, ...)` reports four independent diagnostics:

1. **Energy discrepancy:** maximum absolute difference between ascending
   eigenvalues of Q†HQ and chosen full-reference eigenvalues (lowest k by
   default; another target requires explicit ascending state indices).
2. **Omitted-state population:** the square of the largest singular value of
   (I−QQ†) Ψ_target. This is the squared sine of the largest principal angle
   between the selected full eigenspace and the retained space. It is invariant
   to internal unitary gauges, including rotations within a degenerate target.
3. **Off-manifold residual:** spectral norm of (I−QQ†)HQ, measuring whether H
   leaks the retained basis into omitted directions. It is invariant under
   rotations of Q and has the energy units of H.
4. **Manifold isolation:** the minimum exact full-reference energy difference
   between selected and unselected roots. A difference below `target_gap_floor`
   makes the result unresolved and **always prevents passing**.

All four gates are conjunctive. This distinguishes small energy error from
genuine retained-space fidelity; an energy-only result cannot certify
truncation. The reference must contain at least one omitted state, and double
precision, orthonormal columns, finite matrices, and Hermiticity are required.
Thresholds are explicitly caller-selected and have no universal chemical
interpretation.

## Discriminating analytic control

Start with H=diag(0, 0.06, 0.12), Q=(e0,e1). A unitary mixing the first and
third basis states by 0.3 radians yields an H' with **exactly the same full
eigenvalues**. Yet retaining the *original* Q omits a weight sin²(0.3), about
0.0873, from the lowest exact eigenvector. The truncated energy error remains
less than 0.012, so that deliberately permissive energy-only test passes while
the population and residual tests reject H'. The unmixed reference passes.

Further controls test complex global and retained-frame unitary changes,
explicit non-lowest manifolds, exact boundary degeneracy, invalid input
matrices, state indices and thresholds.

## Reproduce

```bash
pytest tests/test_truncation.py -q
python examples/13_omitted_state_certification.py
```

The example emits JSON with revision, explicit tolerances, diagnostics and
positive/negative-control outcomes. It belongs to the core CI example gate.

## What remains unverified

The full-state Hamiltonian is supplied, not independently proven complete.
No molecular-family transfer, physical cross-geometry overlap, SOC/Kramers
structure, derivative-coupling or trajectory accuracy is certified. Near a
selected/unselected degeneracy the target subspace is unresolved rather
than silently declared safe. Geometry-resolved molecular enlargement,
conical-intersection convergence and provenance-pinned certification
records remain separate v3.4 work.
