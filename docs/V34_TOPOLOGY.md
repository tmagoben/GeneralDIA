# v3.4 development slice 1: closed-loop topology

This is analytic topology **diagnostic evidence**, not complete nonadiabatic or molecular certification.

## Input contract and convention

For a path with at least three geometries, provide *physical* cyclic cross-frame overlaps S[k] = <U[k]|U[(k+1) mod N]> with shape (N, m, m), **including the last-to-first link**. m is the dimension of a consistently selected electronic-state subspace. Direct CI-coefficient inner products from distinct AO/MO/determinant bases are invalid without cross-basis overlaps. Use closed_loop_overlaps(frames) only for orthonormal frames in one common ambient basis, such as the finite-state model.

Each link is polar-unitarized by SVD: S = U diag(s) V†, Q = U V†. The cyclic Wilson product W = Q[0] ... Q[N-1] transforms under local U(m) gauges as W → G[0]† W G[0]. Its eigenvalues, and eigenphases -arg(eig(W)) on [-pi, pi), are gauge invariant. For m=1 the phase is the Abelian Berry phase, modulo 2*pi; for m>1 report Wilson eigenphases, **not** arbitrary individual degenerate-state Berry phases. Reversing loop orientation conjugates eigenvalues.

The default singular-value floor is 0.5; near-singular links fail closed. Singular values above 1 beyond dtype-aware tolerance also fail, because physical cross-frame overlaps are contractions. The tolerance is a diagnostic threshold, not a universal guarantee. The API cannot determine whether an arbitrary overlap array omits the closing link.

## Scientifically discriminating controls

- On (x,y)=(cos(phi),sin(phi)), H_CI = x sigma_z + y sigma_x encloses a real two-state conical intersection, giving ground-state Berry phase pi modulo 2*pi.
- H_trivial = sigma_z gives Berry phase 0, despite matching the CI Hamiltonian's eigenvalues (-1,+1) at every sampled geometry. Energy agreement alone cannot certify topology.
- A complex spinor loop checks a nonquantized Berry phase, pointwise U(1) gauge invariance and orientation reversal.
- A varying two-dimensional subspace in a three-dimensional ambient space checks Wilson eigenvalues under independent complex U(2) gauge changes and basepoint conjugacy.
- Invalid frames, unphysical contractions and low-rank links raise rather than produce plausible numerical results.

## Reproduction

```bash
python -m pytest tests/test_topology.py -q
python examples/11_berry_loop.py
```

The example emits machine-readable JSON: analytic provenance, pass/fail, tolerance, singular-value margin, geometry SHA-256, and commit revision when GITHUB_SHA is provided.

## Boundaries and remaining v3.4 work

This slice does **not** certify molecular nonadiabatic coupling, omitted electronic states, dynamics, or chemical accuracy. Next gates: (1) coupling-numerator checks before division by small gaps, (2) state-truncation sensitivity, (3) CI gap/topology grid convergence with physical cross-geometry molecular overlaps, and (4) revision-pinned certification records. PR #7 remains independently gated.
