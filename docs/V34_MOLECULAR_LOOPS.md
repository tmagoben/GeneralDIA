# v3.4 slice 4 — physical molecular cyclic overlaps and loop refinement

**Status:** small-basis neutral-singlet FCI overlap interface plus analytic and
molecular controls. This does **not** establish a molecular conical intersection,
nonadiabatic derivative coupling, SOC, or globally complete state manifold.

## Physical overlap contract

At a sequence of K >= 3 molecular geometries, return the K physical cyclic
electronic overlaps

```
S[k, i, j] = <Psi_i(R_k) | Psi_j(R_(k+1) mod K)>.
```

The closing link `S[K-1]` must use the electronic wavefunctions calculated at
**the original first geometry**, not an independently regenerated duplicate.
Otherwise an unaccounted phase or U(m) rotation at the duplicated endpoint
can corrupt the Wilson product and spuriously break gauge covariance.

The PySCF `FCIDipoleBackend.calculate_loop(...)` method returns a
`FCIClosedLoop` with its original `MolecularPath` and separate
`cyclic_overlaps` tensor of shape (K, m, m). Open-chain `calculate_path`
retains its existing return type and behaviour. For every link the adapter
evaluates cross-AO integrals, then the cross-orbital overlap
`C_k^T S_AO(R_k, R_l) C_l`, followed by `pyscf.fci.addons.overlap` of
CI wavefunctions/determinants. This is **not** a dot product between CI arrays
from different AO/MO bases.

The loop result can be submitted directly to the existing `wilson_loop`
SVD polar-link calculation. Link singular-value screening still applies;
a small singular value is unresolved, not a well-defined Berry phase.
Full-state U(m)-gauge transformations lead to Wilson conjugacy; eigenphases,
not eigenvectors or arbitrary per-state phases, are invariant.

## Refinement gate

`certify_abelian_loop_convergence` takes at least three `(K, S)` pairs,
with strictly increasing K and explicit K-by-1-by-1 cyclic links.
It checks all successive wrapped-phase differences **and** every coarse-to-
finest phase difference against a declared tolerance. An optional independently
justified reference phase is checked modulo 2 pi. Passing with no reference
phase means only that the supplied loops converge numerically — the provider
may still produce systematically incorrect overlaps. This function is
deliberately rank-one; non-Abelian Wilson eigenphase matching needs a
separate multiplicity-aware spectral convergence procedure.

## Positive and negative controls

1. **Analytic conical intersection:** H(x,y)=x sigma_z + y sigma_x on unit
   circles, with K = 8, 16, 32, 64. The ground-state Berry phase is pi
   modulo 2 pi; arbitrary complex U(1) phases at each point do not change it.
2. **Isospectral counterexample:** H_trivial=sigma_z has the *same two
   eigenvalues* as the circular CI example at every geometry, yet Berry
   phase zero. A pi reference rejects this perfectly energy-matched model.
3. **Finite-resolution spinor:** a complex spinor loop with known
   -2 pi sin^2(alpha/2) phase checks genuinely nonquantized convergence;
   a coarse mesh deliberately fails a strict refinement tolerance.
4. **Bad link:** a phase-corrupted cyclic link may still refine internally
   but fails an externally supplied reference phase.
5. **Physical molecular control:** neutral H2 all-electron singlet FCI/STO-3G
   on a smooth cyclic bond-length scan; K = 4, 8, 12. Last-to-first
   determinant overlaps use original endpoint wavefunctions. Because this
   path varies **one bond coordinate** and does not enclose a conical
   intersection, its reference Berry phase is zero modulo 2 pi. This checks
   electronic-structure overlap integration and refinement, not molecular
   conical-intersection topology.

The optional independent edge control recomputes an endpoint pair in reverse
and verifies the *magnitude* of the physical closing link, allowing the
arbitrary sign of independently solved real roots. The `molecular-loop`
GitHub Actions job installs the PySCF extra and explicitly executes the FCI
tests and JSON example; a skipped optional dependency is **not** success.

## Reproduce

```bash
pytest tests/test_loop_convergence.py -q
python examples/11_berry_loop.py
python -m pip install -e ".[dev,pyscf]"
pytest tests/test_fci_loop.py -q
python examples/14_physical_fci_loop.py
```

The physical example prints JSON including PySCF version, geometry SHA-256s,
sample counts, phase tolerances, convergence errors and link floors. GitHub
Actions supplies `GITHUB_SHA`; local runs use `unrecorded-local`.

## Remaining scientific limitations

- H2 has no molecular conical intersection on this path. Molecular
  conical-intersection certification requires an actual suitable polyatomic
  electronic reference, multi-coordinate loops, independent geometry
  convergence, root-manifold isolation and cross-geometry overlap screening.
- The FCI model is restricted to a small orbital basis, neutral singlet,
  even electron count, and admitted number of converged roots. It contains
  neither SOC nor Kramers structure.
- A selected finite-state subspace can exclude important states. The
  state-truncation diagnostics in [V34_TRUNCATION.md](V34_TRUNCATION.md)
  are a separate gate and require an explicitly larger reference.
- Neither a plausible Berry phase nor a converged phase proves accurate
  physical nonadiabatic couplings or trajectories.
