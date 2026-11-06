# Physical H3+ excited-singlet conical-intersection certificate

**Stage:** v3.4.0.dev3, finite-basis molecular scientific development gate.

This benchmark is the first GeneralDIA certification that uses an actual
molecular FCI reference *and* a symmetry-required conical intersection. It is
**not** a molecular nonadiabatic trajectory, derivative-coupling accuracy,
SOC/Kramers, complete-basis-limit, or globally valid diabatization claim.

## Independent physical hypothesis

At an equilateral D3h geometry, the first two H3+ excited singlets transform
as the two components of an E' representation. Their degeneracy is protected
by symmetry, and two independent nuclear E-type distortions lift that
degeneracy linearly. See Wang & Mazziotti, PCCP 26 (2024), 11491-11497,
[DOI 10.1039/D4CP00391H](https://doi.org/10.1039/D4CP00391H), Section 3.1;
and Barragán et al., JCP 124 (2006), 184303,
[DOI 10.1063/1.2193516](https://doi.org/10.1063/1.2193516).

We use *all-electron restricted singlet FCI/STO-3G* for H3+ (three protons,
two electrons, charge +1), with the four lowest singlet energies. The
physical electronic-state overlaps are independently established by the
existing cross-AO, cross-orbital, FCI-determinant overlap adapter; they are
not overlaps of coefficient arrays in unrelated molecular bases.

## Two-dimensional nuclear branching coordinates

Let the three fixed-atom-label H-H bond lengths, ordered 01, 02, 12, be
`(a,b,c)`, with mean R = 0.875 angstrom. Parameterize

```text
(a,b,c) = R(1,1,1) + x(2,-1,-1)/sqrt(6) + y(0,1,-1)/sqrt(2).
```

The two zero-sum distortion vectors are orthogonal and span the E-mode
branching plane. A deterministic, centroid-centered planar triangle is
reconstructed from the law of cosines; invalid triangles fail closed.
Coordinates are in angstrom. The equilateral CI point is x=y=0; a
counterclockwise loop has (x,y)=r(cos phi,sin phi), r=0.06 angstrom, and
samples **exclude** a duplicate endpoint. Closure is instead the physical
direct last-to-first overlap of the actual FCI wavefunctions.

## Acceptance checks

- At (0,0), the singlet E1-E2 gap must be <= 1e-6 hartree. The E0-E1
  and E2-E3 gaps remain above 1e-3 hartree on every requested loop.
- In both orthogonal distortion directions, the E1-E2 gap at r should be
  approximately twice the gap at r/2 (permitted ratio 1.3 to 2.7).
  This is a local check of nonzero first-order branching, *not* a
  numerical derivative-coupling tensor.
- On closed FCI molecular loops of 8, 16, 32 geometries, *each* of the
  two excited adiabatic states has Berry phase pi modulo 2 pi, with
  circular phase error <= 0.06 radian relative to the independent
  symmetry reference. All successive and coarse-to-finest differences
  must also be <= 0.06 radian.
- Every selected-state physical overlap must be a contraction and have
  minimum singular value >= 0.65. Every loop has an E1-E2 gap > 1e-4
  hartree. FCI energy root ordering, lower/upper isolation, and actual
  physical closing links are tested.
- An independently recomputed *non-enclosing molecular loop* with center
  (x,y)=(4r,0) must have trivial Berry phase.
- Recalculating the same molecular loop in reverse order must reverse
  orientation covariantly, and independent complex U(1) gauge factors
  at every geometry must preserve the Berry phase.
- An adversarial energy-only model deliberately uses the *same* adiabatic
  eigenvalue sequence as the molecular loop but sets all its overlap
  links to identity. Its phase is zero and cannot satisfy the
  independent pi reference. These synthetic links are explicitly
  **not** presented as physical wavefunction overlaps.

All conditions return explicit booleans in a `H3PlusCIEvidence` object;
one failure marks the benchmark **failed**. The complete real PySCF
calculation is an executable CI gate rather than a skipped optional test.

## Reproducibility and provenance

```bash
python -m pip install -e ".[dev,pyscf]"
pytest tests/test_h3plus_geometry.py tests/test_h3plus_ci.py -q
python examples/15_h3plus_molecular_ci.py > h3plus_ci_evidence.json
```

The JSON includes nuclear-geometry SHA-256 digests for each loop, all phase
and gap evidence, numerical thresholds, method, PySCF version, repository
revision `GITHUB_SHA`, and the full per-control pass/fail breakdown.
GitHub Actions archives it as a downloadable `h3plus-physical-ci-evidence`
artifact for 90 days and fails if required output is missing.

## Scientific limits and next steps

- The model uses one small basis and a fixed bond-length seam cross-section;
  test radii, mean bond lengths, and larger bases should be varied before
  extrapolation beyond this benchmark.
- The finite four-root gap isolation check does not prove physical
  completeness of *all* molecular states beyond that explicit window.
- `wilson_loop` and the Abelian refinement gate check a rank-one isolated
  state on a closed path; degeneracy *at the center* is deliberately not
  used as a pointwise state-resolved derivative coupling.
- Neither this holonomy result nor the finite-state truncation checks
  validates nonadiabatic forces, electronic decoherence, propagators,
  conical-intersection branching dynamics, or Berry-phase-aware quantum
  wavepacket trajectories. Those require subsequent independent oracles.
