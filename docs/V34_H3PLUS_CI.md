# Physical H3+ excited-singlet conical-intersection certificate

**Stages:** v3.4.0.dev3 single-basis certificate; v3.4.0.dev5 independent
STO-3G / 6-31G comparison and offline evidence validation.

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

We use *all-electron restricted singlet FCI* for H3+ (three protons,
two electrons, charge +1), with the four lowest singlet energies. The
physical electronic-state overlaps are independently established by the
existing cross-AO, cross-orbital, FCI-determinant overlap adapter; they are
not overlaps of coefficient arrays in unrelated molecular bases. The original
certificate uses STO-3G; the comparison reruns every calculation in 6-31G.

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

Independent robustness tests repeat the physical CI certificate at mean bond
0.90 angstrom and loop radius 0.045 angstrom (8, 12, 16 mesh points),
and verify that applying a fixed cyclic permutation of the identical protons
to all nuclear geometries leaves the FCI energies and Berry holonomy unchanged.

All conditions return explicit booleans in a `H3PlusCIEvidence` object;
one failure marks the benchmark **failed**. The complete real PySCF
calculation is an executable CI gate rather than a skipped optional test.

## Reproducibility and provenance

```bash
python -m pip install -e ".[dev,pyscf]"
pytest tests/test_h3plus_geometry.py tests/test_h3plus_ci.py -q
python examples/15_h3plus_molecular_ci.py > h3plus_ci_evidence.json
```

The JSON includes nuclear-geometry SHA-256 digests for each loop, phase and
gap summaries, method, PySCF version, repository
revision `GITHUB_SHA`, and the full per-control pass/fail breakdown.
GitHub Actions archives it as a downloadable `h3plus-physical-ci-evidence`
artifact for 90 days and fails if required output is missing.

The repository also pins a PySCF 2.15 **computed regression baseline** in
`tests/goldens/h3plus_fci_sto3g_reference.json`. It includes four
equilateral singlet energies, both branching gap ratios, the minimum
on-loop E1-E2 gap, reference phases and smallest selected-state overlap.
The regression test compares these with documented nonzero numerical
tolerances. This snapshot is derived from our own earlier real FCI run,
so it protects against implementation drift but **does not constitute
a separate experimental or electronic-structure method oracle**.
The originating CI run and artifact IDs are recorded in the JSON.

## Two-basis evidence and offline validation

The comparison uses R=0.875 angstrom, r=0.06 angstrom and meshes (8,16,32)
in both bases. Charge +1, spin 0, four singlet roots, SCF/FCI `conv_tol=1e-11`
and the eight-orbital safety cap are unchanged. Each basis is evaluated by
a separate backend instance. There is no cross-basis reuse of CI coefficients,
energies or overlap matrices and no fallback to STO-3G if 6-31G fails.
Thresholds are explicit in the comparison JSON, alongside per-basis settings,
results, pass status and the shared-geometry digest check.

Measured with Python 3.12, NumPy 2.3.5, PyTorch 2.15.0+cpu and PySCF 2.15.0
on 2026-11-11 (one OpenMP thread):

| Basis | x gap at r/2 (Eh) | x gap at r (Eh) | y gap at r/2 (Eh) | y gap at r (Eh) |
| --- | ---: | ---: | ---: | ---: |
| STO-3G | 0.0377772290 | 0.0750179974 | 0.0381000273 | 0.0763157801 |
| 6-31G | 0.0156160136 | 0.0309055969 | 0.0157809634 | 0.0315625232 |

| Basis | x ratio | y ratio | Minimum selected-state link | Both enclosing phases | Nonenclosing phase | Checks |
| --- | ---: | ---: | ---: | --- | --- | --- |
| STO-3G | 1.98579937 | 2.00303741 | 0.91324794 | pi modulo 2 pi | 0 | 10/10 pass |
| 6-31G | 1.97909644 | 2.00003780 | 0.91112611 | pi modulo 2 pi | 0 | 10/10 pass |

The equilateral excited gaps were below 1e-14 hartree in both calculations.
The topology survives this basis change while the gap magnitudes change
substantially; this is not an energy-convergence study.

Run the two calculations serially, then check the evidence without installed
scientific packages (POSIX shell; in PowerShell set `$env:OMP_NUM_THREADS="1"`
before running the Python commands):

```bash
OMP_NUM_THREADS=1 python examples/15_h3plus_molecular_ci.py > h3plus_ci_evidence.json
OMP_NUM_THREADS=1 python examples/16_h3plus_basis_comparison.py > h3plus_basis_comparison.json
python -I -S scripts/check_h3plus_report.py h3plus_ci_evidence.json h3plus_basis_comparison.json
```

All three commands returned exit code 0 in the local validation. The last
command uses only the Python standard library: `-I -S` removes environment
path injection and site-package loading. A deliberately failing copy with
6-31G ground-state isolation set to zero returned exit code 1 and named
`6-31g.checks.ground_and_higher_root_isolation` as the failed required check.

The checker accepts the original dev3 single-basis report and the versioned
comparison format. It rejects duplicate keys, nonfinite numbers, absent
required checks, nonboolean flags, incompatible units, changed thresholds,
inconsistent phase convergence/gap ratios and mismatched geometry digests.
It independently recomputes the scientific decisions from the supplied
measurements. It cannot establish that measurements came from an authentic
FCI calculation; physical regeneration remains a separate mandatory CI step.

The comparison snapshot in `tests/goldens/h3plus_basis_comparison_reference.json`
is a computed regression baseline, not an independent experimental oracle.
It records the code revision and SHA256 fingerprints of the numerical source
files. The original STO-3G snapshot and single-basis report format remain
unchanged. CI regenerates both reports, checks them offline and archives both
in `h3plus-physical-ci-evidence`.

## Scientific limits and next steps

Local dev5 validation passed: core-only **455 passed, 37 skipped, 88.65%**
configured branch-aware coverage; PySCF-installed **484 passed, 8 skipped,
93.62%** coverage. The unchanged gates are 87% and 92%, respectively. Ruff
lint/format and wheel/sdist builds passed. These are separate environments;
hosted Windows/Python-matrix and quantum-backend results remain separate checks.

- The comparison uses two small bases and a fixed bond-length seam cross-section;
  further radii, mean bond lengths, and larger bases should be varied before
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
- The learned-Hamiltonian gate in issue #12 remains open: symmetry-covariant
  E-mode representation, physical-path training, held-out paths and an
  energy-matched topology-wrong learned-model control are not supplied here.
