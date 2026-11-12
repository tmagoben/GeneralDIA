# v3.4.0.dev6 backend validation

This checkpoint closes specific failure-path and coverage gaps identified in
the merged dev5 implementation. It does not implement learned-Hamiltonian
topology training or certify nuclear trajectories.

## Scientific failure tests

The FCI fault-injection tests run genuine small PySCF calculations, then
alter the convergence flag, remove one returned energy/root, or corrupt only
the physical closing overlap. They require the adapter to raise, and verify
that FCI or transition-density evaluation is never reached after an invalid
upstream reference. The nonfinite-overlap tests retain finite, independently
computed open-chain links and inject NaN/Inf only into closure.

RHF and SA-CASSCF wrapper tests also exercise nonconvergence. Independent
central finite differences of LiH energies check their nuclear gradients,
including the angstrom-to-bohr conversion; translation gives a zero total
gradient. SA-CASSCF scaled-NAC key conventions and finite output are checked,
but this is not an independent molecular derivative-coupling accuracy test.

Regression tests exposed and fixed silent fractional-atom-number truncation,
complex-coordinate casting and insufficient constructor validation. The
wrappers now require meaningful basis strings, finite positive convergence
tolerances, integral charge/spin/state/active-orbital controls, a valid active
electron capacity, and boolean ETF selection. Invalid inputs raise ValueError.
Accepted physical reference thresholds and solver methods remain unchanged.

Both STO-3G and 6-31G now run the second bond/radius and proton-permutation
tests, in addition to the original shared-geometry 8/16/32-point certificate.

## Quantum backend tests

- Three-qubit GHZ stabilizers check multiple measurement groups, X/Y/Z basis
  changes and exact expectation signs, with finite shots and no stochastic
  tolerance needed for these eigenstates.
- The two backends' three-qubit, two-layer ansatz statevectors must agree
  after explicitly reversing Qiskit's tensor-index order.
- Qiskit tests cover incompatible circuit dimensions/classical bits, shot
  override rejection, custom backend options, execution failures, and the
  identity-only no-execution path.
- PennyLane's older device-shot API branch is exercised through a proxy that
  hides `set_shots` while still running a real device. This does not substitute
  for installing and testing every historically supported PennyLane version.
- Missing dependencies, invalid shot counts and failed optimizer status are
  explicit tests. Optimizer failures must retain `success=False` and their
  diagnostic message; they are not relabelled as converged results.

## Three coverage scopes

The original core and PySCF-only coverage configurations are unchanged. Their
three optional-wrapper omissions remain visible in `pyproject.toml`; no new
omissions or coverage pragmas were added.

The new `all-backends` CI job installs `.[all]` and uses `.coveragerc.all`,
which measures all `generaldia` modules with statement and branch coverage
and no source exclusions. Its total floor is 92%. Each wrapper has an
additional 90% gate, so a well-covered core cannot hide a poorly tested
adapter. Scripts outside the package are not part of this denominator.

```bash
python -m pip install -e ".[all]"
OMP_NUM_THREADS=1 pytest --cov --cov-config=.coveragerc.all \
  --cov-report=term-missing --cov-report=xml:coverage-all.xml
coverage report --rcfile=.coveragerc.all --include="*/electronic_structure/pyscf_backend.py" --fail-under=90
coverage report --rcfile=.coveragerc.all --include="*/quantum/pennylane_backend.py" --fail-under=90
coverage report --rcfile=.coveragerc.all --include="*/quantum/qiskit_backend.py" --fail-under=90
```

CI archives `coverage-all-backends`, builds the wheel, and installs it into a
separate virtual environment sharing installed dependencies. It runs
`scripts/smoke_installed_package.py` with isolated Python from outside the
checkout. The script requires all inspected GeneralDIA imports to originate
inside that environment, then checks an analytic eigenspectrum and complex
Pauli roundtrip. This catches an accidental fallback to the editable source.

## Local validation, 2026-11-12

Python 3.12.14, NumPy 2.3.5, PyTorch 2.15.0+cpu, PySCF 2.15.0,
PennyLane 0.45.1 and Qiskit 2.6.2; `OMP_NUM_THREADS=1` for numerical runs.

| Environment / scope | Passed | Skipped | Branch-aware coverage | Gate |
| --- | ---: | ---: | ---: | ---: |
| Core only, historical configured source | 506 | 63 | 88.52% | 87% |
| All dependencies, entire package source | 569 | 0 | 94.42% | 92% |
| PySCF wrapper within the all-dependency run | — | — | 100% | 90% |
| PennyLane wrapper within the all-dependency run | — | — | 97.96% | 90% |
| Qiskit wrapper within the all-dependency run | — | — | 97.93% | 90% |

Both suites and all three wrapper gate commands returned exit 0. Core
coverage is a separate measure and is not subtracted from the all-source
percentage. Ruff lint/format, sdist/wheel builds and the installed-wheel
smoke check also returned exit 0. The initial regression run reproduced
26 input-validation failures before the fixes; the corrected focused PySCF
and quantum tests passed. Hosted platform-matrix results are reported on
the PR separately from these local measurements.

## Remaining work

Minimum-supported dependency versions, macOS and GPU execution still lack
dedicated CI jobs. The combined dependency environment does not prove every
supported version combination. Physical reference coverage is still limited
to the documented small molecules and bases; learned H3+ topology on held-out
paths remains the separate issue #12 milestone.
