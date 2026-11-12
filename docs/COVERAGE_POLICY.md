# Coverage policy: core versus PySCF-installed scientific suite

## Why we report two values

GeneralDIA's **core** GitHub Actions matrix installs only `.[dev]`
(Ubuntu Python 3.10–3.13 and Windows Python 3.11).
PySCF-dependent tests are skipped there, but the coverage tool still sees
the PySCF adapter and H3+ FCI certification module in the measured source
set. This is a valid measure of how much the *dependency-light core suite*
executes, **not** the testability of the entire molecular reference stack.

The **full scientific** job installs `.[dev,pyscf]` and executes the *same
complete pytest suite*, including FCI cross-geometry overlaps, charged
H3+ dipoles, electronic references, and actual molecular conical-intersection
holonomy. Its numerical result is a separate measure of the physical
scientific implementation. The repository continues to run a dedicated
real-PySCF molecular CI job and the separate quantum optional-backend job.

The two coverage figures must never be combined or directly subtracted
as if they were obtained under identical installed-dependency conditions.

## Baseline, improvement and gates

| Check | v3.4.0.dev3 baseline | Coverage-test PR first run | Minimum CI gate |
| --- | ---: | ---: | ---: |
| Core-only, Ubuntu Python 3.11 | 84.11% | 87.38% | 87% |
| Full scientific, PySCF-installed Python 3.11 | Not previously measured in the same manner | 93.28% | 92% |

The **core +3.27 percentage-point increase** comes from actual additional
negative-control and numerical-contract tests; it is a like-for-like
comparison. The 93.28% PySCF number is **not** a 9.17-point core gain:
it additionally covers molecular FCI code deliberately skipped by the
dependency-light runner.

The initial 93.28% run passed 347 tests with 8 optional quantum skips
(GitHub Actions `optional-backends` run 42167171549).
The 87.38% core run passed 324 tests with 31 skips
(GitHub Actions `core` run 42167391190).

Coverage includes statement and branch coverage using the project's
`[tool.coverage.run]` configuration. In particular, three optional
integration wrappers are **already omitted** from the coverage universe:
`electronic_structure/pyscf_backend.py`,
`quantum/pennylane_backend.py`, and `quantum/qiskit_backend.py`.
These exclusions predate this PR and have **not been expanded** to inflate
the percentage. Consequently, even the full PySCF percentage is a measure
of the *configured source universe*, not of every Python file in the repo.
Dev6 adds a separate combined-backend gate for these wrappers; see below.

## Reproduce

```bash
# Dependency-light core suite; fail on <87%.
python -m pip install -e ".[dev]"
pytest --cov=generaldia --cov-report=term-missing --cov-fail-under=87

# In a fresh environment with the physical molecular backend; fail on <92%.
python -m pip install -e ".[dev,pyscf]"
pytest --cov=generaldia --cov-report=term-missing \
  --cov-report=xml:coverage-pyscf.xml --cov-fail-under=92
```

The PySCF job uploads `coverage-scientific-pyscf` XML to its GitHub Actions
run. The full job's test count and coverage percentages are stamped into
CI logs. A missing artifact or failing threshold must be investigated
rather than worked around with `# pragma: no cover` or source omissions.

## Highest-priority remaining holes

Dev6 adds `.coveragerc.all` and the `all-backends` job: the complete package
source universe is measured with all optional dependencies installed, with a
92% total gate and a 90% gate for each previously omitted wrapper. The older
core/PySCF figures above retain their original source scope and remain
separate measurements. See [dev6 validation](DEV6_BACKEND_VALIDATION.md).

The next scientific coverage iteration should focus on:

1. The SCF/FCI nonconvergence, missing-root and nonfinite-closing-link
   rejection paths are now covered by controlled real-PySCF fault injection.
   Further electronic-reference error paths remain useful test opportunities.
2. Joint-loss and nonadiabatic state-tracking edge cases under
   spatial/electronic gauge transformations.
3. Quantum Pauli bit-ordering and measurement ambiguity negative controls.
4. Broader dependency-version compatibility for the three optional wrappers;
   the combined `.[all]` environment now measures their coverage directly.

Coverage is only one quality signal. An internally consistent but
scientifically incorrect molecular Hamiltonian can have 100% line coverage;
independent numerical oracles, symmetry, gauge covariance, CI gap
branching, physical overlap holonomy, and reproducibility still determine
scientific acceptance.
