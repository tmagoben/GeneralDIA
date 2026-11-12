# v3.4.0.dev7 compatibility and learned H3+ topology

This iteration starts issue #12 with a fitted, local Hamiltonian and expands
the executable compatibility matrix. The earlier physical FCI certificates
remain separate reference checks.

## Model and fit

`H3PlusEquivariantHamiltonian` uses bond distances in the fixed order
(01, 02, 12). Its fixed three-dimensional latent electronic space decomposes
into A1 and E representations under bond permutations. The A1 ground state
is a spectator; it does not mix with the E doublet. Distances are angstrom,
and Hamiltonian entries are hartree.

Let $P_A=\mathbf{1}\mathbf{1}^T/3$, $P_E=I-P_A$,
$t=(\bar r-0.875)/0.05$, and $d_i=(r_i-\bar r)/0.06$. The model is

$$
H=E_A P_A+E_E P_E+
(c_0+c_1t+c_2\lVert d\rVert^2)P_E\operatorname{diag}(d)P_E+
c_3\left[P_E\operatorname{diag}(d^2)P_E-
\frac{\lVert d\rVert^2}{3}P_E\right].
$$

Each scalar energy uses six invariant features:
$1,t,t^2,\lVert d\rVert^2,t\lVert d\rVert^2,\sum_i d_i^3$.
Sixteen coefficients are fitted: twelve by linear least squares and four
by float64 LBFGS on spectra. Seeds 0, 7 and 23 independently initialize the
four splitting coefficients. For every permutation matrix $P$,
$H(Pr)=PH(r)P^T$. Translation, rotation and spatial reflection preserve
the distances. Hamiltonian derivatives remain smooth at the equilateral
geometry; no radial normalization singularity is introduced there.

The symmetry prior imposes an E degeneracy and permits conical branching.
The experiment tests whether fitting physical spectra gives the correct
branch slopes and loop topology within a held-out local domain. It does
**not** establish that spectra identify a unique physical diabatic basis,
or that a physical gauge connection was learned. Latent model frames share
one fixed basis. Physical reference phases use cross-AO FCI overlaps;
raw CI coefficients in different orbital bases are never compared.

## Predeclared data and acceptance

Separate fits use STO-3G and 6-31G singlet FCI with four reference states,
charge +1, convergence tolerance 1e-11 and an eight-orbital cap.

- Training: mean bonds 0.850, 0.875 and 0.900 A; E-mode radii 0.030,
  0.060 and 0.090 A; twelve angles per circle with a 0.17-radian offset.
  The 108 geometries contain 36 unique sorted-bond triples: symmetry
  repetitions are disclosed and remain entirely in training.
- Held out: mean bonds 0.8625 and 0.8875 A, radii 0.045 and 0.075 A;
  a separate 0.055-A loop at 8/16/32 points; reversed orientation; an
  elliptical loop with axes 0.055/0.044 A; and a nonenclosing circle of
  radius 0.015 A centered at x=0.060 A.
- Branching: the unseen 0.8625-A equilateral geometry and independent
  positive/negative x/y distortions of 0.005 A.
- Leakage guard: compare sorted bond triples rounded to 1e-9 A, including
  proton-permutation, rigid-motion and reflection-equivalent geometries.
  Shared refinement nodes within evaluation do not enter training.

The fixed gates require energy MAE <=0.002 Eh, maximum error <=0.005 Eh,
branching-gap relative error <=10%, central gap <=1e-8 Eh, phase error
<=0.06 radians, cyclic-link strength >=0.65, isolation from S0/S3 >=0.001 Eh,
and loop S1/S2 gap >=0.0001 Eh. Both excited states must pass. Model ground
isolation is checked separately; the model has no S3 state.

Each candidate is compared with a diagonal matrix containing its own
sorted eigenvalues. This control has exactly the same energies but constant
eigenframes. It must pass the energy gates and fail the topology gate.
All six bond permutations and complex U(1) gauge changes are also tested.
These gates certify local H3+ geometry holdouts, not unseen molecules.

```bash
python -m pip install -e ".[dev,pyscf]"
OMP_NUM_THREADS=1 python examples/17_learned_h3plus_topology.py > learned.json
```

The strict JSON includes fitted coefficients, seeds, per-path physical and
model phases, link strengths, energy/gap errors, geometry/source hashes,
solver settings and dependency versions. The program exits nonzero if any
seed or basis fails. CI archives it with the physical H3+ evidence. Full
reproduction recalculates the FCI data; no generated reference is substituted.

## Compatibility checks

| Check | Configuration | Execution |
| --- | --- | --- |
| Scientific dependency floors | Python 3.10; NumPy 1.24.0; Torch 2.1.0; SciPy 1.10.0; PySCF 2.8.0; PennyLane 0.40.0; Qiskit 1.0.0 | Full suite, all-source 92% coverage gate |
| macOS | `macos-15` Apple Silicon, Python 3.11, core/dev dependencies | Core suite, lint, build and examples |
| CUDA | Trusted self-hosted Linux/x64 GPU, Python 3.11 | Manual hardware workflow |

`ci/minimum-constraints.txt` also pins Autoray 0.6.11 and Lightning 0.40.0
to retain the APIs required by PennyLane 0.40. This is one coherent minimum
scientific stack, not proof of every transitive-version combination or of
minimum versions for development tools. The ordinary jobs continue to test
the current resolver-selected dependencies. The minimum job records its
installed packages and runs `pip check`.

The minimum stack exposed a genuine Qiskit 1.0 issue: BasicSimulator does
not directly accept every gate emitted by the ansatz and measurement layer.
The adapter now transpiles circuits to BasicSimulator's supported operations,
including explicitly supplied BasicSimulator instances. Other supplied
backends retain their existing compilation responsibility. Regression tests
include composite Bell preparation, Pauli-Y changes and finite-shot VQE.

The first hosted minimum-version run also exposed noise in the SA-CASSCF
finite-difference oracle. State-specific energies are not stationary under
state-average orbital rotations; an energy threshold of 1e-10 leaves the
default orbital-gradient threshold at 1e-5. One hosted difference amplified
the residual into an 8.28e-5 Eh/bohr discrepancy. The analytic gradient under
test still uses the wrapper defaults. Its independent displaced-energy
reference now uses `conv_tol=1e-14` (orbital threshold 1e-7) and two step
sizes, 2e-4 and 1e-4 A. Both must meet the unchanged 2e-5 Eh/bohr gradient
tolerance, and the two differences must agree within 2e-6 Eh/bohr.
PySCF documents its [square-root orbital tolerance default](https://pyscf.org/pyscf_api_docs/pyscf.mcscf.html).
Local tighter-convergence/step studies gave about 1e-7 Eh/bohr errors;
the change tightens the reference instead of relaxing acceptance.

CUDA tests compare CPU/GPU eigenvalues, coordinate/parameter gradients and
one optimizer step in float32/float64, plus complex128 Berry holonomy.
Ordinary CPU runs explicitly skip these three tests. The manual workflow
sets `GENERALDIA_REQUIRE_CUDA=1`, so missing hardware fails rather than skips.
It records the hardware environment and JUnit results. A trusted GPU runner
must be registered before dispatch; this change provisions no hardware.
macOS CPU success is not MPS validation, and these GPU checks do not certify
all package operations or quantum-provider hardware.

## Remaining scientific work

Issue #12 remains open. A1--E mixing, supervision by additional physical
observables, independent molecular derivative-coupling accuracy, broader
geometry domains and molecule-family generalization are deferred. Correct
local Berry phase alone does not validate state populations, SOC, forces
through a degeneracy, or nonadiabatic trajectories.

## Local validation, 2026-11-12

| Environment | Passed | Skipped | Branch-aware coverage |
| --- | ---: | ---: | ---: |
| Current all-backend stack, Python 3.12.14 | 601 | 3 CUDA | 94.52% of all package source |
| Minimum scientific stack, Python 3.10.22 | 601 | 3 CUDA | 94.45% of all package source |
| Core only, Python 3.12.14 | 534 | 67 optional + 3 CUDA | 88.79% of historical core scope |

The current stack used NumPy 2.3.5, Torch 2.15.0+cpu, PySCF 2.15.0,
PennyLane 0.45.1 and Qiskit 2.6.2. All three optional-wrapper 90% gates
passed. Ruff lint/format checks, 30 documentation tests, sdist/wheel builds
and the isolated installed-wheel smoke test passed. The minimum environment
also passed dependency metadata checks. Setting `GENERALDIA_REQUIRE_CUDA=1`
on this CPU machine produced the expected three fixture errors, confirming
that a designated GPU run cannot silently pass by skipping.

All three seeds passed the physical holdouts in each basis on both dependency
stacks. On the current stack, the initial experiment measured worst-seed
held-out energy MAE about 2.60e-5 Eh (STO-3G) and 1.55e-5 Eh (6-31G);
maximum errors were below 8.58e-5 and 4.92e-5 Eh, respectively. Maximum
branching-gap relative errors were below 0.105% and 0.018%. Every isospectral
control passed the energy criteria and failed the topology criterion.

These are local CPU results. Hosted macOS results are recorded separately
on the PR; no local macOS, MPS or CUDA execution is claimed.
