# An independent check on H3+ optical strengths

Can the lowest singlet absorb into the first excited singlet pair? Energy errors
and Berry phases do not answer this: keeping all four energies and replacing
every transition dipole by zero would leave the spectrum unchanged. This benchmark
targets the missing observable without claiming to have trained a dipole operator.

## What is measured

Let the columns of the Cartesian matrix $D$ be transition moments from the ground
state to the two excited states, in $e a_0$. Let $G$ be their energy operator
minus the ground energy (hartree). The unweighted line tensor and length-gauge
oscillator sum in atomic units are

$$
Q = D D^\dagger, \qquad f = \tfrac{2}{3}\mathrm{tr}(D G D^\dagger).
$$

Rotating the excited-state frame changes $D$ to $DU$ and $G$ to
$U^\dagger G U$; neither output changes. In particular, rotating a split
pair while leaving its diagonal gaps fixed is **not** a change of convention.
At the equilateral degeneracy we compare the whole band, not rootwise dipole signs.
A rigid spatial rotation instead rotates the Cartesian tensor; its trace and
the oscillator sum stay fixed. Translating the ion changes permanent dipoles but
does not add off-diagonal transition moments.

## How to reproduce

```sh
python -m pip install -e '.[dev,pyscf]'
OMP_NUM_THREADS=1 python examples/18_h3plus_transition_band.py > h3plus_transition_band.json
python -m pytest -q tests/test_transition_band.py
```

The driver exits nonzero on a failed gate. The JSON retains coordinates, per-basis
geometry hashes, the physical line tensors and strengths, reference values,
residuals, isolation gaps, and individual gate outcomes. It is independent of
the HTTP report verifier, which accepts conical-intersection evidence only.

The two routes share PySCF's one- and two-electron **AO integrals** and orbital
basis. The established route uses RHF orbitals, singlet FCI, and transition density
matrices. The reference instead orthonormalizes the AOs, builds the full two-electron
Hamiltonian on symmetric spatial products, diagonalizes it with NumPy, and constructs
the one-body position operator explicitly. This tests the solver and dipole algebra;
it is not an independent integral engine or experimental validation.

The driver compares STO-3G and 6-31G **separately** at four static arrangements:
an equilateral triangle of side 0.88125 Å, two distortions of magnitude 0.035 Å
in different bond modes, and a rotated and translated copy of the first distortion.
Each basis must pass its own four-root energy agreement (1e-8 hartree), maximum
Cartesian line-tensor error (2e-6 $e^2a_0^2$), and absolute oscillator error
(2e-6). Gates also check band isolation, a bright transition, the degenerate
center's in-plane isotropy, covariance, and rejection of the zero-dipole control.

These are finite-basis, fixed-geometry electronic calculations. They are not a
basis-set convergence study, absorption spectrum, learned transition operator,
or molecular trajectory. See the [preceding learned-topology checkpoint](DEV7_COMPATIBILITY_AND_LEARNING.md).
