# Shared Hamiltonian and dipole operators

The v3.3.0.dev1 checkpoint adds a minimal neutral-molecule model with real-symmetric
or complex-Hermitian Hamiltonian and dipole heads, a common reporting frame, and
coupled operator supervision. It extends the merged v3.2 path contracts. The
Hamiltonian-only model and its existing interfaces remain available.

## Model and transformations

```python
from generaldia import SharedMolecularOperators, predict_path

model = SharedMolecularOperators(n_states=2, hidden=32, mode="complex")
operators = model.operators(atomic_numbers, positions)
H = operators.hamiltonian  # (S, S); identical to model(Z, R)
mu = operators.dipoles  # (3, S, S); laboratory Cartesian components
prediction = predict_path(model, path, derivatives=True, dipoles=True)
```

The common distance encoder produces a pooled molecular representation and local
atomic representations. The Hamiltonian head uses the pooled features. Hermitian
atomic weights $Q_a$ produce a polar vector:

$$
\mu_c(R)=\sum_a (R_{ac}-\bar R_c)Q_a(R),\qquad
\bar R=\frac{1}{N}\sum_a R_a.
$$

For an orthogonal spatial transformation $O$, including a reflection, $H(OR)=H(R)$
and $\mu_c(OR)=\sum_d O_{cd}\mu_d(R)$. Centering enforces translation invariance,
which is appropriate only for total **neutral** molecular dipoles. `charge != 0`
is rejected. A declared target/model charge mismatch also raises. Missing charge
metadata does not establish neutrality; supplying compatible targets remains the
caller's responsibility. Coordinates and dipoles have the dataset's units, so
$Q_a$ has dipole/coordinate units and is not a physical atomic charge assignment.

Both heads use one fixed electronic basis. If $W$ contains the Hamiltonian's
eigenvectors and tracked transformations, all reported operators use that same $W$:

$$
\mu_c^{\mathrm{ad}}=W^\dagger\mu_c W,\qquad
N_{a c}^{\mathrm{ad}}=W^\dagger\frac{\partial H}{\partial R_{a c}}W.
$$

Dipoles are never diagonalized separately to choose an electronic gauge. The complex
head uses $S$ real diagonal values and independent real/imaginary upper triangles,
giving $S^2$ real outputs. Network parameters remain real. Complex mode is not an
implementation of spin-orbit coupling, time reversal, or Kramers symmetry.

## Targets and losses

`MolecularSample.dipole_matrix_elements` accepts Hermitian `(3,S,S)` tensors. Every
point of a dipole-bearing path must supply them in the same raw electronic gauge as
its derivative targets and overlaps. Tracking, copying, and device transfers preserve
complex values and transform both operator families together.

Let $B,C$ be admitted energy blocks, $m=\mu/s_\mu$, and $n=N/s_N$, with fixed positive
scales recorded in `LossWeights`. Dipole descriptors contain the block trace vectors
and the **full** Cartesian Gram tensors, counting each block pair once:

$$
t_{B,c}=\frac{\operatorname{Tr}(m_{c,BB})}{|B|},\qquad
G_{BC,cd}=\frac{\operatorname{Tr}(m_{c,BC}^\dagger m_{d,BC})}{|B||C|},\quad B\le C.
$$

Joint descriptors link the Hamiltonian derivatives and dipoles:

$$
J_{BC,a c d}=\frac{\operatorname{Tr}(n_{a c,BC}^\dagger m_{d,BC})}{|B||C|},\quad B\le C.
$$

Each new component is the mean absolute-squared descriptor difference over all its
entries and path points. Electronic block transformations cancel inside the traces.
Full Cartesian vectors/tensors transform orthogonally, preserving squared distance
under a common spatial rotation of predictions and targets. Comparing separate
Cartesian spectra would lose this guarantee and relative component information.

```python
from generaldia import LossWeights, path_observable_loss

weights = LossWeights(
    energy=1.0, dipole=0.1, joint_operator=0.1, dipole_scale=2.0, derivative_scale=1.0
)
loss = path_observable_loss(model, path, weights)
loss.total.backward()
```

`joint_operator` requires full derivative-matrix **and** dipole targets. No such
matrices are inferred from scalar gradients. The dipole and joint MSEs are
dimensionless; their scales must be specified before evaluation or derived only
from training data. Existing energy/derivative losses retain their units and
semantics. Their componentwise spectral derivative terms are electronically gauge
invariant but are **not generally spatially rotation invariant**. The new terms do
not change that legacy limitation. The one-geometry `observable_loss` rejects the
new weights rather than silently ignoring them.

All ambiguity, partition, and first-order differentiation limits from
[the path-loss contract](PATH_INVARIANT_LOSS.md) still apply. These low-order moments
are partial constraints: zero loss does not prove joint gauge equivalence or a
unique latent Hamiltonian. A test with unchanged individual dipole spectra but a
flipped relative Cartesian phase is detected by $G$; a dipole-only sign change
invisible to $G$ is detected by its cross tensor with a fixed derivative operator.

## Molecular-reference adapter

```python
from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend

path = FCIDipoleBackend(basis="sto-3g", n_states=2).calculate_path(
    [3, 1], coordinates_angstrom, path_id="lih-scan", family_id="LiH"
)
tracked = path.tracked()
```

The optional PySCF adapter computes all-electron singlet FCI after an RHF reference,
without freezing core orbitals. It accepts neutral even-electron systems and rejects
failed convergence or orbital counts above the configured small-space cap. It stores
positions in angstrom, total energies in hartree, and total dipoles in e*bohr about
laboratory origin zero. Matrix rows are bras and columns are kets. Electronic
transition density contractions carry the negative electron charge; nuclear dipoles
are added on the diagonal. Metadata records basis, method, versions and conventions.

Adjacent overlaps use the determinant overlaps induced by
$C_k^\dagger S_{\mathrm{AO}}^{k,k+1}C_{k+1}$, including physical cross-geometry AO
integrals. Raw CI coefficient dot products would omit the moving orbital basis.
The implementation follows PySCF's [transition-density convention](https://pyscf.org/_modules/pyscf/fci/direct_spin1.html)
and [nonorthogonal determinant overlaps](https://pyscf.org/_modules/pyscf/fci/addons.html).
This adapter supplies no nuclear gradients or derivative-coupling numerators.

Independent checks compare permanent dipoles to finite-field energy derivatives,
transition dipoles to finite-field wavefunction response, and H2 overlaps to explicit
two-electron determinant products. Neutral translated/rotated LiH calculations also
preserve energies and transform dipole invariants correctly.

## Reproduction and limits

```bash
python examples/10_shared_operators.py
python -m pip install -e ".[pyscf]"
python examples/pyscf/03_lih_shared_operators.py
```

The core example uses an analytic complex two-state reference with synthetic units
and derivative/dipole supervision. It saves a checkpoint and verifies exact reload
of energies, derivatives and dipoles. The molecular experiment records complete
reference paths, split membership, units, source/data hashes, settings, histories,
physical metrics and checkpoints in `outputs/lih_shared_operators/`. See
[the benchmark record](V33_VALIDATION.md) for the fixed-budget comparison.

This small architecture restricts vectors to the span of centered nuclear positions.
It cannot represent out-of-plane dipoles for planar geometries. Identical atomic
environments receive identical weights: for homonuclear diatomics it predicts zero
dipoles, including transitions that can physically be allowed. These are expressivity
limits, not chemical selection rules. LiH avoids that specific symmetry obstruction.
The molecular benchmark covers one molecule, one seed, and a minimal basis. Coupled
derivative/dipole validation is synthetic so far. Conical intersections, Berry loops,
omitted-state effects, and dynamics remain later scientific validation work.
