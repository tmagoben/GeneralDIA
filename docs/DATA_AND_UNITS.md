# Data shapes and units

GeneralDIA does not assign implicit physical units to core tensors. A dataset must use
one coordinate unit and one energy unit throughout an experiment.

## Tensor contract

| Quantity | Symbol | Shape | Core dtype |
|---|---:|---:|---|
| Atomic numbers | $Z$ | `(N,)` | integer |
| Positions | $R$ | `(N, 3)` | real floating point |
| Hamiltonian | $H$ | `(S, S)` | real or complex floating point |
| Energies | $E$ | `(S,)` | real floating point |
| Energy gradients | $\partial E/\partial R$ | `(S, N, 3)` | real floating point |
| Hamiltonian Jacobian | $\partial H/\partial R$ | `(N, 3, S, S)` | real or complex floating point |
| Derivative matrix elements | $N_{ij}$ | `(N, 3, S, S)` | real or complex floating point |
| Dipole matrix elements | $\mu_{c,ij}$ | `(3, S, S)` | real or complex floating point |
| Adjacent state overlaps | $S^{k,k+1}$ | `(K - 1, S, S)` | real or complex floating point |

`N` denotes the number of atoms and `S` denotes the number of selected electronic
states. Eigenvectors occupy columns of the `(S, S)` eigenvector matrix.
`K` denotes the number of ordered geometries in one `MolecularPath`.

## `MolecularSample`

Create one sample for each geometry:

```python
sample = MolecularSample(
    atomic_numbers=atomic_numbers,
    positions=positions,
    energies=energies,
    energy_gradients=gradients,  # optional
    derivative_matrix_elements=numerators,  # optional
    metadata={
        "coordinate_unit": "angstrom",
        "energy_unit": "hartree",
        "gradient_unit": "hartree/angstrom",
        "source": "calculation identifier",
    },
)
```

The constructor rejects inconsistent shapes, nonfinite values, nonpositive atomic
numbers, and non-Hermitian derivative matrices.

## `MolecularPath`

A path contains at least two `MolecularSample` objects with the same atomic numbers,
atom order, and state count. Optional gradient and derivative-matrix targets must be
present at every path point or none. The overlap at index `k` connects raw state rows
at geometry `k` to raw state columns at geometry `k + 1`.

The same all-or-none rule applies to dipoles. Supply `dipole_matrix_elements` in
laboratory Cartesian axes with row=bra and column=ket, in the same raw electronic
gauge as the other operators. Record charge, dipole unit and origin in metadata.
All state matrices use the same transformation during tracking. Device transfers
preserve matrix complex dtypes; real coordinate/energy conversions do not drop
operator phases.

Coordinates and energies retain the units declared by the samples. Tracking
thresholds involving energy gaps therefore use that same energy unit.

## PySCF boundary

`ElectronicStructurePoint` stores:

- positions in angstrom;
- energies in hartree;
- nuclear gradients in hartree/bohr;
- PySCF scaled NACs with `(ket, bra)` keys.

`MolecularDataset.from_electronic_structure` keeps positions in angstrom. A model
therefore differentiates energy with respect to angstrom. The conversion applies

$$
\frac{\partial E}{\partial R_{\mathrm{angstrom}}}
=
\frac{\partial E}{\partial R_{\mathrm{bohr}}}
\frac{\mathrm{bohr}}{\mathrm{angstrom}},
$$

using `1 angstrom = 1.8897261254578281 bohr`.

`FCIDipoleBackend.calculate_path()` directly returns a `MolecularPath`, with angstrom
coordinates, hartree total energies, and e*bohr total neutral dipoles about laboratory
origin zero. It supplies physical cross-AO determinant overlaps and no nuclear
derivatives. Its provenance and conventions are described in
[the shared-operator contract](SHARED_OPERATORS.md).

## Forces and gradients

The package uses energy gradients:

$$
G = \frac{\partial E}{\partial R}.
$$

A force has the opposite sign, $F=-G$. Convert force labels before passing them as
`energy_gradients`.

## Normalization

Energy, gradient, and derivative-matrix losses carry different units and numerical
scales. Record any shift, scale, or per-state normalization with the dataset. Apply
the inverse transform before reporting physical errors.

The new dipole and joint path terms divide by fixed positive `dipole_scale` and
`derivative_scale` before forming their moment descriptors, giving dimensionless
MSEs. Choose scales before testing or from training data alone and record them.
