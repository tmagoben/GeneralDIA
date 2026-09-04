# Path-aware targets and visual diagnostics

## Why paths are first-class data

Neighboring geometries from one scan or trajectory are statistically and physically
connected. Randomly splitting those geometries can place nearly identical structures
in training and test sets. It can also discard the adjacency needed to resolve
permutations, phases, and degenerate-subspace rotations.

`MolecularPath` keeps an ordered sample sequence, a stable `path_id`, and optional
physical adjacent-state overlaps together. `MolecularPathDataset.split()` assigns
complete paths to train, validation, and test partitions. Only after splitting should
each partition be flattened with `as_dataset()` for the reference one-geometry
trainer.

```python
from generaldia import MolecularPath, MolecularPathDataset

paths = MolecularPathDataset(
    MolecularPath(
        samples,
        path_id=path_identifier,
        adjacent_overlaps=overlaps,
        metadata={"energy_unit": "hartree"},
    )
    for path_identifier, samples, overlaps in imported_paths
)

training_paths, validation_paths, test_paths = paths.split(seed=23)
training_data = training_paths.as_dataset()
```

Flattened samples retain `generaldia_path_id` and `generaldia_path_index` in their
metadata. Store the resulting path-identifier lists with the experiment; a random
seed alone does not reproduce the split after the dataset changes.

## Tracking and target transformation

Apply state tracking independently within each path:

```python
tracked = path.tracked(
    overlap_floor=0.5,
    assignment_margin_floor=1e-6,
    degeneracy_tolerance=1e-8,
    near_degeneracy_threshold=1e-4,
)
```

The default ambiguity policy raises rather than manufacturing a state identity. With
`on_ambiguous="record"`, the returned object retains every ambiguous transition, and
each tracked sample records whether its incoming transition was ambiguous. This mode
is for diagnosis; it does not turn an uncertain continuation into a physical label.

Energies are reordered by the maximum-overlap assignment. Full state-indexed
derivative matrices receive the covariant transformation

$$
\widetilde A_k = W_k^\dagger A_k W_k.
$$

When full derivative matrices and energy gradients are both present, gradients are
taken from the diagonal of the transformed derivative matrix. Scalar gradients alone
can be reordered across nondegenerate matches. They cannot be rotated inside a
degenerate block because diagonal values do not determine the missing off-diagonal
matrix elements; GeneralDIA raises in that case.

The tracked energies follow state character and can therefore become non-ascending
through a crossing. The current one-geometry `observable_loss()` diagonalizes a model
and compares ascending energy ranks. Do not flatten `tracked.tracked_path` into that
loss and assume the labels remain compatible. This milestone constructs the tracked
targets and their evidence; a later path-aware loss must track or compare the model
predictions under the same explicit gauge/subspace policy.

## Visual communication contract

`state_tracking_report_data()` converts the same tracking result used for target
construction into a versioned, JSON-compatible evidence record. It includes:

- raw energy rank and tracked state-character energies;
- minimum principal overlap and assignment margin for every transition;
- tracked-to-raw permutations;
- absolute aligned-overlap matrices;
- degenerate blocks and near-degenerate pairs;
- ambiguity reasons and all numerical thresholds.

`write_state_tracking_report()` embeds that record in a self-contained HTML page:

```python
from generaldia import write_state_tracking_report

write_state_tracking_report(
    tracked,
    "outputs/path_tracking_report.html",
    coordinates=reaction_coordinates,
    coordinate_label="Reaction coordinate / angstrom",
)
```

The report shows the energy path, transition confidence, and a selectable aligned
overlap heat map. It needs no plotting or web dependency and can be attached to a
pull request, release record, or scientific discussion. The JSON record is the
scientific source of truth; notebooks and future dashboards should consume it rather
than recalculate state assignments in presentation code.

Run the complete synthetic example with:

```bash
python examples/08_path_aware_diagnostics.py
```

## Relationship to shared latent operators

Path-aware, gauge-consistent targets are the data foundation for a future shared
latent-operator model. That model may emit a Hamiltonian and transition operators in
one implicit basis before rotating them into the adiabatic frame, following the
structural motivation of the LUSH architecture. The present milestone deliberately
does not add a Pairformer, train transition dipoles, or claim LUSH reproduction.

GeneralDIA will first require path-independent validation of the target transformations
and held-out-path evaluation. A later model layer can then be judged against those
contracts instead of defining its own gauge and visualization behavior.

## Claim boundary

The path layer establishes a local continuation when supplied adjacent overlaps meet
the declared thresholds. It does not calculate electronic-structure overlaps, prove
a unique global diabatic basis, remove Berry holonomy, or establish dynamical
accuracy. A held-out path tests transfer only over the molecular and geometry domain
represented by the split design.
