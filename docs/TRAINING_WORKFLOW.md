# End-to-end training workflow

This workflow starts with reference observables and ends with a tested checkpoint.

## Step 1: define the state manifold

Choose the number of states, spin symmetry, charge, electronic-structure method, and
geometry domain before generating data. Keep those choices fixed across one dataset.
A model cannot reconcile labels that refer to different state manifolds.

## Step 2: generate reference data

For synthetic development data, calculate targets from a known Hamiltonian. For
molecular work, run PySCF or import results from another electronic-structure code.
Record method settings and convergence status for each geometry.

```python
from generaldia.electronic_structure.pyscf_backend import RHFBackend

backend = RHFBackend(basis="cc-pvdz", charge=0, spin=0)
point = backend.calculate(atomic_numbers, positions_angstrom)
```

RHF supplies one state. A multi-state diabatic model needs a multi-state method and a
state-tracking procedure along the geometry domain.

## Step 3: validate and convert data

```python
from generaldia import MolecularDataset

dataset = MolecularDataset.from_electronic_structure(points)
```

The conversion keeps positions in angstrom and converts PySCF gradients from
hartree/bohr to hartree/angstrom. Inspect `sample.metadata` before training.

## Step 4: split by scientific independence

```python
train_data, validation_data, test_data = dataset.split(seed=17)
```

Random splitting works for the synthetic example. Molecular datasets often need a
grouped split by trajectory, molecule, reaction channel, or geometry region. A random
geometry split can place near-duplicate structures in training and test sets and
understate the prediction error.

For connected scans or trajectories, make paths explicit and split the paths rather
than individual geometries:

```python
from generaldia import MolecularPathDataset

paths = MolecularPathDataset(imported_paths)
training_paths, validation_paths, test_paths = paths.split(seed=17)

training_data = training_paths.as_dataset()
validation_data = validation_paths.as_dataset()
test_data = test_paths.as_dataset()
```

The current one-geometry loss can consume the raw energy-ranked samples after this
split. Track state-indexed targets for diagnosis and preservation, but do not feed
state-character-ordered targets into the energy-rank loss. Path-aware invariant loss
integration is a separate model-layer milestone. See
[Path-aware targets and visual diagnostics](PATH_AWARE_WORKFLOW.md).

## Step 5: construct the model

```python
from generaldia import SimpleMolecularHamiltonian

model = SimpleMolecularHamiltonian(
    n_states=2,
    hidden=64,
    n_rbf=16,
    max_z=20,
    r_min=0.0,
    r_max=8.0,
).double()
```

Set `r_max` to cover the pair distances in the dataset. The current representation
sums pair contributions and accepts one molecule at a time.

## Step 6: choose observable constraints

```python
from generaldia import LossWeights

weights = LossWeights(
    energy=1.0,
    energy_gradient=0.1,
    derivative_matrix=0.0,
)
```

Weights multiply mean squared errors. Normalize targets or set weights with their
physical scales in mind.

- Energy supervision constrains the spectrum.
- Gradient supervision constrains how each eigenvalue changes with geometry.
- Derivative-matrix supervision constrains diagonal and off-diagonal matrix elements
  in the supplied state gauge.

Off-diagonal targets require consistent state phases and subspace alignment. Raw
values from disconnected calculations can change sign or rotate within a degenerate
subspace.

For connected data, use physical adjacent-state overlaps and the transformations
described in [Gauge and connected-path state tracking](GAUGE_AND_STATE_TRACKING.md)
before constructing state-sensitive targets. The current PySCF adapter does not
generate cross-geometry overlaps, so that information must come from a validated
electronic-structure workflow.

## Step 7: train

```python
from generaldia import TrainingConfig, train_model

config = TrainingConfig(
    epochs=300,
    learning_rate=3e-3,
    seed=17,
    gradient_clip_norm=10.0,
    report_every=25,
)

history = train_model(
    model,
    train_data,
    validation_data=validation_data,
    weights=weights,
    config=config,
)
```

The reference trainer processes one geometry per optimizer step so molecules may have
different atom counts. Larger projects should add graph batching and a vectorized
pair representation.

## Step 8: evaluate held-out data

```python
from generaldia import evaluate_model

metrics = evaluate_model(model, test_data)
print(metrics)
```

Report errors in physical units after reversing normalization. Inspect errors against
geometry, energy gap, and state index. A single mean error can hide failure near a
crossing.

## Step 9: save the experiment

```python
from generaldia import save_checkpoint

save_checkpoint(
    "checkpoints/model.pt",
    model,
    config=config,
    weights=weights,
    history=history,
    metadata={
        "dataset_hash": "...",
        "split_seed": 17,
        "git_commit": "...",
    },
)
```

The checkpoint contains model weights, constructor settings when the model exposes
them, loss weights, training settings, history, and caller-supplied metadata. The
loader still needs a model instance with matching parameter shapes.

## Step 10: reload and verify

```python
from generaldia import load_checkpoint

restored = SimpleMolecularHamiltonian(n_states=2, hidden=64, n_rbf=16, max_z=20, r_max=8.0).double()
metadata = load_checkpoint("checkpoints/model.pt", restored)
restored_metrics = evaluate_model(restored, test_data)
```

Compare the restored metrics with the values recorded before saving. They should
agree to floating-point precision on the same hardware and dtype.

## Step 11: validate the scientific interpretation

Before calling the learned matrix diabatic, check:

1. Energy and gradient errors across the full geometry domain.
2. State gaps and behavior near degeneracies.
3. Smoothness of matrix elements along connected paths.
4. Gauge consistency of off-diagonal targets.
5. Invariance under translation, rotation, reflection, and atom reordering.
6. Performance on trajectories or molecules excluded from fitting.

Generate a state-tracking evidence report for every path contributing off-diagonal
targets. Review energy character, overlap confidence, degeneracy flags, and ambiguous
transitions rather than relying on an aggregate training loss.

Energy accuracy alone supports a claim about predicted adiabatic energies. It does
not support a unique diabatic representation.
