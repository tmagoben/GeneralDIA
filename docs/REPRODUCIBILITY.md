# Reproducibility record

A repeatable GeneralDIA experiment needs the records listed below.

## Source and environment

Record:

- GeneralDIA Git commit;
- Python, NumPy, and PyTorch versions;
- optional backend versions;
- operating system and accelerator;
- installed dependency lock file.

Capture the environment with:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment.txt
```

## Dataset

Record the source calculation, state manifold, units, geometry selection, failed
calculations, preprocessing, and a cryptographic hash of the final data artifact.

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('dataset.npz'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

For a connected state path, also record the overlap-generation method, ordered
geometry identifiers, state manifold, degeneracy tolerance, overlap floor,
assignment margin floor, near-degeneracy threshold, and every ambiguous transition.

## Split

Store the exact sample identifiers in each partition. A seed alone cannot reproduce a
split after the dataset changes. Group trajectories or related structures when the
scientific test requires independence.

## Model and optimization

Record constructor arguments, dtype, random seed, loss weights, target normalization,
optimizer settings, epoch count, and stopping rule. `save_checkpoint` stores training
settings and accepts additional metadata, but the caller must include model constructor
arguments and dataset identifiers.

## Evaluation

Report per-state and aggregate errors in physical units. Include errors as a function
of geometry and energy gap. Store the code that generated each table or figure.

## Determinism

Examples set NumPy and Torch seeds and use float64. Bitwise equality can still depend
on hardware, library versions, and parallel numerical kernels. Report numerical
tolerances with each reproducibility check.

## Minimum release artifact

A published result should include:

1. source commit and environment lock;
2. immutable dataset identifier and split files;
3. checkpoint plus model constructor settings;
4. training history and evaluation outputs;
5. state-tracking diagnostics when state-indexed targets are used;
6. one command that regenerates the reported metrics.
