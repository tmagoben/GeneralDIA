# Contributing

## Set up the repository

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the environment before running checks.

## Make a change

Add tests for new behavior and failure cases. Public functions need type annotations,
docstrings, shape definitions, unit definitions, and explicit exceptions for invalid
inputs. Document any gauge or sign convention at the data boundary. State-tracking
changes need adversarial phase, permutation, degeneracy, and ambiguity-failure tests;
spectrum-only agreement is not sufficient validation.

## Run local checks

```bash
ruff check .
ruff format .
pytest --cov=generaldia --cov-report=term-missing
python -m build
```

Optional-backend changes also need the matching extra and test marker:

```bash
python -m pip install -e ".[quantum]"
pytest -m optional
```

## Submit a pull request

Describe the scientific assumption, API change, tests, and numerical tolerance. Note
any compatibility break or result that changed. Keep generated datasets, checkpoints,
and large outputs outside Git.
