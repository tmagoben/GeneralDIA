# Getting started

## 1. Create an isolated environment

Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

The command prompt should now include `.venv`.

## 2. Install the core package

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install points Python at this checkout. Source edits take effect without
another installation.

## 3. Run the verification suite

```bash
pytest
ruff check .
ruff format --check .
```

Pytest checks matrix symmetry, molecular invariances, Cartesian derivatives, gauge
operations, data validation, training, checkpoint loading, and Pauli round trips.

## 4. Run the analytic reference

```bash
python examples/01_avoided_crossing.py
```

The script constructs

$$
H(R)=\begin{pmatrix}aR & c\\c & -aR\end{pmatrix}
$$

at three coordinates. It compares numerical eigenvalues from `torch.linalg.eigvalsh`
with the analytic result $\pm\sqrt{(aR)^2+c^2}$.

Check that each pair of printed energy arrays agrees.

## 5. Check molecular invariance

```bash
python examples/02_molecular_invariance.py
```

The script evaluates one water geometry, rotates it, and permutes its atom order. It
prints the Frobenius norm between the original and transformed Hamiltonians. Floating
point roundoff should keep both values near zero.

This test establishes invariance of the model construction. It does not measure
prediction accuracy.

## 6. Run the training workflow

```bash
python examples/05_end_to_end_training.py
```

The script creates synthetic reference data, partitions the geometries, trains a
model, evaluates held-out samples, and writes `outputs/synthetic_two_state.pt`.

Read [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) before replacing the synthetic data
with electronic-structure results.

## 7. Install an optional backend

PySCF reference calculations:

```bash
python -m pip install -e ".[pyscf]"
python examples/pyscf/01_h2_rhf.py
```

Finite-state quantum solvers:

```bash
python -m pip install -e ".[quantum]"
python examples/quantum/01_pennylane_vqe.py
python examples/quantum/02_qiskit_vqe.py
```

The quantum examples encode the selected state-space matrix. They do not construct a
second-quantized molecular Hamiltonian.
