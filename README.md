# GeneralDIA

GeneralDIA is a compact research-software framework for learning and validating
finite-state diabatic Hamiltonians from molecular geometry and adiabatic observables.

## Scope

The central object is a finite electronic-state Hamiltonian

$$
H_\theta(R) \in \mathbb{C}^{N_s\times N_s},
$$

where $R$ denotes nuclear geometry. The framework is built around the chain

```text
molecular geometry
      |
      v
invariant molecular representation
      |
      v
latent H_theta(R)
      |
      +--> adiabatic energies / states
      +--> Cartesian energy gradients
      +--> Hamiltonian-derivative matrix elements
      +--> optional PySCF reference data
      +--> exact classical eigensolver
      +--> PennyLane/Qiskit state-subspace VQE
```

## Important quantum-computing distinction

The PennyLane and Qiskit backends in this repository encode the **finite electronic
state subspace** represented by $H_\theta$ onto qubits. For example, a two-state
Hamiltonian is a one-qubit Hamiltonian.

This is **not the same operation** as mapping a second-quantized fermionic molecular
Hamiltonian to qubits with Jordan-Wigner or Bravyi-Kitaev. PySCF is used here as an
optional source of ab initio state energies, gradients, and SA-CASSCF nonadiabatic
couplings. A fermionic quantum-chemistry mapping would be a separate layer.

## Core features

- exact real-symmetric and complex-Hermitian matrix construction;
- simple pair-distance molecular model with explicit translation, rotation, and
  atom-order invariance;
- differentiable adiabatic energies and Cartesian gradients;
- Hamiltonian Jacobians and off-diagonal derivative matrix elements;
- complex phase alignment and unitary subspace alignment;
- optional PySCF RHF and SA-CASSCF reference-data adapters;
- exact Pauli expansion for $2^n\times2^n$ Hermitian state-space Hamiltonians;
- optional PennyLane and Qiskit variational state-subspace solvers;
- deterministic tests for the mathematical invariants claimed by the code.

## Install

Core:

```bash
pip install -e ".[dev]"
pytest -q
```

PySCF interface:

```bash
pip install -e ".[pyscf]"
```

Quantum backends:

```bash
pip install -e ".[quantum]"
```

## Examples

```bash
python examples/01_avoided_crossing.py
python examples/02_molecular_invariance.py
python examples/03_observable_only_training.py
python examples/04_pauli_state_encoding.py
```

Optional:

```bash
python examples/pyscf/01_h2_rhf.py
python examples/pyscf/02_lih_sa_casscf.py
python examples/quantum/01_pennylane_vqe.py
python examples/quantum/02_qiskit_vqe.py
```

See `docs/` for the mathematical conventions and limitations.
