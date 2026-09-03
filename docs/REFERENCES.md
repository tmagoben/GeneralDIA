# References

## Measurement optimization

- Vladyslav Verteletskyi, Tzu-Ching Yen, and Artur F. Izmaylov,
  "Measurement Optimization in the Variational Quantum Eigensolver Using a Minimum
  Clique Cover," *J. Chem. Phys.* **152**, 124114 (2020),
  https://doi.org/10.1063/1.5141458; arXiv:1907.03358.

## API references used for optional backends

The optional integrations were written against the public package APIs current during
the August 2026 rebuild.

- PySCF SA-CASSCF NAC documentation: https://pyscf.org/pyscf_api_docs/pyscf.nac.html
- PySCF SA-CASSCF gradient documentation: https://pyscf.org/pyscf_api_docs/pyscf.grad.html
- PennyLane QNode documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.qnode.html
- PennyLane Hamiltonian documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.Hamiltonian.html
- Qiskit Statevector documentation: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Statevector
- Qiskit SparsePauliOp documentation: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp

Core tests do not depend on these optional packages. Optional GitHub Actions jobs
install and smoke-test the backend extras.
