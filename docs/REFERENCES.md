# References

## Latent electronic-state Hamiltonians

- David Juergens, Martin Stöhr, Andreas E. Hillers-Bendtsen, O. Jonathan Fajen, and
  Todd J. Martínez, "Latent unified smooth Hamiltonians for excited state chemistry,"
  arXiv:2609.01871v1 (2026), https://doi.org/10.48550/arXiv.2609.01871.

GeneralDIA cites LUSH as motivation for a future shared implicit-basis operator model.
The current implementation is independent and narrower: it provides path-aware data,
state-target transformations, visual diagnostics, and finite-state quantum encoding;
it does not reproduce the LUSH transformer architecture or its molecular benchmarks.

## Measurement optimization

- Vladyslav Verteletskyi, Tzu-Ching Yen, and Artur F. Izmaylov,
  "Measurement Optimization in the Variational Quantum Eigensolver Using a Minimum
  Clique Cover," *J. Chem. Phys.* **152**, 124114 (2020),
  https://doi.org/10.1063/1.5141458; arXiv:1907.03358.

## API references used for optional backends

The optional integrations were written against the public package APIs current during
the August-September 2026 development cycle.

- PySCF SA-CASSCF NAC documentation: https://pyscf.org/pyscf_api_docs/pyscf.nac.html
- PySCF SA-CASSCF gradient documentation: https://pyscf.org/pyscf_api_docs/pyscf.grad.html
- PennyLane QNode documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.qnode.html
- PennyLane Hamiltonian documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.Hamiltonian.html
- PennyLane counts documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.counts.html
- PennyLane set-shots documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.set_shots.html
- Qiskit Statevector documentation: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Statevector
- Qiskit SparsePauliOp documentation: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp
- Qiskit BasicSimulator documentation: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.basic_provider.BasicSimulator

Core tests do not depend on these optional packages. Optional GitHub Actions jobs
install and smoke-test the backend extras.
