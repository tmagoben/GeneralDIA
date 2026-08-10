# Limitations

1. The default molecular representation is deliberately simple and is not intended to compete with modern E(3)-equivariant architectures.
2. A diabatic Hamiltonian is generally gauge-dependent and may not be globally unique.
3. Near exact degeneracies, individual eigenvectors and derivative couplings are gauge-sensitive; subspace/projector quantities should be preferred.
4. The quantum backends encode a finite state subspace, not a fermionic many-electron Hamiltonian.
5. Optional PySCF/PennyLane/Qiskit integrations should be exercised in environments where those packages are installed; the core repository remains independent of them.
