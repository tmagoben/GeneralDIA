import numpy as np,pytest
from generaldia.quantum.pauli import matrix_to_pauli,exact_ground_energy
H=np.array([[0.2,0.03-0.04j],[0.03+0.04j,-0.1]])
@pytest.mark.optional
def test_pennylane_vqe_if_installed():
 pytest.importorskip('pennylane'); from generaldia.quantum.pennylane_backend import ground_state_vqe; r=ground_state_vqe(matrix_to_pauli(H),maxiter=500); assert abs(r['energy']-exact_ground_energy(H))<1e-5
@pytest.mark.optional
def test_qiskit_vqe_if_installed():
 pytest.importorskip('qiskit'); from generaldia.quantum.qiskit_backend import ground_state_vqe; r=ground_state_vqe(matrix_to_pauli(H),maxiter=500); assert abs(r['energy']-exact_ground_energy(H))<1e-5
