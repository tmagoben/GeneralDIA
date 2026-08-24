import numpy as np

from generaldia.quantum.pauli import exact_ground_energy, matrix_to_pauli
from generaldia.quantum.qiskit_backend import ground_state_vqe

H = np.array([[0.2, 0.03 - 0.04j], [0.03 + 0.04j, -0.1]])
r = ground_state_vqe(matrix_to_pauli(H))
print("exact", exact_ground_energy(H))
print("VQE", r["energy"])
