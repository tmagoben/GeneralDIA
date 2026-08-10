import numpy as np
from generaldia.quantum.pauli import matrix_to_pauli,pauli_to_matrix
H=np.array([[0.2,0.03-0.04j],[0.03+0.04j,-0.1]])
terms=matrix_to_pauli(H); print(terms); print('roundtrip',np.linalg.norm(pauli_to_matrix(terms)-H))
