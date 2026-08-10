import numpy as np
from generaldia.quantum.pauli import matrix_to_pauli,pauli_to_matrix
def random_hermitian(n,seed):
 r=np.random.default_rng(seed); A=r.normal(size=(n,n))+1j*r.normal(size=(n,n)); return (A+A.conj().T)/2
def test_two_and_four_dimensional_roundtrip():
 for n in [2,4]:
  H=random_hermitian(n,n); terms=matrix_to_pauli(H,tol=1e-14); assert np.allclose(pauli_to_matrix(terms),H,atol=1e-12)
