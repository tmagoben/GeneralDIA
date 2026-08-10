import itertools,math,numpy as np
I=np.array([[1,0],[0,1]],complex); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.array([[1,0],[0,-1]],complex)
OPS={'I':I,'X':X,'Y':Y,'Z':Z}
def pauli_matrix(label):
    out=np.array([[1.0+0j]])
    for ch in label: out=np.kron(out,OPS[ch])
    return out
def _nqubits(dim):
    n=int(round(math.log2(dim)))
    if 2**n!=dim: raise ValueError('matrix dimension must be a power of two')
    return n
def matrix_to_pauli(H,tol=1e-12):
    H=np.asarray(H,complex)
    if H.ndim!=2 or H.shape[0]!=H.shape[1]: raise ValueError('H must be square')
    if not np.allclose(H,H.conj().T,atol=1e-10): raise ValueError('H must be Hermitian')
    n=_nqubits(H.shape[0]); terms={}
    for chars in itertools.product('IXYZ',repeat=n):
        label=''.join(chars); P=pauli_matrix(label); c=np.trace(P.conj().T@H)/(2**n)
        if abs(c)>tol: terms[label]=complex(c)
    return terms
def pauli_to_matrix(terms):
    if not terms: raise ValueError('terms cannot be empty')
    n=len(next(iter(terms))); H=np.zeros((2**n,2**n),complex)
    for label,c in terms.items():
        if len(label)!=n: raise ValueError('inconsistent Pauli labels')
        H+=c*pauli_matrix(label)
    return H
def exact_ground_energy(H): return float(np.linalg.eigvalsh(np.asarray(H,complex))[0].real)
