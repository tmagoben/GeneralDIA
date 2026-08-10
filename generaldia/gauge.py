import torch

def align_phase(reference,candidate):
    overlap=torch.vdot(reference,candidate)
    if torch.abs(overlap)==0: return candidate
    return candidate*(overlap.conj()/torch.abs(overlap))

def projector(U): return U@U.conj().T

def projector_distance(U,V): return torch.linalg.matrix_norm(projector(U)-projector(V),ord='fro')

def unitary_procrustes(reference,candidate):
    """Rotate candidate basis inside its subspace to best match reference."""
    M=candidate.conj().T@reference
    U,_,Vh=torch.linalg.svd(M)
    Q=U@Vh
    return candidate@Q
