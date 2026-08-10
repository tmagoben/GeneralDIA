import torch
from generaldia.gauge import projector_distance,unitary_procrustes
torch.set_default_dtype(torch.float64)
def test_projector_invariant_to_subspace_unitary():
 U,_=torch.linalg.qr(torch.randn(7,3,dtype=torch.complex128)); Q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.complex128)); assert projector_distance(U,U@Q)<1e-12
def test_procrustes_alignment():
 U,_=torch.linalg.qr(torch.randn(7,3,dtype=torch.complex128)); Q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.complex128)); A=unitary_procrustes(U,U@Q); assert torch.linalg.norm(A-U)<1e-10
