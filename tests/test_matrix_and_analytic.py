import torch
from generaldia.analytic import TwoStateAvoidedCrossing
from generaldia.matrix import complex_hermitian_from_parts
torch.set_default_dtype(torch.float64)
def test_avoided_crossing_exact():
 m=TwoStateAvoidedCrossing(); R=torch.linspace(-1,1,17); assert torch.allclose(torch.linalg.eigvalsh(m.hamiltonian(R)),m.exact_energies(R),atol=1e-13)
def test_complex_hermitian():
 H=complex_hermitian_from_parts(torch.tensor([0.1,-0.2,0.3]),torch.tensor([.02,.03,.04]),torch.tensor([.05,-.01,.06])); assert torch.allclose(H,H.conj().T)
