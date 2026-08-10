import math,torch
from generaldia.molecular import SimpleMolecularHamiltonian
from generaldia.observables import energy_gradients
torch.set_default_dtype(torch.float64)
def setup():
 torch.manual_seed(5); m=SimpleMolecularHamiltonian(hidden=12,n_rbf=7); Z=torch.tensor([8,1,6,1]); R=torch.tensor([[0.,0.,0.],[.9,.1,0.],[-.4,1.1,.2],[.2,-.7,.6]]); return m,Z,R
def test_symmetry_translation_rotation_permutation():
 m,Z,R=setup(); H=m(Z,R); assert torch.allclose(H,H.T); assert torch.allclose(H,m(Z,R+torch.tensor([1.2,-.3,.8])),atol=1e-12)
 th=.41; Q=torch.tensor([[math.cos(th),-math.sin(th),0.],[math.sin(th),math.cos(th),0.],[0.,0.,1.]]) ; assert torch.allclose(H,m(Z,R@Q.T),atol=1e-12)
 p=torch.tensor([2,0,3,1]); assert torch.allclose(H,m(Z[p],R[p]),atol=1e-12)
def test_translational_force_sum_zero():
 m,Z,R=setup(); E,G=energy_gradients(m,Z,R); assert torch.max(torch.abs(G.sum(dim=1)))<1e-10
