import math,torch
from generaldia.molecular import SimpleMolecularHamiltonian
torch.manual_seed(7); torch.set_default_dtype(torch.float64)
Z=torch.tensor([8,1,1]); R=torch.tensor([[0.,0.,0.],[0.96,0.,0.],[-0.24,0.93,0.]])
m=SimpleMolecularHamiltonian(hidden=16,n_rbf=8); H=m(Z,R); th=0.7; Q=torch.tensor([[math.cos(th),-math.sin(th),0.],[math.sin(th),math.cos(th),0.],[0.,0.,1.]])
print('rotation error',torch.linalg.norm(H-m(Z,R@Q.T)).item()); p=torch.tensor([1,0,2]); print('permutation error',torch.linalg.norm(H-m(Z[p],R[p])).item())
