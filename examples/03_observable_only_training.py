import torch
from generaldia.molecular import SimpleMolecularHamiltonian
torch.manual_seed(11); torch.set_default_dtype(torch.float64)
# A tiny geometry path with synthetic adiabatic energy labels; no H_ref is used in the loss.
Z=torch.tensor([1,1]); distances=torch.linspace(0.7,1.8,32); geometries=[torch.tensor([[0.,0.,-d/2],[0.,0.,d/2]]) for d in distances]; Eref=torch.stack((0.2*(distances-1.2)**2-0.8,0.15*(distances-1.4)**2-0.2),dim=1); Eref,_=torch.sort(Eref,dim=1)
m=SimpleMolecularHamiltonian(n_states=2,hidden=24,n_rbf=10); opt=torch.optim.Adam(m.parameters(),lr=3e-3)
for step in range(300):
 opt.zero_grad(); pred=torch.stack([torch.linalg.eigvalsh(m(Z,R)) for R in geometries]); loss=((pred-Eref)**2).mean(); loss.backward(); opt.step()
print('energy MAE',torch.mean(torch.abs(pred-Eref)).item()); print('reference diabatic Hamiltonian supplied: no')
