import torch

class TwoStateAvoidedCrossing:
    def __init__(self, slope=0.05, coupling=0.01):
        self.slope=float(slope); self.coupling=float(coupling)
    def hamiltonian(self,R):
        R=torch.as_tensor(R,dtype=torch.get_default_dtype())
        H=torch.zeros(R.shape+(2,2),dtype=R.dtype,device=R.device)
        H[...,0,0]=self.slope*R
        H[...,1,1]=-self.slope*R
        H[...,0,1]=H[...,1,0]=self.coupling
        return H
    def exact_energies(self,R):
        R=torch.as_tensor(R,dtype=torch.get_default_dtype())
        e=torch.sqrt((self.slope*R)**2+self.coupling**2)
        return torch.stack((-e,e),dim=-1)
