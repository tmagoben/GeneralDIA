import torch
from torch import nn
from .matrix import unpack_real_symmetric

class GaussianRBF(nn.Module):
    def __init__(self,n=12,r_min=0.0,r_max=6.0,gamma=None):
        super().__init__(); centers=torch.linspace(r_min,r_max,n); self.register_buffer("centers",centers)
        spacing=(r_max-r_min)/max(n-1,1); self.gamma=float(gamma if gamma is not None else 1.0/max(spacing,1e-12)**2)
    def forward(self,r): return torch.exp(-self.gamma*(r[...,None]-self.centers)**2)

class SimpleMolecularHamiltonian(nn.Module):
    """All-pairs invariant model for small molecules.

    Pair features use e_i+e_j and |e_i-e_j|, making each pair representation
    invariant to swapping the atom order within the pair. Summation over all
    unordered pairs makes the molecular representation invariant to a global
    permutation of atom ordering. Coordinates enter only through distances.
    """
    def __init__(self,n_states=2,hidden=32,n_rbf=12,max_z=36):
        super().__init__(); self.n_states=int(n_states); self.embed=nn.Embedding(max_z+1,hidden); self.rbf=GaussianRBF(n_rbf)
        self.pair_net=nn.Sequential(nn.Linear(2*hidden+n_rbf,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,n_states*(n_states+1)//2))
    def representation(self,Z,R):
        if not torch.is_tensor(Z): Z=torch.as_tensor(Z,dtype=torch.long)
        else: Z=Z.to(dtype=torch.long)
        if not torch.is_tensor(R): R=torch.as_tensor(R,dtype=torch.get_default_dtype())
        else: R=R.to(dtype=torch.get_default_dtype())
        if Z.ndim!=1 or R.ndim!=2 or R.shape!=(Z.numel(),3): raise ValueError("Z must be (N,), R must be (N,3)")
        if Z.numel()<2: raise ValueError("this simple all-pairs model requires at least two atoms")
        e=self.embed(Z); d=torch.cdist(R,R); total=torch.zeros(self.pair_net[-2].out_features,dtype=e.dtype,device=e.device)
        for i in range(Z.numel()):
            for j in range(i+1,Z.numel()):
                pair=torch.cat((e[i]+e[j],torch.abs(e[i]-e[j]),self.rbf(d[i,j])),dim=0)
                total=total+self.pair_net(pair)
        return total
    def forward(self,Z,R): return unpack_real_symmetric(self.head(self.representation(Z,R)),self.n_states)
