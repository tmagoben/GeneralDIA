import torch

def adiabatic_energies(H): return torch.linalg.eigvalsh(H)

def energy_gradients(model,Z,R):
    x=R.clone().detach().to(dtype=torch.get_default_dtype()).requires_grad_(True)
    E=torch.linalg.eigvalsh(model(Z,x)); grads=[]
    for i in range(E.numel()):
        (g,)=torch.autograd.grad(E[i],x,retain_graph=True,create_graph=True); grads.append(g)
    return E,torch.stack(grads)

def hamiltonian_jacobian(model,Z,R,create_graph=True):
    x=R.clone().detach().to(dtype=torch.get_default_dtype()).requires_grad_(True)
    def fn(coords): return model(Z,coords)
    jac=torch.autograd.functional.jacobian(fn,x,create_graph=create_graph) # (S,S,N,3)
    return model(Z,x),jac.permute(2,3,0,1) # (N,3,S,S)

def derivative_matrix_elements(model,Z,R):
    H,dH=hamiltonian_jacobian(model,Z,R,create_graph=True); E,U=torch.linalg.eigh(H)
    N=torch.einsum('mi,abmn,nj->abij',U.conj(),dH,U)
    return E,U,N

def derivative_couplings_from_numerators(E,N,gap_floor=1e-6):
    S=E.numel(); tau=torch.zeros_like(N)
    for i in range(S):
        for j in range(S):
            if i==j: continue
            gap=E[j]-E[i]
            if torch.abs(gap)>=gap_floor: tau[...,i,j]=N[...,i,j]/gap
    return tau
