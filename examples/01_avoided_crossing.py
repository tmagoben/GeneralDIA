import torch

from generaldia.analytic import TwoStateAvoidedCrossing

torch.set_default_dtype(torch.float64)
m = TwoStateAvoidedCrossing()
for R in [-0.5, 0.0, 0.5]:
    H = m.hamiltonian(R)
    print(R, H.numpy(), torch.linalg.eigvalsh(H).numpy(), m.exact_energies(R).numpy())
