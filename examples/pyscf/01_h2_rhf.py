import numpy as np

from generaldia.electronic_structure.pyscf_backend import RHFBackend

Z = np.array([1, 1])
R = np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]])
p = RHFBackend().calculate(Z, R)
print(p.energies_hartree)
print(p.gradients_hartree_per_bohr)
