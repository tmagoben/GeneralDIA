import numpy as np
from generaldia.electronic_structure.pyscf_backend import SACASSCFBackend
Z=np.array([3,1]); R=np.array([[0.,0.,0.],[0.,0.,1.5]])
p=SACASSCFBackend(ncas=2,nelecas=2,n_states=2,basis='sto-3g').calculate(Z,R); print('energies',p.energies_hartree); print('scaled NAC keys',p.scaled_nac_pyscf.keys())
