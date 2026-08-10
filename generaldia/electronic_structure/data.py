from dataclasses import dataclass,field
import numpy as np

@dataclass
class ElectronicStructurePoint:
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    energies_hartree: np.ndarray
    gradients_hartree_per_bohr: np.ndarray|None=None
    scaled_nac_pyscf: dict=field(default_factory=dict)
    metadata: dict=field(default_factory=dict)
    def __post_init__(self):
        self.atomic_numbers=np.asarray(self.atomic_numbers,dtype=int); self.positions_angstrom=np.asarray(self.positions_angstrom,dtype=float); self.energies_hartree=np.asarray(self.energies_hartree,dtype=float)
        if self.gradients_hartree_per_bohr is not None: self.gradients_hartree_per_bohr=np.asarray(self.gradients_hartree_per_bohr,dtype=float)
        self.scaled_nac_pyscf={tuple(k):np.asarray(v,dtype=float) for k,v in self.scaled_nac_pyscf.items()}
