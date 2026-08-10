"""Optional PySCF reference-data adapters. Imports are lazy by design."""
import numpy as np
from .data import ElectronicStructurePoint

_SYMBOLS=['X','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca']
def _require():
    try:
        from pyscf import gto,scf,mcscf
        return gto,scf,mcscf
    except ImportError as e: raise ImportError("Install GeneralDIA with the 'pyscf' extra") from e
def _atoms(Z,R):
    out=[]
    for z,r in zip(Z,R):
        if int(z)<=0 or int(z)>=len(_SYMBOLS): raise ValueError("pedagogical symbol table supports Z=1..20")
        out.append((_SYMBOLS[int(z)],tuple(map(float,r))))
    return out

class RHFBackend:
    def __init__(self,basis='sto-3g',charge=0,spin=0,conv_tol=1e-10): self.basis=basis; self.charge=charge; self.spin=spin; self.conv_tol=conv_tol
    def calculate(self,Z,R_angstrom):
        gto,scf,_=_require(); mol=gto.M(atom=_atoms(Z,R_angstrom),basis=self.basis,unit='Angstrom',charge=self.charge,spin=self.spin,verbose=0)
        mf=scf.RHF(mol); mf.conv_tol=self.conv_tol; energy=mf.kernel()
        if not mf.converged: raise RuntimeError('RHF did not converge')
        grad=mf.nuc_grad_method().kernel()
        return ElectronicStructurePoint(Z,R_angstrom,[energy],np.asarray([grad]),metadata={'backend':'PySCF','method':'RHF','basis':self.basis,'converged':True})

class SACASSCFBackend:
    """Equal-weight state-averaged CASSCF energies, state gradients, and PySCF-scaled NACs.

    PySCF defines state=(ket,bra), returning <bra|d ket>. The stored dictionary
    preserves this tuple exactly. With mult_ediff=True, the values are PySCF's
    energy-difference-scaled NAC quantity; GeneralDIA does not silently reinterpret
    its sign/index convention.
    """
    def __init__(self,ncas,nelecas,n_states=2,basis='sto-3g',charge=0,spin=0,conv_tol=1e-10,use_etfs=False):
        self.ncas=ncas; self.nelecas=nelecas; self.n_states=n_states; self.basis=basis; self.charge=charge; self.spin=spin; self.conv_tol=conv_tol; self.use_etfs=use_etfs
    def calculate(self,Z,R_angstrom):
        gto,scf,mcscf=_require(); mol=gto.M(atom=_atoms(Z,R_angstrom),basis=self.basis,unit='Angstrom',charge=self.charge,spin=self.spin,verbose=0)
        mf=scf.RHF(mol); mf.conv_tol=self.conv_tol; mf.kernel()
        if not mf.converged: raise RuntimeError('RHF reference did not converge')
        weights=[1.0/self.n_states]*self.n_states
        mc=mcscf.CASSCF(mf,self.ncas,self.nelecas).state_average(weights); mc.conv_tol=self.conv_tol; mc.kernel()
        energies=np.asarray(mc.e_states,dtype=float); grad_method=mc.nuc_grad_method(); grads=np.stack([np.asarray(grad_method.kernel(state=i)) for i in range(self.n_states)])
        nac_method=mc.nac_method(); scaled={}
        for ket in range(self.n_states):
            for bra in range(ket+1,self.n_states):
                scaled[(ket,bra)]=np.asarray(nac_method.kernel(state=(ket,bra),mult_ediff=True,use_etfs=self.use_etfs))
        return ElectronicStructurePoint(Z,R_angstrom,energies,grads,scaled,metadata={'backend':'PySCF','method':'SA-CASSCF','basis':self.basis,'ncas':self.ncas,'nelecas':self.nelecas,'n_states':self.n_states,'weights':weights,'use_etfs':self.use_etfs})
