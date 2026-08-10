import numpy as np
from .ansatz import deterministic_initial_parameters

def _require():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp,Statevector
        from scipy.optimize import minimize
        return QuantumCircuit,ParameterVector,SparsePauliOp,Statevector,minimize
    except ImportError as e: raise ImportError("Install GeneralDIA with the 'quantum' extra") from e
def ground_state_vqe(terms,layers=2,maxiter=300):
    QuantumCircuit,ParameterVector,SparsePauliOp,Statevector,minimize=_require(); n=len(next(iter(terms))); params=ParameterVector('theta',2*n*layers); qc=QuantumCircuit(n); k=0
    for _ in range(layers):
        for q in range(n): qc.ry(params[k],q); qc.rz(params[k+1],q); k+=2
        if n>1:
            for q in range(n-1): qc.cx(q,q+1)
            if n>2: qc.cx(n-1,0)
    # Internal labels are matrix/Kronecker order. Qiskit displays qubit numbers little-endian,
    # but SparsePauliOp matrix strings already use the same left-to-right Kronecker matrix order.
    op=SparsePauliOp.from_list([(label,complex(c)) for label,c in terms.items()])
    def energy(x):
        bound=qc.assign_parameters(dict(zip(params,x)),inplace=False); state=Statevector.from_instruction(bound); return float(np.real(state.expectation_value(op)))
    x0=deterministic_initial_parameters(n,layers); result=minimize(energy,x0,method='COBYLA',options={'maxiter':maxiter,'tol':1e-10})
    return {'energy':float(result.fun),'parameters':np.asarray(result.x),'success':bool(result.success),'message':str(result.message)}
