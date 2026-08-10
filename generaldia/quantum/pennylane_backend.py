import numpy as np
from .ansatz import deterministic_initial_parameters

def _require():
    try:
        import pennylane as qml
        from scipy.optimize import minimize
        return qml,minimize
    except ImportError as e: raise ImportError("Install GeneralDIA with the 'quantum' extra") from e
def _operator(qml,label):
    factors=[]
    for wire,ch in enumerate(label):
        if ch=='X': factors.append(qml.PauliX(wire))
        elif ch=='Y': factors.append(qml.PauliY(wire))
        elif ch=='Z': factors.append(qml.PauliZ(wire))
    if not factors: return qml.Identity(0)
    op=factors[0]
    for f in factors[1:]: op=op@f
    return op
def ground_state_vqe(terms,layers=2,maxiter=300):
    qml,minimize=_require(); n=len(next(iter(terms))); dev=qml.device('default.qubit',wires=n,shots=None)
    H=qml.Hamiltonian([float(complex(c).real) for c in terms.values()],[_operator(qml,l) for l in terms])
    @qml.qnode(dev)
    def energy(theta):
        p=np.asarray(theta).reshape(layers,n,2)
        for layer in range(layers):
            for q in range(n): qml.RY(p[layer,q,0],wires=q); qml.RZ(p[layer,q,1],wires=q)
            if n>1:
                for q in range(n-1): qml.CNOT(wires=[q,q+1])
                if n>2: qml.CNOT(wires=[n-1,0])
        return qml.expval(H)
    x0=deterministic_initial_parameters(n,layers); result=minimize(lambda x: float(energy(x)),x0,method='COBYLA',options={'maxiter':maxiter,'tol':1e-10})
    return {'energy':float(result.fun),'parameters':np.asarray(result.x),'success':bool(result.success),'message':str(result.message)}
