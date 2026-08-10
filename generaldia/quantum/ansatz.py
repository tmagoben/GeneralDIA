import numpy as np
def parameter_count(n_qubits,layers=2): return 2*n_qubits*layers
def deterministic_initial_parameters(n_qubits,layers=2): return np.linspace(0.07,0.31,parameter_count(n_qubits,layers))
