"""Run with an isolated wheel interpreter; reject imports from the checkout."""

import sys
from pathlib import Path

import numpy as np
import torch

import generaldia
from generaldia.analytic import TwoStateAvoidedCrossing
from generaldia.electronic_structure import pyscf_backend
from generaldia.quantum import pennylane_backend, qiskit_backend
from generaldia.quantum.pauli import matrix_to_pauli, pauli_to_matrix


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    for module in (generaldia, pyscf_backend, pennylane_backend, qiskit_backend):
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(prefix):
            raise RuntimeError(
                f"{module.__name__} was not loaded from the wheel environment: {location}"
            )
    model = TwoStateAvoidedCrossing()
    coordinates = torch.tensor([0.3], dtype=torch.float64)
    matrix = model.hamiltonian(coordinates)[0]
    torch.testing.assert_close(torch.linalg.eigvalsh(matrix), model.exact_energies(coordinates)[0])
    reference = np.array([[0.4, 0.2 - 0.1j], [0.2 + 0.1j, -0.3]])
    np.testing.assert_allclose(pauli_to_matrix(matrix_to_pauli(reference)), reference, atol=1e-12)
    print(
        f"PASS installed GeneralDIA {generaldia.__version__}: "
        "imports, analytic spectrum, complex Pauli roundtrip"
    )


if __name__ == "__main__":
    main()
