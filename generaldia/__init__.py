"""GeneralDIA public API."""

from .analytic import TwoStateAvoidedCrossing
from .dataset import MolecularDataset, MolecularSample
from .losses import LossWeights, observable_loss
from .molecular import GaussianRBF, SimpleMolecularHamiltonian
from .observables import (
    adiabatic_energies,
    coupling_validity_mask,
    derivative_couplings_from_numerators,
    derivative_matrix_elements,
    energy_gradients,
    hamiltonian_jacobian,
)
from .training import (
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_model,
)

__all__ = [
    "GaussianRBF",
    "LossWeights",
    "MolecularDataset",
    "MolecularSample",
    "SimpleMolecularHamiltonian",
    "TrainingConfig",
    "TwoStateAvoidedCrossing",
    "adiabatic_energies",
    "coupling_validity_mask",
    "derivative_couplings_from_numerators",
    "derivative_matrix_elements",
    "energy_gradients",
    "evaluate_model",
    "hamiltonian_jacobian",
    "load_checkpoint",
    "observable_loss",
    "save_checkpoint",
    "train_model",
]

__version__ = "3.0.0"
