"""GeneralDIA public API."""

from .analytic import TwoStateAvoidedCrossing
from .dataset import (
    MolecularDataset,
    MolecularPath,
    MolecularPathDataset,
    MolecularSample,
    PathTrackingSettings,
    TrackedMolecularPath,
    assert_disjoint_paths,
)
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
from .operators import MolecularOperators, SharedMolecularOperators
from .path_losses import PathLossBreakdown, PathPrediction, path_observable_loss, predict_path
from .path_training import evaluate_path_model, train_path_model
from .reporting import state_tracking_report_data, write_state_tracking_report
from .state_tracking import (
    AmbiguousStateTrackingError,
    StateTrackingResult,
    StateTrackingStep,
    SubspaceMatch,
    adjacent_state_overlaps,
    align_state_frames,
    track_states,
    transform_state_matrices,
)
from .training import (
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_model,
)

__all__ = [
    "AmbiguousStateTrackingError",
    "GaussianRBF",
    "LossWeights",
    "MolecularDataset",
    "MolecularOperators",
    "MolecularPath",
    "MolecularPathDataset",
    "MolecularSample",
    "PathLossBreakdown",
    "PathPrediction",
    "PathTrackingSettings",
    "SharedMolecularOperators",
    "SimpleMolecularHamiltonian",
    "StateTrackingResult",
    "StateTrackingStep",
    "SubspaceMatch",
    "TrackedMolecularPath",
    "TrainingConfig",
    "TwoStateAvoidedCrossing",
    "adiabatic_energies",
    "adjacent_state_overlaps",
    "align_state_frames",
    "assert_disjoint_paths",
    "coupling_validity_mask",
    "derivative_couplings_from_numerators",
    "derivative_matrix_elements",
    "energy_gradients",
    "evaluate_model",
    "evaluate_path_model",
    "hamiltonian_jacobian",
    "load_checkpoint",
    "observable_loss",
    "path_observable_loss",
    "predict_path",
    "save_checkpoint",
    "state_tracking_report_data",
    "track_states",
    "train_model",
    "train_path_model",
    "transform_state_matrices",
    "write_state_tracking_report",
]

__version__ = "3.3.0.dev1"
