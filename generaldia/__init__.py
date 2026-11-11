"""GeneralDIA public API."""

from .analytic import TwoStateAvoidedCrossing
from .certification import NumeratorCertification, certify_coupling_numerators
from .dataset import (
    MolecularDataset,
    MolecularPath,
    MolecularPathDataset,
    MolecularSample,
    PathTrackingSettings,
    TrackedMolecularPath,
    assert_disjoint_paths,
)
from .loop_convergence import AbelianLoopConvergence, certify_abelian_loop_convergence
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
from .topology import LoopHolonomy, closed_loop_overlaps, wilson_loop
from .training import (
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from .truncation import TruncationCertification, certify_state_truncation

__all__ = [
    "AbelianLoopConvergence",
    "AmbiguousStateTrackingError",
    "GaussianRBF",
    "LoopHolonomy",
    "LossWeights",
    "MolecularDataset",
    "MolecularOperators",
    "MolecularPath",
    "MolecularPathDataset",
    "MolecularSample",
    "NumeratorCertification",
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
    "TruncationCertification",
    "TwoStateAvoidedCrossing",
    "adiabatic_energies",
    "adjacent_state_overlaps",
    "align_state_frames",
    "assert_disjoint_paths",
    "certify_abelian_loop_convergence",
    "certify_coupling_numerators",
    "certify_state_truncation",
    "closed_loop_overlaps",
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
    "wilson_loop",
    "write_state_tracking_report",
]

__version__ = "3.4.0.dev4"
