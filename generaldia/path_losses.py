"""Path supervision using spectra and state-block operator invariants.

Model Hamiltonians must use one fixed orthonormal latent basis over a path.
All tracking decisions are discrete and detached; gradients are defined within
the accepted assignment and degeneracy partition. No hidden target gauge is fitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from .dataset import MolecularPath, PathTrackingSettings, TrackedMolecularPath
from .losses import LossBreakdown, LossWeights, _mean_squared_error
from .observables import _validate_hamiltonian, hamiltonian_jacobian
from .state_tracking import (
    StateTrackingResult,
    _energy_partition,
    adjacent_state_overlaps,
    track_states,
)


class _SubspaceFrame(torch.autograd.Function):
    """First-order eigenspace derivative with no within-block gauge derivative.

    For block-invariant objectives, only rotations between distinct spectral
    blocks contribute. Masking within-block denominators avoids the 0/0 in an
    individual-eigenvector backward at exact degeneracy. Higher derivatives are
    deliberately unsupported.
    """

    @staticmethod
    def forward(ctx, hamiltonian: Tensor, labels: Tensor) -> Tensor:
        energies, frame = torch.linalg.eigh(hamiltonian)
        ctx.save_for_backward(energies, frame, labels)
        return frame

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        energies, frame, labels = ctx.saved_tensors
        different = labels[:, None] != labels[None, :]
        gaps = energies[None, :] - energies[:, None]
        safe_gaps = torch.where(different, gaps, torch.ones_like(gaps))
        reciprocal = torch.where(different, 1 / safe_gaps, torch.zeros_like(gaps))
        local = (frame.mH @ gradient) * reciprocal
        local = (local + local.mH) / 2
        return frame @ local @ frame.mH, None


@dataclass
class PathPrediction:
    """Tracked model outputs and detached decisions for an ordered path.

    Energies have shape ``(K,S)`` and optional derivative matrices ``(K,N,3,S,S)``.
    Optional dipoles have shape ``(K,3,S,S)`` and share the derivative-matrix frame.
    Energies follow the first geometry's ascending spectrum, then state character.
    Units follow the model and must match the supplied path targets.
    """

    energies: Tensor
    derivative_matrices: Tensor | None
    tracking: StateTrackingResult
    dipole_matrices: Tensor | None = None


@dataclass
class PathLossBreakdown(LossBreakdown):
    """Loss values plus the target and prediction continuation evidence."""

    prediction: PathPrediction
    target: TrackedMolecularPath
    dipole: Tensor | None = None
    joint_operator: Tensor | None = None

    def scalars(self) -> dict[str, float]:
        """Include dimensionless dipole and cross-operator components when requested."""

        result = super().scalars()
        for name in ("dipole", "joint_operator"):
            value = getattr(self, name)
            if value is not None:
                result[name] = float(value.detach())
        return result


def _training_settings(settings: PathTrackingSettings | None) -> PathTrackingSettings:
    settings = settings or PathTrackingSettings()
    if settings.on_ambiguous != "raise":
        raise ValueError("path supervision requires on_ambiguous='raise'")
    return settings


def predict_path(
    model: nn.Module,
    path: MolecularPath,
    *,
    settings: PathTrackingSettings | None = None,
    derivatives: bool = False,
    dipoles: bool = False,
) -> PathPrediction:
    """Track predictions using adjacent eigenvector overlaps in a fixed model basis.

    Raises ``ValueError`` for invalid models, dimensions or policy, and
    ``AmbiguousStateTrackingError`` for unsupported continuation. Optional
    derivatives retain a first-order parameter graph, including exact degenerate
    blocks when used only in block-invariant objectives. Dipoles require an
    ``operators(Z,R)`` method returning Hermitian matrices in the same fixed latent
    basis as ``forward(Z,R)``. A declared model/target charge mismatch raises.
    """

    settings = _training_settings(settings)
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model must have trainable parameters")
    matrices, jacobians, spectra, raw_frames, raw_dipoles = [], [], [], [], []
    if dipoles and not callable(getattr(model, "operators", None)):
        raise ValueError("dipole prediction requires model.operators(atomic_numbers, positions)")
    model_charge = getattr(model, "charge", None)
    if model_charge is not None and any(
        metadata.get("charge", model_charge) != model_charge
        for metadata in (path.metadata, *(sample.metadata for sample in path))
    ):
        raise ValueError("model and target molecular charge must match")
    for raw_sample in path:
        sample = raw_sample.to(parameter.device, parameter.real.dtype)
        if derivatives:
            matrix, jacobian = hamiltonian_jacobian(
                model, sample.atomic_numbers, sample.positions, create_graph=True
            )
            jacobians.append(jacobian)
        else:
            matrix = model(sample.atomic_numbers, sample.positions)
            _validate_hamiltonian(matrix)
        if matrix.shape != (path.n_states, path.n_states):
            raise ValueError("model and path state count must match")
        if dipoles:
            result = model.operators(sample.atomic_numbers, sample.positions)
            _validate_hamiltonian(result.hamiltonian)
            if result.hamiltonian.shape != matrix.shape or not torch.allclose(
                result.hamiltonian, matrix, atol=1e-10, rtol=1e-8
            ):
                raise ValueError(
                    "model.forward and model.operators must return the same Hamiltonian"
                )
            if result.dipoles.shape != (3, path.n_states, path.n_states):
                raise ValueError("predicted dipoles must have shape (3, S, S)")
            for component in result.dipoles:
                _validate_hamiltonian(component)
            raw_dipoles.append(result.dipoles)
        matrices.append(matrix)
        spectra.append(torch.linalg.eigvalsh(matrix))
        raw_frames.append(torch.linalg.eigh(matrix.detach())[1])
    energies = torch.stack(spectra)
    tracking = track_states(
        adjacent_state_overlaps(torch.stack(raw_frames)),
        energies=energies.detach(),
        **asdict(settings),
    )
    permutations = (tuple(range(path.n_states)), *(step.permutation for step in tracking.steps))
    tracked_energies = torch.stack(
        [energy[list(order)] for energy, order in zip(spectra, permutations, strict=True)]
    )
    derivative_matrices = None
    dipole_matrices = None
    if derivatives or dipoles:
        numerators, transformed_dipoles = [], []
        for index, matrix in enumerate(matrices):
            groups = _energy_partition(energies[index].detach(), settings.degeneracy_tolerance)
            labels = torch.empty(path.n_states, dtype=torch.long, device=matrix.device)
            for label, group in enumerate(groups):
                labels[list(group)] = label
            frame = _SubspaceFrame.apply(matrix, labels) @ tracking.transformations[index]
            if derivatives:
                numerators.append(frame.mH @ jacobians[index] @ frame)
            if dipoles:
                operator_dtype = torch.promote_types(frame.dtype, raw_dipoles[index].dtype)
                operator_frame = frame.to(operator_dtype)
                transformed_dipoles.append(
                    operator_frame.mH @ raw_dipoles[index].to(operator_dtype) @ operator_frame
                )
        derivative_matrices = torch.stack(numerators) if derivatives else None
        dipole_matrices = torch.stack(transformed_dipoles) if dipoles else None
    return PathPrediction(tracked_energies, derivative_matrices, tracking, dipole_matrices)


def _dipole_descriptors(dipoles: Tensor, groups: tuple[tuple[int, ...], ...]) -> Tensor:
    """Block trace vectors and Cartesian Gram tensors, including cross components.

    Inputs must already be scaled to dimensionless units. The complete vector and
    tensor entries make squared descriptor distance invariant to a common spatial
    rotation. These are partial joint invariants, not a complete gauge certificate.
    """

    descriptors = []
    for index, group in enumerate(groups):
        diagonal = dipoles[:, list(group), :][:, :, list(group)]
        descriptors.append(diagonal.diagonal(dim1=-2, dim2=-1).sum(-1) / len(group))
        for other in groups[index:]:
            block = dipoles[:, list(group), :][:, :, list(other)]
            gram = torch.einsum("aij,bij->ab", block.conj(), block) / (len(group) * len(other))
            descriptors.append(gram.reshape(-1))
    return torch.cat(descriptors)


def _joint_descriptors(
    derivatives: Tensor,
    dipoles: Tensor,
    groups: tuple[tuple[int, ...], ...],
) -> Tensor:
    """Dimensionless Tr(N_BC^dagger mu_BC) tensors linking both operator families."""

    dtype = torch.promote_types(derivatives.dtype, dipoles.dtype)
    derivatives, dipoles = derivatives.to(dtype), dipoles.to(dtype)
    descriptors = []
    for index, group in enumerate(groups):
        for other in groups[index:]:
            numerator = derivatives[..., list(group), :][..., list(other)]
            dipole = dipoles[..., list(group), :][..., list(other)]
            cross = torch.einsum("xyij,zij->xyz", numerator.conj(), dipole)
            descriptors.append((cross / (len(group) * len(other))).reshape(-1))
    return torch.cat(descriptors)


def _block_descriptors(
    matrix: Tensor,
    groups: tuple[tuple[int, ...], ...],
    *,
    off_diagonal: bool,
) -> Tensor:
    """Concatenate diagonal-block eigenvalues and optional inter-block singular values."""

    descriptors = []
    for index, group in enumerate(groups):
        rows = matrix[..., list(group), :]
        descriptors.append(torch.linalg.eigvalsh(rows[..., list(group)]).reshape(-1))
        if off_diagonal:
            for other in groups[index + 1 :]:
                descriptors.append(torch.linalg.svdvals(rows[..., list(other)]).reshape(-1))
    return torch.cat(descriptors)


def path_observable_loss(
    model: nn.Module,
    path: MolecularPath | TrackedMolecularPath,
    weights: LossWeights | None = None,
    *,
    settings: PathTrackingSettings | None = None,
) -> PathLossBreakdown:
    """Compare complete paths without identifying unobservable state gauges.

    The target is canonicalized by its first-geometry spectrum. Model predictions
    are independently tracked. Diagonal-block eigenvalues and off-block singular
    values define derivative supervision; scalar gradients suffice for singletons
    only. Component MSEs use all descriptor entries over the path. Energy units and
    coordinate units must agree; ``LossWeights`` supplies their relative scaling.
    Dipole supervision uses block trace vectors and full Cartesian Gram tensors;
    joint supervision adds cross tensors linking dipoles to full derivative
    matrices. Both new MSEs use dimensionless, electronically gauge-invariant
    descriptors and are invariant to a common spatial rotation. Legacy derivative
    component spectra alone do not have that spatial-invariance guarantee.

    Raises for missing overlaps/targets, ambiguous transitions, or incompatible
    target/prediction degeneracy partitions. Recorded ambiguous targets are never
    accepted for training. These invariants do not identify a unique Hamiltonian.
    """

    weights = weights or LossWeights()
    if isinstance(path, TrackedMolecularPath):
        if path.ambiguous_steps:
            raise ValueError("recorded ambiguous paths cannot be used for supervision")
        if settings is not None and settings != path.settings:
            raise ValueError("settings must match the supplied tracked target")
        settings = _training_settings(path.settings)
        # Rebuild from raw evidence; detached/mutable cached tensors are not labels.
        raw_path = path.raw_path
    else:
        settings = _training_settings(settings)
        raw_path = path
    target = raw_path.tracked(**asdict(settings))
    needs_derivatives = (
        weights.energy_gradient > 0 or weights.derivative_matrix > 0 or weights.joint_operator > 0
    )
    needs_dipoles = weights.dipole > 0 or weights.joint_operator > 0
    prediction = predict_path(
        model, raw_path, settings=settings, derivatives=needs_derivatives, dipoles=needs_dipoles
    )
    device = prediction.energies.device
    canonical = torch.argsort(target.tracked_energies[0], stable=True)
    target_energies = target.tracked_energies[:, canonical].to(device=device).detach()
    predicted_values: dict[str, list[Tensor]] = {
        key: [] for key in ("energy", "gradient", "matrix", "dipole", "joint_operator")
    }
    target_values: dict[str, list[Tensor]] = {key: [] for key in predicted_values}

    for index, sample in enumerate(target.tracked_path):
        groups = _energy_partition(target_energies[index], settings.degeneracy_tolerance)
        predicted_groups = _energy_partition(
            prediction.energies[index].detach(), settings.degeneracy_tolerance
        )
        if groups != predicted_groups:
            raise ValueError(f"target and prediction degeneracy partitions differ at point {index}")
        for group in groups:
            predicted_values["energy"].append(prediction.energies[index, list(group)].sort().values)
            target_values["energy"].append(target_energies[index, list(group)].sort().values)
        if not (needs_derivatives or needs_dipoles):
            continue
        predicted_matrix = (
            None
            if prediction.derivative_matrices is None
            else prediction.derivative_matrices[index]
        )
        target_matrix = sample.derivative_matrix_elements
        if target_matrix is not None:
            target_matrix = (
                target_matrix[:, :, canonical, :][:, :, :, canonical].to(device).detach()
            )
        if needs_dipoles:
            if sample.dipole_matrix_elements is None:
                raise ValueError("dipole supervision requires dipole_matrix_elements targets")
            target_dipole = (
                sample.dipole_matrix_elements[:, canonical, :][:, :, canonical].to(device).detach()
            )
            predicted_dipole = prediction.dipole_matrices[index]
            if weights.dipole > 0:
                predicted_values["dipole"].append(
                    _dipole_descriptors(predicted_dipole / weights.dipole_scale, groups)
                )
                target_values["dipole"].append(
                    _dipole_descriptors(target_dipole / weights.dipole_scale, groups)
                )
            if weights.joint_operator > 0:
                if target_matrix is None:
                    raise ValueError(
                        "joint-operator supervision requires full derivative-matrix targets"
                    )
                predicted_values["joint_operator"].append(
                    _joint_descriptors(
                        predicted_matrix / weights.derivative_scale,
                        predicted_dipole / weights.dipole_scale,
                        groups,
                    )
                )
                target_values["joint_operator"].append(
                    _joint_descriptors(
                        target_matrix / weights.derivative_scale,
                        target_dipole / weights.dipole_scale,
                        groups,
                    )
                )
        if weights.energy_gradient > 0:
            if target_matrix is None:
                if sample.energy_gradients is None:
                    raise ValueError(
                        "energy-gradient loss requires gradient or full matrix targets"
                    )
                if any(len(group) > 1 for group in groups):
                    raise ValueError("degenerate gradients require full matrix targets")
                gradients = sample.energy_gradients[canonical].to(device).detach()
                target_matrix_for_gradient = torch.diag_embed(gradients.permute(1, 2, 0))
            else:
                target_matrix_for_gradient = target_matrix
            predicted_values["gradient"].append(
                _block_descriptors(predicted_matrix, groups, off_diagonal=False)
            )
            target_values["gradient"].append(
                _block_descriptors(target_matrix_for_gradient, groups, off_diagonal=False)
            )
        if weights.derivative_matrix > 0:
            if target_matrix is None:
                raise ValueError("derivative-matrix loss requires full matrix targets")
            predicted_values["matrix"].append(
                _block_descriptors(predicted_matrix, groups, off_diagonal=True)
            )
            target_values["matrix"].append(
                _block_descriptors(target_matrix, groups, off_diagonal=True)
            )

    components = {
        key: _mean_squared_error(torch.cat(values), torch.cat(target_values[key]))
        if values
        else None
        for key, values in predicted_values.items()
    }
    total = weights.energy * components["energy"]
    if components["gradient"] is not None:
        total = total + weights.energy_gradient * components["gradient"]
    if components["matrix"] is not None:
        total = total + weights.derivative_matrix * components["matrix"]
    for name in ("dipole", "joint_operator"):
        if components[name] is not None:
            total = total + getattr(weights, name) * components[name]
    return PathLossBreakdown(
        total,
        components["energy"],
        components["gradient"],
        components["matrix"],
        prediction,
        target,
        components["dipole"],
        components["joint_operator"],
    )
