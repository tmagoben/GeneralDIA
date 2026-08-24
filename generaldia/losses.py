"""Loss functions for observable-constrained Hamiltonian learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .dataset import MolecularSample
from .observables import derivative_matrix_elements


@dataclass(frozen=True)
class LossWeights:
    """Relative weights for quantities that may use different physical units."""

    energy: float = 1.0
    energy_gradient: float = 0.0
    derivative_matrix: float = 0.0

    def __post_init__(self) -> None:
        values = (self.energy, self.energy_gradient, self.derivative_matrix)
        if any(value < 0 for value in values):
            raise ValueError("loss weights cannot be negative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one loss weight must be positive")


@dataclass
class LossBreakdown:
    """Differentiable total loss and detached component values."""

    total: Tensor
    energy: Tensor
    energy_gradient: Tensor | None
    derivative_matrix: Tensor | None

    def scalars(self) -> dict[str, float]:
        """Return numeric values for logging."""

        values = {"total": float(self.total.detach()), "energy": float(self.energy.detach())}
        if self.energy_gradient is not None:
            values["energy_gradient"] = float(self.energy_gradient.detach())
        if self.derivative_matrix is not None:
            values["derivative_matrix"] = float(self.derivative_matrix.detach())
        return values


def _mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    difference = prediction - target
    return torch.mean(torch.abs(difference) ** 2)


def observable_loss(
    model: nn.Module,
    sample: MolecularSample,
    weights: LossWeights | None = None,
) -> LossBreakdown:
    """Evaluate a weighted observable loss for one geometry.

    Energy-only supervision constrains the spectrum but does not identify a unique
    diabatic Hamiltonian. Gradient and derivative-matrix targets add information;
    matrix targets still require a consistent state gauge across the dataset.
    """

    if weights is None:
        weights = LossWeights()
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model must have trainable parameters")
    sample = sample.to(parameter.device, parameter.dtype)
    needs_derivatives = weights.energy_gradient > 0 or weights.derivative_matrix > 0
    if needs_derivatives:
        energies, _, derivative = derivative_matrix_elements(
            model, sample.atomic_numbers, sample.positions, create_graph=True
        )
    else:
        energies = torch.linalg.eigvalsh(model(sample.atomic_numbers, sample.positions))
        derivative = None

    energy_loss = _mean_squared_error(energies, sample.energies)
    total = weights.energy * energy_loss
    gradient_loss = None
    derivative_loss = None

    if weights.energy_gradient > 0:
        if sample.energy_gradients is None:
            raise ValueError("energy-gradient loss requested for a sample without gradients")
        if derivative is None:
            raise RuntimeError("internal error: derivative tensor was not calculated")
        predicted_gradients = torch.diagonal(derivative, dim1=-2, dim2=-1).permute(2, 0, 1)
        gradient_loss = _mean_squared_error(predicted_gradients.real, sample.energy_gradients)
        total = total + weights.energy_gradient * gradient_loss

    if weights.derivative_matrix > 0:
        if sample.derivative_matrix_elements is None:
            raise ValueError("derivative-matrix loss requested without matrix targets")
        if derivative is None:
            raise RuntimeError("internal error: derivative tensor was not calculated")
        target = sample.derivative_matrix_elements.to(
            device=derivative.device, dtype=derivative.dtype
        )
        derivative_loss = _mean_squared_error(derivative, target)
        total = total + weights.derivative_matrix * derivative_loss

    return LossBreakdown(total, energy_loss, gradient_loss, derivative_loss)
