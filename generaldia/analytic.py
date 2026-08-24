"""Analytic reference models used in examples and numerical tests."""

from __future__ import annotations

import torch
from torch import Tensor


class TwoStateAvoidedCrossing:
    """Two-state model ``[[aR, c], [c, -aR]]`` with analytic eigenvalues."""

    def __init__(self, slope: float = 0.05, coupling: float = 0.01) -> None:
        if slope == 0:
            raise ValueError("slope must be nonzero")
        if coupling < 0:
            raise ValueError("coupling cannot be negative")
        self.slope = float(slope)
        self.coupling = float(coupling)

    def hamiltonian(self, coordinate: Tensor | float) -> Tensor:
        """Return Hamiltonians with shape ``coordinate.shape + (2, 2)``."""

        coordinate = torch.as_tensor(coordinate)
        if not coordinate.is_floating_point():
            coordinate = coordinate.to(torch.get_default_dtype())
        if not torch.isfinite(coordinate).all():
            raise ValueError("coordinate must contain finite values")
        matrix = torch.zeros(
            (*coordinate.shape, 2, 2), dtype=coordinate.dtype, device=coordinate.device
        )
        matrix[..., 0, 0] = self.slope * coordinate
        matrix[..., 1, 1] = -self.slope * coordinate
        matrix[..., 0, 1] = self.coupling
        matrix[..., 1, 0] = self.coupling
        return matrix

    def exact_energies(self, coordinate: Tensor | float) -> Tensor:
        """Return analytic eigenvalues in ascending order."""

        coordinate = torch.as_tensor(coordinate)
        if not coordinate.is_floating_point():
            coordinate = coordinate.to(torch.get_default_dtype())
        magnitude = torch.sqrt((self.slope * coordinate) ** 2 + self.coupling**2)
        return torch.stack((-magnitude, magnitude), dim=-1)
