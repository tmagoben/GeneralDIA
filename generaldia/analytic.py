"""Analytic reference models used in examples and numerical tests."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor


class TwoStateAvoidedCrossing:
    """Two-state model ``[[aR, c], [c, -aR]]`` with analytic eigenvalues."""

    def __init__(self, slope: float = 0.05, coupling: float = 0.01) -> None:
        if not isfinite(slope) or slope == 0:
            raise ValueError("slope must be finite and nonzero")
        if not isfinite(coupling) or coupling < 0:
            raise ValueError("coupling must be finite and nonnegative")
        self.slope = float(slope)
        self.coupling = float(coupling)

    def hamiltonian(self, coordinate: Tensor | float) -> Tensor:
        """Return Hamiltonians with shape ``coordinate.shape + (2, 2)``."""

        coordinate = torch.as_tensor(coordinate)
        if coordinate.is_complex():
            raise ValueError("coordinate must be real")
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
        if not torch.isfinite(coordinate).all():
            raise ValueError("coordinate must contain finite values")
        # hypot avoids intermediate squaring overflow for large finite coordinates.
        magnitude = torch.hypot(self.slope * coordinate, torch.full_like(coordinate, self.coupling))
        if not torch.isfinite(magnitude).all():
            raise ValueError("analytic energies overflowed finite floating-point range")
        return torch.stack((-magnitude, magnitude), dim=-1)
