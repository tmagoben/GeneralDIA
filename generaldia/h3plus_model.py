"""Local, permutation-covariant A1 + E Hamiltonian for H3+.

The three electronic coordinates label bonds (01, 02, 12), in one fixed
latent basis. They are NOT CI coefficients or cross-geometry molecular
orbitals. This restricted model has a spectator A1 ground state and a
symmetry-enforced E intersection; fitting spectra does not learn a gauge
connection. Distances are in angstrom and energies in hartree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def h3plus_bonds(positions: Tensor) -> Tensor:
    """Return distances (01, 02, 12) for real finite (..., 3, 3) positions."""
    positions = torch.as_tensor(positions)
    if positions.shape[-2:] != (3, 3) or positions.ndim < 2:
        raise ValueError("positions must have shape (..., 3, 3)")
    if not positions.is_floating_point() or not torch.isfinite(positions).all():
        raise ValueError("positions must be finite real floating-point coordinates")
    return torch.stack(
        [
            torch.linalg.vector_norm(positions[..., i, :] - positions[..., j, :], dim=-1)
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
        dim=-1,
    )


class H3PlusEquivariantHamiltonian(nn.Module):
    """Smooth local A1 + E model, covariant under every bond permutation.

    Six invariant polynomial features fit each of the ground energy and E
    centroid. Four coefficients fit the linear/quadratic E splitting. The
    E block is exactly degenerate at equilateral geometries. Validity is
    limited to the fitted local domain; there is no A1--E mixing or dipole.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scalar_coefficients = nn.Parameter(torch.zeros(2, 6, dtype=torch.float64))
        self.splitting_coefficients = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        a1 = torch.ones(3, 3, dtype=torch.float64) / 3
        self.register_buffer("a1_projector", a1)
        self.register_buffer("e_projector", torch.eye(3, dtype=torch.float64) - a1)

    def _features(self, bonds: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bonds = torch.as_tensor(bonds, device=self.a1_projector.device)
        if bonds.ndim < 1 or bonds.shape[-1] != 3:
            raise ValueError("bonds must have shape (..., 3)")
        if not bonds.is_floating_point() or not torch.isfinite(bonds).all():
            raise ValueError("bonds must contain finite real floating-point distances")
        if bool((bonds <= 0).any()) or bool(
            (2 * bonds.max(dim=-1).values >= bonds.sum(dim=-1)).any()
        ):
            raise ValueError("bonds must form a nondegenerate triangle")
        bonds = bonds.to(dtype=self.a1_projector.dtype)
        mean = bonds.mean(dim=-1)
        t = (mean - 0.875) / 0.05
        d = (bonds - mean[..., None]) / 0.06
        r2 = d.square().sum(dim=-1)
        features = torch.stack(
            (torch.ones_like(t), t, t.square(), r2, t * r2, d.pow(3).sum(dim=-1)), dim=-1
        )
        return features, d, t, r2

    def forward(self, bonds: Tensor) -> Tensor:
        """Return a symmetric (..., 3, 3) Hamiltonian from bond distances."""
        features, d, t, r2 = self._features(bonds)
        scalar = features @ self.scalar_coefficients.T
        linear = self.e_projector @ torch.diag_embed(d) @ self.e_projector
        quadratic = self.e_projector @ torch.diag_embed(d.square()) @ self.e_projector
        quadratic = quadratic - (r2 / 3)[..., None, None] * self.e_projector
        c = self.splitting_coefficients
        amplitude = c[0] + c[1] * t + c[2] * r2
        return (
            scalar[..., 0, None, None] * self.a1_projector
            + scalar[..., 1, None, None] * self.e_projector
            + amplitude[..., None, None] * linear
            + c[3] * quadratic
        )

    def from_positions(self, positions: Tensor) -> Tensor:
        """Evaluate the Hamiltonian from three proton coordinates."""
        return self(h3plus_bonds(positions))


@dataclass(frozen=True)
class H3PlusFit:
    """A fitted model and independently recomputable training residual."""

    model: H3PlusEquivariantHamiltonian
    seed: int
    training_mae_hartree: float
    training_max_error_hartree: float


def fit_h3plus_spectra(bonds: Tensor, energies: Tensor, *, seed: int = 0) -> H3PlusFit:
    """Fit three ordered energies using least squares and seeded gap optimization.

    Only the supplied training geometries/energies are used. The fit runs
    in CPU float64. Scalar terms are linear least squares; a small LBFGS
    problem fits the E gap. Seed controls its initialization, not a split.
    Returned errors are diagnostics, not a held-out acceptance certificate.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    model = H3PlusEquivariantHamiltonian()
    bonds = torch.as_tensor(bonds).detach().cpu()
    features, _, _, _ = model._features(bonds)
    energies = torch.as_tensor(energies).detach().cpu()
    if (
        bonds.ndim != 2
        or len(bonds) < 6
        or energies.shape != bonds.shape
        or not energies.is_floating_point()
        or not torch.isfinite(energies).all()
        or bool((torch.diff(energies, dim=-1) < 0).any())
    ):
        raise ValueError("training requires >=6 triangles and ordered finite real (N, 3) energies")
    energies = energies.to(torch.float64)
    if torch.linalg.matrix_rank(features) < 6:
        raise ValueError("training geometries must span all six invariant features")
    targets = torch.stack((energies[:, 0], energies[:, 1:].mean(dim=-1)), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.scalar_coefficients.copy_(torch.linalg.lstsq(features, targets).solution.T)
        model.splitting_coefficients.copy_(
            0.01 * torch.randn(4, generator=generator, dtype=torch.float64)
        )
        model.splitting_coefficients[0] += 0.05
    model.scalar_coefficients.requires_grad_(False)
    optimizer = torch.optim.LBFGS(
        [model.splitting_coefficients],
        max_iter=200,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        prediction = torch.linalg.eigvalsh(model(bonds))
        loss = (prediction - energies).square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.scalar_coefficients.requires_grad_(True)
    with torch.no_grad():
        errors = (torch.linalg.eigvalsh(model(bonds)) - energies).abs()
    if not torch.isfinite(errors).all():
        raise ValueError("spectral fit produced nonfinite predictions")
    return H3PlusFit(model, seed, float(errors.mean()), float(errors.max()))
