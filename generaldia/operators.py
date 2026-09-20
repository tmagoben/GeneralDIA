"""Minimal scalar Hamiltonian and polar-vector dipoles in a shared electronic basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .matrix import complex_hermitian_from_parts, unpack_real_symmetric
from .molecular import SimpleMolecularHamiltonian


@dataclass
class MolecularOperators:
    """Latent H ``(S,S)`` and dipoles ``(3,S,S)`` in one fixed electronic basis.

    The dipole axes are laboratory Cartesian axes. Units are set by the training
    data. Neither field is diagonalized independently to define a reporting gauge.
    """

    hamiltonian: Tensor
    dipoles: Tensor


class SharedMolecularOperators(SimpleMolecularHamiltonian):
    """Neutral-molecule real/complex Hermitian operators from a shared pair encoder.

    H is invariant to rigid spatial rotations. Dipoles use Hermitian atomic weights
    Q_a and centered polar vectors: mu = sum_a (R_a - mean(R)) Q_a. This enforces
    O(3) covariance, translation invariance, and atom-permutation invariance. It is
    a small reference architecture, restricted to neutral systems and vectors in
    the span of the centered atomic positions. Equivalent atomic environments
    share weights, forcing all dipoles to zero for homonuclear diatomics; allowed
    transition dipoles in such systems need a richer representation. Complex mode
    alone imposes no SOC, time-reversal or Kramers constraints. Coordinates use
    ``(N,3)`` input units; learned weights carry dipole/coordinate units. Raises
    ``ValueError`` for nonzero charge, invalid mode or invalid molecular inputs.
    """

    def __init__(
        self,
        n_states: int = 2,
        hidden: int = 32,
        n_rbf: int = 12,
        max_z: int = 36,
        r_min: float = 0.0,
        r_max: float = 6.0,
        mode: Literal["real", "complex"] = "real",
        charge: int = 0,
    ) -> None:
        if mode not in {"real", "complex"}:
            raise ValueError("mode must be 'real' or 'complex'")
        if charge != 0:
            raise ValueError("SharedMolecularOperators currently supports neutral molecules only")
        super().__init__(n_states, hidden, n_rbf, max_z, r_min, r_max)
        self.mode = mode
        self.charge = 0
        packed = self.n_states**2 if mode == "complex" else self.n_states * (self.n_states + 1) // 2
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, packed))
        self.dipole_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, packed)
        )

    @property
    def configuration(self) -> dict[str, int | float | str]:
        """Settings needed to restore this exact model architecture."""

        return {**super().configuration, "mode": self.mode, "charge": self.charge}

    def _features(self, atomic_numbers: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        numbers = torch.as_tensor(atomic_numbers, dtype=torch.long, device=self.model_device)
        coordinates = torch.as_tensor(positions, dtype=self.model_dtype, device=self.model_device)
        self._validate_inputs(numbers, coordinates)
        left, right = torch.triu_indices(
            len(numbers), len(numbers), offset=1, device=numbers.device
        )
        embeddings = self.embed(numbers)
        distances = torch.linalg.vector_norm(coordinates[left] - coordinates[right], dim=-1)
        pairs = self.pair_net(
            torch.cat(
                (
                    embeddings[left] + embeddings[right],
                    torch.abs(embeddings[left] - embeddings[right]),
                    self.rbf(distances),
                ),
                dim=-1,
            )
        )
        pooled = pairs.sum(dim=0)
        local = torch.zeros_like(embeddings).index_add(0, left, pairs).index_add(0, right, pairs)
        nodes = torch.cat((embeddings, local, pooled.expand(len(numbers), -1)), dim=-1)
        return coordinates, pooled, nodes

    def _matrix(self, packed: Tensor) -> Tensor:
        if self.mode == "real":
            return unpack_real_symmetric(packed, self.n_states)
        n, upper = self.n_states, self.n_states * (self.n_states - 1) // 2
        return complex_hermitian_from_parts(
            packed[..., :n], packed[..., n : n + upper], packed[..., n + upper :]
        )

    def representation(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Return the shared invariant molecular representation, shape ``(hidden,)``."""

        return self._features(atomic_numbers, positions)[1]

    def forward(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Return H ``(S,S)``; compatible with the existing Hamiltonian interfaces."""

        return self._matrix(self.head(self.representation(atomic_numbers, positions)))

    def operators(self, atomic_numbers: Tensor, positions: Tensor) -> MolecularOperators:
        """Return H and vector dipoles in one latent basis from one encoder evaluation."""

        coordinates, pooled, nodes = self._features(atomic_numbers, positions)
        hamiltonian = self._matrix(self.head(pooled))
        weights = self._matrix(self.dipole_head(nodes))
        centered = coordinates - coordinates.mean(dim=0)
        dipoles = torch.einsum("ac,aij->cij", centered.to(weights.dtype), weights)
        return MolecularOperators(hamiltonian, dipoles)
