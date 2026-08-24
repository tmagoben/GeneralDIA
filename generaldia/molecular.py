"""Invariant molecular representations and finite-state Hamiltonian models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .matrix import unpack_real_symmetric


class GaussianRBF(nn.Module):
    """Expand scalar distances in fixed Gaussian radial basis functions."""

    def __init__(
        self,
        n: int = 12,
        r_min: float = 0.0,
        r_max: float = 6.0,
        gamma: float | None = None,
    ) -> None:
        super().__init__()
        if int(n) < 1:
            raise ValueError("n must be a positive integer")
        if r_max <= r_min:
            raise ValueError("r_max must be greater than r_min")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be positive")

        centers = torch.linspace(float(r_min), float(r_max), int(n))
        spacing = (r_max - r_min) / max(int(n) - 1, 1)
        width = float(gamma) if gamma is not None else 1.0 / max(spacing, 1e-12) ** 2
        self.register_buffer("centers", centers)
        self.gamma = width

    def forward(self, distances: Tensor) -> Tensor:
        """Return RBF values with one new trailing feature dimension."""

        centers = self.centers.to(dtype=distances.dtype, device=distances.device)
        return torch.exp(-self.gamma * (distances[..., None] - centers) ** 2)


class SimpleMolecularHamiltonian(nn.Module):
    """Predict a real symmetric Hamiltonian from unordered atomic pairs.

    Coordinates enter through pair distances. Each pair uses ``e_i + e_j`` and
    ``abs(e_i - e_j)``, and the model sums all unordered pair contributions.
    These choices enforce translation, orthogonal-transformation, and atom-order
    invariance. The representation does not distinguish enantiomers because pair
    distances remain unchanged under reflection.

    The model accepts one molecule at a time: ``atomic_numbers`` has shape ``(N,)``
    and ``positions`` has shape ``(N, 3)``.
    """

    def __init__(
        self,
        n_states: int = 2,
        hidden: int = 32,
        n_rbf: int = 12,
        max_z: int = 36,
        r_min: float = 0.0,
        r_max: float = 6.0,
    ) -> None:
        super().__init__()
        for value, name in ((n_states, "n_states"), (hidden, "hidden"), (max_z, "max_z")):
            if int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.n_states = int(n_states)
        self.max_z = int(max_z)
        self.hidden = int(hidden)
        self.n_rbf = int(n_rbf)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.embed = nn.Embedding(self.max_z + 1, int(hidden), padding_idx=0)
        self.rbf = GaussianRBF(n_rbf, r_min=r_min, r_max=r_max)
        self.pair_net = nn.Sequential(
            nn.Linear(2 * int(hidden) + int(n_rbf), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(hidden)),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), self.n_states * (self.n_states + 1) // 2),
        )

    @property
    def model_dtype(self) -> torch.dtype:
        """Floating-point dtype used by the trainable parameters."""

        return self.embed.weight.dtype

    @property
    def model_device(self) -> torch.device:
        """Device used by the trainable parameters."""

        return self.embed.weight.device

    @property
    def configuration(self) -> dict[str, int | float]:
        """Constructor settings required to recreate the model architecture."""

        return {
            "n_states": self.n_states,
            "hidden": self.hidden,
            "n_rbf": self.n_rbf,
            "max_z": self.max_z,
            "r_min": self.r_min,
            "r_max": self.r_max,
        }

    def _validate_inputs(self, atomic_numbers: Tensor, positions: Tensor) -> None:
        if atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must have shape (N,)")
        if positions.ndim != 2 or positions.shape != (atomic_numbers.numel(), 3):
            raise ValueError("positions must have shape (N, 3)")
        if atomic_numbers.numel() < 2:
            raise ValueError("the all-pairs model requires at least two atoms")
        if torch.any(atomic_numbers < 1) or torch.any(atomic_numbers > self.max_z):
            raise ValueError(f"atomic numbers must lie in [1, {self.max_z}]")
        if not torch.isfinite(positions).all():
            raise ValueError("positions must contain finite values")

    def representation(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Return the invariant molecular feature vector with shape ``(hidden,)``."""

        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long, device=self.model_device)
        positions = torch.as_tensor(positions, dtype=self.model_dtype, device=self.model_device)
        self._validate_inputs(atomic_numbers, positions)

        pair_indices = torch.triu_indices(
            atomic_numbers.numel(), atomic_numbers.numel(), offset=1, device=self.model_device
        )
        left, right = pair_indices
        embeddings = self.embed(atomic_numbers)
        pair_distances = torch.linalg.vector_norm(positions[left] - positions[right], dim=-1)
        pair_features = torch.cat(
            (
                embeddings[left] + embeddings[right],
                torch.abs(embeddings[left] - embeddings[right]),
                self.rbf(pair_distances),
            ),
            dim=-1,
        )
        return self.pair_net(pair_features).sum(dim=0)

    def forward(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Return a real symmetric Hamiltonian with shape ``(S, S)``."""

        packed = self.head(self.representation(atomic_numbers, positions))
        return unpack_real_symmetric(packed, self.n_states)
