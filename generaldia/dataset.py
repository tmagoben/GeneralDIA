"""Validated in-memory datasets for observable-constrained training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .electronic_structure.data import ElectronicStructurePoint

ANGSTROM_TO_BOHR = 1.889_726_125_457_828_1


@dataclass
class MolecularSample:
    """Geometry and reference observables for one molecule.

    ``positions`` has shape ``(N, 3)`` and ``energies`` has shape ``(S,)``.
    Optional gradients have shape ``(S, N, 3)``. Optional Hamiltonian-derivative
    matrix elements have shape ``(N, 3, S, S)`` and require a documented gauge.
    """

    atomic_numbers: Tensor
    positions: Tensor
    energies: Tensor
    energy_gradients: Tensor | None = None
    derivative_matrix_elements: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atomic_numbers = torch.as_tensor(self.atomic_numbers, dtype=torch.long)
        self.positions = self._floating_tensor(self.positions)
        self.energies = self._floating_tensor(self.energies)
        if self.atomic_numbers.ndim != 1 or self.atomic_numbers.numel() < 2:
            raise ValueError("atomic_numbers must have shape (N,) with N >= 2")
        if torch.any(self.atomic_numbers < 1):
            raise ValueError("atomic numbers must be positive")
        if self.positions.shape != (self.atomic_numbers.numel(), 3):
            raise ValueError("positions must have shape (N, 3)")
        if self.energies.ndim != 1 or self.energies.numel() < 1:
            raise ValueError("energies must have shape (S,) with S >= 1")
        self._require_finite(self.positions, "positions")
        self._require_finite(self.energies, "energies")

        if self.energy_gradients is not None:
            self.energy_gradients = self._floating_tensor(self.energy_gradients)
            expected = (self.n_states, self.n_atoms, 3)
            if self.energy_gradients.shape != expected:
                raise ValueError(f"energy_gradients must have shape {expected}")
            self._require_finite(self.energy_gradients, "energy_gradients")

        if self.derivative_matrix_elements is not None:
            self.derivative_matrix_elements = torch.as_tensor(self.derivative_matrix_elements)
            if not (
                self.derivative_matrix_elements.is_floating_point()
                or self.derivative_matrix_elements.is_complex()
            ):
                self.derivative_matrix_elements = self.derivative_matrix_elements.to(
                    torch.get_default_dtype()
                )
            expected = (self.n_atoms, 3, self.n_states, self.n_states)
            if self.derivative_matrix_elements.shape != expected:
                raise ValueError(f"derivative_matrix_elements must have shape {expected}")
            self._require_finite(self.derivative_matrix_elements, "derivative_matrix_elements")
            if not torch.allclose(
                self.derivative_matrix_elements,
                self.derivative_matrix_elements.mH,
                atol=1e-9,
                rtol=1e-7,
            ):
                raise ValueError("derivative matrix elements must be Hermitian")

    @staticmethod
    def _floating_tensor(value: Tensor) -> Tensor:
        tensor = torch.as_tensor(value)
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.get_default_dtype())
        return tensor

    @staticmethod
    def _require_finite(value: Tensor, name: str) -> None:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the geometry."""

        return int(self.atomic_numbers.numel())

    @property
    def n_states(self) -> int:
        """Number of target electronic states."""

        return int(self.energies.numel())

    def to(self, device: torch.device | str, dtype: torch.dtype) -> MolecularSample:
        """Return a copy on ``device`` with floating tensors converted to ``dtype``."""

        return MolecularSample(
            atomic_numbers=self.atomic_numbers.to(device=device),
            positions=self.positions.to(device=device, dtype=dtype),
            energies=self.energies.to(device=device, dtype=dtype),
            energy_gradients=(
                None
                if self.energy_gradients is None
                else self.energy_gradients.to(device=device, dtype=dtype)
            ),
            derivative_matrix_elements=(
                None
                if self.derivative_matrix_elements is None
                else self.derivative_matrix_elements.to(device=device)
            ),
            metadata=dict(self.metadata),
        )


class MolecularDataset(Sequence[MolecularSample]):
    """A validated sequence of molecular samples with a shared state count."""

    def __init__(self, samples: Iterable[MolecularSample]) -> None:
        self._samples = tuple(samples)
        if not self._samples:
            raise ValueError("dataset must contain at least one sample")
        if not all(isinstance(sample, MolecularSample) for sample in self._samples):
            raise TypeError("all dataset items must be MolecularSample instances")
        state_counts = {sample.n_states for sample in self._samples}
        if len(state_counts) != 1:
            raise ValueError("all samples must contain the same number of states")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int | slice) -> MolecularSample | MolecularDataset:
        result = self._samples[index]
        if isinstance(index, slice):
            return MolecularDataset(result)
        return result

    def __iter__(self) -> Iterator[MolecularSample]:
        return iter(self._samples)

    @property
    def n_states(self) -> int:
        """Shared number of electronic states."""

        return self._samples[0].n_states

    def split(
        self,
        fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
        *,
        seed: int = 0,
    ) -> tuple[MolecularDataset, MolecularDataset, MolecularDataset]:
        """Create deterministic train, validation, and test partitions."""

        if len(self) < 3:
            raise ValueError("a three-way split requires at least three samples")
        if len(fractions) != 3 or any(value <= 0 for value in fractions):
            raise ValueError("fractions must contain three positive values")
        total = sum(fractions)
        normalized = tuple(value / total for value in fractions)
        train_count = max(1, int(np.floor(normalized[0] * len(self))))
        validation_count = max(1, int(np.floor(normalized[1] * len(self))))
        if train_count + validation_count >= len(self):
            train_count = len(self) - 2
            validation_count = 1

        indices = np.random.default_rng(seed).permutation(len(self))
        train_end = train_count
        validation_end = train_end + validation_count
        return (
            MolecularDataset(self._samples[index] for index in indices[:train_end]),
            MolecularDataset(self._samples[index] for index in indices[train_end:validation_end]),
            MolecularDataset(self._samples[index] for index in indices[validation_end:]),
        )

    @classmethod
    def from_electronic_structure(
        cls, points: Iterable[ElectronicStructurePoint]
    ) -> MolecularDataset:
        """Convert PySCF-style points to angstrom-coordinate training samples.

        PySCF gradients use hartree/bohr. The model differentiates with respect to
        angstrom coordinates, so this method multiplies gradients by
        ``ANGSTROM_TO_BOHR`` to obtain hartree/angstrom.
        """

        samples = []
        for point in points:
            gradients = point.gradients_hartree_per_bohr
            if gradients is not None:
                gradients = gradients * ANGSTROM_TO_BOHR
            metadata = dict(point.metadata)
            metadata.update(
                {
                    "coordinate_unit": "angstrom",
                    "energy_unit": "hartree",
                    "gradient_unit": "hartree/angstrom",
                }
            )
            samples.append(
                MolecularSample(
                    atomic_numbers=point.atomic_numbers,
                    positions=point.positions_angstrom,
                    energies=point.energies_hartree,
                    energy_gradients=gradients,
                    metadata=metadata,
                )
            )
        return cls(samples)
