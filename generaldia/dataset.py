"""Validated in-memory datasets for observable-constrained training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from .electronic_structure.data import ElectronicStructurePoint
from .state_tracking import StateTrackingResult, track_states, transform_state_matrices

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


@dataclass(frozen=True)
class PathTrackingSettings:
    """Numerical policy used to turn raw path targets into one tracked gauge."""

    overlap_floor: float = 0.5
    assignment_margin_floor: float = 1e-6
    degeneracy_tolerance: float = 1e-8
    near_degeneracy_threshold: float | None = None
    on_ambiguous: Literal["raise", "record"] = "raise"

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_floor <= 1:
            raise ValueError("overlap_floor must lie between zero and one")
        if self.assignment_margin_floor < 0:
            raise ValueError("assignment_margin_floor cannot be negative")
        if self.degeneracy_tolerance < 0:
            raise ValueError("degeneracy_tolerance cannot be negative")
        if (
            self.near_degeneracy_threshold is not None
            and self.near_degeneracy_threshold < self.degeneracy_tolerance
        ):
            raise ValueError(
                "near_degeneracy_threshold cannot be smaller than degeneracy_tolerance"
            )
        if self.on_ambiguous not in {"raise", "record"}:
            raise ValueError('on_ambiguous must be either "raise" or "record"')


@dataclass(frozen=True)
class TrackedMolecularPath:
    """Raw path, covariantly transformed path, and the diagnostics linking them."""

    raw_path: MolecularPath
    tracked_path: MolecularPath
    tracking: StateTrackingResult
    settings: PathTrackingSettings

    @property
    def ambiguous_steps(self) -> tuple[int, ...]:
        """Starting indices of recorded ambiguous transitions."""

        return self.tracking.ambiguous_steps

    @property
    def tracked_energies(self) -> Tensor:
        """State-character energies, which need not remain in ascending order."""

        return self.tracked_path.energies


class MolecularPath(Sequence[MolecularSample]):
    """Ordered geometries that must remain together through splitting and tracking.

    The path stores physical adjacent-state overlaps separately from per-geometry
    targets. Overlaps may be omitted for energy-only grouped splitting, but
    :meth:`tracked` then remains unavailable.
    """

    def __init__(
        self,
        samples: Iterable[MolecularSample],
        *,
        path_id: str,
        adjacent_overlaps: Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._samples = tuple(samples)
        if len(self._samples) < 2:
            raise ValueError("a molecular path must contain at least two samples")
        if not all(isinstance(sample, MolecularSample) for sample in self._samples):
            raise TypeError("all path items must be MolecularSample instances")
        if not isinstance(path_id, str) or not path_id.strip():
            raise ValueError("path_id must be a non-empty string")
        self.path_id = path_id
        self.metadata = dict(metadata or {})

        first = self._samples[0]
        for sample in self._samples[1:]:
            if sample.n_states != first.n_states:
                raise ValueError("all samples in a path must contain the same number of states")
            if not torch.equal(sample.atomic_numbers, first.atomic_numbers):
                raise ValueError("atomic numbers and atom order must remain fixed within a path")
        gradient_presence = {sample.energy_gradients is not None for sample in self._samples}
        derivative_presence = {
            sample.derivative_matrix_elements is not None for sample in self._samples
        }
        if len(gradient_presence) != 1:
            raise ValueError("energy-gradient targets must be present at every path point or none")
        if len(derivative_presence) != 1:
            raise ValueError(
                "derivative-matrix targets must be present at every path point or none"
            )

        if adjacent_overlaps is None:
            self.adjacent_overlaps = None
        else:
            overlaps = torch.as_tensor(adjacent_overlaps)
            expected = (len(self._samples) - 1, first.n_states, first.n_states)
            if overlaps.shape != expected:
                raise ValueError(f"adjacent_overlaps must have shape {expected}")
            if not (overlaps.is_floating_point() or overlaps.is_complex()):
                overlaps = overlaps.to(torch.get_default_dtype())
            if not torch.isfinite(overlaps).all():
                raise ValueError("adjacent_overlaps must contain finite values")
            epsilon = torch.finfo(overlaps.real.dtype).eps
            tolerance = max(1e-8, 10 * epsilon)
            if float(torch.linalg.svdvals(overlaps).amax()) > 1.0 + tolerance:
                raise ValueError(
                    "each adjacent overlap must be a contraction with singular values <= 1"
                )
            self.adjacent_overlaps = overlaps

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int | slice) -> MolecularSample | tuple[MolecularSample, ...]:
        return self._samples[index]

    def __iter__(self) -> Iterator[MolecularSample]:
        return iter(self._samples)

    @property
    def n_states(self) -> int:
        """Number of electronic states shared by all path points."""

        return self._samples[0].n_states

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the fixed path ordering."""

        return self._samples[0].n_atoms

    @property
    def energies(self) -> Tensor:
        """Raw per-geometry energies with shape ``(n_geometries, n_states)``."""

        return torch.stack(tuple(sample.energies for sample in self._samples))

    def as_dataset(self) -> MolecularDataset:
        """Return samples tagged with immutable path identity and path position."""

        tagged = []
        for index, sample in enumerate(self._samples):
            metadata = dict(sample.metadata)
            metadata.update(
                {
                    "generaldia_path_id": self.path_id,
                    "generaldia_path_index": index,
                }
            )
            tagged.append(_copy_sample(sample, metadata=metadata))
        return MolecularDataset(tagged)

    def tracked(
        self,
        *,
        overlap_floor: float = 0.5,
        assignment_margin_floor: float = 1e-6,
        degeneracy_tolerance: float = 1e-8,
        near_degeneracy_threshold: float | None = None,
        on_ambiguous: Literal["raise", "record"] = "raise",
    ) -> TrackedMolecularPath:
        """Track states and transform every state-indexed target covariantly.

        Scalar energy gradients can only be reordered across nondegenerate matches.
        A degenerate block needs the full derivative-matrix target because individual
        gradients do not contain enough information to rotate the subspace.
        """

        if self.adjacent_overlaps is None:
            raise ValueError("state tracking requires adjacent_overlaps")
        settings = PathTrackingSettings(
            overlap_floor=overlap_floor,
            assignment_margin_floor=assignment_margin_floor,
            degeneracy_tolerance=degeneracy_tolerance,
            near_degeneracy_threshold=near_degeneracy_threshold,
            on_ambiguous=on_ambiguous,
        )
        tracking = track_states(
            self.adjacent_overlaps,
            energies=self.energies,
            overlap_floor=overlap_floor,
            assignment_margin_floor=assignment_margin_floor,
            degeneracy_tolerance=degeneracy_tolerance,
            near_degeneracy_threshold=near_degeneracy_threshold,
            on_ambiguous=on_ambiguous,
        )

        permutations = (
            tuple(range(self.n_states)),
            *(step.permutation for step in tracking.steps),
        )
        tracked_energies = torch.stack(
            tuple(
                sample.energies[list(permutation)]
                for sample, permutation in zip(self._samples, permutations, strict=True)
            )
        )

        raw_derivatives = None
        derivative_targets = None
        if self._samples[0].derivative_matrix_elements is not None:
            raw_derivatives = torch.stack(
                tuple(sample.derivative_matrix_elements for sample in self._samples)  # type: ignore[arg-type]
            )
            derivative_targets = transform_state_matrices(raw_derivatives, tracking.transformations)

        gradient_targets = None
        if self._samples[0].energy_gradients is not None:
            if derivative_targets is not None:
                if raw_derivatives is None:
                    raise RuntimeError("internal error: raw derivative targets are unavailable")
                raw_gradients = torch.stack(
                    tuple(sample.energy_gradients for sample in self._samples)  # type: ignore[arg-type]
                )
                derivative_diagonal = torch.diagonal(
                    raw_derivatives, dim1=-2, dim2=-1
                ).real.permute(0, 3, 1, 2)
                if not torch.allclose(
                    derivative_diagonal,
                    raw_gradients,
                    atol=1e-9,
                    rtol=1e-7,
                ):
                    raise ValueError(
                        "energy_gradients must match the diagonal of "
                        "derivative_matrix_elements before path tracking"
                    )
                gradient_targets = torch.diagonal(
                    derivative_targets, dim1=-2, dim2=-1
                ).real.permute(0, 3, 1, 2)
            else:
                if any(step.degenerate_blocks for step in tracking.steps):
                    raise ValueError(
                        "degenerate-subspace gradient tracking requires full "
                        "derivative_matrix_elements"
                    )
                gradient_targets = torch.stack(
                    tuple(
                        sample.energy_gradients[list(permutation)]  # type: ignore[index]
                        for sample, permutation in zip(self._samples, permutations, strict=True)
                    )
                )

        tracked_samples = []
        for index, (sample, energies) in enumerate(
            zip(self._samples, tracked_energies, strict=True)
        ):
            metadata = dict(sample.metadata)
            metadata.update(
                {
                    "generaldia_path_id": self.path_id,
                    "generaldia_path_index": index,
                    "state_tracking_ambiguous_incoming": (
                        False if index == 0 else tracking.steps[index - 1].ambiguous
                    ),
                }
            )
            tracked_samples.append(
                MolecularSample(
                    atomic_numbers=sample.atomic_numbers,
                    positions=sample.positions,
                    energies=energies,
                    energy_gradients=(
                        None if gradient_targets is None else gradient_targets[index]
                    ),
                    derivative_matrix_elements=(
                        None if derivative_targets is None else derivative_targets[index]
                    ),
                    metadata=metadata,
                )
            )

        tracked_path = MolecularPath(
            tracked_samples,
            path_id=self.path_id,
            adjacent_overlaps=tracking.aligned_overlaps,
            metadata={**self.metadata, "state_tracking_applied": True},
        )
        return TrackedMolecularPath(self, tracked_path, tracking, settings)


class MolecularPathDataset(Sequence[MolecularPath]):
    """Collection of complete paths with deterministic path-level splitting."""

    def __init__(self, paths: Iterable[MolecularPath]) -> None:
        self._paths = tuple(paths)
        if not self._paths:
            raise ValueError("path dataset must contain at least one path")
        if not all(isinstance(path, MolecularPath) for path in self._paths):
            raise TypeError("all path dataset items must be MolecularPath instances")
        identifiers = [path.path_id for path in self._paths]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("path_id values must be unique within a path dataset")
        state_counts = {path.n_states for path in self._paths}
        if len(state_counts) != 1:
            raise ValueError("all paths must contain the same number of states")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int | slice) -> MolecularPath | MolecularPathDataset:
        result = self._paths[index]
        if isinstance(index, slice):
            return MolecularPathDataset(result)
        return result

    def __iter__(self) -> Iterator[MolecularPath]:
        return iter(self._paths)

    @property
    def n_states(self) -> int:
        """Number of states shared by every path."""

        return self._paths[0].n_states

    @property
    def n_samples(self) -> int:
        """Total number of geometries without discarding path membership."""

        return sum(len(path) for path in self._paths)

    @property
    def path_ids(self) -> tuple[str, ...]:
        """Ordered path identifiers."""

        return tuple(path.path_id for path in self._paths)

    def as_dataset(self) -> MolecularDataset:
        """Flatten paths after splitting while retaining path metadata."""

        return MolecularDataset(sample for path in self._paths for sample in path.as_dataset())

    def split(
        self,
        fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
        *,
        seed: int = 0,
    ) -> tuple[MolecularPathDataset, MolecularPathDataset, MolecularPathDataset]:
        """Split complete paths so no trajectory leaks across partitions."""

        if len(self) < 3:
            raise ValueError("a three-way path split requires at least three paths")
        train_count, validation_count = _partition_counts(len(self), fractions)
        indices = np.random.default_rng(seed).permutation(len(self))
        validation_end = train_count + validation_count
        return (
            MolecularPathDataset(self._paths[index] for index in indices[:train_count]),
            MolecularPathDataset(
                self._paths[index] for index in indices[train_count:validation_end]
            ),
            MolecularPathDataset(self._paths[index] for index in indices[validation_end:]),
        )


def _partition_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int]:
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    total = sum(fractions)
    normalized = tuple(value / total for value in fractions)
    train_count = max(1, int(np.floor(normalized[0] * size)))
    validation_count = max(1, int(np.floor(normalized[1] * size)))
    if train_count + validation_count >= size:
        train_count = size - 2
        validation_count = 1
    return train_count, validation_count


def _copy_sample(sample: MolecularSample, *, metadata: dict[str, Any]) -> MolecularSample:
    return MolecularSample(
        atomic_numbers=sample.atomic_numbers,
        positions=sample.positions,
        energies=sample.energies,
        energy_gradients=sample.energy_gradients,
        derivative_matrix_elements=sample.derivative_matrix_elements,
        metadata=metadata,
    )
