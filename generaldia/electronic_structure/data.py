"""Validated electronic-structure result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ElectronicStructurePoint:
    """Reference observables for one molecular geometry.

    Positions use angstrom, energies use hartree, and gradients use hartree/bohr.
    ``scaled_nac_pyscf`` preserves PySCF's ``(ket, bra)`` state-key convention.
    """

    atomic_numbers: NDArray[np.integer]
    positions_angstrom: NDArray[np.floating]
    energies_hartree: NDArray[np.floating]
    gradients_hartree_per_bohr: NDArray[np.floating] | None = None
    scaled_nac_pyscf: dict[tuple[int, int], NDArray[np.floating]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atomic_numbers = np.asarray(self.atomic_numbers, dtype=np.int64)
        self.positions_angstrom = np.asarray(self.positions_angstrom, dtype=np.float64)
        self.energies_hartree = np.asarray(self.energies_hartree, dtype=np.float64)
        if self.atomic_numbers.ndim != 1 or self.atomic_numbers.size == 0:
            raise ValueError("atomic_numbers must have shape (N,) with N > 0")
        if np.any(self.atomic_numbers < 1):
            raise ValueError("atomic numbers must be positive")
        if self.positions_angstrom.shape != (self.atomic_numbers.size, 3):
            raise ValueError("positions_angstrom must have shape (N, 3)")
        if self.energies_hartree.ndim != 1 or self.energies_hartree.size == 0:
            raise ValueError("energies_hartree must have shape (S,) with S > 0")
        self._require_finite(self.positions_angstrom, "positions_angstrom")
        self._require_finite(self.energies_hartree, "energies_hartree")

        if self.gradients_hartree_per_bohr is not None:
            self.gradients_hartree_per_bohr = np.asarray(
                self.gradients_hartree_per_bohr, dtype=np.float64
            )
            expected = (self.energies_hartree.size, self.atomic_numbers.size, 3)
            if self.gradients_hartree_per_bohr.shape != expected:
                raise ValueError(f"gradients_hartree_per_bohr must have shape {expected}")
            self._require_finite(self.gradients_hartree_per_bohr, "gradients_hartree_per_bohr")

        converted: dict[tuple[int, int], NDArray[np.floating]] = {}
        for raw_key, raw_value in self.scaled_nac_pyscf.items():
            key = tuple(raw_key)
            if len(key) != 2 or not all(isinstance(index, (int, np.integer)) for index in key):
                raise ValueError("scaled NAC keys must be integer (ket, bra) pairs")
            ket, bra = map(int, key)
            if not (
                0 <= ket < self.energies_hartree.size and 0 <= bra < self.energies_hartree.size
            ):
                raise ValueError("scaled NAC state index lies outside the energy array")
            value = np.asarray(raw_value, dtype=np.float64)
            if value.shape != (self.atomic_numbers.size, 3):
                raise ValueError("each scaled NAC must have shape (N, 3)")
            self._require_finite(value, "scaled_nac_pyscf")
            converted[(ket, bra)] = value
        self.scaled_nac_pyscf = converted

    @staticmethod
    def _require_finite(value: NDArray[np.floating], name: str) -> None:
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this geometry."""

        return int(self.atomic_numbers.size)

    @property
    def n_states(self) -> int:
        """Number of electronic states in this result."""

        return int(self.energies_hartree.size)

    def save_npz(self, path: str | Path) -> None:
        """Save numerical arrays to a compressed NumPy archive.

        Metadata is omitted because arbitrary Python objects do not have a safe,
        portable NumPy representation. Store experiment metadata in JSON beside
        the archive.
        """

        nac_keys = np.asarray(list(self.scaled_nac_pyscf), dtype=np.int64).reshape(-1, 2)
        nac_values = (
            np.stack(list(self.scaled_nac_pyscf.values()))
            if self.scaled_nac_pyscf
            else np.empty((0, self.n_atoms, 3), dtype=np.float64)
        )
        gradients = (
            self.gradients_hartree_per_bohr
            if self.gradients_hartree_per_bohr is not None
            else np.empty((0, self.n_atoms, 3), dtype=np.float64)
        )
        np.savez_compressed(
            Path(path),
            atomic_numbers=self.atomic_numbers,
            positions_angstrom=self.positions_angstrom,
            energies_hartree=self.energies_hartree,
            gradients_hartree_per_bohr=gradients,
            has_gradients=np.asarray(self.gradients_hartree_per_bohr is not None),
            scaled_nac_keys=nac_keys,
            scaled_nac_values=nac_values,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> ElectronicStructurePoint:
        """Load arrays written by :meth:`save_npz` without enabling pickle."""

        with np.load(Path(path), allow_pickle=False) as archive:
            gradients = (
                archive["gradients_hartree_per_bohr"] if bool(archive["has_gradients"]) else None
            )
            keys = archive["scaled_nac_keys"]
            values = archive["scaled_nac_values"]
            scaled_nac = {
                tuple(map(int, key)): value for key, value in zip(keys, values, strict=True)
            }
            return cls(
                atomic_numbers=archive["atomic_numbers"],
                positions_angstrom=archive["positions_angstrom"],
                energies_hartree=archive["energies_hartree"],
                gradients_hartree_per_bohr=gradients,
                scaled_nac_pyscf=scaled_nac,
            )
