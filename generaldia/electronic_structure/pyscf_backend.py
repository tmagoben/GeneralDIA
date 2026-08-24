"""Optional PySCF reference-data adapters with explicit units and provenance."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .data import ElectronicStructurePoint

_SYMBOLS = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
]


def _require() -> tuple[Any, Any, Any, str]:
    try:
        import pyscf
        from pyscf import gto, mcscf, scf
    except ImportError as error:
        raise ImportError("install GeneralDIA with the 'pyscf' extra") from error
    return gto, scf, mcscf, pyscf.__version__


def _atoms(atomic_numbers: ArrayLike, positions: ArrayLike) -> list[tuple[str, tuple[float, ...]]]:
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
        raise ValueError("atomic_numbers must have shape (N,) with N > 0")
    if positions.shape != (atomic_numbers.size, 3):
        raise ValueError("positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain finite values")
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers >= len(_SYMBOLS)):
        raise ValueError("the bundled symbol table supports atomic numbers 1 through 20")
    return [
        (_SYMBOLS[int(atomic_number)], tuple(map(float, position)))
        for atomic_number, position in zip(atomic_numbers, positions, strict=True)
    ]


class RHFBackend:
    """Restricted Hartree-Fock energies and analytic nuclear gradients."""

    def __init__(
        self,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        conv_tol: float = 1e-10,
    ) -> None:
        if not basis:
            raise ValueError("basis cannot be empty")
        if conv_tol <= 0:
            raise ValueError("conv_tol must be positive")
        self.basis = basis
        self.charge = int(charge)
        self.spin = int(spin)
        self.conv_tol = float(conv_tol)

    def calculate(
        self, atomic_numbers: ArrayLike, positions_angstrom: ArrayLike
    ) -> ElectronicStructurePoint:
        """Run RHF for one geometry supplied in angstrom."""

        gto, scf, _, version = _require()
        molecule = gto.M(
            atom=_atoms(atomic_numbers, positions_angstrom),
            basis=self.basis,
            unit="Angstrom",
            charge=self.charge,
            spin=self.spin,
            verbose=0,
        )
        mean_field = scf.RHF(molecule)
        mean_field.conv_tol = self.conv_tol
        energy = mean_field.kernel()
        if not mean_field.converged:
            raise RuntimeError("PySCF RHF calculation did not converge")
        gradient = mean_field.nuc_grad_method().kernel()
        return ElectronicStructurePoint(
            atomic_numbers,
            positions_angstrom,
            [energy],
            np.asarray([gradient]),
            metadata={
                "backend": "PySCF",
                "backend_version": version,
                "method": "RHF",
                "basis": self.basis,
                "charge": self.charge,
                "spin": self.spin,
                "coordinate_unit": "angstrom",
                "energy_unit": "hartree",
                "gradient_unit": "hartree/bohr",
                "converged": True,
            },
        )


class SACASSCFBackend:
    """Equal-weight SA-CASSCF energies, gradients, and PySCF-scaled NACs.

    PySCF defines ``state=(ket, bra)`` and returns ``<bra|d ket>``. GeneralDIA
    preserves that key order. With ``mult_ediff=True``, PySCF supplies its scaled
    NAC quantity; the adapter does not reinterpret the sign or index convention.
    """

    def __init__(
        self,
        ncas: int,
        nelecas: int | tuple[int, int],
        n_states: int = 2,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        conv_tol: float = 1e-10,
        use_etfs: bool = False,
    ) -> None:
        if int(ncas) < 1 or int(n_states) < 2:
            raise ValueError("ncas must be positive and n_states must be at least two")
        if conv_tol <= 0:
            raise ValueError("conv_tol must be positive")
        self.ncas = int(ncas)
        self.nelecas = nelecas
        self.n_states = int(n_states)
        self.basis = basis
        self.charge = int(charge)
        self.spin = int(spin)
        self.conv_tol = float(conv_tol)
        self.use_etfs = bool(use_etfs)

    def calculate(
        self, atomic_numbers: ArrayLike, positions_angstrom: ArrayLike
    ) -> ElectronicStructurePoint:
        """Run an equal-weight SA-CASSCF calculation for one geometry."""

        gto, scf, mcscf, version = _require()
        molecule = gto.M(
            atom=_atoms(atomic_numbers, positions_angstrom),
            basis=self.basis,
            unit="Angstrom",
            charge=self.charge,
            spin=self.spin,
            verbose=0,
        )
        mean_field = scf.RHF(molecule)
        mean_field.conv_tol = self.conv_tol
        mean_field.kernel()
        if not mean_field.converged:
            raise RuntimeError("PySCF RHF reference did not converge")
        weights = [1.0 / self.n_states] * self.n_states
        casscf = mcscf.CASSCF(mean_field, self.ncas, self.nelecas).state_average(weights)
        casscf.conv_tol = self.conv_tol
        casscf.kernel()
        if not casscf.converged:
            raise RuntimeError("PySCF SA-CASSCF calculation did not converge")
        energies = np.asarray(casscf.e_states, dtype=np.float64)
        gradient_method = casscf.nuc_grad_method()
        gradients = np.stack(
            [np.asarray(gradient_method.kernel(state=state)) for state in range(self.n_states)]
        )
        nac_method = casscf.nac_method()
        scaled = {}
        for ket in range(self.n_states):
            for bra in range(ket + 1, self.n_states):
                scaled[(ket, bra)] = np.asarray(
                    nac_method.kernel(state=(ket, bra), mult_ediff=True, use_etfs=self.use_etfs)
                )
        return ElectronicStructurePoint(
            atomic_numbers,
            positions_angstrom,
            energies,
            gradients,
            scaled,
            metadata={
                "backend": "PySCF",
                "backend_version": version,
                "method": "SA-CASSCF",
                "basis": self.basis,
                "charge": self.charge,
                "spin": self.spin,
                "ncas": self.ncas,
                "nelecas": self.nelecas,
                "n_states": self.n_states,
                "weights": weights,
                "use_etfs": self.use_etfs,
                "coordinate_unit": "angstrom",
                "energy_unit": "hartree",
                "gradient_unit": "hartree/bohr",
                "scaled_nac_convention": "PySCF mult_ediff=True, key=(ket, bra)",
                "converged": True,
            },
        )
