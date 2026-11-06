"""Small closed-shell singlet FCI paths with physical overlaps and dipoles from PySCF."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from numpy.typing import ArrayLike

from ..dataset import MolecularPath, MolecularSample
from .pyscf_backend import _atoms


@dataclass(frozen=True)
class FCIClosedLoop:
    """Physical cyclic FCI overlaps and their open-chain MolecularPath.

    `cyclic_overlaps[k]` is <Psi_k | Psi_(k+1) mod K> with final-to-first
    linkage computed using the *original* first electronic wavefunctions.
    """

    path: MolecularPath
    cyclic_overlaps: torch.Tensor


class FCIDipoleBackend:
    """All-electron singlet FCI in a small orbital basis; no frozen-core approximation.

    Positions are angstrom, total energies hartree, and total molecular dipoles
    e*bohr about laboratory origin zero. Closed-shell singlet references with
    explicitly declared integer molecular charge are admitted. State matrices use row=bra, column=ket. Cross-geometry overlaps use
    cross-AO integrals, not raw CI coefficient dot products. This adapter computes
    no nuclear gradients or derivative-coupling numerators.
    """

    def __init__(
        self,
        basis: str = "sto-3g",
        n_states: int = 2,
        conv_tol: float = 1e-10,
        max_orbitals: int = 8,
        charge: int = 0,
    ) -> None:
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("basis must be a non-empty string")
        if isinstance(n_states, bool) or int(n_states) != n_states or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        if not isfinite(conv_tol) or conv_tol <= 0:
            raise ValueError("conv_tol must be finite and positive")
        if isinstance(max_orbitals, bool) or int(max_orbitals) != max_orbitals or max_orbitals < 1:
            raise ValueError("max_orbitals must be a positive integer")
        if isinstance(charge, bool) or not isinstance(charge, (int, np.integer)):
            raise ValueError("charge must be an integer")
        self.basis, self.n_states = basis, int(n_states)
        self.conv_tol, self.max_orbitals = float(conv_tol), int(max_orbitals)
        self.charge = int(charge)

    def calculate_path(
        self,
        atomic_numbers: ArrayLike,
        positions_angstrom: ArrayLike,
        *,
        path_id: str,
        family_id: str | None = None,
    ) -> MolecularPath:
        """Calculate an open FCI path with physical adjacent-state overlaps."""

        path, _ = self._calculate_path(
            atomic_numbers,
            positions_angstrom,
            path_id=path_id,
            family_id=family_id,
            include_closure=False,
        )
        return path

    def calculate_loop(
        self,
        atomic_numbers: ArrayLike,
        positions_angstrom: ArrayLike,
        *,
        path_id: str,
        family_id: str | None = None,
    ) -> FCIClosedLoop:
        """Calculate a physical cyclic loop (K >= 3), including final-to-first.

        All geometries share atom ordering, orbital basis specification and
        neutral singlet FCI conventions; cross-AO integrals provide overlap
        between different AO spaces. The returned path remains an ordinary
        open-chain MolecularPath; only cyclic_overlaps contains the closure.
        """
        positions = np.asarray(positions_angstrom)
        if positions.ndim != 3 or positions.shape[0] < 3:
            raise ValueError("closed loops require at least three molecular geometries")
        path, cyclic = self._calculate_path(
            atomic_numbers,
            positions,
            path_id=path_id,
            family_id=family_id,
            include_closure=True,
        )
        if cyclic is None:
            raise RuntimeError("internal error: closed-loop overlaps were not produced")
        return FCIClosedLoop(path=path, cyclic_overlaps=cyclic)

    def _calculate_path(
        self,
        atomic_numbers: ArrayLike,
        positions_angstrom: ArrayLike,
        *,
        path_id: str,
        family_id: str | None,
        include_closure: bool,
    ) -> tuple[MolecularPath, torch.Tensor | None]:
        """Calculate a scan; optionally include direct physical closure."""
        try:
            import pyscf
            from pyscf import ao2mo, fci, gto, scf
        except ImportError as error:
            raise ImportError("install GeneralDIA with the 'pyscf' extra") from error
        numbers = np.asarray(atomic_numbers)
        positions = np.asarray(positions_angstrom, dtype=np.float64)
        if (
            numbers.ndim != 1
            or len(numbers) < 2
            or not np.isfinite(numbers).all()
            or np.any(numbers < 1)
            or np.any(numbers != numbers.astype(int))
        ):
            raise ValueError("atomic_numbers must contain at least two positive integers")
        electrons = int(numbers.sum()) - self.charge
        if electrons < 2 or electrons % 2:
            raise ValueError("closed-shell singlet FCI requires a positive even electron count")
        if (
            positions.ndim != 3
            or positions.shape[0] < 2
            or positions.shape[1:] != (len(numbers), 3)
        ):
            raise ValueError("positions_angstrom must have shape (K,N,3), K >= 2")
        metadata = {
            "backend": "PySCF",
            "backend_version": pyscf.__version__,
            "method": "all-electron singlet FCI",
            "basis": self.basis,
            "charge": self.charge,
            "spin": 0,
            "n_states": self.n_states,
            "conv_tol": self.conv_tol,
            "coordinate_unit": "angstrom",
            "energy_unit": "hartree",
            "dipole_unit": "e*bohr",
            "dipole_origin_bohr": [0.0, 0.0, 0.0],
            "dipole_translation_law": "mu(R+delta) = mu(R) + charge*delta_bohr*I",
            "state_matrix_convention": "row=bra,column=ket",
            "overlap_method": "FCI determinant overlaps using cross-AO integrals",
        }
        if family_id is not None:
            if not isinstance(family_id, str) or not family_id.strip():
                raise ValueError("family_id must be a non-empty string")
            metadata["molecular_family"] = family_id
        samples, overlaps, previous = [], [], None
        first_frame = None
        for coordinates in positions:
            mol = gto.M(
                atom=_atoms(numbers, coordinates),
                basis=self.basis,
                unit="Angstrom",
                charge=self.charge,
                spin=0,
                verbose=0,
            )
            if mol.nao_nr() > self.max_orbitals:
                raise ValueError("FCI orbital count exceeds max_orbitals")
            mf = scf.RHF(mol)
            mf.conv_tol = self.conv_tol
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("PySCF RHF reference did not converge")
            mo, norb, nelec = mf.mo_coeff, mol.nao_nr(), mol.nelec
            solver = fci.direct_spin0.FCI(mol)
            solver.conv_tol, solver.nroots = self.conv_tol, self.n_states
            energies, roots = solver.kernel(
                mo.T @ mf.get_hcore() @ mo,
                ao2mo.kernel(mol, mo),
                norb,
                nelec,
                ecore=mol.energy_nuc(),
            )
            if not np.all(solver.converged):
                raise RuntimeError("PySCF FCI calculation did not converge")
            energies = np.atleast_1d(energies)
            roots = [roots] if self.n_states == 1 else list(roots)
            if len(energies) != self.n_states or len(roots) != self.n_states:
                raise ValueError("requested state count exceeds the available singlet roots")
            order = np.argsort(energies, kind="stable")
            energies, roots = energies[order], [roots[i] for i in order]
            with mol.with_common_orig((0.0, 0.0, 0.0)):
                integrals = np.einsum("pi,xpq,qj->xij", mo, mol.intor("int1e_r", comp=3), mo)
            nuclear = np.einsum("a,ax->x", mol.atom_charges(), mol.atom_coords())
            dipoles = np.empty((3, self.n_states, self.n_states))
            for bra in range(self.n_states):
                for ket in range(self.n_states):
                    density = solver.trans_rdm1(roots[bra], roots[ket], norb, nelec)
                    dipoles[:, bra, ket] = -np.einsum("xpq,qp->x", integrals, density)
                    if bra == ket:
                        dipoles[:, bra, ket] += nuclear
            if first_frame is None:
                first_frame = (mol, mo, roots)
            if previous is not None:
                prior_mol, prior_mo, prior_roots = previous
                orbital_overlap = prior_mo.T @ gto.intor_cross("int1e_ovlp", prior_mol, mol) @ mo
                overlaps.append(
                    np.asarray(
                        [
                            [
                                fci.addons.overlap(bra, ket, norb, nelec, s=orbital_overlap)
                                for ket in roots
                            ]
                            for bra in prior_roots
                        ]
                    )
                )
            previous = mol, mo, roots
            samples.append(
                MolecularSample(
                    torch.as_tensor(numbers, dtype=torch.long),
                    torch.from_numpy(coordinates.copy()),
                    torch.from_numpy(energies.copy()),
                    metadata={**metadata, "converged": True},
                    dipole_matrix_elements=torch.from_numpy(dipoles),
                )
            )
        path = MolecularPath(
            samples,
            path_id=path_id,
            adjacent_overlaps=torch.from_numpy(np.stack(overlaps)),
            metadata=metadata,
        )
        if not include_closure:
            return path, None
        if first_frame is None or previous is None:
            raise RuntimeError("cannot close an empty electronic-state path")
        first_mol, first_mo, first_roots = first_frame
        last_mol, last_mo, last_roots = previous
        closing_orbital_overlap = (
            last_mo.T @ gto.intor_cross("int1e_ovlp", last_mol, first_mol) @ first_mo
        )
        closing = np.asarray(
            [
                [
                    fci.addons.overlap(
                        bra, ket, last_mo.shape[1], last_mol.nelec, s=closing_orbital_overlap
                    )
                    for ket in first_roots
                ]
                for bra in last_roots
            ]
        )
        if not np.isfinite(closing).all():
            raise ValueError("physical closing overlap contains nonfinite values")
        cyclic = torch.from_numpy(np.concatenate((np.stack(overlaps), closing[None]), axis=0))
        return path, cyclic
