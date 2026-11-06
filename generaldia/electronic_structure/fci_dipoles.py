"""Small neutral singlet FCI paths with physical overlaps and dipoles from PySCF."""

from __future__ import annotations

from math import isfinite

import numpy as np
import torch
from numpy.typing import ArrayLike

from ..dataset import MolecularPath, MolecularSample
from .pyscf_backend import _atoms


class FCIDipoleBackend:
    """All-electron singlet FCI in a small orbital basis; no frozen-core approximation.

    Positions are angstrom, total energies hartree, and total molecular dipoles
    e*bohr about laboratory origin zero. Only neutral closed-shell references are
    admitted. State matrices use row=bra, column=ket. Cross-geometry overlaps use
    cross-AO integrals, not raw CI coefficient dot products. This adapter computes
    no nuclear gradients or derivative-coupling numerators.
    """

    def __init__(
        self,
        basis: str = "sto-3g",
        n_states: int = 2,
        conv_tol: float = 1e-10,
        max_orbitals: int = 8,
    ) -> None:
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("basis must be a non-empty string")
        if isinstance(n_states, bool) or int(n_states) != n_states or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        if not isfinite(conv_tol) or conv_tol <= 0:
            raise ValueError("conv_tol must be finite and positive")
        if isinstance(max_orbitals, bool) or int(max_orbitals) != max_orbitals or max_orbitals < 1:
            raise ValueError("max_orbitals must be a positive integer")
        self.basis, self.n_states = basis, int(n_states)
        self.conv_tol, self.max_orbitals = float(conv_tol), int(max_orbitals)

    def calculate_path(
        self,
        atomic_numbers: ArrayLike,
        positions_angstrom: ArrayLike,
        *,
        path_id: str,
        family_id: str | None = None,
    ) -> MolecularPath:
        """Calculate a ``(K,N,3)`` scan and return a validated overlap-bearing path.

        Raises on unsupported electron counts, oversized orbital spaces, or failed
        RHF/FCI convergence. Call ``tracked()`` to assess the selected manifold's
        continuation; the adapter does not suppress tracking ambiguity.
        """

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
        if int(numbers.sum()) % 2:
            raise ValueError("the neutral singlet adapter requires an even electron count")
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
            "charge": 0,
            "spin": 0,
            "n_states": self.n_states,
            "conv_tol": self.conv_tol,
            "coordinate_unit": "angstrom",
            "energy_unit": "hartree",
            "dipole_unit": "e*bohr",
            "dipole_origin_bohr": [0.0, 0.0, 0.0],
            "state_matrix_convention": "row=bra,column=ket",
            "overlap_method": "FCI determinant overlaps using cross-AO integrals",
        }
        if family_id is not None:
            if not isinstance(family_id, str) or not family_id.strip():
                raise ValueError("family_id must be a non-empty string")
            metadata["molecular_family"] = family_id
        samples, overlaps, previous = [], [], None
        for coordinates in positions:
            mol = gto.M(
                atom=_atoms(numbers, coordinates),
                basis=self.basis,
                unit="Angstrom",
                charge=0,
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
        return MolecularPath(
            samples,
            path_id=path_id,
            adjacent_overlaps=torch.from_numpy(np.stack(overlaps)),
            metadata=metadata,
        )
