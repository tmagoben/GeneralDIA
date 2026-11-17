"""State-frame invariant length-gauge dipole observables, in atomic units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BandStrength:
    """Unweighted Cartesian line tensor (e² bohr²) and dimensionless band oscillator strength."""

    tensor: NDArray[np.complex128]
    oscillator: float

    @property
    def line_strength(self) -> float:
        return float(self.tensor.trace().real)


def transition_band(dipoles: ArrayLike, gap_hamiltonian: ArrayLike) -> BandStrength:
    """D[a,j]=<g|mu_a|j>. Rotate D -> DU and G -> U†GU together, even for split bands.

    G is the excited-space Hamiltonian minus the ground energy times identity.
    No energy averaging, rootwise phase fixing, or atomic-unit conversion is done.
    """
    d = np.asarray(dipoles, dtype=np.complex128)
    g = np.asarray(gap_hamiltonian, dtype=np.complex128)
    if d.ndim != 2 or d.shape[0] != 3 or d.shape[1] < 1 or g.shape != (d.shape[1], d.shape[1]):
        raise ValueError("expected dipoles (3,K) and gap Hamiltonian (K,K), K >= 1")
    if not np.isfinite(d).all() or not np.isfinite(g).all():
        raise ValueError("transition data must be finite")
    if not np.allclose(g, g.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("gap Hamiltonian must be Hermitian")
    if np.linalg.eigvalsh(g).min() <= 0:
        raise ValueError("all transition energies must be positive")
    tensor = d @ d.conj().T
    oscillator = float((2 / 3) * np.trace(d @ g @ d.conj().T).real)
    if not np.isfinite(tensor).all() or not np.isfinite(oscillator):
        raise ValueError("nonfinite transition strength")
    return BandStrength(tensor, oscillator)
