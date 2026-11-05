"""Finite-state Berry and Wilson-loop diagnostics from physical cross-frame overlaps.

Links S[k] = <U[k]|U[k+1]> MUST include the final-to-first link.
The frames need not share an ambient basis if physical cross-overlaps are
supplied directly. Gauges must preserve the same isolated state subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
from torch import Tensor


@dataclass(frozen=True)
class LoopHolonomy:
    """Polar-link Wilson loop with Berry phases in radians (minus arg convention)."""

    wilson: Tensor
    eigenphases: Tensor
    minimum_link_singular_value: float
    number_of_links: int

    @property
    def berry_phase(self) -> float:
        """Abelian Berry phase; unavailable for multi-state subspaces."""

        if self.wilson.shape != (1, 1):
            raise ValueError("berry_phase requires a rank-one electronic subspace")
        return float(self.eigenphases[0])


def _numerical_tolerance(values: Tensor, dimension: int) -> float:
    eps = torch.finfo(values.real.dtype).eps
    return max(5e-12, 64.0 * eps * dimension)


def closed_loop_overlaps(frames: Tensor) -> Tensor:
    """Compute cyclic overlaps for orthonormal frames in ONE common ambient basis.

    Shape: (n_geometries>=3, ambient_dimension>=n_states, n_states>=1).
    Cross-geometry AO/CI overlaps require the physical electronic-structure
    interface rather than unqualified products of unrelated coefficients.
    """

    frames = torch.as_tensor(frames)
    if (
        frames.ndim != 3
        or frames.shape[0] < 3
        or frames.shape[2] == 0
        or frames.shape[1] < frames.shape[2]
    ):
        raise ValueError("frames must have shape (n>=3, ambient>=states, states>=1)")
    if not (frames.is_floating_point() or frames.is_complex()):
        frames = frames.to(torch.float64)
    if frames.dtype in (torch.float16, torch.bfloat16):
        frames = frames.to(torch.float32)
    if not torch.isfinite(frames).all():
        raise ValueError("frames must contain finite values")
    identity = torch.eye(frames.shape[2], dtype=frames.dtype, device=frames.device)
    tolerance = _numerical_tolerance(frames, frames.shape[2])
    if not torch.allclose(frames.mH @ frames, identity, atol=tolerance, rtol=tolerance):
        raise ValueError("frame columns must be orthonormal")
    return frames.mH @ torch.roll(frames, shifts=-1, dims=0)


def wilson_loop(
    cyclic_overlaps: Tensor, *, minimum_link_singular_value: float = 0.5
) -> LoopHolonomy:
    """Return polar-link Wilson matrix and its gauge-invariant eigenphases.

    SVD link S = U diag(s) V^dagger gives Q = U V^dagger; the cyclic product
    W = Q_01 ... Q_last,0 transforms as G_0^dagger W G_0. The sorted phases
    -arg(eig(W)) lie on [-pi, pi). For one state, the phase is Abelian.
    This function requires an explicit physical closing link; it cannot
    detect a missing closure in arbitrary user-supplied overlap arrays.
    """

    overlaps = torch.as_tensor(cyclic_overlaps)
    if (
        overlaps.ndim != 3
        or overlaps.shape[0] < 3
        or overlaps.shape[1] == 0
        or overlaps.shape[1] != overlaps.shape[2]
    ):
        raise ValueError("cyclic_overlaps must have shape (n_links>=3, states, states)")
    if not (overlaps.is_floating_point() or overlaps.is_complex()):
        overlaps = overlaps.to(torch.float64)
    if overlaps.dtype in (torch.float16, torch.bfloat16):
        overlaps = overlaps.to(torch.float32)
    if not torch.isfinite(overlaps).all():
        raise ValueError("cyclic_overlaps must contain finite values")
    if not (0.0 < minimum_link_singular_value <= 1.0):
        raise ValueError("minimum_link_singular_value must be in (0, 1]")

    u, singular, vh = torch.linalg.svd(overlaps, full_matrices=False)
    tolerance = _numerical_tolerance(overlaps, overlaps.shape[-1])
    if bool((singular > 1.0 + tolerance).any()):
        raise ValueError("cross-frame overlaps must be contractions (singular values <= 1)")
    smallest = float(singular.min())
    if smallest < minimum_link_singular_value:
        raise ValueError("unresolved loop: link singular value below required minimum")

    unitary_links = u @ vh
    result = torch.eye(overlaps.shape[-1], dtype=unitary_links.dtype, device=overlaps.device)
    for link in unitary_links:
        result = result @ link
    phases = -torch.angle(torch.linalg.eigvals(result))
    phases = torch.remainder(phases + pi, 2 * pi) - pi
    return LoopHolonomy(
        wilson=result,
        eigenphases=torch.sort(phases).values,
        minimum_link_singular_value=smallest,
        number_of_links=overlaps.shape[0],
    )
