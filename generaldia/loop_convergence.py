"""Resolution diagnostics for Abelian Berry loops from explicit cyclic overlaps.

This checks refinement stability, NOT correctness of the supplied physical
overlaps. Reference-phase checking is optional and must be separately justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, sin, cos
from typing import Sequence

from torch import Tensor

from .topology import wilson_loop


@dataclass(frozen=True)
class AbelianLoopConvergence:
    """Rank-one phase convergence and independently declared reference control."""

    point_counts: tuple[int, ...]
    phases: tuple[float, ...]
    successive_phase_errors: tuple[float, ...]
    coarse_to_finest_phase_error: float
    minimum_link_singular_value: float
    reference_phase: float | None
    reference_phase_error: float | None
    phase_tolerance: float
    passed: bool


def _circular_distance(a: float, b: float) -> float:
    return abs(atan2(sin(a - b), cos(a - b)))


def certify_abelian_loop_convergence(
    refinements: Sequence[tuple[int, Tensor]],
    *,
    phase_tolerance: float = 1e-2,
    minimum_link_singular_value: float = 0.5,
    reference_phase: float | None = None,
) -> AbelianLoopConvergence:
    """Certify phase agreement at >=3 strictly increasing loop resolutions.

    Inputs are `(n_geometries, cyclic_overlaps)` pairs, where the n-by-1-by-1
    tensor includes the physical final-to-first link. This function cannot
    authenticate the electronic-structure overlap provider or detect missing
    closure from arbitrary tensors.

    A passing result without a separately justified reference phase implies
    only *internal numerical convergence*; identically wrong but converged
    loops are possible. Rank>1 Wilson spectral convergence is not supported
    by this Abelian function.
    """
    if not isfinite(phase_tolerance) or not 0 < phase_tolerance < 3.141592653589793:
        raise ValueError("phase_tolerance must be finite and in (0, pi)")
    if reference_phase is not None and not isfinite(reference_phase):
        raise ValueError("reference_phase must be finite")
    if len(refinements) < 3:
        raise ValueError("at least three refinement levels are required")

    counts: list[int] = []
    phases: list[float] = []
    link_floors: list[float] = []
    for count, overlaps in refinements:
        if type(count) is not int or count < 3:
            raise ValueError("point counts must be integers >= 3")
        if counts and count <= counts[-1]:
            raise ValueError("point counts must be strictly increasing")
        if overlaps.ndim != 3 or overlaps.shape != (count, 1, 1):
            raise ValueError("rank-one cyclic overlaps must have shape (count, 1, 1)")
        result = wilson_loop(
            overlaps, minimum_link_singular_value=minimum_link_singular_value
        )
        counts.append(count)
        phases.append(result.berry_phase)
        link_floors.append(result.minimum_link_singular_value)

    successive = tuple(
        _circular_distance(phases[i], phases[i + 1])
        for i in range(len(phases) - 1)
    )
    coarse_to_finest = max(_circular_distance(p, phases[-1]) for p in phases[:-1])
    reference_error = (
        None if reference_phase is None else _circular_distance(phases[-1], reference_phase)
    )
    passed = (
        max(successive) <= phase_tolerance
        and coarse_to_finest <= phase_tolerance
        and (reference_error is None or reference_error <= phase_tolerance)
    )
    return AbelianLoopConvergence(
        point_counts=tuple(counts),
        phases=tuple(phases),
        successive_phase_errors=successive,
        coarse_to_finest_phase_error=coarse_to_finest,
        minimum_link_singular_value=min(link_floors),
        reference_phase=reference_phase,
        reference_phase_error=reference_error,
        phase_tolerance=phase_tolerance,
        passed=passed,
    )
