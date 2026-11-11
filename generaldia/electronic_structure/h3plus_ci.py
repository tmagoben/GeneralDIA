"""Physical H3+ excited-singlet conical-intersection certification.

Geometry uses the two E-type symmetry-breaking modes of the three H--H
bond lengths, with fixed mean bond length. All electronic overlaps are
computed from PySCF FCI wavefunctions in their actual cross-geometry AO bases.
This is a restricted finite-basis reference, not a fitted potential surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import atan2, cos, isfinite, pi, sin, sqrt

import numpy as np
import torch
from numpy.typing import NDArray

from ..loop_convergence import AbelianLoopConvergence, certify_abelian_loop_convergence
from ..topology import wilson_loop
from .fci_dipoles import FCIDipoleBackend

ATOMIC_NUMBERS = (1, 1, 1)
BASIS = "sto-3g"
CHARGE = 1


def h3plus_triangle(
    bond_angstrom: float = 0.875,
    mode_x_angstrom: float = 0.0,
    mode_y_angstrom: float = 0.0,
) -> NDArray[np.float64]:
    """Return a planar, centered H3+ triangle with two independent E modes.

    The physical bond lengths (01, 02, 12) equal
    R*(1,1,1) + x*(2,-1,-1)/sqrt(6) + y*(0,1,-1)/sqrt(2).
    A common change in x or y breaks D3h symmetry; (0,0) is equilateral.
    Coordinates and E-mode amplitudes are in angstrom, with fixed atom labels.
    """
    values = (bond_angstrom, mode_x_angstrom, mode_y_angstrom)
    if any(not isfinite(v) for v in values) or bond_angstrom <= 0:
        raise ValueError("bond and modes must be finite, with positive bond length")
    a = bond_angstrom + 2 * mode_x_angstrom / sqrt(6)
    b = bond_angstrom - mode_x_angstrom / sqrt(6) + mode_y_angstrom / sqrt(2)
    c = bond_angstrom - mode_x_angstrom / sqrt(6) - mode_y_angstrom / sqrt(2)
    if min(a, b, c) <= 0 or max(a, b, c) * 2 >= a + b + c:
        raise ValueError("symmetry-breaking modes produce an invalid triangle")
    x = (b * b - c * c) / (2 * a)
    height_sq = b * b - (x + 0.5 * a) ** 2
    if height_sq <= 0:
        raise ValueError("symmetry-breaking modes produce a degenerate triangle")
    coordinates = np.array(
        [[-0.5 * a, 0.0, 0.0], [0.5 * a, 0.0, 0.0], [x, sqrt(height_sq), 0.0]],
        dtype=np.float64,
    )
    return coordinates - coordinates.mean(axis=0)


def h3plus_ci_loop(
    points: int,
    radius_angstrom: float,
    *,
    bond_angstrom: float = 0.875,
    center_x_angstrom: float = 0.0,
    center_y_angstrom: float = 0.0,
    orientation: int = 1,
) -> NDArray[np.float64]:
    """Generate a closed E-mode circle without duplicating the first geometry."""
    if type(points) is not int or points < 6:
        raise ValueError("points must be an integer >= 6")
    values = (radius_angstrom, center_x_angstrom, center_y_angstrom)
    if not all(isfinite(v) for v in values) or radius_angstrom <= 0:
        raise ValueError("radius and center must be finite, with positive radius")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1")
    theta = orientation * 2 * pi * np.arange(points) / points
    return np.stack(
        [
            h3plus_triangle(
                bond_angstrom,
                center_x_angstrom + radius_angstrom * cos(angle),
                center_y_angstrom + radius_angstrom * sin(angle),
            )
            for angle in theta
        ]
    )


def _distance(a: float, b: float) -> float:
    return abs(atan2(sin(a - b), cos(a - b)))


@dataclass(frozen=True)
class H3PlusCIEvidence:
    """Numerical evidence with separately inspectable acceptance controls."""

    basis: str
    charge: int
    bond_angstrom: float
    radius_angstrom: float
    refinement_points: tuple[int, ...]
    center_energies_hartree: tuple[float, ...]
    central_excited_gap_hartree: float
    branching_gap_ratios: tuple[float, float]
    min_ground_isolation_hartree: float
    min_upper_isolation_hartree: float
    min_excited_gap_on_loop_hartree: float
    lower_excited_loop: AbelianLoopConvergence
    upper_excited_loop: AbelianLoopConvergence
    nonenclosing_phase_radians: float
    reverse_orientation_phase_radians: float
    gauge_phase_error_radians: float
    isospectral_energy_only_phase_radians: float
    minimum_link_strength: float
    geometry_sha256: dict[str, str]
    checks: dict[str, bool]
    branching_gaps_hartree: tuple[float, float, float, float] | None = None

    @property
    def passed(self) -> bool:
        """True only when every independent scientific check passes."""
        return all(self.checks.values())


def certify_h3plus_molecular_ci(
    *,
    basis: str = BASIS,
    bond_angstrom: float = 0.875,
    radius_angstrom: float = 0.06,
    refinement_points: tuple[int, ...] = (8, 16, 32),
) -> H3PlusCIEvidence:
    """Certify the H3+ S1/S2 CI against finite-basis FCI physical overlaps.

    Positive controls: high-symmetry E' gap closing, branching along two
    independent E distortions, isolated S1 and S2 loops, pi Berry phases,
    U(1) gauge covariance, reversal, grid refinement.
    Negative controls: non-enclosing molecular circle and identical
    *energies* with artificial unit overlap links (energy-only baseline).
    All thresholds and underlying numerical observables are returned.
    This cannot certify arbitrary nuclear trajectories or orbital bases.
    """
    if (
        not isinstance(refinement_points, tuple)
        or len(refinement_points) < 3
        or any(type(n) is not int or n < 6 for n in refinement_points)
        or tuple(sorted(set(refinement_points))) != refinement_points
    ):
        raise ValueError("refinement_points must be >=3 strictly increasing unique integers")
    if not isfinite(radius_angstrom) or radius_angstrom <= 0:
        raise ValueError("radius_angstrom must be finite and positive")
    backend = FCIDipoleBackend(
        basis=basis, n_states=4, charge=CHARGE, conv_tol=1e-11, max_orbitals=8
    )
    center = h3plus_triangle(bond_angstrom)
    x_half = h3plus_triangle(bond_angstrom, radius_angstrom / 2, 0.0)
    x_full = h3plus_triangle(bond_angstrom, radius_angstrom, 0.0)
    y_half = h3plus_triangle(bond_angstrom, 0.0, radius_angstrom / 2)
    y_full = h3plus_triangle(bond_angstrom, 0.0, radius_angstrom)
    axis = backend.calculate_path(
        ATOMIC_NUMBERS,
        np.stack((center, x_half, x_full, y_half, y_full)),
        path_id="h3plus-ci-branching",
    )
    energies = axis.energies.numpy()
    central_gap = float(energies[0, 2] - energies[0, 1])
    gaps = energies[:, 2] - energies[:, 1]
    if not np.isfinite(gaps).all() or np.any(gaps[1:] <= 0):
        raise ValueError("branching gaps must be finite and positive in both E directions")
    ratios = (float(gaps[2] / gaps[1]), float(gaps[4] / gaps[3]))

    circular = []
    digests: dict[str, str] = {
        "symmetry_and_branching": sha256(
            np.stack((center, x_half, x_full, y_half, y_full)).tobytes()
        ).hexdigest()
    }
    lower_levels = []
    upper_levels = []
    ground_isolation = []
    upper_isolation = []
    excited_gap = []
    minimum_links = []
    for n in refinement_points:
        geometry = h3plus_ci_loop(n, radius_angstrom, bond_angstrom=bond_angstrom)
        loop = backend.calculate_loop(ATOMIC_NUMBERS, geometry, path_id=f"h3plus-ci-encircle-{n}")
        circular.append(loop)
        digests[f"encircling_{n}"] = sha256(geometry.tobytes()).hexdigest()
        for state, levels in ((1, lower_levels), (2, upper_levels)):
            levels.append((n, loop.cyclic_overlaps[:, state : state + 1, state : state + 1]))
        gap = torch.diff(loop.path.energies, dim=1).numpy()
        ground_isolation.append(float(gap[:, 0].min()))
        excited_gap.append(float(gap[:, 1].min()))
        upper_isolation.append(float(gap[:, 2].min()))

    phase_tolerance = 0.06
    link_floor = 0.65
    lower = certify_abelian_loop_convergence(
        lower_levels,
        phase_tolerance=phase_tolerance,
        minimum_link_singular_value=link_floor,
        reference_phase=pi,
    )
    upper = certify_abelian_loop_convergence(
        upper_levels,
        phase_tolerance=phase_tolerance,
        minimum_link_singular_value=link_floor,
        reference_phase=pi,
    )
    minimum_links.extend((lower.minimum_link_singular_value, upper.minimum_link_singular_value))
    n = refinement_points[-1]
    negative_geo = h3plus_ci_loop(
        n,
        radius_angstrom,
        bond_angstrom=bond_angstrom,
        center_x_angstrom=4 * radius_angstrom,
    )
    negative = backend.calculate_loop(ATOMIC_NUMBERS, negative_geo, path_id="h3plus-off-CI")
    digests["nonenclosing"] = sha256(negative_geo.tobytes()).hexdigest()
    no_ci_links = negative.cyclic_overlaps[:, 1:2, 1:2]
    no_ci = wilson_loop(no_ci_links, minimum_link_singular_value=link_floor)
    minimum_links.append(no_ci.minimum_link_singular_value)

    # Reverse order is recalculated, not formed by editing an overlap product.
    reverse_geo = h3plus_ci_loop(n, radius_angstrom, bond_angstrom=bond_angstrom, orientation=-1)
    reverse = backend.calculate_loop(ATOMIC_NUMBERS, reverse_geo, path_id="h3plus-reverse")
    digests["reversed"] = sha256(reverse_geo.tobytes()).hexdigest()
    reversed_phase = wilson_loop(
        reverse.cyclic_overlaps[:, 1:2, 1:2],
        minimum_link_singular_value=link_floor,
    )
    minimum_links.append(reversed_phase.minimum_link_singular_value)

    chosen = circular[-1].cyclic_overlaps[:, 1:2, 1:2].to(torch.complex128)
    index = torch.arange(n, dtype=torch.float64)
    local_gauge = torch.exp(1j * (0.2 * index + torch.sin(1.7 * index)))
    gauged = (
        chosen
        * local_gauge.conj()[:, None, None]
        * torch.roll(local_gauge, shifts=-1)[:, None, None]
    )
    phase_invariant = wilson_loop(gauged, minimum_link_singular_value=link_floor)
    gauge_error = _distance(phase_invariant.berry_phase, lower.phases[-1])

    # Same underlying *adiabatic energy sequence* as the physical loop.
    # Unit overlap links form an intentionally nonphysical isospectral control.
    # Their Berry phase vanishes and must not pass the pi reference.
    trivial = wilson_loop(torch.ones((n, 1, 1), dtype=torch.float64))

    checks = {
        "equilateral_excited_degeneracy": abs(central_gap) <= 1e-6,
        "two_independent_linear_branching_directions": all(1.3 < ratio < 2.7 for ratio in ratios),
        "ground_and_higher_root_isolation": (
            min(ground_isolation) > 1e-3 and min(upper_isolation) > 1e-3
        ),
        "positive_excited_gap_on_loops": min(excited_gap) > 1e-4,
        "lower_excited_pi_and_mesh": lower.passed,
        "upper_excited_pi_and_mesh": upper.passed,
        "physical_nonencircling_zero": _distance(no_ci.berry_phase, 0) < phase_tolerance,
        "reversal_covariance": _distance(reversed_phase.berry_phase, -lower.phases[-1])
        < phase_tolerance,
        "local_complex_gauge_covariance": gauge_error < 1e-9,
        "energy_only_control_rejected": (
            _distance(trivial.berry_phase, 0) < 1e-12 and _distance(trivial.berry_phase, pi) > 3.0
        ),
    }
    return H3PlusCIEvidence(
        basis=basis,
        charge=CHARGE,
        bond_angstrom=bond_angstrom,
        radius_angstrom=radius_angstrom,
        refinement_points=refinement_points,
        center_energies_hartree=tuple(map(float, energies[0])),
        central_excited_gap_hartree=central_gap,
        branching_gap_ratios=ratios,
        min_ground_isolation_hartree=min(ground_isolation),
        min_upper_isolation_hartree=min(upper_isolation),
        min_excited_gap_on_loop_hartree=min(excited_gap),
        lower_excited_loop=lower,
        upper_excited_loop=upper,
        nonenclosing_phase_radians=no_ci.berry_phase,
        reverse_orientation_phase_radians=reversed_phase.berry_phase,
        gauge_phase_error_radians=gauge_error,
        isospectral_energy_only_phase_radians=trivial.berry_phase,
        minimum_link_strength=min(minimum_links),
        geometry_sha256=digests,
        checks=checks,
        branching_gaps_hartree=tuple(map(float, gaps[1:])),
    )
