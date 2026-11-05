"""Convergence, gauge covariance and topology-breaking negative controls."""

import math

import pytest
import torch

from generaldia.loop_convergence import certify_abelian_loop_convergence
from generaldia.topology import closed_loop_overlaps


def _loop(n, *, crossing=True, gauge=False):
    phase = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
    h = torch.zeros((n, 2, 2), dtype=torch.float64)
    h[:, 0, 0] = torch.cos(phase) if crossing else 1.0
    h[:, 1, 1] = -h[:, 0, 0]
    h[:, 0, 1] = torch.sin(phase) if crossing else 0.0
    h[:, 1, 0] = h[:, 0, 1]
    _, eigenvectors = torch.linalg.eigh(h)
    frame = eigenvectors[:, :, :1]
    if gauge:
        frame = frame.to(torch.complex128) * torch.exp(
            1j * (0.8 * phase + torch.sin(3 * phase))
        )[:, None, None]
    return n, closed_loop_overlaps(frame)


def _spinor(n):
    phase = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
    alpha = 0.7
    frame = torch.zeros((n, 2, 1), dtype=torch.complex128)
    frame[:, 0, 0] = math.cos(alpha / 2)
    frame[:, 1, 0] = math.sin(alpha / 2) * torch.exp(1j * phase)
    return n, closed_loop_overlaps(frame)


def test_conical_intersection_refinements_phase_pi_and_gauge_invariant():
    counts = (8, 16, 32, 64)
    clean = certify_abelian_loop_convergence(
        [_loop(n) for n in counts], phase_tolerance=1e-8, reference_phase=math.pi
    )
    gauged = certify_abelian_loop_convergence(
        [_loop(n, gauge=True) for n in counts],
        phase_tolerance=1e-8,
        reference_phase=-math.pi,
    )
    assert clean.passed and gauged.passed
    assert clean.point_counts == counts
    assert clean.coarse_to_finest_phase_error < 1e-12
    assert clean.minimum_link_singular_value > 0.92
    assert all(abs(abs(p) - math.pi) < 1e-12 for p in clean.phases)
    assert clean.phases == pytest.approx(gauged.phases, abs=1e-12)


def test_isospectral_trivial_loop_is_rejected_by_ci_reference():
    counts = (8, 16, 32)
    triv = certify_abelian_loop_convergence(
        [_loop(n, crossing=False) for n in counts],
        phase_tolerance=1e-8,
        reference_phase=math.pi,
    )
    assert not triv.passed
    assert triv.coarse_to_finest_phase_error == pytest.approx(0, abs=1e-13)
    assert triv.reference_phase_error == pytest.approx(math.pi)


def test_complex_spinor_second_order_grid_convergence():
    exact = -2 * math.pi * math.sin(0.7 / 2) ** 2
    result = certify_abelian_loop_convergence(
        [_spinor(n) for n in (16, 32, 64)],
        phase_tolerance=0.015,
        reference_phase=exact,
    )
    assert result.passed
    assert result.reference_phase_error < 0.001
    assert result.successive_phase_errors[1] < result.successive_phase_errors[0]
    coarse = certify_abelian_loop_convergence(
        [_spinor(n) for n in (8, 16, 32)],
        phase_tolerance=0.015,
        reference_phase=exact,
    )
    assert not coarse.passed


def test_corrupted_single_link_fails_reference_phase_even_if_converged():
    distorted = []
    for n in (8, 16, 32):
        _, links = _loop(n)
        links = links.to(torch.complex128).clone()
        links[0] *= torch.exp(torch.tensor(0.2j, dtype=torch.complex128))
        distorted.append((n, links))
    result = certify_abelian_loop_convergence(
        distorted, phase_tolerance=1e-5, reference_phase=math.pi
    )
    assert not result.passed
    assert result.reference_phase_error == pytest.approx(0.2, abs=1e-12)


@pytest.mark.parametrize(
    "refinements, kwargs, message",
    [
        ([_loop(8), _loop(16)], {}, "three"),
        ([_loop(8), _loop(8), _loop(16)], {}, "increasing"),
        ([_loop(8), _loop(16), _loop(32)], {"phase_tolerance": -1}, "phase_tolerance"),
        (
            [_loop(8), _loop(16), _loop(32)],
            {"reference_phase": float("nan")},
            "reference_phase",
        ),
        ([(8, torch.ones(8, 2, 2)), _loop(16), _loop(32)], {}, "rank-one"),
        ([(8, torch.zeros(8, 1, 1)), _loop(16), _loop(32)], {}, "unresolved"),
    ],
)
def test_incorrect_convergence_inputs_fail_closed(refinements, kwargs, message):
    with pytest.raises(ValueError, match=message):
        certify_abelian_loop_convergence(refinements, **kwargs)
