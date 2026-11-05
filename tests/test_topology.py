"""Analytic positive and negative controls for the initial v3.4 topology slice."""

import math

import pytest
import torch

from generaldia.topology import closed_loop_overlaps, wilson_loop


def _ci_frames(n=80):
    phi = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
    h = torch.zeros((n, 2, 2), dtype=torch.float64)
    h[:, 0, 0] = torch.cos(phi)
    h[:, 1, 1] = -torch.cos(phi)
    h[:, 0, 1] = torch.sin(phi)
    h[:, 1, 0] = torch.sin(phi)
    energies, vectors = torch.linalg.eigh(h)
    return energies, vectors[:, :, :1]


def _distance(actual, expected):
    return abs(math.atan2(math.sin(actual - expected), math.cos(actual - expected)))


def test_ci_and_trivial_paths_have_same_spectrum_but_different_topology():
    energies, ground = _ci_frames()
    trivial = torch.tensor([[1.0], [0.0]], dtype=torch.float64).expand(80, -1, -1)
    ci = wilson_loop(closed_loop_overlaps(ground))
    control = wilson_loop(closed_loop_overlaps(trivial))
    assert torch.allclose(
        energies, torch.tensor([-1.0, 1.0], dtype=torch.float64), atol=1e-12
    )
    assert _distance(ci.berry_phase, math.pi) < 1e-10
    assert _distance(control.berry_phase, 0.0) < 1e-10
    assert ci.minimum_link_singular_value > 0.99
    assert ci.number_of_links == 80


def test_complex_gauge_reversal_and_nonquantized_phase():
    n, alpha = 120, 0.7
    phi = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
    base = torch.zeros((n, 2, 1), dtype=torch.complex128)
    base[:, 0, 0] = math.cos(alpha / 2)
    base[:, 1, 0] = math.sin(alpha / 2) * torch.exp(1j * phi)
    gauged = base * torch.exp(
        1j * (1.2 * torch.sin(3 * phi) + 0.7 * phi)
    ).reshape(n, 1, 1)
    clean_phase = wilson_loop(closed_loop_overlaps(base)).berry_phase
    gauged_phase = wilson_loop(closed_loop_overlaps(gauged)).berry_phase
    reverse_phase = wilson_loop(closed_loop_overlaps(gauged.flip(0))).berry_phase

    expected = -2 * math.pi * math.sin(alpha / 2) ** 2
    assert _distance(clean_phase, expected) < 1e-3
    assert _distance(clean_phase, gauged_phase) < 1e-10
    assert _distance(reverse_phase, -clean_phase) < 1e-10


def test_nonabelian_wilson_eigenvalues_and_conjugacy():
    n = 64
    phi = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
    frames = torch.zeros((n, 3, 2), dtype=torch.complex128)
    frames[:, 0, 0] = torch.cos(phi / 2)
    frames[:, 1, 0] = torch.sin(phi / 2)
    frames[:, 2, 1] = 1.0
    baseline = wilson_loop(closed_loop_overlaps(frames))

    torch.manual_seed(37)
    z = torch.randn(n, 2, 2, dtype=torch.float64) + 1j * torch.randn(
        n, 2, 2, dtype=torch.float64
    )
    q, r = torch.linalg.qr(z)
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    gauges = q @ torch.diag_embed(diagonal / torch.abs(diagonal)).mH
    rotated = wilson_loop(closed_loop_overlaps(frames @ gauges))
    assert torch.allclose(
        baseline.eigenphases,
        torch.tensor([-math.pi, 0.0], dtype=torch.float64),
        atol=1e-9,
    )
    assert torch.allclose(
        rotated.wilson, gauges[0].mH @ baseline.wilson @ gauges[0], atol=1e-10
    )
    assert torch.allclose(
        torch.sort(torch.linalg.eigvals(rotated.wilson).real).values,
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="rank-one"):
        _ = rotated.berry_phase


def test_fail_closed_on_bad_frames_and_links():
    with pytest.raises(ValueError, match="shape"):
        closed_loop_overlaps(torch.eye(2))
    with pytest.raises(ValueError, match="orthonormal"):
        closed_loop_overlaps(torch.ones(3, 2, 1))
    with pytest.raises(ValueError, match="finite"):
        closed_loop_overlaps(torch.full((3, 2, 1), float("nan")))
    with pytest.raises(ValueError, match="shape"):
        wilson_loop(torch.eye(2))
    with pytest.raises(ValueError, match="finite"):
        wilson_loop(torch.full((3, 1, 1), float("nan")))
    with pytest.raises(ValueError, match="contractions"):
        wilson_loop(torch.full((3, 1, 1), 1.01))
    with pytest.raises(ValueError, match="unresolved"):
        wilson_loop(torch.tensor([[[0.1]], [[1.0]], [[1.0]]]))
    with pytest.raises(ValueError, match="minimum_link_singular_value"):
        wilson_loop(torch.ones((3, 1, 1)), minimum_link_singular_value=0)
