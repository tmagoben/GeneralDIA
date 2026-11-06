"""Optional molecular FCI closed-loop checks using physical cross-AO overlaps."""

import math

import numpy as np
import pytest
import torch

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend
from generaldia.loop_convergence import certify_abelian_loop_convergence
from generaldia.topology import wilson_loop

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def pyscf_serial():
    pyscf = pytest.importorskip("pyscf")
    prior = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    yield
    pyscf.lib.num_threads(prior)


def _h2_geometries(n):
    phi = 2 * math.pi * np.arange(n) / n
    bonds = 0.75 + 0.035 * np.sin(phi) + 0.015 * np.cos(phi)
    positions = np.zeros((n, 2, 3), dtype=np.float64)
    positions[:, 1, 2] = bonds
    return positions


def test_closed_loop_uses_direct_physical_endpoint_link():
    backend = FCIDipoleBackend(n_states=1, conv_tol=1e-11)
    positions = _h2_geometries(4)
    loop = backend.calculate_loop([1, 1], positions, path_id="h2-closed")
    path = loop.path
    assert len(path) == 4
    assert loop.cyclic_overlaps.shape == (4, 1, 1)
    torch.testing.assert_close(loop.cyclic_overlaps[:-1], path.adjacent_overlaps)
    independent = backend.calculate_path([1, 1], positions[[-1, 0]], path_id="h2-reverse-edge")
    # Independently solved electronic roots may differ by a real sign.
    torch.testing.assert_close(
        loop.cyclic_overlaps[-1].abs(),
        independent.adjacent_overlaps[0].abs(),
        atol=1e-9,
        rtol=1e-9,
    )
    assert path.metadata["overlap_method"] == "FCI determinant overlaps using cross-AO integrals"
    phase = wilson_loop(loop.cyclic_overlaps)
    assert phase.minimum_link_singular_value > 0.95
    assert abs(phase.berry_phase) < 1e-8


def test_molecular_h2_loop_refines_and_is_gauge_invariant():
    backend = FCIDipoleBackend(n_states=1, conv_tol=1e-11)
    loops = [
        (n, backend.calculate_loop([1, 1], _h2_geometries(n), path_id=f"h2-{n}"))
        for n in (4, 8, 12)
    ]
    result = certify_abelian_loop_convergence(
        [(n, loop.cyclic_overlaps) for n, loop in loops],
        phase_tolerance=1e-5,
        reference_phase=0.0,
    )
    assert result.passed
    assert result.minimum_link_singular_value > 0.95

    n, loop = loops[-1]
    angle = torch.linspace(0.1, 1.2, n, dtype=torch.float64)
    phase = torch.exp(1j * angle)
    links = (
        loop.cyclic_overlaps.to(torch.complex128)
        * phase.conj()[:, None, None]
        * torch.roll(phase, -1)[:, None, None]
    )
    before = wilson_loop(loop.cyclic_overlaps)
    after = wilson_loop(links)
    delta = before.berry_phase - after.berry_phase
    assert abs(math.atan2(math.sin(delta), math.cos(delta))) < 1e-10


def test_molecular_loop_invalid_closure_fails_closed():
    backend = FCIDipoleBackend(n_states=1)
    with pytest.raises(ValueError, match="three molecular"):
        backend.calculate_loop([1, 1], _h2_geometries(2), path_id="too-short")
    loop = backend.calculate_loop([1, 1], _h2_geometries(4), path_id="h2-corrupt")
    bad = loop.cyclic_overlaps.clone()
    bad[-1] = 0.1
    with pytest.raises(ValueError, match="unresolved"):
        wilson_loop(bad)
