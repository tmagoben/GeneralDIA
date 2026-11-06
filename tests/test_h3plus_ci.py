"""Physical H3+ singlet-FCI CI evidence; mandatory in the real-PySCF job."""

import math

import numpy as np
import pytest
import torch

from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend
from generaldia.electronic_structure.h3plus_ci import (
    certify_h3plus_molecular_ci,
    h3plus_ci_loop,
)
from generaldia.topology import wilson_loop

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def single_threaded_pyscf():
    pyscf = pytest.importorskip("pyscf")
    old_threads = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    yield
    pyscf.lib.num_threads(old_threads)


def test_molecular_h3plus_excited_ci_has_converged_physical_pi_holonomy():
    evidence = certify_h3plus_molecular_ci(refinement_points=(8, 16, 32))
    assert evidence.passed, evidence.checks
    assert all(evidence.checks.values())
    assert evidence.charge == 1
    assert evidence.basis == "sto-3g"
    assert abs(evidence.central_excited_gap_hartree) <= 1e-6
    assert evidence.min_ground_isolation_hartree > 1e-3
    assert evidence.min_upper_isolation_hartree > 1e-3
    assert evidence.min_excited_gap_on_loop_hartree > 1e-4
    assert evidence.lower_excited_loop.passed
    assert evidence.upper_excited_loop.passed
    assert evidence.lower_excited_loop.minimum_link_singular_value > 0.65
    assert evidence.upper_excited_loop.minimum_link_singular_value > 0.65
    assert abs(abs(evidence.lower_excited_loop.phases[-1]) - math.pi) < 0.06
    assert abs(abs(evidence.upper_excited_loop.phases[-1]) - math.pi) < 0.06
    assert abs(evidence.nonenclosing_phase_radians) < 0.06
    assert abs(evidence.isospectral_energy_only_phase_radians) < 1e-12
    assert evidence.gauge_phase_error_radians < 1e-9
    assert all(len(digest) == 64 for digest in evidence.geometry_sha256.values())


@pytest.mark.parametrize(
    "settings",
    [
        {"radius_angstrom": 0.0},
        {"radius_angstrom": float("nan")},
        {"refinement_points": (8, 16)},
        {"refinement_points": (16, 8, 32)},
        {"refinement_points": (8, 8, 16)},
    ],
)
def test_invalid_ci_benchmark_settings_fail_before_pyscf_calculation(settings):
    with pytest.raises(ValueError):
        certify_h3plus_molecular_ci(**settings)


def test_molecular_ci_is_stable_at_second_bond_and_loop_radius():
    evidence = certify_h3plus_molecular_ci(
        bond_angstrom=0.90,
        radius_angstrom=0.045,
        refinement_points=(8, 12, 16),
    )
    assert evidence.passed, evidence.checks
    assert evidence.min_ground_isolation_hartree > 1e-3
    assert evidence.min_upper_isolation_hartree > 1e-3
    assert abs(abs(evidence.lower_excited_loop.phases[-1]) - math.pi) < 0.06


def test_atom_relabeling_preserves_molecular_energies_and_holonomy():
    backend = FCIDipoleBackend(charge=1, n_states=4, conv_tol=1e-11)
    geometry = h3plus_ci_loop(16, 0.06)
    ordinary = backend.calculate_loop([1, 1, 1], geometry, path_id="h3plus-ordered")
    permuted = backend.calculate_loop(
        [1, 1, 1], geometry[:, [2, 0, 1], :], path_id="h3plus-permuted"
    )
    torch.testing.assert_close(
        ordinary.path.energies, permuted.path.energies, atol=1e-9, rtol=1e-9
    )
    for state in (1, 2):
        selector = np.s_[:, state : state + 1, state : state + 1]
        phase_a = wilson_loop(ordinary.cyclic_overlaps[selector]).berry_phase
        phase_b = wilson_loop(permuted.cyclic_overlaps[selector]).berry_phase
        distance = abs(math.atan2(math.sin(phase_a - phase_b), math.cos(phase_a - phase_b)))
        assert distance < 1e-9
