"""Physical H3+ singlet-FCI CI evidence; mandatory in the real-PySCF job."""

import math

import pytest

from generaldia.electronic_structure.h3plus_ci import certify_h3plus_molecular_ci

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
