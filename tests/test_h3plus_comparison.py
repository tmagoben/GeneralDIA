"""Basis independence, numerical evidence contracts and real-FCI regression."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generaldia.electronic_structure import h3plus_comparison as comparison
from generaldia.electronic_structure.h3plus_ci import H3PlusCIEvidence, certify_h3plus_molecular_ci
from generaldia.loop_convergence import AbelianLoopConvergence

GOLDEN = Path(__file__).parent / "goldens/h3plus_basis_comparison_reference.json"


@pytest.fixture
def reference():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _evidence(reference, basis):
    values = reference["results"][basis]["result"]
    for key in ("lower_excited_loop", "upper_excited_loop"):
        values[key] = AbelianLoopConvergence(**values[key])
    return H3PlusCIEvidence(**values)


def test_builder_runs_independent_bases_and_preserves_all_measurements(monkeypatch, reference):
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(__version__="test-double"))
    calls = []

    def calculate(**kwargs):
        calls.append(kwargs)
        return _evidence(reference, kwargs["basis"])

    monkeypatch.setattr(comparison, "certify_h3plus_molecular_ci", calculate)
    output = comparison.certify_h3plus_basis_comparison(revision="test-revision")
    assert [call.pop("basis") for call in calls] == ["sto-3g", "6-31g"]
    assert (
        calls[0]
        == calls[1]
        == {
            "bond_angstrom": 0.875,
            "radius_angstrom": 0.06,
            "refinement_points": (8, 16, 32),
        }
    )
    assert output["pass"]
    assert output["revision"] == "test-revision"
    assert output["settings"]["thresholds"] == comparison.THRESHOLDS
    assert (
        output["results"]["sto-3g"]["result"]["center_energies_hartree"]
        != output["results"]["6-31g"]["result"]["center_energies_hartree"]
    )
    assert (
        output["results"]["sto-3g"]["result"]["branching_gaps_hartree"]
        != output["results"]["6-31g"]["result"]["branching_gaps_hartree"]
    )


@pytest.mark.parametrize("corruption", ["failed_check", "geometry_mismatch"])
def test_comparison_does_not_hide_a_failing_basis(monkeypatch, reference, corruption):
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(__version__="test-double"))
    second = reference["results"]["6-31g"]["result"]
    if corruption == "failed_check":
        second["checks"]["positive_excited_gap_on_loops"] = False
    else:
        second["geometry_sha256"]["nonenclosing"] = "0" * 64
    monkeypatch.setattr(
        comparison, "certify_h3plus_molecular_ci", lambda **kw: _evidence(reference, kw["basis"])
    )
    output = comparison.certify_h3plus_basis_comparison()
    assert not output["pass"]
    assert not all(output["checks"].values())


def test_comparison_propagates_solver_failure_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(__version__="test-double"))

    def fail(**kwargs):
        raise RuntimeError("FCI did not converge")

    monkeypatch.setattr(comparison, "certify_h3plus_molecular_ci", fail)
    with pytest.raises(RuntimeError, match="FCI did not converge"):
        comparison.certify_h3plus_basis_comparison()
    with pytest.raises(ValueError, match="revision"):
        comparison.certify_h3plus_basis_comparison(revision="")


@pytest.mark.parametrize("mesh", [(6, 8, 10.5), (True, 8, 16), (4, 8, 16), [8, 16, 32]])
def test_invalid_mesh_rejected_before_backend_evaluation(mesh):
    with pytest.raises(ValueError, match="refinement_points"):
        certify_h3plus_molecular_ci(refinement_points=mesh)


def test_zero_branching_gap_fails_before_division(monkeypatch):
    from generaldia.electronic_structure import h3plus_ci

    class Backend:
        def calculate_path(self, *args, **kwargs):
            import torch

            return SimpleNamespace(energies=torch.zeros((5, 4)))

    monkeypatch.setattr(h3plus_ci, "FCIDipoleBackend", lambda **kw: Backend())
    with pytest.raises(ValueError, match="branching gaps must be finite and positive"):
        certify_h3plus_molecular_ci()


@pytest.mark.optional
def test_real_fci_basis_comparison_has_independent_gaps_and_matching_geometry(reference):
    pyscf = pytest.importorskip("pyscf")
    original_threads = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    try:
        report = comparison.certify_h3plus_basis_comparison(revision="pytest-physical")
    finally:
        pyscf.lib.num_threads(original_threads)
    assert report["pass"], report["checks"]
    for basis in ("sto-3g", "6-31g"):
        measured = report["results"][basis]["result"]
        golden = reference["results"][basis]["result"]
        assert all(measured["checks"].values())
        np.testing.assert_allclose(
            measured["center_energies_hartree"],
            golden["center_energies_hartree"],
            rtol=0,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            measured["branching_gaps_hartree"], golden["branching_gaps_hartree"], rtol=0, atol=2e-6
        )
        gaps = measured["branching_gaps_hartree"]
        np.testing.assert_allclose(
            measured["branching_gap_ratios"], (gaps[1] / gaps[0], gaps[3] / gaps[2]), rtol=1e-12
        )
