"""Analytic state-truncation positive and negative controls."""

import math

import pytest
import torch

from generaldia.truncation import certify_state_truncation


def _reference():
    return torch.diag(torch.tensor([0.0, 0.06, 0.12], dtype=torch.float64))


def _frame():
    return torch.eye(3, dtype=torch.float64)[:, :2]


def _rotated(theta):
    u = torch.eye(3, dtype=torch.float64)
    u[0, 0] = math.cos(theta)
    u[0, 2] = math.sin(theta)
    u[2, 0] = -math.sin(theta)
    u[2, 2] = math.cos(theta)
    return u @ _reference() @ u.T


def test_exact_manifold_is_certified():
    result = certify_state_truncation(_reference(), _frame())
    assert result.passed
    assert result.target_state_indices == (0, 1)
    assert result.maximum_omitted_population == pytest.approx(0, abs=1e-14)
    assert result.residual_operator_norm == pytest.approx(0, abs=1e-14)
    assert result.minimum_target_gap == pytest.approx(0.06)


def test_isospectral_rotation_exposes_energy_only_false_positive():
    baseline = _reference()
    mixed = _rotated(0.3)
    assert torch.allclose(torch.linalg.eigvalsh(baseline), torch.linalg.eigvalsh(mixed))
    result = certify_state_truncation(
        mixed,
        _frame(),
        energy_tolerance=0.012,
        omitted_population_tolerance=0.02,
        residual_tolerance=0.01,
    )
    assert result.maximum_energy_error < 0.012  # Energy-only check would pass.
    assert result.maximum_omitted_population == pytest.approx(math.sin(0.3) ** 2)
    assert result.maximum_omitted_population > 0.08
    assert result.residual_operator_norm > 0.03
    assert not result.passed


def test_full_and_retained_complex_unitary_gauge_invariance():
    h = _rotated(0.3).to(torch.complex128)
    q = _frame().to(torch.complex128)
    z = torch.tensor([[1j, 0, 1], [0, 1j, 0.4], [0.6, 0, -1j]], dtype=torch.complex128)
    unitary, _ = torch.linalg.qr(z)
    v = torch.tensor([[0, 1j], [1, 0]], dtype=torch.complex128)
    base = certify_state_truncation(h, q)
    changed = certify_state_truncation(unitary.mH @ h @ unitary, unitary.mH @ q @ v)
    assert changed.maximum_energy_error == pytest.approx(base.maximum_energy_error, abs=1e-12)
    assert changed.maximum_omitted_population == pytest.approx(
        base.maximum_omitted_population, abs=1e-12
    )
    assert changed.residual_operator_norm == pytest.approx(base.residual_operator_norm, abs=1e-12)
    assert changed.minimum_target_gap == pytest.approx(base.minimum_target_gap, abs=1e-12)
    assert changed.passed == base.passed


def test_exact_boundary_degeneracy_fails_closed():
    h = torch.diag(torch.tensor([0.0, 0.06, 0.06], dtype=torch.float64))
    result = certify_state_truncation(h, _frame())
    assert result.unresolved_target_gap
    assert result.minimum_target_gap == 0
    assert not result.passed


def test_explicit_nonlowest_manifold():
    h = torch.diag(torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64))
    q = torch.eye(4, dtype=torch.float64)[:, 1:3]
    selected = certify_state_truncation(h, q, target_state_indices=(1, 2))
    default = certify_state_truncation(h, q)
    assert selected.passed
    assert not default.passed


@pytest.mark.parametrize(
    "hamiltonian, frame, message",
    [
        (torch.ones(2, 3, dtype=torch.float64), _frame(), "square"),
        (_reference(), torch.eye(3, dtype=torch.float64), "retained_frame"),
        (_reference(), _frame() * 2, "orthonormal"),
        (_reference().to(torch.float32), _frame(), "double-precision"),
        (
            _reference(),
            torch.full((3, 2), float("nan"), dtype=torch.float64),
            "finite",
        ),
        (
            torch.tensor([[0.0, 1, 0], [0, 1, 0], [0, 0, 2]], dtype=torch.float64),
            _frame(),
            "Hermitian",
        ),
    ],
)
def test_invalid_matrices_fail_closed(hamiltonian, frame, message):
    with pytest.raises(ValueError, match=message):
        certify_state_truncation(hamiltonian, frame)


@pytest.mark.parametrize("indices", [(0, 0), (2, 1), (0,), (0, 3), (0, 1.0)])
def test_bad_target_indices_fail_closed(indices):
    with pytest.raises(ValueError, match="target_state_indices"):
        certify_state_truncation(_reference(), _frame(), target_state_indices=indices)


@pytest.mark.parametrize(
    "controls",
    [
        {"energy_tolerance": -1},
        {"omitted_population_tolerance": 1.1},
        {"residual_tolerance": float("nan")},
        {"target_gap_floor": 0.0},
    ],
)
def test_invalid_controls_fail_closed(controls):
    with pytest.raises(ValueError):
        certify_state_truncation(_reference(), _frame(), **controls)
