"""Scientific positive and negative controls for v3.4 numerator checks."""

import math

import pytest
import torch

from generaldia.certification import certify_coupling_numerators


class AvoidedCrossing:
    def __init__(self, *, slope=0.07, coupling=1e-9, detach=False, wrong_basis=False):
        self.slope = slope
        self.coupling = coupling
        self.detach = detach
        self.wrong_basis = wrong_basis

    def __call__(self, numbers, positions):
        del numbers
        x = positions[0, 0]
        if self.detach:
            x = x.detach()
        a = self.slope * x
        c = x.new_tensor(self.coupling)
        if self.wrong_basis:
            return torch.stack((torch.stack((c, a)), torch.stack((a, -c))))
        return torch.stack((torch.stack((a, c)), torch.stack((c, -a))))


def reference(slope=0.07):
    result = torch.zeros(1, 3, 2, 2, dtype=torch.float64)
    result[0, 0, 0, 0] = slope
    result[0, 0, 1, 1] = -slope
    return result


def test_small_gap_numerator_finite_and_mask_not_physical_zero():
    result = certify_coupling_numerators(
        AvoidedCrossing(),
        torch.tensor([1]),
        torch.zeros(1, 3, dtype=torch.float64),
        reference_derivative=reference(),
    )
    assert result.passed
    assert result.reference_checked
    assert result.minimum_energy_gap == pytest.approx(2e-9)
    assert result.unresolved_pairs == ((0, 1),)
    assert not result.valid_gap_pairs.any()
    assert abs(result.numerators_autograd[0, 0, 0, 1]) == pytest.approx(0.07)
    assert torch.isfinite(result.numerators_finite_difference).all()
    assert result.finite_difference_max_error < 1e-11
    assert result.reference_max_error == pytest.approx(0.0)


def test_finite_gap_analytic_oracle():
    x = 0.3
    result = certify_coupling_numerators(
        AvoidedCrossing(coupling=0.01),
        torch.tensor([1]),
        torch.tensor([[x, 0.0, 0.0]], dtype=torch.float64),
        reference_derivative=reference(),
    )
    expected = 0.07 * 0.01 / math.sqrt((0.07 * x) ** 2 + 0.01**2)
    assert result.passed
    assert abs(result.numerators_autograd[0, 0, 0, 1]) == pytest.approx(expected)
    assert result.unresolved_pairs == ()
    assert result.valid_gap_pairs[0, 1]


def test_detached_gradient_fails_even_when_energies_are_correct():
    result = certify_coupling_numerators(
        AvoidedCrossing(coupling=0.01, detach=True),
        torch.tensor([1]),
        torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64),
    )
    assert not result.passed
    assert result.finite_difference_max_error > 0.05
    assert not result.reference_checked
    assert result.reference_max_error is None


def test_isospectral_wrong_basis_passes_internal_but_fails_oracle():
    x = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    good = certify_coupling_numerators(
        AvoidedCrossing(coupling=0.03), torch.tensor([1]), x,
        reference_derivative=reference(),
    )
    wrong = certify_coupling_numerators(
        AvoidedCrossing(coupling=0.03, wrong_basis=True), torch.tensor([1]), x,
        reference_derivative=reference(),
    )
    assert torch.allclose(good.energies, wrong.energies, atol=1e-13)
    assert good.passed and not wrong.passed
    assert wrong.finite_difference_max_error < 1e-10
    assert wrong.reference_max_error > 0.06


def test_constant_complex_basis_gauge_and_oracle_covariance():
    phase = torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
    unitary = torch.diag(
        torch.stack((torch.tensor(1.0j, dtype=torch.complex128), phase))
    )
    base = AvoidedCrossing(coupling=0.02)

    def rotated(numbers, positions):
        return unitary.mH @ base(numbers, positions).to(torch.complex128) @ unitary

    coordinates = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
    original = certify_coupling_numerators(
        base, torch.tensor([1]), coordinates, reference_derivative=reference()
    )
    transformed = certify_coupling_numerators(
        rotated, torch.tensor([1]), coordinates,
        reference_derivative=unitary.mH @ reference().to(torch.complex128) @ unitary,
    )
    assert original.passed and transformed.passed
    assert transformed.numerators_autograd.is_complex()
    assert torch.allclose(original.energies, transformed.energies)
    assert torch.allclose(
        original.numerators_autograd.abs(),
        transformed.numerators_autograd.abs(),
        atol=1e-12,
    )


def test_exact_degeneracy_is_not_certified_as_a_state_resolved_coupling():
    result = certify_coupling_numerators(
        AvoidedCrossing(coupling=0.0),
        torch.tensor([1]),
        torch.zeros(1, 3, dtype=torch.float64),
        reference_derivative=reference(),
    )
    assert result.passed  # Operator derivative consistency, NOT root identification.
    assert result.minimum_energy_gap == 0.0
    assert result.unresolved_pairs == ((0, 1),)
    assert not result.valid_gap_pairs.any()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"step": 0.0}, "step"),
        ({"step": float("nan")}, "step"),
        ({"gap_floor": 0.0}, "gap_floor"),
        ({"atol": -1}, "atol"),
        ({"rtol": float("inf")}, "rtol"),
    ],
)
def test_control_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        certify_coupling_numerators(
            AvoidedCrossing(), torch.tensor([1]), torch.zeros(1, 3, dtype=torch.float64),
            **kwargs,
        )


def test_invalid_inputs_and_reference_fail_closed():
    model = AvoidedCrossing()
    with pytest.raises(ValueError, match="float64 positions"):
        certify_coupling_numerators(model, torch.tensor([1]), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="positions must have shape"):
        certify_coupling_numerators(
            model, torch.tensor([1]), torch.zeros(1, 2, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="match the number of atoms"):
        certify_coupling_numerators(
            model, torch.tensor([1, 1]), torch.zeros(1, 3, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="reference_derivative must have shape"):
        certify_coupling_numerators(
            model, torch.tensor([1]), torch.zeros(1, 3, dtype=torch.float64),
            reference_derivative=torch.zeros(2, 2),
        )


def test_bad_stencil_not_silently_accepted():
    def broken(numbers, positions):
        del numbers
        x = positions[0, 0]
        return torch.stack((torch.stack((x, 1.0 * x)), torch.stack((0.0 * x, -x))))

    with pytest.raises(ValueError, match="Hermitian"):
        certify_coupling_numerators(
            broken, torch.tensor([1]), torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        )


def test_complex_multiatom_tensor_and_independent_oracle():
    def model(numbers, positions):
        del numbers
        x, y = positions[0, 0], positions[1, 1]
        diagonal = 0.1 * x + 0.02 * y**2
        coupling = 0.04 + 0.03j * y
        return torch.stack(
            (torch.stack((diagonal, coupling)), torch.stack((coupling.conj(), -diagonal)))
        )

    coordinates = torch.tensor(
        [[0.3, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=torch.float64
    )
    target = torch.zeros(2, 3, 2, 2, dtype=torch.complex128)
    target[0, 0, 0, 0] = 0.1
    target[0, 0, 1, 1] = -0.1
    target[1, 1, 0, 0] = 0.04 * 0.2
    target[1, 1, 1, 1] = -0.04 * 0.2
    target[1, 1, 0, 1] = 0.03j
    target[1, 1, 1, 0] = -0.03j
    result = certify_coupling_numerators(
        model, torch.tensor([1, 1]), coordinates, reference_derivative=target
    )
    assert result.passed
    assert result.numerators_autograd.shape == (2, 3, 2, 2)
    assert result.finite_difference_max_error < 1e-11
    assert result.reference_max_error < 1e-11


def test_five_point_error_scales_at_fourth_order():
    def model(numbers, positions):
        del numbers
        x = positions[0, 0]
        coupling = x.new_tensor(0.2)
        diagonal = torch.sin(x)
        return torch.stack(
            (torch.stack((diagonal, coupling)), torch.stack((coupling, -diagonal)))
        )

    coordinates = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64)
    coarse = certify_coupling_numerators(
        model, torch.tensor([1]), coordinates, step=0.15
    )
    fine = certify_coupling_numerators(
        model, torch.tensor([1]), coordinates, step=0.075
    )
    assert fine.finite_difference_max_error > 0
    assert 12 < coarse.finite_difference_max_error / fine.finite_difference_max_error < 20
