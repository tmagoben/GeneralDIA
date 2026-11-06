"""Adversarial scientific certification cases with explicit failure policies."""

import math

import pytest
import torch

from generaldia.certification import _checked_matrix, certify_coupling_numerators


def _linear_hamiltonian(numbers, positions):
    del numbers
    x = positions[0, 0]
    z = 0.2 + 0.03 * x
    coupling = 0.05 + 0 * x
    return torch.stack((torch.stack((z, coupling)), torch.stack((coupling, -z))))


@pytest.mark.parametrize(
    "matrix, error",
    [
        (torch.ones(2, 3), "square"),
        (torch.ones(0, 0), "nonempty"),
        (torch.tensor([[float("nan")]]), "nonfinite"),
        (torch.tensor([[0.0, 1.0], [0.0, 0.0]]), "Hermitian"),
        (torch.eye(2, dtype=torch.float32), "float64"),
    ],
)
def test_certification_matrix_guard_rejects_unphysical_stencil(matrix, error):
    with pytest.raises(ValueError, match=error):
        _checked_matrix(matrix)


@pytest.mark.parametrize(
    "position, message",
    [
        (torch.full((1, 3), float("nan")), "positions must be finite"),
        (torch.full((1, 3), float("inf")), "positions must be finite"),
        (torch.zeros((0, 3)), "positions must have shape"),
    ],
)
def test_certification_rejects_invalid_nuclear_geometry(position, message):
    with pytest.raises(ValueError, match=message):
        certify_coupling_numerators(_linear_hamiltonian, torch.tensor([1]), position)


@pytest.mark.parametrize(
    "numbers, message",
    [
        (torch.tensor([[1]]), "atomic_numbers must match"),
        (torch.tensor([float("nan")]), "atomic_numbers must be finite"),
        (torch.tensor([1, 1]), "atomic_numbers must match"),
    ],
)
def test_certification_rejects_bad_atom_labels(numbers, message):
    with pytest.raises(ValueError, match=message):
        certify_coupling_numerators(
            _linear_hamiltonian, numbers, torch.zeros((1, 3), dtype=torch.float64)
        )


@pytest.mark.parametrize(
    "reference, message",
    [
        (torch.zeros((1, 3, 2, 2), dtype=torch.float32), "double precision"),
        (torch.full((1, 3, 2, 2), float("nan")), "finite"),
    ],
)
def test_certification_rejects_invalid_independent_reference(reference, message):
    with pytest.raises(ValueError, match=message):
        certify_coupling_numerators(
            _linear_hamiltonian,
            torch.tensor([1]),
            torch.zeros((1, 3), dtype=torch.float64),
            reference_derivative=reference,
        )


def test_certification_rejects_nonhermitian_independent_reference():
    wrong = torch.zeros((1, 3, 2, 2), dtype=torch.float64)
    wrong[0, 0, 0, 1] = 0.4
    with pytest.raises(ValueError, match="reference_derivative must be Hermitian"):
        certify_coupling_numerators(
            _linear_hamiltonian,
            torch.tensor([1]),
            torch.zeros((1, 3), dtype=torch.float64),
            reference_derivative=wrong,
        )


def test_certification_detects_state_count_changing_finite_difference_stencil():
    def model(numbers, positions):
        del numbers
        x = positions[0, 0]
        count = 3 if abs(x.item()) > 0.0015 else 2
        return torch.eye(count, dtype=torch.float64) * (x + 0.5)

    with pytest.raises(ValueError, match="dimension changes"):
        certify_coupling_numerators(
            model,
            torch.tensor([1]),
            torch.zeros((1, 3), dtype=torch.float64),
            step=0.001,
        )


def test_one_state_numerator_certification_has_no_unresolved_state_pair():
    def scalar_model(numbers, positions):
        del numbers
        return (0.1 + positions[0, 0]).reshape(1, 1)

    result = certify_coupling_numerators(
        scalar_model,
        torch.tensor([1]),
        torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float64),
        reference_derivative=torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]]),
    )
    assert result.passed
    assert result.unresolved_pairs == ()
    assert math.isinf(result.minimum_energy_gap)
    assert not result.valid_gap_pairs.any()
    assert result.reference_max_error == pytest.approx(0.0)
