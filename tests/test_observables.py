import pytest
import torch

from generaldia.observables import (
    adiabatic_energies,
    coupling_validity_mask,
    derivative_couplings_from_numerators,
    derivative_matrix_elements,
    energy_gradients,
    hamiltonian_jacobian,
)


class LinearModel(torch.nn.Module):
    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del atomic_numbers
        coordinate = positions[0, 0]
        coupling = coordinate.new_tensor(0.01)
        return torch.stack(
            (
                torch.stack((0.05 * coordinate, coupling)),
                torch.stack((coupling, -0.05 * coordinate)),
            )
        )


def test_derivative_matrix_element_matches_analytic_magnitude() -> None:
    atomic_numbers = torch.tensor([1])
    positions = torch.tensor([[0.3, 0.0, 0.0]])
    _, _, numerators = derivative_matrix_elements(LinearModel(), atomic_numbers, positions)
    expected = torch.tensor(0.05 * 0.01 / (((0.05 * 0.3) ** 2 + 0.01**2) ** 0.5))
    assert torch.allclose(torch.abs(numerators[0, 0, 0, 1]), expected, atol=1e-10)


def test_energy_gradient_matches_finite_difference() -> None:
    atomic_numbers = torch.tensor([1])
    positions = torch.tensor([[0.3, 0.0, 0.0]])
    energies, gradients = energy_gradients(
        LinearModel(), atomic_numbers, positions, create_graph=False
    )
    step = 1e-6
    plus = positions.clone()
    minus = positions.clone()
    plus[0, 0] += step
    minus[0, 0] -= step
    finite_difference = (
        torch.linalg.eigvalsh(LinearModel()(atomic_numbers, plus))
        - torch.linalg.eigvalsh(LinearModel()(atomic_numbers, minus))
    ) / (2 * step)
    assert torch.allclose(gradients[:, 0, 0], finite_difference, atol=1e-9)
    assert energies.shape == (2,)


def test_hamiltonian_jacobian_shape_and_hermiticity() -> None:
    matrix, jacobian = hamiltonian_jacobian(
        LinearModel(), torch.tensor([1]), torch.tensor([[0.2, 0.0, 0.0]])
    )
    assert matrix.shape == (2, 2)
    assert jacobian.shape == (1, 3, 2, 2)
    assert torch.allclose(jacobian, jacobian.mH)


def test_derivative_couplings_return_degeneracy_mask() -> None:
    energies = torch.tensor([0.0, 1e-8, 0.2])
    numerators = torch.ones(2, 3, 3, 3)
    couplings, mask = derivative_couplings_from_numerators(
        energies, numerators, gap_floor=1e-6, return_mask=True
    )
    assert not mask[0, 1]
    assert mask[0, 2]
    assert couplings[..., 0, 1].count_nonzero() == 0
    assert torch.allclose(couplings[..., 0, 2], torch.full((2, 3), 5.0))
    assert torch.equal(mask, coupling_validity_mask(energies))


def test_observable_validation_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="square"):
        adiabatic_energies(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="positive"):
        coupling_validity_mask(torch.tensor([0.0, 1.0]), gap_floor=0)
    with pytest.raises(ValueError, match="state dimensions"):
        derivative_couplings_from_numerators(torch.tensor([0.0, 1.0]), torch.zeros(3, 3))
