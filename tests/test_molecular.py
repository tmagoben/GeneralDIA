import math

import pytest
import torch

from generaldia.molecular import GaussianRBF, SimpleMolecularHamiltonian
from generaldia.observables import energy_gradients


def make_system() -> tuple[SimpleMolecularHamiltonian, torch.Tensor, torch.Tensor]:
    torch.manual_seed(5)
    model = SimpleMolecularHamiltonian(hidden=12, n_rbf=7)
    atomic_numbers = torch.tensor([8, 1, 6, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.4, 1.1, 0.2], [0.2, -0.7, 0.6]])
    return model, atomic_numbers, positions


def test_translation_rotation_reflection_and_permutation_invariance() -> None:
    model, atomic_numbers, positions = make_system()
    reference = model(atomic_numbers, positions)
    translated = positions + torch.tensor([1.2, -0.3, 0.8])
    angle = 0.41
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    permutation = torch.tensor([2, 0, 3, 1])
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    assert torch.allclose(reference, reference.mT)
    assert torch.allclose(reference, model(atomic_numbers, translated), atol=1e-12)
    assert torch.allclose(reference, model(atomic_numbers, positions @ rotation.mT), atol=1e-12)
    assert torch.allclose(reference, model(atomic_numbers, positions @ reflection), atol=1e-12)
    assert torch.allclose(
        reference, model(atomic_numbers[permutation], positions[permutation]), atol=1e-12
    )


def test_translational_gradient_sum_is_zero() -> None:
    model, atomic_numbers, positions = make_system()
    _, gradients = energy_gradients(model, atomic_numbers, positions)
    assert torch.max(torch.abs(gradients.sum(dim=1))) < 1e-10


def test_vectorized_representation_propagates_coordinate_gradients() -> None:
    model, atomic_numbers, positions = make_system()
    positions.requires_grad_(True)
    model(atomic_numbers, positions).sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


@pytest.mark.parametrize(
    ("atomic_numbers", "positions", "message"),
    [
        (torch.tensor([1]), torch.zeros(1, 3), "at least two"),
        (torch.tensor([1, 99]), torch.zeros(2, 3), "atomic numbers"),
        (torch.tensor([1, 1]), torch.zeros(2, 2), "positions"),
        (torch.tensor([1, 1]), torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0, 0]]), "finite"),
    ],
)
def test_model_rejects_invalid_inputs(
    atomic_numbers: torch.Tensor, positions: torch.Tensor, message: str
) -> None:
    model = SimpleMolecularHamiltonian()
    with pytest.raises(ValueError, match=message):
        model(atomic_numbers, positions)


def test_rbf_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        GaussianRBF(n=0)
    with pytest.raises(ValueError):
        GaussianRBF(r_min=2.0, r_max=1.0)
