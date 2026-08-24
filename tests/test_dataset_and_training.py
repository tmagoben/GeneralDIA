from pathlib import Path

import pytest
import torch
from torch import nn

from generaldia import (
    LossWeights,
    MolecularDataset,
    MolecularSample,
    SimpleMolecularHamiltonian,
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    observable_loss,
    save_checkpoint,
    train_model,
)
from generaldia.observables import derivative_matrix_elements, energy_gradients


class ReferenceModel(nn.Module):
    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del atomic_numbers
        distance = torch.linalg.vector_norm(positions[1] - positions[0])
        coordinate = distance - 1.0
        coupling = coordinate.new_tensor(0.04)
        return torch.stack(
            (
                torch.stack((0.2 * coordinate, coupling)),
                torch.stack((coupling, -0.1 * coordinate + 0.3)),
            )
        )


def make_dataset(count: int = 9) -> MolecularDataset:
    atomic_numbers = torch.tensor([1, 1])
    samples = []
    for distance in torch.linspace(0.7, 1.5, count):
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
        energies, gradients = energy_gradients(
            ReferenceModel(), atomic_numbers, positions, create_graph=False
        )
        samples.append(
            MolecularSample(
                atomic_numbers,
                positions,
                energies.detach(),
                gradients.detach(),
            )
        )
    return MolecularDataset(samples)


def test_sample_validates_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="positions"):
        MolecularSample(torch.tensor([1, 1]), torch.zeros(2, 2), torch.zeros(2))
    with pytest.raises(ValueError, match="energy_gradients"):
        MolecularSample(
            torch.tensor([1, 1]), torch.zeros(2, 3), torch.zeros(2), torch.zeros(2, 2, 2)
        )


def test_dataset_split_is_deterministic_and_complete() -> None:
    dataset = make_dataset()
    first = dataset.split(seed=4)
    second = dataset.split(seed=4)
    assert [len(partition) for partition in first] == [len(partition) for partition in second]
    assert sum(len(partition) for partition in first) == len(dataset)
    assert torch.equal(first[0][0].positions, second[0][0].positions)


def test_observable_loss_uses_energy_and_gradient_targets() -> None:
    torch.manual_seed(2)
    model = SimpleMolecularHamiltonian(hidden=8, n_rbf=5)
    breakdown = observable_loss(
        model,
        make_dataset(3)[0],
        LossWeights(energy=1.0, energy_gradient=0.1),
    )
    breakdown.total.backward()
    assert breakdown.energy_gradient is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_observable_loss_accepts_gauge_consistent_derivative_targets() -> None:
    source = make_dataset(3)[0]
    _, _, target = derivative_matrix_elements(
        ReferenceModel(), source.atomic_numbers, source.positions, create_graph=False
    )
    sample = MolecularSample(
        source.atomic_numbers,
        source.positions,
        source.energies,
        derivative_matrix_elements=target.detach(),
    )
    model = SimpleMolecularHamiltonian(hidden=8, n_rbf=5)
    breakdown = observable_loss(
        model,
        sample,
        LossWeights(energy=1.0, derivative_matrix=0.05),
    )
    assert breakdown.derivative_matrix is not None
    assert "derivative_matrix" in breakdown.scalars()


def test_training_evaluation_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    training, validation, test = make_dataset().split(seed=3)
    model = SimpleMolecularHamiltonian(hidden=8, n_rbf=5)
    config = TrainingConfig(epochs=3, learning_rate=1e-3, seed=3, report_every=1)
    weights = LossWeights()
    history = train_model(
        model,
        training,
        validation_data=validation,
        config=config,
        weights=weights,
    )
    metrics = evaluate_model(model, test)
    assert len(history) == 3
    assert metrics["energy_mae"] >= 0

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config=config,
        weights=weights,
        history=history,
        metadata={"test": True},
    )
    restored = SimpleMolecularHamiltonian(hidden=8, n_rbf=5)
    metadata = load_checkpoint(checkpoint_path, restored)
    assert metadata["metadata"] == {"test": True}
    assert metadata["model_configuration"] == model.configuration
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_invalid_training_configuration_and_loss_weights() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        LossWeights(energy=0, energy_gradient=0, derivative_matrix=0)
