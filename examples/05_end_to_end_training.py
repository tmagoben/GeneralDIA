"""Train, evaluate, save, and reload a synthetic two-state model."""

from pathlib import Path

import torch
from torch import nn

from generaldia import (
    LossWeights,
    MolecularDataset,
    MolecularSample,
    SimpleMolecularHamiltonian,
    TrainingConfig,
    energy_gradients,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    train_model,
)


class SyntheticDiatomicHamiltonian(nn.Module):
    """Known target used to test the complete software workflow."""

    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del atomic_numbers
        distance = torch.linalg.vector_norm(positions[1] - positions[0])
        coordinate = distance - 1.2
        coupling = 0.04 * torch.exp(-0.7 * coordinate**2)
        return torch.stack(
            (
                torch.stack((0.18 * coordinate - 0.6, coupling)),
                torch.stack((coupling, -0.12 * coordinate - 0.2)),
            )
        )


def make_dataset() -> MolecularDataset:
    """Sample a bond path and calculate exact targets by differentiation."""

    reference = SyntheticDiatomicHamiltonian()
    atomic_numbers = torch.tensor([1, 1])
    samples = []
    for distance in torch.linspace(0.7, 2.0, 36):
        positions = torch.tensor([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])
        energies, gradients = energy_gradients(
            reference, atomic_numbers, positions, create_graph=False
        )
        samples.append(
            MolecularSample(
                atomic_numbers=atomic_numbers,
                positions=positions,
                energies=energies.detach(),
                energy_gradients=gradients.detach(),
                metadata={
                    "coordinate_unit": "synthetic length unit",
                    "energy_unit": "synthetic energy unit",
                },
            )
        )
    return MolecularDataset(samples)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    training_data, validation_data, test_data = make_dataset().split(seed=17)
    model = SimpleMolecularHamiltonian(n_states=2, hidden=24, n_rbf=10, r_max=3.0)
    weights = LossWeights(energy=1.0, energy_gradient=0.2)
    config = TrainingConfig(epochs=200, learning_rate=3e-3, seed=17, report_every=20)
    history = train_model(
        model,
        training_data,
        validation_data=validation_data,
        weights=weights,
        config=config,
    )
    metrics = evaluate_model(model, test_data)
    print("test metrics", metrics)
    print("last history record", history[-1])

    checkpoint_path = Path("outputs") / "synthetic_two_state.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config=config,
        weights=weights,
        history=history,
        metadata={"example": "05_end_to_end_training", "seed": 17},
    )
    restored = SimpleMolecularHamiltonian(n_states=2, hidden=24, n_rbf=10, r_max=3.0)
    checkpoint_metadata = load_checkpoint(checkpoint_path, restored)
    restored_metrics = evaluate_model(restored, test_data)
    print("restored test metrics", restored_metrics)
    print("checkpoint format", checkpoint_metadata["format_version"])


if __name__ == "__main__":
    main()
