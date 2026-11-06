"""Train complex H/dipoles jointly on an analytic synthetic diatomic reference.

Coordinates and observables use synthetic units. This example checks integration;
the separate PySCF example provides the small molecular-reference comparison.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from generaldia import (
    LossWeights,
    MolecularOperators,
    MolecularPath,
    MolecularPathDataset,
    MolecularSample,
    PathTrackingSettings,
    SharedMolecularOperators,
    TrainingConfig,
    assert_disjoint_paths,
    evaluate_path_model,
    load_checkpoint,
    predict_path,
    save_checkpoint,
    train_path_model,
)
from generaldia.observables import hamiltonian_jacobian


class AnalyticReference(nn.Module):
    """Avoided-crossing Hamiltonian with a distinct, noncommuting vector operator."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        delta = 0.4 * (torch.linalg.vector_norm(r[1] - r[0]) - 1.5)
        h = torch.tensor([[0.0, 0.2 + 0.07j], [0.2 - 0.07j, 0.0]])
        return self.scale * (h + torch.diag(torch.stack((delta, -delta))))

    def operators(self, z: torch.Tensor, r: torch.Tensor) -> MolecularOperators:
        weight = torch.tensor([[0.4, 0.15 - 0.08j], [0.15 + 0.08j, -0.3]])
        dipoles = (r[1] - r[0])[:, None, None] * weight
        return MolecularOperators(self(z, r), dipoles)


def make_path(name: str, lengths: list[float]) -> MolecularPath:
    """Supply exact reference operators and deliberately scramble raw state phases."""

    reference = AnalyticReference()
    samples, frames = [], []
    for index, length in enumerate(lengths):
        z = torch.tensor([3, 1])
        r = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, length]])
        h, jac = hamiltonian_jacobian(reference, z, r, create_graph=False)
        energies, frame = torch.linalg.eigh(h.detach())
        frame = frame * torch.exp(1j * torch.tensor([0.4 * index, -0.3 * index]))
        permutation = [1, 0] if index % 2 else [0, 1]
        frame = frame[:, permutation]
        samples.append(
            MolecularSample(
                z,
                r,
                energies[permutation],
                derivative_matrix_elements=frame.mH @ jac.detach() @ frame,
                dipole_matrix_elements=frame.mH
                @ reference.operators(z, r).dipoles.detach()
                @ frame,
                metadata={
                    "coordinate_unit": "synthetic",
                    "energy_unit": "synthetic",
                    "dipole_unit": "synthetic",
                    "charge": 0,
                },
            )
        )
        frames.append(frame)
    frames = torch.stack(frames)
    return MolecularPath(samples, path_id=name, adjacent_overlaps=frames[:-1].mH @ frames[1:])


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(29)
    training = MolecularPathDataset(
        [make_path("short", [1.2, 1.25]), make_path("long", [1.85, 1.9])]
    )
    validation = MolecularPathDataset([make_path("validation", [1.4, 1.45])])
    test = MolecularPathDataset([make_path("test", [1.6, 1.65])])
    assert_disjoint_paths(training, validation, test)
    model = SharedMolecularOperators(hidden=8, n_rbf=6, mode="complex")
    config = TrainingConfig(epochs=80, learning_rate=0.004, seed=29, report_every=20)
    weights = LossWeights(energy=1.0, dipole=0.3, joint_operator=0.2)
    initial = evaluate_path_model(model, validation, weights=weights)
    history = train_path_model(
        model, training, validation_data=validation, config=config, weights=weights
    )
    final = evaluate_path_model(model, validation, weights=weights)
    assert final["total"] < initial["total"]
    result = {
        "schema": "generaldia.synthetic_shared_operators.v1",
        "scope": "analytic synthetic reference; no chemistry or unique-gauge claim",
        "torch_version": str(torch.__version__),
        "training_config": asdict(config),
        "loss_weights": asdict(weights),
        "tracking_settings": asdict(PathTrackingSettings()),
        "split": {
            "train": list(training.path_ids),
            "validation": list(validation.path_ids),
            "test": list(test.path_ids),
        },
        "initial_validation": initial,
        "final_validation": final,
        "test": evaluate_path_model(model, test, weights=weights),
    }
    output = Path("outputs/shared_operators")
    save_checkpoint(
        output / "model.pt", model, config=config, weights=weights, history=history, metadata=result
    )
    restored = SharedMolecularOperators(**model.configuration)
    load_checkpoint(output / "model.pt", restored)
    before = predict_path(model, test[0], derivatives=True, dipoles=True)
    after = predict_path(restored, test[0], derivatives=True, dipoles=True)
    for name in ("energies", "derivative_matrices", "dipole_matrices"):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    result["checkpoint_roundtrip_exact"] = True
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
