"""Reproducible complete-path training and a finite-state crossing negative control."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from generaldia import (
    LossWeights,
    MolecularPath,
    MolecularPathDataset,
    MolecularSample,
    PathTrackingSettings,
    TrainingConfig,
    assert_disjoint_paths,
    evaluate_path_model,
    path_observable_loss,
    save_checkpoint,
    train_path_model,
)


class CrossingHamiltonian(nn.Module):
    """Synthetic H=diag(s*x,-s*x); positions use a synthetic length unit."""

    def __init__(self, slope: float = 0.6, *, wrong_character: bool = False) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.wrong_character = wrong_character

    def forward(self, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = positions[1, 0]
        if self.wrong_character:
            x = x.abs()
        value = self.slope * x
        return torch.diag(torch.stack((value, -value)))


def make_path(index: int) -> MolecularPath:
    samples, raw_frames = [], []
    # Sample on either side of the crossing. A sampled block split/merge would
    # correctly fail the current continuation contract.
    distance = 0.8 + 0.13 * index
    for step, x in enumerate((-distance, distance)):
        positions = torch.tensor([[0.0, 0.0, 0.0], [x, 0.0, 0.1 * index]])
        h = torch.diag(torch.tensor([x, -x]))
        energies, frame = torch.linalg.eigh(h)
        permutation = [1, 0] if (index + step) % 2 else [0, 1]
        phases = torch.exp(1j * torch.tensor([0.3 * index, -0.4 * step]))
        frame = (frame.to(torch.complex128) * phases)[:, permutation]
        derivative = torch.zeros(2, 3, 2, 2, dtype=torch.complex128)
        derivative[1, 0] = frame.mH @ torch.diag(torch.tensor([1.0, -1.0])).to(frame.dtype) @ frame
        samples.append(
            MolecularSample(
                torch.tensor([1, 1]),
                positions,
                energies[permutation],
                derivative_matrix_elements=derivative,
                metadata={"energy_unit": "synthetic", "coordinate_unit": "synthetic"},
            )
        )
        raw_frames.append(frame)
    return MolecularPath(
        samples,
        path_id=f"scan-{index}",
        adjacent_overlaps=(raw_frames[0].mH @ raw_frames[1])[None],
        metadata={"molecular_family": f"synthetic-family-{index // 2}"},
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(23)
    dataset = MolecularPathDataset(make_path(index) for index in range(8))
    partitions = dataset.split(seed=23, family_key="molecular_family")
    training, validation, test = partitions
    assert_disjoint_paths(*partitions, family_key="molecular_family")
    model = CrossingHamiltonian()
    settings = PathTrackingSettings()
    weights = LossWeights(energy=1.0, derivative_matrix=0.1)
    config = TrainingConfig(epochs=50, learning_rate=0.02, seed=23, report_every=10)
    initial = evaluate_path_model(model, test, weights=weights)
    history = train_path_model(
        model,
        training,
        validation_data=validation,
        config=config,
        weights=weights,
        settings=settings,
        family_key="molecular_family",
    )
    final = evaluate_path_model(model, test, weights=weights)
    matched = path_observable_loss(CrossingHamiltonian(1.0), test[0], weights).scalars()
    wrong = path_observable_loss(
        CrossingHamiltonian(1.0, wrong_character=True), test[0], weights
    ).scalars()
    assert final["total"] < initial["total"] / 100
    assert matched["total"] < 1e-24 and wrong["total"] > 0.1
    result = {
        "schema": "generaldia.path_invariant_example.v1",
        "torch_version": str(torch.__version__),
        "tracking_settings": asdict(settings),
        "loss_weights": asdict(weights),
        "training_config": asdict(config),
        "family_key": "molecular_family",
        "split": {
            name: {
                "path_ids": list(part.path_ids),
                "families": sorted({p.metadata["molecular_family"] for p in part}),
            }
            for name, part in zip(("train", "validation", "test"), partitions, strict=True)
        },
        "initial_test": initial,
        "final_test": final,
        "matched_control": matched,
        "wrong_continuation_control": wrong,
        "fitted_slope": float(model.slope.detach()),
    }
    output = Path("outputs/invariant_path_training")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_checkpoint(
        output / "model.pt", model, config=config, weights=weights, history=history, metadata=result
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
