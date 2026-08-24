"""Small, inspectable training and evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import MolecularDataset
from .losses import LossWeights, observable_loss
from .observables import derivative_matrix_elements


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for deterministic full-dataset training."""

    epochs: int = 300
    learning_rate: float = 3e-3
    seed: int = 0
    gradient_clip_norm: float | None = 10.0
    report_every: int = 25

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.report_every < 1:
            raise ValueError("report_every must be positive")


@dataclass
class EpochRecord:
    """Mean training and validation losses recorded at one epoch."""

    epoch: int
    train_loss: float
    validation_loss: float | None


def _dataset_loss(model: nn.Module, dataset: MolecularDataset, weights: LossWeights) -> float:
    model.eval()
    values = []
    for sample in dataset:
        values.append(float(observable_loss(model, sample, weights).total.detach()))
    return float(np.mean(values))


def train_model(
    model: nn.Module,
    training_data: MolecularDataset,
    *,
    validation_data: MolecularDataset | None = None,
    weights: LossWeights | None = None,
    config: TrainingConfig | None = None,
) -> list[EpochRecord]:
    """Train a model and return checkpoint-ready epoch records.

    The loop processes one geometry at a time to support variable atom counts.
    It favors readability and deterministic examples over throughput.
    """

    if weights is None:
        weights = LossWeights()
    if config is None:
        config = TrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    history: list[EpochRecord] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(training_data), generator=generator).tolist()
        training_losses = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            breakdown = observable_loss(model, training_data[index], weights)
            breakdown.total.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            training_losses.append(float(breakdown.total.detach()))

        should_record = epoch == 1 or epoch == config.epochs or epoch % config.report_every == 0
        if should_record:
            validation_loss = (
                None if validation_data is None else _dataset_loss(model, validation_data, weights)
            )
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=float(np.mean(training_losses)),
                    validation_loss=validation_loss,
                )
            )
    return history


def evaluate_model(model: nn.Module, dataset: MolecularDataset) -> dict[str, float]:
    """Return mean absolute errors for all reference quantities present."""

    model.eval()
    parameter = next(model.parameters())
    energy_errors = []
    gradient_errors = []
    derivative_errors = []
    for raw_sample in dataset:
        sample = raw_sample.to(parameter.device, parameter.dtype)
        if sample.energy_gradients is not None or sample.derivative_matrix_elements is not None:
            energies, _, derivative = derivative_matrix_elements(
                model, sample.atomic_numbers, sample.positions, create_graph=False
            )
        else:
            with torch.no_grad():
                energies = torch.linalg.eigvalsh(model(sample.atomic_numbers, sample.positions))
            derivative = None
        energy_errors.append(torch.abs(energies - sample.energies).detach().cpu())
        if sample.energy_gradients is not None and derivative is not None:
            gradients = torch.diagonal(derivative, dim1=-2, dim2=-1).permute(2, 0, 1)
            gradient_errors.append(
                torch.abs(gradients.real - sample.energy_gradients).detach().cpu()
            )
        if sample.derivative_matrix_elements is not None and derivative is not None:
            target = sample.derivative_matrix_elements.to(derivative.dtype)
            derivative_errors.append(torch.abs(derivative - target).detach().cpu())

    metrics = {"energy_mae": float(torch.cat(energy_errors).mean())}
    if gradient_errors:
        metrics["energy_gradient_mae"] = float(
            torch.cat([value.reshape(-1) for value in gradient_errors]).mean()
        )
    if derivative_errors:
        metrics["derivative_matrix_mae"] = float(
            torch.cat([value.reshape(-1) for value in derivative_errors]).mean()
        )
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    config: TrainingConfig,
    weights: LossWeights,
    history: list[EpochRecord],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights and experiment settings to a Torch checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "model_configuration": getattr(model, "configuration", None),
            "model_state_dict": model.state_dict(),
            "training_config": asdict(config),
            "loss_weights": asdict(weights),
            "history": [asdict(record) for record in history],
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load trusted checkpoint weights into ``model`` and return stored metadata."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("format_version") != 1 or "model_state_dict" not in checkpoint:
        raise ValueError("unsupported or invalid GeneralDIA checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
