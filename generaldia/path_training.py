"""Small reference trainer that preserves complete paths per optimizer step."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .dataset import MolecularPathDataset, PathTrackingSettings, assert_disjoint_paths
from .losses import LossWeights
from .path_losses import path_observable_loss
from .training import EpochRecord, TrainingConfig


def evaluate_path_model(
    model: nn.Module,
    dataset: MolecularPathDataset,
    *,
    weights: LossWeights | None = None,
    settings: PathTrackingSettings | None = None,
) -> dict[str, float]:
    """Return means of invariant component MSEs, weighting every path equally.

    Legacy components use squared input units; scaled dipole/joint components are
    dimensionless. These are not elementwise matrix MAEs. Coordinate derivatives
    require autograd even during evaluation.
    """

    model.eval()
    records = [
        path_observable_loss(model, path, weights, settings=settings).scalars() for path in dataset
    ]
    return {name: float(np.mean([record[name] for record in records])) for name in records[0]}


def train_path_model(
    model: nn.Module,
    training_data: MolecularPathDataset,
    *,
    validation_data: MolecularPathDataset | None = None,
    weights: LossWeights | None = None,
    config: TrainingConfig | None = None,
    settings: PathTrackingSettings | None = None,
    family_key: str | None = None,
    geometry_tolerance: float = 1e-8,
) -> list[EpochRecord]:
    """Train on complete paths with fail-closed tracking and finite-gradient checks.

    Training/validation IDs, optional family labels, and ordered-atom duplicate
    geometries are checked before any parameter update. Validation may be omitted
    for fitting experiments. Call ``assert_disjoint_paths`` on all three partitions
    before held-out evaluation. Loss units and settings follow ``path_observable_loss``.
    """

    weights, config = weights or LossWeights(), config or TrainingConfig()
    partitions = (training_data,) if validation_data is None else (training_data, validation_data)
    assert_disjoint_paths(*partitions, family_key=family_key, geometry_tolerance=geometry_tolerance)
    # Validate every initial path before mutating model parameters.
    for partition in partitions:
        for path in partition:
            path_observable_loss(model, path, weights, settings=settings)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for index in torch.randperm(len(training_data), generator=generator).tolist():
            optimizer.zero_grad(set_to_none=True)
            result = path_observable_loss(model, training_data[index], weights, settings=settings)
            if not torch.isfinite(result.total):
                raise ValueError("path training encountered a non-finite loss")
            result.total.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("path training encountered non-finite parameter gradients")
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(result.total.detach()))
        if epoch == 1 or epoch == config.epochs or epoch % config.report_every == 0:
            validation_loss = (
                None
                if validation_data is None
                else evaluate_path_model(
                    model, validation_data, weights=weights, settings=settings
                )["total"]
            )
            history.append(EpochRecord(epoch, float(np.mean(losses)), validation_loss))
    return history
