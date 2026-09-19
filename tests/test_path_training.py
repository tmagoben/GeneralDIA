from pathlib import Path

import pytest
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
    load_checkpoint,
    save_checkpoint,
    train_path_model,
)


class SlopeHamiltonian(nn.Module):
    def __init__(self, slope=0.6):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))

    def forward(self, atoms, positions):
        x = self.slope * positions[1, 0]
        return torch.diag(torch.stack((-x, x)))


def make_path(path_id, start, family=None, shift=0.0):
    samples = []
    for x in (start, start + 0.03):
        positions = torch.tensor([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]) + shift
        samples.append(MolecularSample(torch.tensor([1, 1]), positions, torch.tensor([-x, x])))
    return MolecularPath(
        samples,
        path_id=path_id,
        adjacent_overlaps=torch.eye(2)[None],
        metadata={} if family is None else {"molecular_family": family},
    )


def test_family_split_keeps_all_related_paths_together_and_is_reproducible():
    paths = MolecularPathDataset(
        make_path(f"path-{i}", 1 + 0.1 * i, f"family-{i // 2}") for i in range(12)
    )
    first = paths.split(seed=11, family_key="molecular_family")
    second = paths.split(seed=11, family_key="molecular_family")
    assert [p.path_ids for p in first] == [p.path_ids for p in second]
    assert set.union(*(set(p.path_ids) for p in first)) == set(paths.path_ids)
    assert sum(p.n_samples for p in first) == paths.n_samples
    assert_disjoint_paths(*first, family_key="molecular_family")


def test_family_split_rejects_missing_labels_and_insufficient_groups():
    missing = MolecularPathDataset(make_path(str(i), 1 + 0.1 * i) for i in range(4))
    with pytest.raises(ValueError, match="family metadata"):
        missing.split(family_key="molecular_family")
    few = MolecularPathDataset(make_path(str(i), 1 + 0.1 * i, "one") for i in range(4))
    with pytest.raises(ValueError, match="three families"):
        few.split(family_key="molecular_family")
    with pytest.raises(ValueError, match="finite positive"):
        missing.split(fractions=(1.0, float("nan"), 1.0))


def test_leakage_guard_catches_renaming_families_and_rigid_motion_duplicates():
    training = MolecularPathDataset([make_path("a", 1.0, "same")])
    with pytest.raises(ValueError, match="path ID leakage"):
        assert_disjoint_paths(training, MolecularPathDataset([make_path("a", 2.0, "other")]))
    with pytest.raises(ValueError, match="family leakage"):
        assert_disjoint_paths(
            training,
            MolecularPathDataset([make_path("b", 2.0, "same")]),
            family_key="molecular_family",
        )
    duplicate = make_path("renamed", 1.0, "different", shift=3.0)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for sample in duplicate:
        sample.positions = sample.positions @ rotation
    with pytest.raises(ValueError, match="duplicate geometry"):
        assert_disjoint_paths(
            training, MolecularPathDataset([duplicate]), family_key="molecular_family"
        )


def test_path_training_reduces_held_out_loss_and_checkpoint_roundtrip(tmp_path: Path):
    training = MolecularPathDataset([make_path("train", 1.0, "train-family")])
    validation = MolecularPathDataset([make_path("valid", 1.5, "valid-family")])
    config = TrainingConfig(epochs=35, learning_rate=0.03, seed=4, report_every=10)
    settings = PathTrackingSettings()
    model = SlopeHamiltonian()
    initial = evaluate_path_model(model, validation)["total"]
    history = train_path_model(
        model,
        training,
        validation_data=validation,
        config=config,
        settings=settings,
        family_key="molecular_family",
    )
    final = evaluate_path_model(model, validation)["total"]
    assert final < initial / 50
    repeated = SlopeHamiltonian()
    repeated_history = train_path_model(
        repeated, training, validation_data=validation, config=config, family_key="molecular_family"
    )
    assert repeated_history == history
    torch.testing.assert_close(repeated.slope, model.slope, atol=0, rtol=0)
    checkpoint = tmp_path / "path-model.pt"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        weights=LossWeights(),
        history=history,
        metadata={
            "loss": "path_block_invariants_v1",
            "train_ids": training.path_ids,
            "validation_ids": validation.path_ids,
        },
    )
    restored = SlopeHamiltonian()
    load_checkpoint(checkpoint, restored)
    assert evaluate_path_model(restored, validation)["total"] == final


def test_leakage_fails_before_parameter_mutation():
    model = SlopeHamiltonian()
    original = model.slope.detach().clone()
    train = MolecularPathDataset([make_path("a", 1.0, "one")])
    validation = MolecularPathDataset([make_path("b", 1.4, "one")])
    with pytest.raises(ValueError, match="family leakage"):
        train_path_model(model, train, validation_data=validation, family_key="molecular_family")
    torch.testing.assert_close(model.slope, original)
