"""Fixed-budget LiH FCI comparison on held-out bond scans within one molecule.

Run with the pyscf extra. This is a small-basis integration benchmark, not a test
of chemical accuracy, unseen molecules, derivative targets, or nonadiabatic dynamics.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from generaldia import (
    LossWeights,
    MolecularPathDataset,
    PathTrackingSettings,
    SharedMolecularOperators,
    SimpleMolecularHamiltonian,
    TrainingConfig,
    assert_disjoint_paths,
    evaluate_path_model,
    load_checkpoint,
    predict_path,
    save_checkpoint,
    train_path_model,
)
from generaldia.electronic_structure.fci_dipoles import FCIDipoleBackend

# Declared before fitting; no test-set model selection. All paths belong to LiH.
BOND_LENGTHS = {
    "train-short": [1.20, 1.25, 1.30],
    "train-long": [1.80, 1.90, 2.00],
    "validation": [1.35, 1.40, 1.45],
    "test": [1.55, 1.60, 1.65],
}


def physical_metrics(model, dataset: MolecularPathDataset) -> dict[str, float]:
    """Report energy MAE and gauge-invariant dipole errors in declared units."""

    energies, permanent, strength = [], [], []
    for path in dataset:
        prediction = predict_path(model, path, dipoles=hasattr(model, "operators"))
        target = path.tracked()
        energies.append((prediction.energies.detach() - target.tracked_energies).abs().reshape(-1))
        if prediction.dipole_matrices is not None:
            p = prediction.dipole_matrices.detach()
            t = torch.stack([sample.dipole_matrix_elements for sample in target.tracked_path])
            delta = p.diagonal(dim1=-2, dim2=-1) - t.diagonal(dim1=-2, dim2=-1)
            permanent.append(torch.linalg.vector_norm(delta, dim=1).reshape(-1))
            strength.append(
                (p[:, :, 0, 1].abs().square().sum(1) - t[:, :, 0, 1].abs().square().sum(1)).abs()
            )
    result = {"energy_mae_hartree": float(torch.cat(energies).mean())}
    if permanent:
        result["permanent_dipole_mean_vector_error_e_bohr"] = float(torch.cat(permanent).mean())
        result["transition_strength_mae_e2_bohr2"] = float(torch.cat(strength).mean())
    return result


def main() -> None:
    from pyscf import lib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("outputs/lih_shared_operators"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(23)
    lib.num_threads(1)
    backend = FCIDipoleBackend(conv_tol=1e-12)
    paths = {
        name: backend.calculate_path(
            [3, 1],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, length]] for length in lengths],
            path_id=f"lih-{name}",
            family_id="LiH",
        )
        for name, lengths in BOND_LENGTHS.items()
    }
    training = MolecularPathDataset([paths["train-short"], paths["train-long"]])
    validation = MolecularPathDataset([paths["validation"]])
    test = MolecularPathDataset([paths["test"]])
    # Intentionally NOT a molecular-family split: this measures interpolation in LiH.
    assert_disjoint_paths(training, validation, test)
    for path in paths.values():
        assert path.tracked().ambiguous_steps == ()
    model = SharedMolecularOperators(hidden=8, n_rbf=6, r_min=0.5, r_max=3.0)
    # Energy offset/gap initialization uses training labels only.
    mean_energies = torch.stack([sample.energies for path in training for sample in path]).mean(0)
    with torch.no_grad():
        model.head[-1].weight.mul_(0.01)
        model.head[-1].bias.copy_(
            torch.stack((mean_energies[0], mean_energies[0] * 0, mean_energies[1]))
        )
    baseline = SimpleMolecularHamiltonian(
        **{
            key: value
            for key, value in model.configuration.items()
            if key not in {"mode", "charge"}
        }
    )
    baseline.load_state_dict(
        {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("dipole_head.")
        }
    )
    probe = training[0][0]
    torch.testing.assert_close(
        baseline(probe.atomic_numbers, probe.positions),
        model(probe.atomic_numbers, probe.positions),
        rtol=0,
        atol=0,
    )
    config = TrainingConfig(epochs=args.epochs, learning_rate=0.005, seed=23, report_every=50)
    supervised = LossWeights(energy=1.0, dipole=0.1, dipole_scale=2.0)
    energy_only = LossWeights(energy=1.0)
    settings = PathTrackingSettings()
    args.output.mkdir(parents=True, exist_ok=True)
    reference_file = args.output / "reference_paths.pt"
    torch.save(
        {
            name: {
                "path_id": path.path_id,
                "metadata": path.metadata,
                "adjacent_overlaps": path.adjacent_overlaps,
                "samples": [asdict(sample) for sample in path],
            }
            for name, path in paths.items()
        },
        reference_file,
    )
    source_hash = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for source in [*sorted((root / "generaldia").rglob("*.py")), Path(__file__).resolve()]:
        source_hash.update(
            source.relative_to(root).as_posix().encode() + b"\0" + source.read_bytes()
        )
    results = {
        "schema": "generaldia.lih_shared_operators.v1",
        "scope": "two-state all-electron singlet FCI/STO-3G; held-out LiH geometry paths",
        "limitations": [
            "one molecule",
            "one seed",
            "minimal basis",
            "no derivative labels",
            "no joint-operator molecular validation",
            "no chemical-accuracy claim",
        ],
        "source_sha256": source_hash.hexdigest(),
        "reference_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "reference_provenance": paths["test"].metadata,
        "bond_lengths_angstrom": BOND_LENGTHS,
        "split": {
            "train": list(training.path_ids),
            "validation": list(validation.path_ids),
            "test": list(test.path_ids),
            "molecular_family": "LiH",
        },
        "training_configuration": asdict(config),
        "training_mean_energies_hartree": mean_energies.tolist(),
        "tracking_settings": asdict(settings),
        "checkpoint_selection": "fixed final epoch; no test or validation selection",
        "runs": {},
    }
    for name, candidate, weights in (
        ("v32_hamiltonian_only", baseline, energy_only),
        ("shared_energy_only", deepcopy(model), energy_only),
        ("shared_energy_dipole", deepcopy(model), supervised),
    ):
        evaluation_weights = supervised if hasattr(candidate, "operators") else energy_only
        initial = evaluate_path_model(candidate, validation, weights=evaluation_weights)
        history = train_path_model(
            candidate,
            training,
            validation_data=validation,
            weights=weights,
            config=config,
            settings=settings,
        )
        record = {
            "model_configuration": candidate.configuration,
            "loss_weights": asdict(weights),
            "initial_validation": initial,
            "final_validation": evaluate_path_model(
                candidate, validation, weights=evaluation_weights
            ),
            "test_loss": evaluate_path_model(candidate, test, weights=evaluation_weights),
            "test_metrics": physical_metrics(candidate, test),
            "history": [asdict(epoch) for epoch in history],
        }
        checkpoint = args.output / f"{name}.pt"
        save_checkpoint(
            checkpoint,
            candidate,
            config=config,
            weights=weights,
            history=history,
            metadata={
                "source_sha256": results["source_sha256"],
                "reference_sha256": results["reference_sha256"],
                "run": record,
            },
        )
        restored = type(candidate)(**candidate.configuration)
        metadata = load_checkpoint(checkpoint, restored)
        assert metadata["model_configuration"] == candidate.configuration
        for path in test:
            before = predict_path(candidate, path, dipoles=hasattr(candidate, "operators"))
            after = predict_path(restored, path, dipoles=hasattr(restored, "operators"))
            torch.testing.assert_close(before.energies, after.energies, rtol=0, atol=0)
            if before.dipole_matrices is not None:
                torch.testing.assert_close(
                    before.dipole_matrices, after.dipole_matrices, rtol=0, atol=0
                )
        record["checkpoint_roundtrip_exact"] = True
        results["runs"][name] = record
        print(name, json.dumps(record["test_metrics"], allow_nan=False), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
