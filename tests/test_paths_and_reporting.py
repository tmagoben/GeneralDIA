import json
import math
from pathlib import Path

import pytest
import torch

from generaldia import MolecularPath, MolecularPathDataset, MolecularSample
from generaldia.reporting import state_tracking_report_data, write_state_tracking_report


def _raw_column_gauge(permutation: tuple[int, ...], phases: tuple[complex, ...]) -> torch.Tensor:
    gauge = torch.zeros((len(permutation), len(permutation)), dtype=torch.complex128)
    for raw_index, canonical_index in enumerate(permutation):
        gauge[canonical_index, raw_index] = phases[raw_index]
    return gauge


def _tracked_target_path(path_id: str = "crossing-A") -> tuple[MolecularPath, torch.Tensor]:
    atomic_numbers = torch.tensor([1, 1])
    canonical_energies = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]], dtype=torch.float64)
    permutations = ((0, 1), (1, 0), (0, 1))
    phases = ((1.0, 1.0), (1j, -1.0), (complex(torch.exp(torch.tensor(0.3j))), -1j))
    gauges = torch.stack(
        tuple(
            _raw_column_gauge(permutation, phase)
            for permutation, phase in zip(permutations, phases, strict=True)
        )
    )

    canonical_derivatives = torch.empty((3, 2, 3, 2, 2), dtype=torch.complex128)
    base = torch.tensor([[0.2, 0.03 + 0.04j], [0.03 - 0.04j, -0.1]], dtype=torch.complex128)
    for geometry in range(3):
        for atom in range(2):
            for axis in range(3):
                canonical_derivatives[geometry, atom, axis] = base * (geometry + atom + axis + 1)

    samples = []
    for index, (permutation, gauge) in enumerate(zip(permutations, gauges, strict=True)):
        raw_derivatives = gauge.mH @ canonical_derivatives[index] @ gauge
        raw_gradients = torch.diagonal(raw_derivatives, dim1=-2, dim2=-1).real.permute(2, 0, 1)
        samples.append(
            MolecularSample(
                atomic_numbers=atomic_numbers,
                positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8 + 0.1 * index]]),
                energies=canonical_energies[index, list(permutation)],
                energy_gradients=raw_gradients,
                derivative_matrix_elements=raw_derivatives,
                metadata={"energy_unit": "synthetic energy"},
            )
        )
    overlaps = gauges[:-1].mH @ gauges[1:]
    return (
        MolecularPath(
            samples,
            path_id=path_id,
            adjacent_overlaps=overlaps,
            metadata={"energy_unit": "synthetic energy"},
        ),
        canonical_derivatives,
    )


def _energy_only_path(path_id: str) -> MolecularPath:
    samples = []
    for index in range(2):
        samples.append(
            MolecularSample(
                atomic_numbers=torch.tensor([1, 1]),
                positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8 + index]]),
                energies=torch.tensor([0.1 * index, 1.0 - 0.1 * index]),
            )
        )
    return MolecularPath(samples, path_id=path_id)


def test_path_tracking_transforms_complete_matrix_targets_covariantly() -> None:
    path, canonical_derivatives = _tracked_target_path()

    result = path.tracked()

    expected_energies = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]])
    assert torch.allclose(result.tracked_energies, expected_energies)
    recovered_derivatives = torch.stack(
        tuple(
            sample.derivative_matrix_elements
            for sample in result.tracked_path  # type: ignore[arg-type]
        )
    )
    assert torch.linalg.matrix_norm(recovered_derivatives - canonical_derivatives).amax() < 1e-12
    for index, sample in enumerate(result.tracked_path):
        expected_gradients = torch.diagonal(
            canonical_derivatives[index], dim1=-2, dim2=-1
        ).real.permute(2, 0, 1)
        assert torch.allclose(sample.energy_gradients, expected_gradients)
        assert sample.metadata["generaldia_path_id"] == "crossing-A"
        assert sample.metadata["generaldia_path_index"] == index
    assert result.ambiguous_steps == ()


def test_path_level_split_is_deterministic_complete_and_leak_free() -> None:
    paths = MolecularPathDataset(_energy_only_path(f"path-{index}") for index in range(6))

    first = paths.split(seed=19)
    second = paths.split(seed=19)

    assert [partition.path_ids for partition in first] == [
        partition.path_ids for partition in second
    ]
    assert sum(len(partition) for partition in first) == len(paths)
    assert sum(partition.n_samples for partition in first) == paths.n_samples
    assert set(first[0].path_ids).isdisjoint(first[1].path_ids)
    assert set(first[0].path_ids).isdisjoint(first[2].path_ids)
    assert set(first[1].path_ids).isdisjoint(first[2].path_ids)

    flattened = first[0].as_dataset()
    assert {sample.metadata["generaldia_path_id"] for sample in flattened} == set(first[0].path_ids)


def test_tracked_character_energies_can_be_nonascending_across_a_crossing() -> None:
    samples = []
    frames = []
    for index, coordinate in enumerate((-1.0, 1.0)):
        hamiltonian = torch.diag(torch.tensor([coordinate, -coordinate]))
        energies, frame = torch.linalg.eigh(hamiltonian)
        samples.append(
            MolecularSample(
                atomic_numbers=torch.tensor([1, 1]),
                positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 + index]]),
                energies=energies,
            )
        )
        frames.append(frame)
    overlaps = torch.stack(frames[:-1]).mH @ torch.stack(frames[1:])
    result = MolecularPath(
        samples,
        path_id="exact-crossing",
        adjacent_overlaps=overlaps,
    ).tracked()

    assert torch.equal(result.tracked_energies, torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
    assert result.tracked_energies[1, 0] > result.tracked_energies[1, 1]


def test_degenerate_gradient_tracking_requires_full_matrix_targets() -> None:
    angle = 0.4
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    samples = tuple(
        MolecularSample(
            atomic_numbers=torch.tensor([1, 1]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 + index]]),
            energies=torch.zeros(2),
            energy_gradients=torch.zeros(2, 2, 3),
        )
        for index in range(2)
    )
    path = MolecularPath(samples, path_id="degenerate", adjacent_overlaps=rotation.unsqueeze(0))

    with pytest.raises(ValueError, match="requires full derivative_matrix_elements"):
        path.tracked(degeneracy_tolerance=1e-12)


def test_path_validation_rejects_inconsistent_content_and_identifiers() -> None:
    valid = _energy_only_path("valid")
    with pytest.raises(ValueError, match="at least two"):
        MolecularPath([valid[0]], path_id="short")
    with pytest.raises(ValueError, match="non-empty"):
        MolecularPath(tuple(valid), path_id=" ")
    with pytest.raises(ValueError, match="atom order"):
        MolecularPath(
            [
                valid[0],
                MolecularSample(torch.tensor([1, 2]), valid[1].positions, valid[1].energies),
            ],
            path_id="changed-atoms",
        )
    with pytest.raises(ValueError, match="adjacent_overlaps"):
        MolecularPath(tuple(valid), path_id="bad-overlaps", adjacent_overlaps=torch.eye(2))
    with pytest.raises(ValueError, match="contraction"):
        MolecularPath(
            tuple(valid),
            path_id="nonphysical-overlaps",
            adjacent_overlaps=(1.1 * torch.eye(2)).unsqueeze(0),
        )
    with pytest.raises(ValueError, match="unique"):
        MolecularPathDataset([valid, _energy_only_path("valid")])
    with pytest.raises(ValueError, match="requires adjacent_overlaps"):
        valid.tracked()


def test_tracking_rejects_inconsistent_gradient_and_matrix_targets() -> None:
    path, _ = _tracked_target_path()
    inconsistent_samples = list(path)
    sample = inconsistent_samples[1]
    inconsistent_samples[1] = MolecularSample(
        atomic_numbers=sample.atomic_numbers,
        positions=sample.positions,
        energies=sample.energies,
        energy_gradients=sample.energy_gradients + 0.1,  # type: ignore[operator]
        derivative_matrix_elements=sample.derivative_matrix_elements,
        metadata=sample.metadata,
    )
    inconsistent = MolecularPath(
        inconsistent_samples,
        path_id="inconsistent-targets",
        adjacent_overlaps=path.adjacent_overlaps,
    )

    with pytest.raises(ValueError, match="must match the diagonal"):
        inconsistent.tracked()


def test_report_data_and_html_preserve_numeric_evidence(tmp_path: Path) -> None:
    path, _ = _tracked_target_path("report-path")
    result = path.tracked()

    data = state_tracking_report_data(
        result,
        coordinates=[-1.0, 0.0, 1.0],
        coordinate_label="Reaction coordinate / a.u.",
    )

    assert data["schema_version"] == 1
    assert data["path_id"] == "report-path"
    assert data["tracked_energies"] == [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]
    assert data["transitions"][0]["permutation"] == [1, 0]
    assert len(data["transitions"][0]["absolute_aligned_overlap"]) == 2
    json.dumps(data, allow_nan=False)

    output = write_state_tracking_report(
        result,
        tmp_path / "diagnostics.html",
        coordinates=[-1.0, 0.0, 1.0],
        coordinate_label="Reaction coordinate / a.u.",
    )
    html = output.read_text(encoding="utf-8")
    assert "Connected-path state diagnostics" in html
    assert "generaldia.state_tracking_report" in html
    assert "https://" not in html
    assert "report-path" in html


def test_report_rejects_invalid_coordinate_metadata() -> None:
    path, _ = _tracked_target_path()
    result = path.tracked()
    with pytest.raises(ValueError, match="one value"):
        state_tracking_report_data(result, coordinates=[0.0])
    with pytest.raises(ValueError, match="finite"):
        state_tracking_report_data(result, coordinates=[0.0, float("nan"), 1.0])
    with pytest.raises(ValueError, match="non-empty"):
        state_tracking_report_data(result, coordinate_label="")
