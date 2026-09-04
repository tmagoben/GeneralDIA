"""Split complete paths, track state character, and write a visual HTML report."""

from pathlib import Path

import torch

from generaldia import (
    MolecularPath,
    MolecularPathDataset,
    MolecularSample,
    adjacent_state_overlaps,
    write_state_tracking_report,
)


def make_crossing_path(path_id: str, phase_offset: float) -> MolecularPath:
    """Return one exact two-state crossing with backend-like gauge changes."""

    coordinates = torch.tensor([-1.0, -0.35, 0.35, 1.0])
    phase_angles = torch.tensor(
        [
            [0.0, 0.0],
            [0.4 + phase_offset, -0.6],
            [-0.7, 0.2 + phase_offset],
            [0.9, -0.3],
        ]
    )
    raw_frames = []
    samples = []
    atomic_numbers = torch.tensor([1, 1])
    for coordinate, phases in zip(coordinates, phase_angles, strict=True):
        hamiltonian = torch.diag(torch.stack((coordinate, -coordinate))).to(torch.complex128)
        energies, frame = torch.linalg.eigh(hamiltonian)
        raw_frames.append(frame * torch.exp(1j * phases))
        samples.append(
            MolecularSample(
                atomic_numbers=atomic_numbers,
                positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5 + 0.1 * coordinate]]),
                energies=energies,
                metadata={"energy_unit": "synthetic energy unit"},
            )
        )
    return MolecularPath(
        samples,
        path_id=path_id,
        adjacent_overlaps=adjacent_state_overlaps(torch.stack(raw_frames)),
        metadata={"energy_unit": "synthetic energy unit"},
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    coordinates = [-1.0, -0.35, 0.35, 1.0]
    paths = MolecularPathDataset(
        make_crossing_path(f"crossing-{index}", 0.05 * index) for index in range(6)
    )
    training_paths, validation_paths, test_paths = paths.split(seed=23)

    assert set(training_paths.path_ids).isdisjoint(validation_paths.path_ids)
    assert set(training_paths.path_ids).isdisjoint(test_paths.path_ids)
    assert set(validation_paths.path_ids).isdisjoint(test_paths.path_ids)

    tracked = training_paths[0].tracked()
    output = write_state_tracking_report(
        tracked,
        Path("outputs") / "path_tracking_report.html",
        coordinates=coordinates,
        coordinate_label="Crossing coordinate / a.u.",
    )

    print("train paths", training_paths.path_ids)
    print("validation paths", validation_paths.path_ids)
    print("test paths", test_paths.path_ids)
    print("recorded ambiguous transitions", tracked.ambiguous_steps)
    print("visual report", output)


if __name__ == "__main__":
    main()
