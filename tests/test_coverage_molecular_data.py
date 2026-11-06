"""Validation of molecular data, path consistency and portable reference archives."""

from pathlib import Path

import numpy as np
import pytest
import torch

from generaldia.dataset import MolecularDataset, MolecularPath, MolecularSample
from generaldia.electronic_structure.data import ElectronicStructurePoint


def _sample(
    *,
    n_states=2,
    gradients=False,
    dipoles=False,
    derivatives=False,
    positions=None,
    numbers=None,
):
    atomic_numbers = [1, 1] if numbers is None else numbers
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]] if positions is None else positions
    n_atoms = len(atomic_numbers)
    kwargs = {}
    if gradients:
        kwargs["energy_gradients"] = torch.zeros(n_states, n_atoms, 3)
    if dipoles:
        kwargs["dipole_matrix_elements"] = torch.zeros(3, n_states, n_states)
    if derivatives:
        kwargs["derivative_matrix_elements"] = torch.zeros(n_atoms, 3, n_states, n_states)
    return MolecularSample(
        atomic_numbers=torch.tensor(atomic_numbers),
        positions=torch.tensor(positions),
        energies=torch.arange(n_states, dtype=torch.float64),
        **kwargs,
    )


@pytest.mark.parametrize(
    "numbers, positions, energies, message",
    [
        ([1], [[0, 0, 0]], [-1.0], "atomic_numbers"),
        ([1, 0], [[0, 0, 0], [0, 0, 1]], [-1.0], "positive"),
        ([1, 1], [[0, 0, 0]], [-1.0], "positions must"),
        ([1, 1], [[0, 0, 0], [0, 0, 1]], [], "energies must"),
        ([1, 1], [[0, 0, 0], [0, 0, 1]], [float("inf")], "finite"),
        ([1, 1], [[0, 0, 0], [0, float("nan"), 1]], [-1.0], "finite"),
    ],
)
def test_sample_fail_closed_on_invalid_geometry_or_spectra(numbers, positions, energies, message):
    with pytest.raises(ValueError, match=message):
        MolecularSample(torch.tensor(numbers), torch.tensor(positions), torch.tensor(energies))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"positions": torch.zeros(2, 3, dtype=torch.complex128)}, "must be real"),
        ({"energies": torch.tensor([1 + 0j, 2 + 0j])}, "must be real"),
        ({"energy_gradients": torch.zeros(3, 2, 3)}, "energy_gradients"),
        ({"energy_gradients": torch.full((2, 2, 3), float("nan"))}, "finite"),
        ({"derivative_matrix_elements": torch.zeros(2, 3, 1, 1)}, "derivative_matrix_elements"),
        ({"dipole_matrix_elements": torch.zeros(3, 1, 1)}, "dipole_matrix_elements"),
        ({"dipole_matrix_elements": torch.full((3, 2, 2), float("inf"))}, "finite"),
    ],
)
def test_sample_rejects_invalid_targets(kwargs, message):
    sample = _sample()
    values = {
        "atomic_numbers": sample.atomic_numbers,
        "positions": sample.positions,
        "energies": sample.energies,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        MolecularSample(**values)


def test_sample_rejects_nonhermitian_dipole_and_derivative_operators():
    bad_dipole = torch.zeros(3, 2, 2)
    bad_dipole[0, 0, 1] = 1
    with pytest.raises(ValueError, match="dipole matrix elements must be Hermitian"):
        MolecularSample([1, 1], torch.zeros(2, 3), [0.0, 1.0], dipole_matrix_elements=bad_dipole)

    bad_derivative = torch.zeros(2, 3, 2, 2, dtype=torch.complex128)
    bad_derivative[0, 0, 0, 1] = 0.3j
    with pytest.raises(ValueError, match="derivative matrix elements must be Hermitian"):
        MolecularSample(
            [1, 1], torch.zeros(2, 3), [0.0, 1.0], derivative_matrix_elements=bad_derivative
        )


def test_sample_device_dtype_copy_preserves_electronic_complex_phase():
    operators = torch.zeros(2, 3, 2, 2, dtype=torch.complex128)
    operators[0, 0, 0, 1] = 0.7j
    operators[0, 0, 1, 0] = -0.7j
    dipoles = torch.zeros(3, 2, 2, dtype=torch.complex128)
    dipoles[1, 0, 1] = 0.2 + 0.4j
    dipoles[1, 1, 0] = 0.2 - 0.4j
    original = MolecularSample(
        [1, 1],
        torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.float64),
        [0.0, 1.0],
        energy_gradients=torch.zeros(2, 2, 3),
        derivative_matrix_elements=operators,
        dipole_matrix_elements=dipoles,
        metadata={"source": "original"},
    )
    copy = original.to("cpu", torch.float32)
    assert copy.positions.dtype == torch.float32
    assert copy.energies.dtype == torch.float32
    assert copy.energy_gradients.dtype == torch.float32
    assert copy.derivative_matrix_elements.dtype == torch.complex128
    assert copy.dipole_matrix_elements.dtype == torch.complex128
    torch.testing.assert_close(copy.dipole_matrix_elements, dipoles)
    copy.metadata["source"] = "changed"
    assert original.metadata["source"] == "original"


def test_dataset_rejects_empty_wrong_item_types_and_state_count_inconsistency():
    with pytest.raises(ValueError, match="at least one"):
        MolecularDataset([])
    with pytest.raises(TypeError, match="MolecularSample"):
        MolecularDataset([123])
    with pytest.raises(ValueError, match="same number of states"):
        MolecularDataset([_sample(), _sample(n_states=3)])


def test_dataset_split_checks_degenerate_training_cases_and_reproducibility():
    sample = [_sample() for _ in range(7)]
    dataset = MolecularDataset(sample)
    with pytest.raises(ValueError, match="at least three"):
        MolecularDataset(sample[:2]).split()
    with pytest.raises(ValueError, match="three positive"):
        dataset.split((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="three positive"):
        dataset.split((1.0, 1.0))
    splits_a = dataset.split((98.0, 1.0, 1.0), seed=19)
    splits_b = dataset.split((98.0, 1.0, 1.0), seed=19)
    assert [len(v) for v in splits_a] == [5, 1, 1]
    assert tuple(tuple(id(s) for s in v) for v in splits_a) == tuple(
        tuple(id(s) for s in v) for v in splits_b
    )
    assert all(len(v) > 0 for v in splits_a)


def test_dataset_slicing_preserves_state_count_and_tags():
    dataset = MolecularDataset([_sample(), _sample(), _sample()])
    subset = dataset[1:]
    assert isinstance(subset, MolecularDataset)
    assert len(subset) == 2
    assert subset.n_states == 2
    assert list(iter(dataset)) == list(iter(dataset))
    assert dataset[0] is subset[0] or dataset[1] is subset[0]


@pytest.mark.parametrize(
    "points, message",
    [
        ([1], "at least two"),
        ([_sample(), 12], "MolecularSample"),
        ([_sample(), _sample(n_states=3)], "same number of states"),
        ([_sample(), _sample(numbers=[1, 8])], "atom order"),
        ([_sample(gradients=True), _sample()], "energy-gradient targets"),
        ([_sample(derivatives=True), _sample()], "derivative-matrix targets"),
        ([_sample(dipoles=True), _sample()], "dipole targets"),
    ],
)
def test_path_checks_group_consistency(points, message):
    with pytest.raises((TypeError, ValueError), match=message):
        MolecularPath(points, path_id="invalid")


@pytest.mark.parametrize(
    "overlaps, message",
    [
        (torch.eye(2)[None].expand(2, -1, -1), "adjacent_overlaps must"),
        (torch.full((1, 2, 2), float("nan")), "finite"),
        (torch.eye(2)[None] * 1.02, "contraction"),
    ],
)
def test_path_fails_closed_on_malformed_or_nonphysical_overlap_links(overlaps, message):
    with pytest.raises(ValueError, match=message):
        MolecularPath([_sample(), _sample()], path_id="bad-overlap", adjacent_overlaps=overlaps)


def test_path_requires_nonblank_name_and_explicit_overlaps_for_tracking():
    with pytest.raises(ValueError, match="path_id"):
        MolecularPath([_sample(), _sample()], path_id=" ")
    path = MolecularPath([_sample(), _sample()], path_id="path")
    assert path.n_atoms == 2
    assert path.n_states == 2
    assert path.energies.shape == (2, 2)
    assert path.adjacent_overlaps is None
    with pytest.raises(ValueError, match="requires adjacent_overlaps"):
        path.tracked()
    data = path.as_dataset()
    assert data[0].metadata["generaldia_path_id"] == "path"
    assert data[0].metadata["generaldia_path_index"] == 0
    assert data[1].metadata["generaldia_path_index"] == 1


def _point(**changes):
    arguments = dict(
        atomic_numbers=[1, 1],
        positions_angstrom=[[0, 0, 0], [0, 0, 0.75]],
        energies_hartree=[-1.0, -0.1],
    )
    arguments.update(changes)
    return ElectronicStructurePoint(**arguments)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"atomic_numbers": []}, "atomic_numbers"),
        ({"atomic_numbers": [1, 0]}, "positive"),
        ({"positions_angstrom": [[0, 0, 0]]}, "positions_angstrom"),
        ({"energies_hartree": []}, "energies_hartree"),
        ({"positions_angstrom": [[0, 0, 0], [0, 0, float("nan")]]}, "finite"),
        ({"energies_hartree": [-1.0, float("inf")]}, "finite"),
        ({"gradients_hartree_per_bohr": np.zeros((2, 3))}, "gradients"),
        (
            {"gradients_hartree_per_bohr": np.full((2, 2, 3), float("nan"))},
            "finite",
        ),
        ({"scaled_nac_pyscf": {(0,): np.zeros((2, 3))}}, "integer"),
        ({"scaled_nac_pyscf": {(0, "1"): np.zeros((2, 3))}}, "integer"),
        ({"scaled_nac_pyscf": {(0, 2): np.zeros((2, 3))}}, "outside"),
        ({"scaled_nac_pyscf": {(0, 1): np.zeros((3, 3))}}, "shape"),
        ({"scaled_nac_pyscf": {(0, 1): np.full((2, 3), float("inf"))}}, "finite"),
    ],
)
def test_electronic_structure_rejects_malformed_reference(changes, message):
    with pytest.raises(ValueError, match=message):
        _point(**changes)


def test_electronic_structure_portable_npz_no_gradients_no_nac(tmp_path: Path):
    original = _point(metadata={"arbitrary_python": object()})
    path = tmp_path / "reference.npz"
    original.save_npz(path)
    loaded = ElectronicStructurePoint.load_npz(path)
    assert loaded.gradients_hartree_per_bohr is None
    assert loaded.scaled_nac_pyscf == {}
    assert loaded.metadata == {}
    np.testing.assert_array_equal(loaded.atomic_numbers, original.atomic_numbers)
    np.testing.assert_allclose(loaded.energies_hartree, original.energies_hartree)
