from pathlib import Path

import numpy as np
import pytest

from generaldia.dataset import ANGSTROM_TO_BOHR, MolecularDataset
from generaldia.electronic_structure.data import ElectronicStructurePoint


def make_point() -> ElectronicStructurePoint:
    return ElectronicStructurePoint(
        atomic_numbers=np.array([1, 1]),
        positions_angstrom=np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]),
        energies_hartree=np.array([-1.0, -0.5]),
        gradients_hartree_per_bohr=np.ones((2, 2, 3)),
        scaled_nac_pyscf={(0, 1): np.zeros((2, 3))},
        metadata={"method": "test"},
    )


def test_electronic_structure_point_validates_and_reports_dimensions() -> None:
    point = make_point()
    assert point.n_atoms == 2
    assert point.n_states == 2


def test_pyscf_gradient_unit_conversion() -> None:
    dataset = MolecularDataset.from_electronic_structure([make_point()])
    assert torch_allclose_numpy(
        dataset[0].energy_gradients.numpy(),
        np.ones((2, 2, 3)) * ANGSTROM_TO_BOHR,
    )
    assert dataset[0].metadata["gradient_unit"] == "hartree/angstrom"


def torch_allclose_numpy(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right))


def test_electronic_structure_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="positions_angstrom"):
        ElectronicStructurePoint([1, 1], np.zeros((2, 2)), [-1.0])
    with pytest.raises(ValueError, match="gradients"):
        ElectronicStructurePoint([1, 1], np.zeros((2, 3)), [-1.0], np.zeros((2, 3)))


def test_electronic_structure_npz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    expected = make_point()
    expected.save_npz(path)
    actual = ElectronicStructurePoint.load_npz(path)
    assert np.array_equal(actual.atomic_numbers, expected.atomic_numbers)
    assert np.allclose(actual.positions_angstrom, expected.positions_angstrom)
    assert np.allclose(actual.energies_hartree, expected.energies_hartree)
    assert actual.gradients_hartree_per_bohr is not None
    assert np.allclose(actual.gradients_hartree_per_bohr, expected.gradients_hartree_per_bohr)
    assert np.allclose(actual.scaled_nac_pyscf[(0, 1)], expected.scaled_nac_pyscf[(0, 1)])


@pytest.mark.optional
def test_pyscf_rhf_smoke_if_installed() -> None:
    pytest.importorskip("pyscf")
    from generaldia.electronic_structure.pyscf_backend import RHFBackend

    point = RHFBackend().calculate(
        np.array([1, 1]), np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]])
    )
    assert point.metadata["converged"]
    assert point.gradients_hartree_per_bohr is not None
