"""Local model covariance, smooth derivatives and identifiable spectral fits."""

import itertools
import math

import pytest
import torch

from generaldia.h3plus_model import H3PlusEquivariantHamiltonian, fit_h3plus_spectra, h3plus_bonds
from generaldia.topology import closed_loop_overlaps, wilson_loop


def known_model():
    model = H3PlusEquivariantHamiltonian()
    with torch.no_grad():
        model.scalar_coefficients.copy_(
            torch.tensor(
                [
                    [-1.2, 0.03, 0.02, 0.001, 0.002, 0.0001],
                    [-0.4, 0.05, -0.01, 0.002, 0.001, 0.0002],
                ]
            )
        )
        model.splitting_coefficients.copy_(torch.tensor([0.06, 0.005, 0.001, 0.008]))
    return model


def sample_bonds():
    generator = torch.Generator().manual_seed(214)
    return 0.875 + 0.03 * torch.randn(40, 3, generator=generator)


def test_all_proton_permutations_and_spatial_reflections():
    model = known_model()
    positions = torch.tensor([[0.1, -0.2, 0.3], [0.9, -0.1, 0.2], [0.5, 0.7, 0.4]])
    h = model.from_positions(positions)
    pairs = ((0, 1), (0, 2), (1, 2))
    for perm in itertools.permutations(range(3)):
        induced = [pairs.index(tuple(sorted((perm[i], perm[j])))) for i, j in pairs]
        torch.testing.assert_close(
            model.from_positions(positions[list(perm)]), h[induced][:, induced], atol=1e-14, rtol=0
        )
    orthogonal, _ = torch.linalg.qr(
        torch.tensor([[0.3, 0.5, 0.7], [0.4, 0.8, 0.1], [0.9, 0.2, 0.6]])
    )
    for sign in (-1, 1):
        transformed = positions @ (sign * orthogonal) + torch.tensor([3.0, -2.0, 8.0])
        torch.testing.assert_close(model.from_positions(transformed), h, atol=1e-14, rtol=0)


def test_smooth_hamiltonian_derivatives_including_equilateral_center():
    model = known_model()
    for bonds in (torch.full((3,), 0.875), torch.tensor([0.82, 0.91, 0.88])):
        bonds.requires_grad_()
        assert torch.autograd.gradcheck(model, (bonds,), eps=1e-6, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(model(bonds), model(bonds).T, atol=1e-14, rtol=0)
    energies = torch.linalg.eigvalsh(model(torch.full((3,), 0.875)))
    assert float((energies[2] - energies[1]).detach()) == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_spectral_fit_recovers_known_surface_on_unseen_geometries(seed):
    truth = known_model()
    training = sample_bonds()
    energies = torch.linalg.eigvalsh(truth(training)).detach()
    fitted = fit_h3plus_spectra(training, energies, seed=seed)
    test = training[:11] + torch.tensor([0.004, -0.003, 0.002])
    torch.testing.assert_close(
        torch.linalg.eigvalsh(fitted.model(test)),
        torch.linalg.eigvalsh(truth(test)),
        atol=2e-6,
        rtol=0,
    )
    assert fitted.training_max_error_hartree < 2e-6
    assert fitted.training_mae_hartree < 2e-6
    assert fitted.seed == seed


def test_correct_and_isospectral_wrong_topology_use_identical_energy_data():
    theta = torch.arange(32) * 2 * math.pi / 32
    bonds = 0.875 + 0.06 * (
        torch.cos(theta)[:, None] * torch.tensor([2.0, -1.0, -1.0]) / math.sqrt(6)
        + torch.sin(theta)[:, None] * torch.tensor([0.0, 1.0, -1.0]) / math.sqrt(2)
    )
    h = known_model()(bonds).detach()
    energies, frames = torch.linalg.eigh(h)
    wrong_energies, wrong_frames = torch.linalg.eigh(torch.diag_embed(energies))
    torch.testing.assert_close(energies, wrong_energies, atol=0, rtol=0)
    for state in (1, 2):
        phase = wilson_loop(closed_loop_overlaps(frames[:, :, state : state + 1])).berry_phase
        wrong = wilson_loop(closed_loop_overlaps(wrong_frames[:, :, state : state + 1])).berry_phase
        assert abs(phase) == pytest.approx(math.pi, abs=1e-12)
        assert wrong == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "bonds",
    [
        torch.ones(2),
        torch.tensor([1, 1, 1]),
        torch.ones(3, dtype=torch.complex128),
        torch.tensor([1.0, 1.0, float("nan")]),
        torch.tensor([-1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 2.0]),
    ],
)
def test_invalid_bonds_are_rejected(bonds):
    with pytest.raises(ValueError):
        known_model()(bonds)


@pytest.mark.parametrize(
    "positions",
    [
        torch.ones(3),
        torch.ones(3, 3, dtype=torch.int64),
        torch.ones(3, 3, dtype=torch.complex128),
        torch.full((3, 3), float("inf")),
    ],
)
def test_invalid_positions_are_rejected(positions):
    with pytest.raises(ValueError):
        h3plus_bonds(positions)


@pytest.mark.parametrize("seed", [-1, True, 0.5])
def test_invalid_training_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        fit_h3plus_spectra(sample_bonds(), torch.zeros(40, 3), seed=seed)


@pytest.mark.parametrize("kind", ["shape", "few", "unordered", "nonfinite", "complex", "rank"])
def test_invalid_training_data(kind):
    bonds = sample_bonds()
    energies = torch.linalg.eigvalsh(known_model()(bonds)).detach()
    if kind == "shape":
        energies = energies[:, :2]
    elif kind == "few":
        bonds, energies = bonds[:5], energies[:5]
    elif kind == "unordered":
        energies = energies.flip(-1)
    elif kind == "nonfinite":
        energies[0, 0] = float("nan")
    elif kind == "complex":
        energies = energies.to(torch.complex128)
    elif kind == "rank":
        bonds[:] = 0.875
    with pytest.raises(ValueError):
        fit_h3plus_spectra(bonds, energies)
