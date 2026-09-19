"""Adversarial physics and independent gradient checks for path supervision."""

import math

import pytest
import torch
from torch import nn

from generaldia import (
    AmbiguousStateTrackingError,
    LossWeights,
    MolecularPath,
    MolecularSample,
    PathTrackingSettings,
    path_observable_loss,
    predict_path,
)


class LinearHamiltonian(nn.Module):
    def __init__(self, scale=1.0, coupling=0.4, complex_mode=False, unitary=None):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.coupling = nn.Parameter(torch.tensor(coupling))
        dtype = torch.complex128 if complex_mode else torch.float64
        self.register_buffer("z", torch.diag(torch.tensor([1.0, -1.0])).to(dtype))
        self.register_buffer(
            "x",
            torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
            if complex_mode
            else torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.register_buffer("unitary", torch.eye(2, dtype=dtype) if unitary is None else unitary)

    def forward(self, atoms, positions):
        h = self.scale * (positions[1, 0] * self.z + self.coupling * self.x)
        return self.unitary @ h @ self.unitary.mH

    def derivative(self):
        return self.unitary @ (self.scale * self.z) @ self.unitary.mH


class DoubletHamiltonian(nn.Module):
    """Two degenerate pairs, with an independently known off-path derivative."""

    def __init__(self, angle=0.2, strength=0.8, complex_mode=False):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(angle))
        self.strength = nn.Parameter(torch.tensor(strength))
        dtype = torch.complex128 if complex_mode else torch.float64
        generator = torch.zeros(4, 4, dtype=dtype)
        generator[0, 2] = 1j if complex_mode else 1
        generator[2, 0] = -generator[0, 2].conj()
        self.register_buffer("generator", generator)
        self.register_buffer("d", torch.diag(torch.tensor([-1.0, -1.0, 2.0, 2.0])).to(dtype))
        a = torch.tensor(
            [
                [0.3, 0.1, 0.6, 0.2],
                [0.1, -0.2, 0.2, 0.7],
                [0.6, 0.2, -0.4, 0.1],
                [0.2, 0.7, 0.1, 0.5],
            ]
        ).to(dtype)
        self.register_buffer("a", a)

    def forward(self, atoms, positions):
        u = torch.matrix_exp(self.angle * self.generator)
        return u @ self.d @ u.mH + positions[1, 0] * self.strength * self.a

    def derivative(self):
        return self.strength * self.a


def reference_path(model, *, scramble=False, doublet=False, path_id="reference", seed=9):
    generator = torch.Generator().manual_seed(seed)
    samples, frames = [], []
    xs = [0.0, 0.0, 0.0] if doublet else [-0.2, 0.0, 0.2]
    for index, x in enumerate(xs):
        r = torch.tensor([[0.0, 0.0, 0.0], [x, 0.0, 1.0 + 0.1 * index]])
        h = model(torch.tensor([1, 1]), r).detach()
        energies, frame = torch.linalg.eigh(h)
        n = len(energies)
        gauge = torch.eye(n, dtype=torch.complex128)
        if scramble:
            if doublet:
                for block in ([0, 1], [2, 3]):
                    a = torch.randn(2, 2, generator=generator, dtype=torch.complex128)
                    rotation, _ = torch.linalg.qr(a)
                    gauge[torch.tensor(block)[:, None], torch.tensor(block)[None, :]] = rotation
            phases = torch.exp(1j * torch.randn(n, generator=generator))
            order = torch.randperm(n, generator=generator)
            gauge = (gauge * phases)[..., order]
        frame = frame.to(torch.complex128) @ gauge
        raw_energies = torch.diagonal(gauge.mH @ torch.diag(energies).to(gauge.dtype) @ gauge).real
        derivative = torch.zeros(2, 3, n, n, dtype=torch.complex128)
        derivative[1, 0] = frame.mH @ model.derivative().detach().to(frame.dtype) @ frame
        samples.append(
            MolecularSample(
                torch.tensor([1, 1]),
                r,
                raw_energies,
                energy_gradients=derivative.diagonal(dim1=-2, dim2=-1).real.permute(2, 0, 1),
                derivative_matrix_elements=derivative,
            )
        )
        frames.append(frame)
    frames = torch.stack(frames)
    return MolecularPath(samples, path_id=path_id, adjacent_overlaps=frames[:-1].mH @ frames[1:])


@pytest.mark.parametrize("complex_mode", [False, True])
@pytest.mark.parametrize("doublet", [False, True])
def test_loss_invariant_to_permutation_phase_and_degenerate_block_gauge(complex_mode, doublet):
    cls = DoubletHamiltonian if doublet else LinearHamiltonian
    reference = cls(complex_mode=complex_mode)
    model = cls(complex_mode=complex_mode)
    with torch.no_grad():
        (model.strength if doublet else model.scale).add_(0.2)
    weights = LossWeights(1, 0.2, 0.7)
    plain = reference_path(reference, doublet=doublet)
    baseline = path_observable_loss(model, plain, weights)
    baseline.total.backward()
    baseline_grad = [p.grad.clone() for p in model.parameters()]
    assert baseline.total > 1e-5
    for seed in [1, 17, 36]:
        scrambled = reference_path(reference, doublet=doublet, scramble=True, seed=seed)
        model.zero_grad()
        result = path_observable_loss(model, scrambled, weights)
        torch.testing.assert_close(result.total, baseline.total, atol=1e-12, rtol=1e-10)
        result.total.backward()
        for expected, parameter in zip(baseline_grad, model.parameters(), strict=True):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("complex_mode", [False, True])
@pytest.mark.parametrize("doublet", [False, True])
def test_parameter_gradients_match_central_finite_differences(complex_mode, doublet):
    cls = DoubletHamiltonian if doublet else LinearHamiltonian
    target = reference_path(cls(complex_mode=complex_mode), doublet=doublet, scramble=True)
    model = cls(complex_mode=complex_mode)
    with torch.no_grad():
        (model.angle if doublet else model.coupling).add_(0.11)
    weights = LossWeights(1, 0.3, 0.7)
    path_observable_loss(model, target, weights).total.backward()
    step = 1e-6
    for parameter in model.parameters():
        analytical = parameter.grad.clone()
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(original + step)
        plus = path_observable_loss(model, target, weights).total.detach()
        with torch.no_grad():
            parameter.copy_(original - step)
        minus = path_observable_loss(model, target, weights).total.detach()
        with torch.no_grad():
            parameter.copy_(original)
        torch.testing.assert_close(analytical, (plus - minus) / (2 * step), atol=2e-8, rtol=2e-5)


def test_constant_latent_basis_rotation_does_not_change_loss_or_gradients():
    a = torch.tensor([[1.0, 1j], [1j, 1.0]], dtype=torch.complex128) / math.sqrt(2)
    reference = LinearHamiltonian(complex_mode=True)
    path = reference_path(reference, scramble=True)
    model = LinearHamiltonian(scale=1.2, complex_mode=True)
    rotated = LinearHamiltonian(scale=1.2, complex_mode=True, unitary=a)
    weights = LossWeights(1, 0.2, 0.5)
    left, right = [path_observable_loss(m, path, weights) for m in (model, rotated)]
    torch.testing.assert_close(left.total, right.total, atol=1e-12, rtol=1e-10)
    left.total.backward()
    right.total.backward()
    for p, q in zip(model.parameters(), rotated.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_identical_spectra_with_wrong_state_continuation_are_penalized():
    class Crossing(nn.Module):
        def __init__(self, wrong=False):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.wrong = wrong

        def forward(self, atoms, r):
            x = r[1, 0].abs() if self.wrong else r[1, 0]
            return self.scale * torch.diag(torch.stack((x, -x)))

    model = Crossing()
    samples, frames = [], []
    for x in [-1.0, 1.0]:
        r = torch.tensor([[0.0, 0.0, 0.0], [x, 0.0, 0.0]])
        e, u = torch.linalg.eigh(model(None, r).detach())
        samples.append(MolecularSample(torch.tensor([1, 1]), r, e))
        frames.append(u)
    path = MolecularPath(
        samples, path_id="crossing", adjacent_overlaps=(frames[0].mH @ frames[1])[None]
    )
    assert path_observable_loss(model, path).total < 1e-25
    torch.testing.assert_close(
        path_observable_loss(Crossing(wrong=True), path).total, torch.tensor(2.0)
    )


def test_ambiguity_partition_and_missing_target_guards():
    model = LinearHamiltonian()
    path = reference_path(model)
    tied = torch.ones(1, 2, 2, dtype=torch.complex128) / 2
    ambiguous = MolecularPath(list(path)[:2], path_id="ambiguous", adjacent_overlaps=tied)
    with pytest.raises(AmbiguousStateTrackingError):
        path_observable_loss(model, ambiguous)
    recorded = ambiguous.tracked(on_ambiguous="record")
    with pytest.raises(ValueError, match="ambiguous"):
        path_observable_loss(model, recorded)
    with pytest.raises(ValueError, match="requires on_ambiguous"):
        predict_path(model, path, settings=PathTrackingSettings(on_ambiguous="record"))
    with pytest.raises(ValueError, match="settings must match"):
        path_observable_loss(
            model, path.tracked(), settings=PathTrackingSettings(overlap_floor=0.4)
        )
    missing = MolecularPath(
        [MolecularSample(s.atomic_numbers, s.positions, s.energies) for s in path],
        path_id="missing",
        adjacent_overlaps=path.adjacent_overlaps,
    )
    with pytest.raises(ValueError, match="matrix targets"):
        path_observable_loss(model, missing, LossWeights(derivative_matrix=1))
    with pytest.raises(ValueError, match="gradient or full matrix"):
        path_observable_loss(model, missing, LossWeights(energy_gradient=1))
    with pytest.raises(ValueError, match="state count"):
        predict_path(DoubletHamiltonian(), path)


def test_gradient_only_and_pretracked_targets_are_supported():
    model = LinearHamiltonian(scale=1.2)
    path = reference_path(LinearHamiltonian(), scramble=True)
    gradients = MolecularPath(
        [
            MolecularSample(
                s.atomic_numbers, s.positions, s.energies, energy_gradients=s.energy_gradients
            )
            for s in path
        ],
        path_id="gradients",
        adjacent_overlaps=path.adjacent_overlaps,
    )
    weights = LossWeights(1, 0.2)
    expected = path_observable_loss(model, path, weights)
    actual = path_observable_loss(model, gradients.tracked(), weights)
    torch.testing.assert_close(actual.total, expected.total)


def test_degeneracy_partition_mismatch_is_rejected():
    model = LinearHamiltonian(scale=0.0)
    path = reference_path(LinearHamiltonian())
    with pytest.raises(ValueError, match="degeneracy partitions differ"):
        path_observable_loss(model, path)


def test_exact_degenerate_match_has_finite_zero_loss_gradients():
    model = DoubletHamiltonian(complex_mode=True)
    target = reference_path(DoubletHamiltonian(complex_mode=True), doublet=True, scramble=True)
    result = path_observable_loss(model, target, LossWeights(1.0, 0.2, 0.7))
    assert result.total < 1e-25
    result.total.backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs() < 1e-12
