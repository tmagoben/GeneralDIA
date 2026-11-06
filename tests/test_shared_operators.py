"""Independent covariance, identifiability and differentiation controls for v3.3."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from generaldia import (
    LossWeights,
    MolecularOperators,
    MolecularPath,
    MolecularPathDataset,
    MolecularSample,
    SharedMolecularOperators,
    SimpleMolecularHamiltonian,
    TrainingConfig,
    load_checkpoint,
    path_observable_loss,
    predict_path,
    save_checkpoint,
    train_path_model,
)
from generaldia.observables import hamiltonian_jacobian
from generaldia.path_losses import _dipole_descriptors, _joint_descriptors


@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize("n_states", [2, 4])
def test_shared_model_hermiticity_rotations_reflections_translations_and_atom_order(mode, n_states):
    torch.manual_seed(8)
    model = SharedMolecularOperators(n_states=n_states, hidden=8, n_rbf=5, mode=mode)
    z = torch.tensor([8, 1, 1])
    r = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.1, 0.2], [-0.3, 0.8, -0.1]])
    original = model.operators(z, r)
    torch.testing.assert_close(original.hamiltonian, model(z, r))
    assert original.hamiltonian.shape == (n_states, n_states)
    assert original.dipoles.shape == (3, n_states, n_states)
    assert torch.equal(original.hamiltonian, original.hamiltonian.mH)
    assert torch.equal(original.dipoles, original.dipoles.mH)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    for rotation in (q, -q):
        moved = model.operators(z, r @ rotation.T + torch.tensor([2.0, -3.0, 4.0]))
        expected = torch.einsum(
            "ab,bij->aij", rotation.to(original.dipoles.dtype), original.dipoles
        )
        torch.testing.assert_close(moved.hamiltonian, original.hamiltonian, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(moved.dipoles, expected, atol=1e-12, rtol=1e-10)
    permuted = model.operators(z[[2, 0, 1]], r[[2, 0, 1]])
    torch.testing.assert_close(permuted.hamiltonian, original.hamiltonian)
    torch.testing.assert_close(permuted.dipoles, original.dipoles)
    if mode == "complex":
        assert original.hamiltonian.imag.abs().max() > 1e-5
        assert original.dipoles.imag.abs().max() > 1e-5


@pytest.mark.parametrize("mode", ["real", "complex"])
def test_dipole_coordinate_derivative_matches_finite_difference(mode):
    torch.manual_seed(7)
    model = SharedMolecularOperators(hidden=5, n_rbf=4, mode=mode)
    z = torch.tensor([3, 1])
    r = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.3, 1.5]], requires_grad=True)
    value = model.operators(z, r).dipoles.abs().square().sum()
    gradient = torch.autograd.grad(value, r)[0]
    step = 1e-6
    for atom in range(2):
        for axis in range(3):
            plus, minus = r.detach().clone(), r.detach().clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite = (
                model.operators(z, plus).dipoles.abs().square().sum()
                - model.operators(z, minus).dipoles.abs().square().sum()
            ) / (2 * step)
            torch.testing.assert_close(gradient[atom, axis], finite, atol=1e-9, rtol=1e-5)


class ToyOperators(nn.Module):
    """Exactly degenerate four-state model with noncommuting dipole operators."""

    def __init__(self, angle=0.2, amplitude=0.8):
        super().__init__()
        self.angle = nn.Parameter(torch.tensor(angle))
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        generator = torch.zeros(4, 4, dtype=torch.complex128)
        generator[0, 2] = generator[2, 0] = 1j
        self.register_buffer("generator", generator)
        self.register_buffer(
            "diagonal", torch.diag(torch.tensor([-1.0, -1.0, 2.0, 2.0])).to(generator.dtype)
        )
        g = torch.Generator().manual_seed(32)
        a = torch.randn(4, 4, 4, generator=g, dtype=torch.complex128)
        self.register_buffer("a", (a + a.mH) / 2)

    def forward(self, z, r):
        u = torch.matrix_exp(self.angle * self.generator)
        return u @ self.diagonal @ u.mH + r[0, 0] * self.a[0]

    def operators(self, z, r):
        return MolecularOperators(self(z, r), self.amplitude * self.a[1:])


def make_operator_path(model, *, scramble=False, derivatives=True):
    g = torch.Generator().manual_seed(5)
    samples, frames = [], []
    for k in range(3):
        z = torch.tensor([3, 1])
        r = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 1.3 + 0.05 * k]])
        h, jac = hamiltonian_jacobian(model, z, r, create_graph=False)
        e, u = torch.linalg.eigh(h.detach())
        gauge = torch.eye(len(e), dtype=torch.complex128)
        if scramble:
            if len(e) == 4:
                for block in ([0, 1], [2, 3]):
                    q, _ = torch.linalg.qr(torch.randn(2, 2, generator=g, dtype=torch.complex128))
                    gauge[torch.tensor(block)[:, None], torch.tensor(block)[None, :]] = q
            phases = torch.exp(1j * torch.randn(len(e), generator=g))
            gauge = (gauge * phases)[:, torch.randperm(len(e), generator=g)]
        u = u.to(gauge.dtype) @ gauge
        raw_e = torch.diagonal(gauge.mH @ torch.diag(e).to(gauge.dtype) @ gauge).real
        mu = model.operators(z, r).dipoles.detach().to(gauge.dtype)
        samples.append(
            MolecularSample(
                z,
                r,
                raw_e,
                derivative_matrix_elements=(u.mH @ jac.detach().to(gauge.dtype) @ u)
                if derivatives
                else None,
                dipole_matrix_elements=u.mH @ mu @ u,
                metadata={"charge": 0, "energy_unit": "synthetic", "dipole_unit": "synthetic"},
            )
        )
        frames.append(u)
    frames = torch.stack(frames)
    return MolecularPath(
        samples, path_id="operator-path", adjacent_overlaps=frames[:-1].mH @ frames[1:]
    )


def test_degenerate_joint_loss_and_gradients_are_gauge_invariant():
    model, reference = ToyOperators(angle=0.31, amplitude=0.95), ToyOperators()
    weights = LossWeights(
        energy=1.0, dipole=0.7, joint_operator=0.3, dipole_scale=2.0, derivative_scale=0.5
    )
    values, gradients = [], []
    for scramble in (False, True):
        model.zero_grad()
        loss = path_observable_loss(
            model, make_operator_path(reference, scramble=scramble), weights
        )
        loss.total.backward()
        values.append(loss.total.detach())
        gradients.append(torch.stack([p.grad for p in model.parameters()]))
        assert torch.isfinite(gradients[-1]).all()
    torch.testing.assert_close(values[0], values[1], atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-10, rtol=1e-8)
    assert values[0] > 1e-5
    target = make_operator_path(reference, scramble=True)
    step = 1e-6
    for parameter, expected in zip(model.parameters(), gradients[0], strict=True):
        saved = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(saved + step)
        plus = path_observable_loss(model, target, weights).total.detach()
        with torch.no_grad():
            parameter.copy_(saved - step)
        minus = path_observable_loss(model, target, weights).total.detach()
        with torch.no_grad():
            parameter.copy_(saved)
        torch.testing.assert_close(expected, (plus - minus) / (2 * step), atol=2e-8, rtol=2e-5)


@pytest.mark.parametrize("mode", ["real", "complex"])
def test_joint_loss_spatial_rotation_and_shared_encoder_gradients(mode):
    torch.manual_seed(6)
    reference = SharedMolecularOperators(hidden=5, n_rbf=4, mode=mode)
    model = deepcopy(reference)
    with torch.no_grad():
        model.dipole_head[-1].weight.mul_(1.2)
    target = make_operator_path(reference, scramble=True)
    weights = LossWeights(energy=1.0, dipole=1.0, joint_operator=0.2)
    original = path_observable_loss(model, target, weights)
    original.total.backward()
    assert model.pair_net[0].weight.grad.abs().max() > 1e-10
    assert torch.isfinite(model.head[-1].weight.grad).all()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    samples = []
    for sample in target:
        rotation = q.to(sample.dipole_matrix_elements.dtype)
        samples.append(
            MolecularSample(
                sample.atomic_numbers,
                sample.positions @ q.T,
                sample.energies,
                derivative_matrix_elements=torch.einsum(
                    "ab,nbij->naij", rotation, sample.derivative_matrix_elements
                ),
                dipole_matrix_elements=torch.einsum(
                    "ab,bij->aij", rotation, sample.dipole_matrix_elements
                ),
            )
        )
    rotated = MolecularPath(samples, path_id="rotated", adjacent_overlaps=target.adjacent_overlaps)
    actual = path_observable_loss(model, rotated, weights)
    torch.testing.assert_close(actual.total, original.total, atol=1e-12, rtol=1e-9)


def test_joint_descriptors_detect_inconsistent_relative_operator_phases():
    groups = ((0,), (1,))
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    mu = torch.stack((x, x, x * 0))
    flipped_component = mu.clone()
    flipped_component[1] *= -1
    # Each separate component has identical spectra/magnitudes, but relative components differ.
    torch.testing.assert_close(torch.linalg.eigvalsh(mu), torch.linalg.eigvalsh(flipped_component))
    assert not torch.allclose(
        _dipole_descriptors(mu, groups), _dipole_descriptors(flipped_component, groups)
    )
    # A common dipole-only sign change preserves its Gram tensors; the shared derivative fixes it.
    torch.testing.assert_close(_dipole_descriptors(mu, groups), _dipole_descriptors(-mu, groups))
    numerator = torch.stack((x, x * 0, x * 0))[None]
    assert not torch.allclose(
        _joint_descriptors(numerator, mu, groups), _joint_descriptors(numerator, -mu, groups)
    )


def test_dipole_targets_survive_copy_device_and_tracking_and_bad_inputs_fail():
    target = make_operator_path(ToyOperators(), scramble=True)
    tracked = target.tracked()
    for raw, transformed, gauge in zip(
        target, tracked.tracked_path, tracked.tracking.transformations, strict=True
    ):
        torch.testing.assert_close(
            transformed.dipole_matrix_elements, gauge.mH @ raw.dipole_matrix_elements @ gauge
        )
    torch.testing.assert_close(
        target.as_dataset()[0].dipole_matrix_elements, target[0].dipole_matrix_elements
    )
    torch.testing.assert_close(
        target[0].to("cpu", torch.float64).dipole_matrix_elements, target[0].dipole_matrix_elements
    )
    point = target[0]
    for invalid in (torch.zeros(2, 4, 4), torch.full((3, 4, 4), float("nan"))):
        with pytest.raises(ValueError):
            MolecularSample(
                point.atomic_numbers,
                point.positions,
                point.energies,
                dipole_matrix_elements=invalid,
            )
    with pytest.raises(ValueError, match="neutral"):
        SharedMolecularOperators(charge=1)
    with pytest.raises(ValueError, match="mode"):
        SharedMolecularOperators(mode="unsupported")
    with pytest.raises(ValueError, match="scales"):
        LossWeights(dipole_scale=0)
    with pytest.raises(ValueError, match=r"model\.operators"):
        predict_path(SimpleMolecularHamiltonian(n_states=4), target, dipoles=True)
    with pytest.raises(ValueError, match="full derivative"):
        path_observable_loss(
            ToyOperators(),
            make_operator_path(ToyOperators(), derivatives=False),
            LossWeights(joint_operator=1),
        )


@pytest.mark.parametrize("mode", ["real", "complex"])
def test_shared_training_and_checkpoint_preserve_both_operators(mode, tmp_path):
    torch.manual_seed(8)
    reference = SharedMolecularOperators(hidden=5, n_rbf=4, mode=mode)
    model = deepcopy(reference)
    with torch.no_grad():
        model.dipole_head[-1].weight.mul_(1.4)
    path = make_operator_path(reference, scramble=True)
    data = MolecularPathDataset([path])
    weights = LossWeights(energy=1, dipole=1, joint_operator=0.2)
    config = TrainingConfig(epochs=15, learning_rate=0.002, seed=8)
    initial = path_observable_loss(model, path, weights).total.detach()
    history = train_path_model(model, data, weights=weights, config=config)
    final = path_observable_loss(model, path, weights).total.detach()
    assert final < initial
    file = tmp_path / "operators.pt"
    save_checkpoint(file, model, config=config, weights=weights, history=history)
    restored = SharedMolecularOperators(**model.configuration)
    metadata = load_checkpoint(file, restored)
    assert metadata["model_configuration"] == model.configuration
    assert metadata["loss_weights"]["joint_operator"] == 0.2
    before = predict_path(model, path, derivatives=True, dipoles=True)
    after = predict_path(restored, path, derivatives=True, dipoles=True)
    for name in ("energies", "derivative_matrices", "dipole_matrices"):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)


def test_shared_prediction_rejects_charge_mismatch_and_missing_dipoles():
    model = SharedMolecularOperators(hidden=5, n_rbf=4)
    path = make_operator_path(model)
    path.metadata["charge"] = 1
    with pytest.raises(ValueError, match="charge"):
        predict_path(model, path, dipoles=True)
    path.metadata["charge"] = 0
    for sample in path:
        sample.dipole_matrix_elements = None
    with pytest.raises(ValueError, match="dipole_matrix_elements"):
        path_observable_loss(model, path, LossWeights(dipole=1))


def test_constant_latent_basis_change_preserves_joint_loss_and_parameter_gradient():
    class RotatedModel(nn.Module):
        def __init__(self, original, unitary):
            super().__init__()
            self.original = original
            self.register_buffer("unitary", unitary)

        def forward(self, z, r):
            return self.unitary.mH @ self.original(z, r) @ self.unitary

        def operators(self, z, r):
            return MolecularOperators(
                self(z, r), self.unitary.mH @ self.original.operators(z, r).dipoles @ self.unitary
            )

    torch.manual_seed(18)
    model = ToyOperators(angle=0.31, amplitude=0.95)
    q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.complex128))
    rotated = RotatedModel(deepcopy(model), q)
    target = make_operator_path(ToyOperators(), scramble=True)
    weights = LossWeights(dipole=1, joint_operator=1)
    first, second = [path_observable_loss(m, target, weights).total for m in (model, rotated)]
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-10)
    first.backward()
    second.backward()
    for p, r in zip(model.parameters(), rotated.parameters(), strict=True):
        torch.testing.assert_close(p.grad, r.grad, atol=1e-10, rtol=1e-8)
