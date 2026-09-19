"""Independent counterexamples found while auditing the first v3.2 slice."""

import pytest
import torch
from torch import nn

from generaldia import (
    LossWeights,
    MolecularDataset,
    MolecularSample,
    PathTrackingSettings,
    evaluate_model,
    hamiltonian_jacobian,
    observable_loss,
    track_states,
)


class DiagonalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, atomic_numbers, positions):
        x = positions[1, 0]
        return self.scale * torch.diag(torch.stack((x, x + 2)))


def sample(**kwargs):
    return MolecularSample(
        torch.tensor([1, 1]),
        torch.zeros(2, 3),
        kwargs.pop("energies", torch.tensor([0.0, 2.0])),
        **kwargs,
    )


@pytest.mark.parametrize(
    "name", ["assignment_margin_floor", "degeneracy_tolerance", "near_degeneracy_threshold"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_tracking_thresholds_fail_closed(name, value):
    with pytest.raises(ValueError, match="finite"):
        PathTrackingSettings(**{name: value})
    with pytest.raises(ValueError, match="finite"):
        track_states(
            torch.eye(2)[None], energies=torch.tensor([[0.0, 1.0], [0.0, 1.0]]), **{name: value}
        )


def test_complex_real_observable_is_rejected_without_truncation():
    with pytest.raises(ValueError, match="real"):
        sample(energies=torch.tensor([0.0 + 1.0j, 2.0 + 0.0j]))


def test_complex_jacobian_matches_explicit_pauli_y_and_backpropagates():
    scale = torch.tensor(0.7, requires_grad=True)
    pauli_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128)

    def model(z, r):
        return torch.diag(r.new_tensor([0.0, 2.0])).to(torch.complex128) + scale * r[1, 0] * pauli_y

    _, jacobian = hamiltonian_jacobian(model, torch.tensor([1, 1]), torch.zeros(2, 3))
    torch.testing.assert_close(jacobian[1, 0], scale * pauli_y)
    assert torch.count_nonzero(jacobian[0]) == 0
    jacobian.abs().square().sum().backward()
    torch.testing.assert_close(scale.grad, 4 * scale.detach())


def test_imaginary_target_error_is_preserved_for_real_model():
    target = torch.zeros(2, 3, 2, 2, dtype=torch.complex128)
    target[1, 0] = torch.tensor([[1.0, -1.0j], [1.0j, 1.0]])
    point = sample(derivative_matrix_elements=target)
    loss = observable_loss(DiagonalModel(), point, LossWeights(energy=0, derivative_matrix=1))
    torch.testing.assert_close(loss.total, torch.tensor(2 / 24))
    metrics = evaluate_model(DiagonalModel(), MolecularDataset([point]))
    assert metrics["derivative_matrix_mae"] == pytest.approx(2 / 24)


def test_ranked_loss_rejects_nonascending_targets():
    point = sample(energies=torch.tensor([2.0, 0.0]))
    with pytest.raises(ValueError, match="ascending"):
        observable_loss(DiagonalModel(), point)
    with pytest.raises(ValueError, match="ascending"):
        evaluate_model(DiagonalModel(), MolecularDataset([point]))


def test_loss_rejects_broadcastable_state_count_mismatch():
    with pytest.raises(ValueError, match="state count"):
        observable_loss(DiagonalModel(), sample(energies=torch.tensor([0.0])))
