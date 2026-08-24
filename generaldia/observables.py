"""Observable calculations derived from finite-state Hamiltonians."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

HamiltonianModel = nn.Module | Callable[[Tensor, Tensor], Tensor]


def _validate_hamiltonian(matrix: Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("the model must return one square matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("the Hamiltonian must contain finite values")
    if not torch.allclose(matrix, matrix.mH, atol=1e-10, rtol=1e-8):
        raise ValueError("the Hamiltonian must be Hermitian")


def adiabatic_energies(hamiltonian: Tensor) -> Tensor:
    """Return ascending eigenvalues of one Hermitian matrix."""

    hamiltonian = torch.as_tensor(hamiltonian)
    _validate_hamiltonian(hamiltonian)
    return torch.linalg.eigvalsh(hamiltonian)


def _coordinate_tensor(positions: Tensor) -> Tensor:
    positions = torch.as_tensor(positions)
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not positions.is_floating_point():
        positions = positions.to(torch.get_default_dtype())
    return positions.detach().clone().requires_grad_(True)


def energy_gradients(
    model: HamiltonianModel,
    atomic_numbers: Tensor,
    positions: Tensor,
    *,
    create_graph: bool = True,
) -> tuple[Tensor, Tensor]:
    """Calculate energies and Cartesian energy gradients.

    Returns energies with shape ``(S,)`` and gradients with shape ``(S, N, 3)``.
    Gradients use the energy and coordinate units supplied to ``model``.
    """

    coordinates = _coordinate_tensor(positions)
    hamiltonian = model(atomic_numbers, coordinates)
    _validate_hamiltonian(hamiltonian)
    energies = torch.linalg.eigvalsh(hamiltonian)
    gradients = []
    for state in range(energies.shape[0]):
        (gradient,) = torch.autograd.grad(
            energies[state], coordinates, retain_graph=True, create_graph=create_graph
        )
        gradients.append(gradient)
    return energies, torch.stack(gradients)


def hamiltonian_jacobian(
    model: HamiltonianModel,
    atomic_numbers: Tensor,
    positions: Tensor,
    *,
    create_graph: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return ``H`` and ``dH/dR`` with shapes ``(S,S)`` and ``(N,3,S,S)``."""

    coordinates = _coordinate_tensor(positions)

    def evaluate(value: Tensor) -> Tensor:
        result = model(atomic_numbers, value)
        _validate_hamiltonian(result)
        return result

    hamiltonian = evaluate(coordinates)
    jacobian = torch.autograd.functional.jacobian(
        evaluate, coordinates, create_graph=create_graph, vectorize=True
    )
    return hamiltonian, jacobian.permute(2, 3, 0, 1)


def derivative_matrix_elements(
    model: HamiltonianModel,
    atomic_numbers: Tensor,
    positions: Tensor,
    *,
    create_graph: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate energies, column eigenvectors, and ``<phi_i|dH/dR|phi_j>``."""

    hamiltonian, derivative = hamiltonian_jacobian(
        model, atomic_numbers, positions, create_graph=create_graph
    )
    energies, eigenvectors = torch.linalg.eigh(hamiltonian)
    numerators = torch.einsum("mi,abmn,nj->abij", eigenvectors.conj(), derivative, eigenvectors)
    return energies, eigenvectors, numerators


def coupling_validity_mask(energies: Tensor, gap_floor: float = 1e-6) -> Tensor:
    """Return a state-pair mask for gaps whose magnitude meets ``gap_floor``."""

    energies = torch.as_tensor(energies)
    if energies.ndim != 1:
        raise ValueError("energies must have shape (S,)")
    if gap_floor <= 0:
        raise ValueError("gap_floor must be positive")
    gaps = energies[None, :] - energies[:, None]
    mask = torch.abs(gaps) >= gap_floor
    mask.fill_diagonal_(False)
    return mask


def derivative_couplings_from_numerators(
    energies: Tensor,
    numerators: Tensor,
    gap_floor: float = 1e-6,
    *,
    return_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Divide derivative elements by energy gaps and mark unreliable state pairs.

    State pairs below ``gap_floor`` receive zero. ``return_mask=True`` returns the
    ``(S, S)`` validity mask so callers can distinguish suppression from physical zero.
    """

    energies = torch.as_tensor(energies)
    numerators = torch.as_tensor(numerators)
    expected = (energies.numel(), energies.numel())
    if numerators.ndim < 2 or numerators.shape[-2:] != expected:
        raise ValueError("numerators must end with state dimensions (S, S)")
    mask = coupling_validity_mask(energies, gap_floor)
    gaps = energies[None, :] - energies[:, None]
    safe_gaps = torch.where(mask, gaps, torch.ones_like(gaps))
    couplings = torch.where(mask, numerators / safe_gaps, torch.zeros_like(numerators))
    if return_mask:
        return couplings, mask
    return couplings
