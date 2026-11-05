"""Independent finite-difference and reference checks of coupling numerators.

This module operates in a fixed, declared finite-state basis. It never divides
by an energy gap, nor does numerical consistency imply physical correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from .observables import HamiltonianModel, coupling_validity_mask, hamiltonian_jacobian


@dataclass(frozen=True)
class NumeratorCertification:
    """Diagnostics for finite-state Hamiltonian derivatives at one geometry.

    All derivative matrices and numerators have shape (atoms, 3, states, states).
    The valid-gap mask distinguishes nondegenerate pairs from unresolved pairs.
    Passing establishes internal derivative consistency, plus any independently
    supplied reference, and does not certify a molecular Hamiltonian.
    """

    energies: Tensor
    numerators_autograd: Tensor
    numerators_finite_difference: Tensor
    valid_gap_pairs: Tensor
    unresolved_pairs: tuple[tuple[int, int], ...]
    minimum_energy_gap: float
    finite_difference_max_error: float
    finite_difference_relative_error: float
    reference_max_error: float | None
    reference_checked: bool
    passed: bool


def _validate_controls(step: float, gap_floor: float, atol: float, rtol: float) -> None:
    if not isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite")
    if not isfinite(gap_floor) or gap_floor <= 0:
        raise ValueError("gap_floor must be positive and finite")
    if not isfinite(atol) or atol < 0 or not isfinite(rtol) or rtol < 0:
        raise ValueError("atol and rtol must be finite and nonnegative")


def _checked_matrix(matrix: Tensor) -> Tensor:
    matrix = torch.as_tensor(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("model must return a nonempty square Hamiltonian")
    if not torch.isfinite(matrix).all():
        raise ValueError("Hamiltonian stencil contains nonfinite values")
    if not torch.allclose(matrix, matrix.mH, atol=1e-10, rtol=1e-8):
        raise ValueError("Hamiltonian stencil must be Hermitian")
    if matrix.dtype not in (torch.float64, torch.complex128):
        raise ValueError("certification requires float64 or complex128 Hamiltonians")
    return matrix


def _evaluate_at(model: HamiltonianModel, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
    return _checked_matrix(model(atomic_numbers, positions))


def _five_point_derivative(
    model: HamiltonianModel,
    atomic_numbers: Tensor,
    coordinates: Tensor,
    step: float,
    expected_shape: tuple[int, int],
) -> Tensor:
    result = []
    for atom in range(coordinates.shape[0]):
        row = []
        for axis in range(3):
            stencil = []
            for multiplier in (-2, -1, 1, 2):
                displaced = coordinates.clone()
                displaced[atom, axis] += multiplier * step
                matrix = _evaluate_at(model, atomic_numbers, displaced)
                if matrix.shape != expected_shape:
                    raise ValueError(
                        "Hamiltonian dimension changes within finite-difference stencil"
                    )
                stencil.append(matrix)
            minus_two, minus_one, plus_one, plus_two = stencil
            row.append((minus_two - 8 * minus_one + 8 * plus_one - plus_two) / (12 * step))
        result.append(torch.stack(row))
    return torch.stack(result)


def _max_error(candidate: Tensor, target: Tensor) -> tuple[float, float]:
    difference = (candidate.detach() - target.detach()).abs()
    max_error = float(difference.max())
    reference_scale = float(target.detach().abs().max())
    return max_error, max_error / max(reference_scale, 1e-15)


def certify_coupling_numerators(
    model: HamiltonianModel,
    atomic_numbers: Tensor,
    positions: Tensor,
    *,
    step: float = 1e-3,
    gap_floor: float = 1e-6,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    reference_derivative: Tensor | None = None,
) -> NumeratorCertification:
    """Compare autograd numerators against a five-point Hamiltonian derivative.

    A supplied independent reference_derivative must have shape
    (atoms, 3, states, states) in the SAME fixed diabatic basis as the model.
    It is an operator derivative, not an adiabatic derivative coupling.
    Both numerical derivatives are projected using the eigenvectors at the
    central geometry, without phase matching or energy-gap division.

    Double-precision input is required. The step uses the same Cartesian units
    as positions. Models in geometry-dependent bases require connection terms.
    """

    _validate_controls(step, gap_floor, atol, rtol)
    positions = torch.as_tensor(positions)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (atoms>=1, 3)")
    if positions.dtype != torch.float64:
        raise ValueError("certification requires float64 positions")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must be finite")
    atomic_numbers = torch.as_tensor(atomic_numbers, device=positions.device)
    if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != positions.shape[0]:
        raise ValueError("atomic_numbers must match the number of atoms")
    if not torch.isfinite(atomic_numbers).all():
        raise ValueError("atomic_numbers must be finite")

    coordinates = positions.detach().clone()
    with torch.enable_grad():
        hamiltonian, derivative = hamiltonian_jacobian(
            model, atomic_numbers, coordinates, create_graph=False
        )
    hamiltonian = _checked_matrix(hamiltonian)
    if derivative.dtype not in (torch.float64, torch.complex128):
        raise ValueError("certification requires double-precision matrix gradients")
    if not torch.isfinite(derivative).all():
        raise ValueError("autograd derivative contains nonfinite values")
    if derivative.shape != (positions.shape[0], 3, *hamiltonian.shape):
        raise ValueError("autograd derivative shape is inconsistent with Hamiltonian")
    with torch.no_grad():
        finite_difference = _five_point_derivative(
            model, atomic_numbers, coordinates, step, tuple(hamiltonian.shape)
        )
    if not torch.allclose(derivative, derivative.mH, atol=atol, rtol=rtol):
        raise ValueError("autograd derivative is not Hermitian")
    if not torch.allclose(finite_difference, finite_difference.mH, atol=atol, rtol=rtol):
        raise ValueError("finite-difference derivative is not Hermitian")

    energies, eigenvectors = torch.linalg.eigh(hamiltonian.detach())
    numerical = torch.einsum("mi,abmn,nj->abij", eigenvectors.conj(), derivative, eigenvectors)
    independent = torch.einsum(
        "mi,abmn,nj->abij", eigenvectors.conj(), finite_difference, eigenvectors
    )
    maximum, relative = _max_error(numerical, independent)
    consistent = bool(torch.allclose(numerical, independent, atol=atol, rtol=rtol))

    reference_checked = reference_derivative is not None
    reference_max_error = None
    if reference_checked:
        target = torch.as_tensor(reference_derivative, device=positions.device)
        if target.shape != derivative.shape:
            raise ValueError("reference_derivative must have shape (atoms, 3, states, states)")
        if target.dtype not in (torch.float64, torch.complex128):
            raise ValueError("reference_derivative must use double precision")
        if not torch.isfinite(target).all():
            raise ValueError("reference_derivative must be finite")
        if not torch.allclose(target, target.mH, atol=atol, rtol=rtol):
            raise ValueError("reference_derivative must be Hermitian")
        reference_max_error, _ = _max_error(derivative, target)
        consistent = consistent and bool(torch.allclose(derivative, target, atol=atol, rtol=rtol))
        consistent = consistent and bool(
            torch.allclose(finite_difference, target, atol=atol, rtol=rtol)
        )

    mask = coupling_validity_mask(energies, gap_floor=gap_floor)
    n_states = energies.shape[0]
    unresolved = tuple(
        (i, j) for i in range(n_states) for j in range(i + 1, n_states) if not mask[i, j]
    )
    gaps = (energies[None, :] - energies[:, None]).abs()
    minimum_gap = (
        float(gaps[torch.triu_indices(n_states, n_states, offset=1).unbind()].min())
        if n_states > 1
        else float("inf")
    )
    return NumeratorCertification(
        energies=energies.detach(),
        numerators_autograd=numerical.detach(),
        numerators_finite_difference=independent.detach(),
        valid_gap_pairs=mask.detach(),
        unresolved_pairs=unresolved,
        minimum_energy_gap=minimum_gap,
        finite_difference_max_error=maximum,
        finite_difference_relative_error=relative,
        reference_max_error=reference_max_error,
        reference_checked=reference_checked,
        passed=consistent,
    )
