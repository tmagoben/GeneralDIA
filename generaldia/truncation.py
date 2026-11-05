"""Gauge-invariant finite-manifold truncation diagnostics with a full-state oracle.

This compares a retained orthonormal frame against an explicitly supplied
larger Hermitian Hamiltonian in the SAME physical Hilbert space. It cannot
infer states absent from the reference Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True)
class TruncationCertification:
    """Full-reference diagnostics; energies have the input Hamiltonian's units."""

    target_state_indices: tuple[int, ...]
    full_energies: Tensor
    retained_energies: Tensor
    maximum_energy_error: float
    maximum_omitted_population: float
    residual_operator_norm: float
    minimum_target_gap: float
    unresolved_target_gap: bool
    passed: bool


def certify_state_truncation(
    hamiltonian: Tensor,
    retained_frame: Tensor,
    *,
    target_state_indices: tuple[int, ...] | None = None,
    energy_tolerance: float = 1e-3,
    omitted_population_tolerance: float = 1e-2,
    residual_tolerance: float = 1e-2,
    target_gap_floor: float = 1e-6,
) -> TruncationCertification:
    """Compare a retained projector to selected eigenstates of a full reference.

    `retained_frame` contains orthonormal columns in the full Hamiltonian basis.
    By default the target manifold is the lowest k full eigenstates, where k is
    the number of columns. Other targets require explicit ascending indices.

    The largest omitted population is the squared operator norm of
    `(I-QQ†) Psi_target`, invariant under internal target and retained gauges.
    The residual is `||(I-QQ†) H Q||_2`, invariant under retained-frame gauge.
    Both may reject an energy-only false positive. An unresolved boundary gap
    always prevents certification, even if other metrics pass.
    """

    controls = {
        "energy_tolerance": energy_tolerance,
        "omitted_population_tolerance": omitted_population_tolerance,
        "residual_tolerance": residual_tolerance,
        "target_gap_floor": target_gap_floor,
    }
    for name, value in controls.items():
        if (
            not isfinite(value)
            or value < 0
            or (name == "target_gap_floor" and value == 0)
        ):
            raise ValueError(f"{name} must be finite and nonnegative (gap floor positive)")
    if omitted_population_tolerance > 1:
        raise ValueError("omitted_population_tolerance cannot exceed one")

    h = torch.as_tensor(hamiltonian)
    q = torch.as_tensor(retained_frame)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian must be square with at least two states")
    if q.ndim != 2 or q.shape[0] != h.shape[0] or not 1 <= q.shape[1] < h.shape[0]:
        raise ValueError("retained_frame must have shape (states, 1 <= retained < states)")
    if h.dtype not in (torch.float64, torch.complex128) or q.dtype not in (
        torch.float64,
        torch.complex128,
    ):
        raise ValueError("certification requires double-precision real/complex inputs")
    if h.device != q.device:
        raise ValueError("hamiltonian and retained_frame must use the same device")
    if not torch.isfinite(h).all() or not torch.isfinite(q).all():
        raise ValueError("hamiltonian and retained_frame must be finite")
    dtype = torch.promote_types(h.dtype, q.dtype)
    h, q = h.to(dtype), q.to(dtype)
    if not torch.allclose(h, h.mH, atol=1e-10, rtol=1e-10):
        raise ValueError("hamiltonian must be Hermitian")
    eye = torch.eye(q.shape[1], dtype=dtype, device=q.device)
    if not torch.allclose(q.mH @ q, eye, atol=1e-10, rtol=1e-10):
        raise ValueError("retained_frame columns must be orthonormal")

    k, n = q.shape[1], h.shape[0]
    selected = tuple(range(k)) if target_state_indices is None else tuple(target_state_indices)
    if len(selected) != k or any(type(i) is not int for i in selected):
        raise ValueError("target_state_indices must contain exactly k integer indices")
    if tuple(sorted(set(selected))) != selected or selected[0] < 0 or selected[-1] >= n:
        raise ValueError("target_state_indices must be unique, ascending and in range")
    omitted = tuple(i for i in range(n) if i not in selected)

    full_energies, full_vectors = torch.linalg.eigh(h)
    retained_h = q.mH @ h @ q
    retained_energies = torch.linalg.eigvalsh(retained_h)
    target_energies = full_energies[list(selected)]
    mismatch = float((retained_energies - target_energies).abs().max())
    target_vectors = full_vectors[:, list(selected)]
    perpendicular = target_vectors - q @ (q.mH @ target_vectors)
    max_population = float(torch.linalg.matrix_norm(perpendicular, ord=2) ** 2)
    residual = h @ q - q @ retained_h
    residual_norm = float(torch.linalg.matrix_norm(residual, ord=2))
    gap = float(
        (target_energies[:, None] - full_energies[list(omitted)][None, :]).abs().min()
    )
    unresolved = gap < target_gap_floor
    passed = (
        not unresolved
        and mismatch <= energy_tolerance
        and max_population <= omitted_population_tolerance
        and residual_norm <= residual_tolerance
    )
    return TruncationCertification(
        target_state_indices=selected,
        full_energies=full_energies,
        retained_energies=retained_energies,
        maximum_energy_error=mismatch,
        maximum_omitted_population=max_population,
        residual_operator_norm=residual_norm,
        minimum_target_gap=gap,
        unresolved_target_gap=unresolved,
        passed=passed,
    )
