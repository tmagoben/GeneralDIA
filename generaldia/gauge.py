"""Phase and subspace alignment utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def align_phase(reference: Tensor, candidate: Tensor, *, overlap_floor: float = 1e-12) -> Tensor:
    """Align one candidate vector to a reference vector by a complex phase.

    The function raises an error when the normalized overlap falls below
    ``overlap_floor`` because the reference does not define a stable phase there.
    """

    reference = torch.as_tensor(reference)
    candidate = torch.as_tensor(candidate, device=reference.device)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("reference and candidate must be vectors with matching shapes")
    if overlap_floor < 0:
        raise ValueError("overlap_floor cannot be negative")
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    if reference_norm == 0 or candidate_norm == 0:
        raise ValueError("phase alignment requires nonzero vectors")
    overlap = torch.vdot(reference, candidate)
    normalized_overlap = torch.abs(overlap) / (reference_norm * candidate_norm)
    if normalized_overlap <= overlap_floor:
        raise ValueError("phase alignment is undefined for an orthogonal state")
    return candidate * (overlap.conj() / torch.abs(overlap))


def projector(basis: Tensor) -> Tensor:
    """Return the projector onto the columns of an orthonormal basis."""

    basis = torch.as_tensor(basis)
    if basis.ndim != 2:
        raise ValueError("basis must have shape (dimension, n_vectors)")
    gram = basis.mH @ basis
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    if not torch.allclose(gram, identity, atol=1e-8, rtol=1e-7):
        raise ValueError("basis columns must be orthonormal")
    return basis @ basis.mH


def projector_distance(left: Tensor, right: Tensor) -> Tensor:
    """Return the Frobenius distance between two subspace projectors."""

    left_projector = projector(left)
    right_projector = projector(right)
    if left_projector.shape != right_projector.shape:
        raise ValueError("the two bases must span the same ambient dimension")
    return torch.linalg.matrix_norm(left_projector - right_projector, ord="fro")


def unitary_procrustes(reference: Tensor, candidate: Tensor) -> Tensor:
    """Rotate a candidate basis within its subspace to best match a reference."""

    reference = torch.as_tensor(reference)
    candidate = torch.as_tensor(candidate, device=reference.device)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must be matrices with matching shapes")
    projector(reference)
    projector(candidate)
    overlap = candidate.mH @ reference
    left, _, right_h = torch.linalg.svd(overlap)
    return candidate @ (left @ right_h)
