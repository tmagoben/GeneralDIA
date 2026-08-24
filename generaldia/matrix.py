"""Matrix constructors with exact symmetry constraints."""

from __future__ import annotations

import torch
from torch import Tensor


def _positive_integer(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def unpack_real_symmetric(packed: Tensor, n: int) -> Tensor:
    """Construct real symmetric matrices from packed upper triangles.

    ``packed`` has shape ``(..., n * (n + 1) // 2)``. Values follow upper
    triangle order: ``(0, 0), (0, 1), ..., (1, 1), ...``.
    """

    n = _positive_integer(n, "n")
    packed = torch.as_tensor(packed)
    if packed.ndim < 1:
        raise ValueError("packed must have at least one dimension")
    expected = n * (n + 1) // 2
    if packed.shape[-1] != expected:
        raise ValueError(f"packed has {packed.shape[-1]} values; expected {expected} for n={n}")

    rows, cols = torch.triu_indices(n, n, device=packed.device)
    matrix = packed.new_zeros((*packed.shape[:-1], n, n))
    matrix[..., rows, cols] = packed
    matrix[..., cols, rows] = packed
    return matrix


def complex_hermitian_from_parts(
    diagonal: Tensor,
    real_upper: Tensor,
    imag_upper: Tensor,
) -> Tensor:
    """Construct complex Hermitian matrices from independent real components.

    ``diagonal`` has shape ``(..., n)``. The two upper-triangle tensors each
    have shape ``(..., n * (n - 1) // 2)`` and use ``(0, 1), (0, 2), ...`` order.
    """

    diagonal = torch.as_tensor(diagonal)
    real_upper = torch.as_tensor(real_upper, device=diagonal.device)
    imag_upper = torch.as_tensor(imag_upper, device=diagonal.device)
    if diagonal.ndim < 1:
        raise ValueError("diagonal must have at least one dimension")
    if diagonal.is_complex():
        raise ValueError("diagonal must be real")
    if not diagonal.is_floating_point():
        diagonal = diagonal.to(torch.get_default_dtype())

    n = diagonal.shape[-1]
    expected = n * (n - 1) // 2
    target_shape = (*diagonal.shape[:-1], expected)
    if real_upper.shape != target_shape or imag_upper.shape != target_shape:
        raise ValueError(
            "real_upper and imag_upper must have shape "
            f"{target_shape}; received {real_upper.shape} and {imag_upper.shape}"
        )
    if real_upper.is_complex() or imag_upper.is_complex():
        raise ValueError("off-diagonal component tensors must be real")

    real_upper = real_upper.to(dtype=diagonal.dtype)
    imag_upper = imag_upper.to(dtype=diagonal.dtype)
    complex_dtype = torch.complex128 if diagonal.dtype == torch.float64 else torch.complex64
    matrix = torch.zeros(
        (*diagonal.shape[:-1], n, n),
        dtype=complex_dtype,
        device=diagonal.device,
    )
    matrix.diagonal(dim1=-2, dim2=-1).copy_(diagonal.to(complex_dtype))
    rows, cols = torch.triu_indices(n, n, offset=1, device=diagonal.device)
    values = real_upper.to(complex_dtype) + 1j * imag_upper.to(complex_dtype)
    matrix[..., rows, cols] = values
    matrix[..., cols, rows] = values.conj()
    return matrix
