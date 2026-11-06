"""Adversarial validation and symmetry checks for small scientific primitives."""

import math

import pytest
import torch

from generaldia.analytic import TwoStateAvoidedCrossing
from generaldia.gauge import align_phase, projector, projector_distance, unitary_procrustes
from generaldia.matrix import complex_hermitian_from_parts, unpack_real_symmetric


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"slope": 0.0}, "slope"),
        ({"coupling": -0.001}, "coupling"),
    ],
)
def test_analytic_reference_rejects_nonphysical_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TwoStateAvoidedCrossing(**kwargs)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_analytic_hamiltonian_rejects_nonfinite_coordinate(coordinate):
    with pytest.raises(ValueError, match="finite"):
        TwoStateAvoidedCrossing().hamiltonian(coordinate)


def test_analytic_reference_accepts_integer_input_and_preserves_exact_spectrum():
    model = TwoStateAvoidedCrossing(slope=0.08, coupling=0.03)
    positions = torch.tensor([-2, 0, 3])
    hamiltonians = model.hamiltonian(positions)
    assert hamiltonians.shape == (3, 2, 2)
    assert hamiltonians.dtype == torch.float64
    torch.testing.assert_close(hamiltonians, hamiltonians.mT)
    torch.testing.assert_close(torch.linalg.eigvalsh(hamiltonians), model.exact_energies(positions))
    assert model.exact_energies(2).shape == (2,)


@pytest.mark.parametrize("n", [0, -1])
def test_real_symmetric_rejects_nonpositive_state_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        unpack_real_symmetric(torch.ones(1), n)


@pytest.mark.parametrize(
    "packed, message",
    [
        (torch.tensor(1.0), "one dimension"),
        (torch.ones(2), "expected"),
    ],
)
def test_real_symmetric_rejects_bad_packing(packed, message):
    with pytest.raises(ValueError, match=message):
        unpack_real_symmetric(packed, 2)


def test_real_symmetric_complex_conjugation_is_not_silently_invented():
    packed = torch.tensor([[1.0, -0.3, 2.0], [3.0, 0.4, 1.0]])
    result = unpack_real_symmetric(packed, 2)
    assert result.shape == (2, 2, 2)
    torch.testing.assert_close(result, result.mT)
    torch.testing.assert_close(result[:, 0, 1], packed[:, 1])


@pytest.mark.parametrize(
    "diagonal, real_upper, imag_upper, message",
    [
        (torch.tensor(1.0), torch.ones(0), torch.ones(0), "one dimension"),
        (
            torch.tensor([1 + 0j, 2 + 0j]),
            torch.ones(1),
            torch.ones(1),
            "diagonal must be real",
        ),
        (torch.ones(2), torch.ones(2), torch.ones(1), "must have shape"),
        (torch.ones(2), torch.ones(1, dtype=torch.complex128), torch.ones(1), "must be real"),
    ],
)
def test_hermitian_constructor_rejects_invalid_inputs(diagonal, real_upper, imag_upper, message):
    with pytest.raises(ValueError, match=message):
        complex_hermitian_from_parts(diagonal, real_upper, imag_upper)


def test_hermitian_constructor_batched_integer_diagonal_and_complex_offdiagonals():
    diagonal = torch.tensor([[1, 2], [3, 4]])
    real = torch.tensor([[0.2], [-0.5]])
    imag = torch.tensor([[0.7], [0.3]])
    matrix = complex_hermitian_from_parts(diagonal, real, imag)
    assert matrix.shape == (2, 2, 2)
    assert matrix.dtype == torch.complex128
    torch.testing.assert_close(matrix, matrix.mH, atol=0, rtol=0)
    torch.testing.assert_close(matrix[:, 0, 1], real[:, 0] + 1j * imag[:, 0])
    torch.testing.assert_close(torch.diagonal(matrix, dim1=-2, dim2=-1).real, diagonal.double())


@pytest.mark.parametrize(
    "reference, candidate, floor, message",
    [
        (torch.ones(2), torch.ones(3), 0.0, "matching shapes"),
        (torch.ones(1, 2), torch.ones(1, 2), 0.0, "vectors"),
        (torch.ones(2), torch.ones(2), -0.01, "overlap_floor"),
        (torch.zeros(2), torch.ones(2), 0.0, "nonzero vectors"),
        (torch.ones(2), torch.zeros(2), 0.0, "nonzero vectors"),
        (torch.ones(2), torch.ones(2), 1.0, "overlap_floor"),
    ],
)
def test_phase_alignment_rejects_invalid_or_unresolved_vectors(
    reference, candidate, floor, message
):
    with pytest.raises(ValueError, match=message):
        align_phase(reference, candidate, overlap_floor=floor)


def test_phase_alignment_complex_covariance_and_amplitude_preservation():
    reference = torch.tensor([1.0 + 0j, 2.0 - 3.0j, -0.2 + 0.4j])
    rotated = reference * torch.exp(torch.tensor(1.3j))
    result = align_phase(reference, 2 * rotated)
    torch.testing.assert_close(result, 2 * reference)
    assert torch.linalg.vector_norm(result) == pytest.approx(
        2 * torch.linalg.vector_norm(reference).item()
    )


def test_projector_rejects_non_matrix_or_invalid_orthonormal_basis():
    with pytest.raises(ValueError, match="shape"):
        projector(torch.ones(3))
    with pytest.raises(ValueError, match="orthonormal"):
        projector(torch.ones(3, 2))


def test_projector_distance_rejects_mismatched_ambient_spaces():
    a = torch.eye(3)[:, :1]
    b = torch.eye(4)[:, :1]
    with pytest.raises(ValueError, match="same ambient dimension"):
        projector_distance(a, b)


@pytest.mark.parametrize(
    "reference, candidate, message",
    [
        (torch.ones(3), torch.ones(3), "matching shapes"),
        (torch.eye(3), torch.eye(4), "matching shapes"),
        (torch.ones(3, 2), torch.ones(3, 2), "orthonormal"),
    ],
)
def test_procrustes_rejects_unidentifiable_input(reference, candidate, message):
    with pytest.raises(ValueError, match=message):
        unitary_procrustes(reference, candidate)


def test_projector_unitary_procrustes_complex_geometric_error_is_phase_free():
    theta = 0.35
    reference = torch.eye(3, dtype=torch.complex128)[:, :2]
    rotated = torch.stack(
        (
            torch.tensor([math.cos(theta), math.sin(theta), 0], dtype=torch.complex128),
            torch.tensor([-math.sin(theta), math.cos(theta), 0], dtype=torch.complex128),
        ),
        dim=1,
    )
    assert projector_distance(reference, rotated) < 1e-12
    aligned = unitary_procrustes(reference, rotated)
    torch.testing.assert_close(aligned, reference, atol=1e-12, rtol=0)
