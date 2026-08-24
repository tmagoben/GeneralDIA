import pytest
import torch

from generaldia.gauge import align_phase, projector_distance, unitary_procrustes


def test_phase_alignment_recovers_reference_phase() -> None:
    reference = torch.tensor([1.0 + 0j, 2.0j])
    candidate = reference * torch.exp(torch.tensor(0.7j))
    aligned = align_phase(reference, candidate)
    assert torch.allclose(aligned, reference)


def test_phase_alignment_rejects_orthogonal_vectors() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        align_phase(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))


def test_projector_is_invariant_to_subspace_unitary() -> None:
    basis, _ = torch.linalg.qr(torch.randn(7, 3, dtype=torch.complex128))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.complex128))
    assert projector_distance(basis, basis @ rotation) < 1e-12


def test_procrustes_alignment_recovers_reference_basis() -> None:
    reference, _ = torch.linalg.qr(torch.randn(7, 3, dtype=torch.complex128))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.complex128))
    aligned = unitary_procrustes(reference, reference @ rotation)
    assert torch.linalg.norm(aligned - reference) < 1e-10


def test_projector_rejects_nonorthonormal_basis() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        projector_distance(torch.ones(3, 2), torch.ones(3, 2))
