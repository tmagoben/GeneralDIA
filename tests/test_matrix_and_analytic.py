import pytest
import torch

from generaldia.analytic import TwoStateAvoidedCrossing
from generaldia.matrix import complex_hermitian_from_parts, unpack_real_symmetric


def test_avoided_crossing_matches_analytic_eigenvalues() -> None:
    model = TwoStateAvoidedCrossing()
    coordinate = torch.linspace(-1, 1, 17)
    numerical = torch.linalg.eigvalsh(model.hamiltonian(coordinate))
    assert torch.allclose(numerical, model.exact_energies(coordinate), atol=1e-13)


def test_avoided_crossing_preserves_input_dtype() -> None:
    coordinate = torch.tensor([0.2], dtype=torch.float32)
    assert TwoStateAvoidedCrossing().hamiltonian(coordinate).dtype == torch.float32


@pytest.mark.parametrize("kwargs", [{"slope": 0}, {"coupling": -0.1}])
def test_avoided_crossing_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TwoStateAvoidedCrossing(**kwargs)


def test_symmetric_unpack_supports_batches_and_gradients() -> None:
    packed = torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_(True)
    matrix = unpack_real_symmetric(packed, 3)
    assert matrix.shape == (2, 3, 3)
    assert torch.allclose(matrix, matrix.mT)
    matrix.square().sum().backward()
    assert packed.grad is not None


def test_symmetric_unpack_rejects_wrong_parameter_count() -> None:
    with pytest.raises(ValueError, match="expected 6"):
        unpack_real_symmetric(torch.zeros(5), 3)


def test_complex_hermitian_constructor_supports_batches() -> None:
    diagonal = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    real = torch.tensor([[0.02], [0.05]])
    imag = torch.tensor([[0.03], [-0.01]])
    matrix = complex_hermitian_from_parts(diagonal, real, imag)
    assert matrix.shape == (2, 2, 2)
    assert matrix.dtype == torch.complex128
    assert torch.allclose(matrix, matrix.mH)


def test_complex_hermitian_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        complex_hermitian_from_parts(torch.zeros(3), torch.zeros(2), torch.zeros(3))
