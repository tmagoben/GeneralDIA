"""Real CUDA parity checks; an explicitly required GPU must never silently skip."""

import math
import os

import pytest
import torch

from generaldia.h3plus_model import H3PlusEquivariantHamiltonian
from generaldia.topology import closed_loop_overlaps, wilson_loop

pytestmark = pytest.mark.gpu


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        if os.environ.get("GENERALDIA_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA validation requested, but no CUDA device is available")
        pytest.skip("requires real CUDA hardware")
    return torch.device("cuda")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_spectrum_gradients_and_optimizer_step_match_cpu(cuda_device, dtype):
    cpu = H3PlusEquivariantHamiltonian().to(dtype=dtype)
    with torch.no_grad():
        cpu.scalar_coefficients[:, 0] = torch.tensor([-1.2, -0.4])
        cpu.splitting_coefficients.copy_(torch.tensor([0.06, 0.005, 0.001, 0.008]))
    gpu = H3PlusEquivariantHamiltonian().to(device=cuda_device, dtype=dtype)
    gpu.load_state_dict(cpu.state_dict())
    bonds = torch.tensor([[0.85, 0.89, 0.87], [0.92, 0.88, 0.86]], dtype=dtype, requires_grad=True)
    device_bonds = bonds.detach().to(cuda_device).requires_grad_()
    output_cpu = torch.linalg.eigvalsh(cpu(bonds))
    output_gpu = torch.linalg.eigvalsh(gpu(device_bonds))
    tolerance = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(output_gpu.cpu(), output_cpu, atol=tolerance, rtol=tolerance)
    output_cpu.square().sum().backward()
    output_gpu.square().sum().backward()
    torch.testing.assert_close(device_bonds.grad.cpu(), bonds.grad, atol=tolerance, rtol=tolerance)
    for p, q in zip(cpu.parameters(), gpu.parameters(), strict=True):
        torch.testing.assert_close(q.grad.cpu(), p.grad, atol=tolerance, rtol=tolerance)
    torch.optim.SGD(cpu.parameters(), lr=0.01).step()
    torch.optim.SGD(gpu.parameters(), lr=0.01).step()
    for p, q in zip(cpu.parameters(), gpu.parameters(), strict=True):
        torch.testing.assert_close(q.cpu(), p, atol=tolerance, rtol=tolerance)


def test_cuda_complex_gauge_berry_loop_matches_cpu(cuda_device):
    theta = torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32
    frames = torch.stack((-torch.sin(theta / 2), torch.cos(theta / 2)), dim=-1)[:, :, None]
    frames = frames.to(torch.complex128) * torch.exp(1j * 0.37 * theta)[:, None, None]
    cpu = wilson_loop(closed_loop_overlaps(frames))
    gpu = wilson_loop(closed_loop_overlaps(frames.to(cuda_device)))
    assert gpu.wilson.is_cuda
    torch.testing.assert_close(gpu.wilson.cpu(), cpu.wilson, atol=1e-12, rtol=0)
    assert abs(gpu.berry_phase) == pytest.approx(math.pi, abs=1e-12)
    assert gpu.minimum_link_singular_value == pytest.approx(
        cpu.minimum_link_singular_value, abs=1e-12
    )
