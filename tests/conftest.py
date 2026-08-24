import torch


def pytest_sessionstart() -> None:
    torch.set_default_dtype(torch.float64)
