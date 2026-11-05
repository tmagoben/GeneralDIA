"""Reproducible positive and negative controls for v3.4 numerator checks."""

import json
import os

import torch

from generaldia.certification import certify_coupling_numerators


def crossing(numbers, positions):
    del numbers
    x = positions[0, 0]
    coupling = x.new_tensor(1e-9)
    return torch.stack((torch.stack((0.07 * x, coupling)), torch.stack((coupling, -0.07 * x))))


def detached(numbers, positions):
    return crossing(numbers, positions.detach())


def main():
    coordinates = torch.zeros(1, 3, dtype=torch.float64)
    numbers = torch.tensor([1])
    reference = torch.zeros(1, 3, 2, 2, dtype=torch.float64)
    reference[0, 0] = torch.diag(torch.tensor([0.07, -0.07], dtype=torch.float64))
    positive = certify_coupling_numerators(
        crossing, numbers, coordinates, reference_derivative=reference
    )
    negative = certify_coupling_numerators(detached, numbers, coordinates)
    checks = {
        "independent_reference": positive.reference_checked and positive.passed,
        "near_gap_masked": positive.unresolved_pairs == ((0, 1),),
        "nonzero_finite_numerator": abs(float(positive.numerators_autograd[0, 0, 0, 1])) > 0.069,
        "detached_gradient_detected": not negative.passed,
    }
    result = {
        "stage": "v3.4-slice-2",
        "revision": os.getenv("GITHUB_SHA", "unrecorded-local"),
        "model": "H(x)=0.07*x*sigma_z+1e-9*sigma_x",
        "finite_difference": "five-point centered Hamiltonian stencil",
        "step": 1e-3,
        "gap_floor": 1e-6,
        "minimum_energy_gap": positive.minimum_energy_gap,
        "maximum_numerator_error": positive.finite_difference_max_error,
        "reference_max_error": positive.reference_max_error,
        "unresolved_pairs": positive.unresolved_pairs,
        "checks": checks,
        "pass": all(checks.values()),
        "limits": "fixed-basis analytic controls only; no molecular NAC certification",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
