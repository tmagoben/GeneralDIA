"""v3.4 omitted-state negative control: isospectral but wrong subspace."""

import json
import math
import os

import torch

from generaldia.truncation import certify_state_truncation


def main():
    reference = torch.diag(torch.tensor([0.0, 0.06, 0.12], dtype=torch.float64))
    retained = torch.eye(3, dtype=torch.float64)[:, :2]
    angle = 0.3
    unitary = torch.eye(3, dtype=torch.float64)
    unitary[0, 0], unitary[2, 2] = math.cos(angle), math.cos(angle)
    unitary[0, 2], unitary[2, 0] = math.sin(angle), -math.sin(angle)
    isospectral = unitary @ reference @ unitary.T

    thresholds = {
        "energy_tolerance": 0.012,
        "omitted_population_tolerance": 0.02,
        "residual_tolerance": 0.01,
        "target_gap_floor": 1e-6,
    }
    baseline = certify_state_truncation(reference, retained, **thresholds)
    rotated = certify_state_truncation(isospectral, retained, **thresholds)
    checks = {
        "exact_reference_passes": baseline.passed,
        "full_spectra_identical": bool(
            torch.allclose(
                torch.linalg.eigvalsh(reference),
                torch.linalg.eigvalsh(isospectral),
                atol=1e-12,
                rtol=0,
            )
        ),
        "energy_only_false_positive": rotated.maximum_energy_error
        < thresholds["energy_tolerance"],
        "omitted_state_detected": not rotated.passed
        and rotated.maximum_omitted_population > thresholds["omitted_population_tolerance"],
        "coupling_residual_detected": rotated.residual_operator_norm
        > thresholds["residual_tolerance"],
    }
    record = {
        "stage": "v3.4-slice-3",
        "revision": os.getenv("GITHUB_SHA", "unrecorded-local"),
        "reference": "three-level diagonal H, rigid 0-2 electronic rotation by 0.3 rad",
        "thresholds": thresholds,
        "baseline_pass": baseline.passed,
        "rotated_energy_error": rotated.maximum_energy_error,
        "rotated_omitted_population": rotated.maximum_omitted_population,
        "rotated_residual_norm": rotated.residual_operator_norm,
        "rotated_minimum_boundary_gap": rotated.minimum_target_gap,
        "checks": checks,
        "pass": all(checks.values()),
        "limits": "Requires a complete supplied finite reference; no molecular completeness claim",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
