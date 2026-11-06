"""Analytic v3.4 topology controls; no molecular claim."""

from __future__ import annotations

import hashlib
import json
import math
import os

import torch

from generaldia.topology import closed_loop_overlaps, wilson_loop


def _circular_ci(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    phi = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
    h = torch.zeros((n, 2, 2), dtype=torch.float64)
    h[:, 0, 0] = torch.cos(phi)
    h[:, 1, 1] = -torch.cos(phi)
    h[:, 0, 1] = torch.sin(phi)
    h[:, 1, 0] = torch.sin(phi)
    _, frames = torch.linalg.eigh(h)
    return phi, frames[:, :, :1]


def _distance(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def main() -> None:
    n = 80
    phi, frames = _circular_ci(n)
    trivial = torch.tensor([[1.0], [0.0]], dtype=torch.float64).expand(n, -1, -1)
    gauge = torch.exp(1j * (0.3 * phi + torch.sin(5 * phi)))
    ci = wilson_loop(closed_loop_overlaps(frames))
    gauged = wilson_loop(closed_loop_overlaps(frames.to(torch.complex128) * gauge[:, None, None]))
    control = wilson_loop(closed_loop_overlaps(trivial))
    reverse = wilson_loop(closed_loop_overlaps(frames.flip(0)))
    tolerance = 1e-9
    checks = {
        "CI_berry_pi": _distance(ci.berry_phase, math.pi) < tolerance,
        "trivial_berry_zero": _distance(control.berry_phase, 0.0) < tolerance,
        "gauge_invariant": _distance(ci.berry_phase, gauged.berry_phase) < tolerance,
        "reverse_orientation": _distance(ci.berry_phase, -reverse.berry_phase) < tolerance,
    }
    record = {
        "status": "analytic-topology-controls-only",
        "version_stage": "v3.4-slice-1",
        "revision": os.getenv("GITHUB_SHA", "unrecorded-local"),
        "reference": "H_CI=cos(phi)*sigma_z+sin(phi)*sigma_x; H_trivial=sigma_z",
        "geometry_sha256": hashlib.sha256(phi.numpy().tobytes()).hexdigest(),
        "samples": n,
        "phase_tolerance_radians": tolerance,
        "minimum_link_singular_value": ci.minimum_link_singular_value,
        "CI_berry_phase_radians": ci.berry_phase,
        "trivial_berry_phase_radians": control.berry_phase,
        "checks": checks,
        "pass": all(checks.values()),
        "limits": "No molecular NAC, omitted-state, or trajectory certification",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
