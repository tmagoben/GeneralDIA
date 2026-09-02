"""Track electronic-state character through a two-state energy crossing."""

import torch

from generaldia import (
    adjacent_state_overlaps,
    align_state_frames,
    track_states,
    transform_state_matrices,
)

torch.set_default_dtype(torch.float64)

# The exact crossing lies between the two central samples. Energy sorting changes
# the raw column order there, and independent complex phases mimic backend gauge
# choices at each geometry.
coordinates = torch.tensor([-1.0, -0.2, 0.2, 1.0])
phase_angles = torch.tensor([[0.0, 0.0], [0.4, -0.6], [-0.7, 0.2], [0.9, -0.3]])
raw_energies = []
raw_frames = []
for coordinate, phases in zip(coordinates, phase_angles, strict=True):
    hamiltonian = torch.diag(torch.stack((coordinate, -coordinate))).to(torch.complex128)
    energies, frame = torch.linalg.eigh(hamiltonian)
    raw_energies.append(energies)
    raw_frames.append(frame * torch.exp(1j * phases))

raw_energies = torch.stack(raw_energies)
raw_frames = torch.stack(raw_frames)
overlaps = adjacent_state_overlaps(raw_frames)
tracking = track_states(overlaps, energies=raw_energies)

tracked_frames = align_state_frames(raw_frames, tracking.transformations)
tracked_energy_matrices = transform_state_matrices(
    torch.diag_embed(raw_energies).to(torch.complex128), tracking.transformations
)
tracked_energies = torch.diagonal(tracked_energy_matrices, dim1=-2, dim2=-1).real
expected_character_energies = torch.stack((coordinates, -coordinates), dim=-1)

print(" coordinate | raw energy order | tracked character order")
for coordinate, raw, tracked in zip(coordinates, raw_energies, tracked_energies, strict=True):
    print(
        f" {float(coordinate):+9.3f} |"
        f" ({float(raw[0]):+6.3f}, {float(raw[1]):+6.3f}) |"
        f" ({float(tracked[0]):+6.3f}, {float(tracked[1]):+6.3f})"
    )

energy_residual = torch.max(torch.abs(tracked_energies - expected_character_energies))
frame_residual = torch.linalg.matrix_norm(
    tracked_frames - tracked_frames[0].expand_as(tracked_frames)
).amax()
minimum_overlap = min(step.minimum_overlap for step in tracking.steps)
print(f"minimum matched overlap: {minimum_overlap:.3f}")
print(f"tracked-energy residual: {float(energy_residual):.3e}")
print(f"tracked-frame residual: {float(frame_residual):.3e}")

assert tracking.ambiguous_steps == ()
assert energy_residual < 1e-12
assert frame_residual < 1e-12
