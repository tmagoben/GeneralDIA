import itertools
import math

import pytest
import torch

from generaldia import (
    AmbiguousStateTrackingError,
    adjacent_state_overlaps,
    align_state_frames,
    track_states,
    transform_state_matrices,
)


def _raw_column_gauge(
    permutation: tuple[int, ...],
    phases: tuple[complex, ...],
    *,
    block_rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    n_states = len(permutation)
    gauge = torch.eye(n_states, dtype=torch.complex128)
    if block_rotation is not None:
        gauge[: block_rotation.shape[0], : block_rotation.shape[1]] = block_rotation
    reorder = torch.zeros_like(gauge)
    for raw_index, canonical_index in enumerate(permutation):
        reorder[canonical_index, raw_index] = phases[raw_index]
    return gauge @ reorder


def _complex_two_state_rotation(theta: float, phase: float) -> torch.Tensor:
    cosine = math.cos(theta)
    sine = math.sin(theta)
    phase_factor = torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))
    return torch.tensor(
        [
            [cosine, -sine * phase_factor.conj()],
            [sine * phase_factor, cosine],
        ],
        dtype=torch.complex128,
    )


def test_tracks_permuted_complex_states_by_character() -> None:
    angles = (0.0, 0.07, 0.15, 0.24)
    smooth_frames = []
    for angle in angles:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        smooth_frames.append(
            torch.tensor(
                [
                    [cosine, -sine, 0.0],
                    [sine, cosine, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.complex128,
            )
        )
    smooth_frames = torch.stack(smooth_frames)
    permutations = ((0, 1, 2), (2, 0, 1), (1, 2, 0), (0, 2, 1))
    phase_angles = ((0.0, 0.0, 0.0), (0.4, -0.7, 1.1), (-0.2, 0.9, 0.3), (0.8, 0.1, -1.0))
    gauges = []
    raw_frames = []
    raw_energies = []
    canonical_energies = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    for frame, permutation, angles_at_geometry in zip(
        smooth_frames, permutations, phase_angles, strict=True
    ):
        phases = tuple(complex(torch.exp(torch.tensor(1j * value))) for value in angles_at_geometry)
        gauge = _raw_column_gauge(permutation, phases)
        gauges.append(gauge)
        raw_frames.append(frame @ gauge)
        raw_energies.append(canonical_energies[list(permutation)])
    raw_frames = torch.stack(raw_frames)
    raw_energies = torch.stack(raw_energies)

    result = track_states(adjacent_state_overlaps(raw_frames), energies=raw_energies)
    aligned = align_state_frames(raw_frames, result.transformations)

    assert torch.linalg.matrix_norm(aligned - smooth_frames).amax() < 1e-12
    assert result.ambiguous_steps == ()
    assert result.aligned_overlaps.shape == (3, 3, 3)
    for step, permutation in zip(result.steps, permutations[1:], strict=True):
        expected = tuple(permutation.index(state) for state in range(3))
        assert step.permutation == expected
        assert not step.ambiguous
        assert step.end_index == step.start_index + 1
        assert step.degenerate_blocks == ()
    identity = torch.eye(3, dtype=torch.complex128).expand(4, 3, 3)
    assert torch.allclose(result.transformations.mH @ result.transformations, identity)


def test_state_matrix_transformation_is_gauge_covariant_with_extra_axes() -> None:
    gauges = torch.stack(
        (
            torch.eye(3, dtype=torch.complex128),
            _raw_column_gauge((2, 0, 1), (1j, -1.0, complex(torch.exp(torch.tensor(0.4j))))),
            _raw_column_gauge((1, 2, 0), (-1j, 1.0, -1.0)),
        )
    )
    raw_frames = gauges.clone()
    energies = torch.stack(
        (
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([2.0, 0.0, 1.0]),
            torch.tensor([1.0, 2.0, 0.0]),
        )
    )
    result = track_states(adjacent_state_overlaps(raw_frames), energies=energies)

    canonical = torch.empty((3, 2, 3, 3, 3), dtype=torch.complex128)
    base = torch.tensor(
        [[0.2, 0.3 + 0.1j, -0.2j], [0.3 - 0.1j, -0.4, 0.5], [0.2j, 0.5, 0.7]],
        dtype=torch.complex128,
    )
    for geometry in range(3):
        for atom in range(2):
            for axis in range(3):
                canonical[geometry, atom, axis] = base * (1 + geometry + atom + axis)
    raw = torch.empty_like(canonical)
    for geometry, gauge in enumerate(gauges):
        for atom in range(2):
            for axis in range(3):
                raw[geometry, atom, axis] = gauge.mH @ canonical[geometry, atom, axis] @ gauge

    recovered = transform_state_matrices(raw, result.transformations)
    assert torch.linalg.matrix_norm(recovered - canonical).amax() < 1e-12


def test_procrustes_aligns_exact_degenerate_subspaces() -> None:
    permutations = ((0, 1, 2, 3), (2, 0, 3, 1), (1, 3, 0, 2))
    rotations = (
        torch.eye(2, dtype=torch.complex128),
        _complex_two_state_rotation(0.63, 0.41),
        _complex_two_state_rotation(-0.38, 0.92),
    )
    phases = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1j, -1.0, complex(torch.exp(torch.tensor(0.7j)))),
        (-1j, 1.0, complex(torch.exp(torch.tensor(-0.2j))), -1.0),
    )
    gauges = torch.stack(
        tuple(
            _raw_column_gauge(permutation, phase, block_rotation=rotation)
            for permutation, phase, rotation in zip(permutations, phases, rotations, strict=True)
        )
    )
    canonical_energies = torch.tensor([0.0, 0.0, 1.0, 2.0])
    energies = torch.stack(
        tuple(canonical_energies[list(permutation)] for permutation in permutations)
    )

    result = track_states(
        adjacent_state_overlaps(gauges),
        energies=energies,
        degeneracy_tolerance=1e-12,
    )
    aligned = align_state_frames(gauges, result.transformations)

    identity_frames = torch.eye(4, dtype=torch.complex128).expand(3, 4, 4)
    assert torch.linalg.matrix_norm(aligned - identity_frames).amax() < 1e-12
    assert all(step.degenerate_blocks == ((0, 1),) for step in result.steps)
    for step in result.steps:
        block_match = next(match for match in step.matches if match.is_degenerate)
        assert torch.allclose(block_match.principal_overlaps, torch.ones(2))


def test_energy_sorted_crossing_is_reordered_by_state_character() -> None:
    coordinates = torch.tensor([-1.0, -0.2, 0.2, 1.0], dtype=torch.float64)
    raw_energies = []
    raw_frames = []
    for coordinate in coordinates:
        hamiltonian = torch.diag(torch.stack((coordinate, -coordinate))).to(torch.complex128)
        energies, frame = torch.linalg.eigh(hamiltonian)
        raw_energies.append(energies)
        raw_frames.append(frame)
    raw_energies = torch.stack(raw_energies)
    raw_frames = torch.stack(raw_frames)

    forward = track_states(adjacent_state_overlaps(raw_frames), energies=raw_energies)
    reverse = track_states(
        adjacent_state_overlaps(raw_frames.flip(0)), energies=raw_energies.flip(0)
    )
    tracked_energy_matrices = transform_state_matrices(
        torch.diag_embed(raw_energies).to(torch.complex128), forward.transformations
    )

    assert torch.allclose(
        torch.diagonal(tracked_energy_matrices, dim1=-2, dim2=-1).real,
        torch.stack((coordinates, -coordinates), dim=-1),
    )
    assert (
        torch.linalg.matrix_norm(
            align_state_frames(raw_frames, forward.transformations)
            - torch.eye(2, dtype=torch.complex128)
        ).amax()
        < 1e-12
    )
    assert (
        torch.linalg.matrix_norm(
            align_state_frames(raw_frames.flip(0), reverse.transformations)
            - torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
        ).amax()
        < 1e-12
    )


def test_tied_assignment_raises_or_can_be_recorded() -> None:
    overlap = torch.tensor(
        [[[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]]],
        dtype=torch.float64,
    )
    energies = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(AmbiguousStateTrackingError, match="assignment margin") as caught:
        track_states(overlap, energies=energies)
    assert caught.value.diagnostic.start_index == 0

    recorded = track_states(overlap, energies=energies, on_ambiguous="record")
    assert recorded.ambiguous_steps == (0,)
    assert recorded.steps[0].ambiguous
    assert any("assignment margin" in reason for reason in recorded.steps[0].reasons)


def test_single_state_path_removes_complex_phase() -> None:
    overlap = torch.exp(torch.tensor(0.7j)).reshape(1, 1, 1)
    result = track_states(overlap, energies=torch.tensor([[0.2], [0.3]]))

    assert result.steps[0].permutation == (0,)
    assert result.steps[0].assignment_margin == float("inf")
    assert torch.allclose(result.aligned_overlaps, torch.ones(1, 1, 1, dtype=torch.complex128))


def test_four_state_assignment_matches_independent_exhaustive_oracle() -> None:
    raw_overlap = torch.tensor(
        [
            [0.80, 0.10, 0.30, 0.20],
            [0.20, 0.75, 0.10, 0.35],
            [0.40, 0.05, 0.70, 0.15],
            [0.10, 0.45, 0.20, 0.65],
        ],
        dtype=torch.float64,
    )
    overlap = raw_overlap / (1.01 * torch.linalg.matrix_norm(raw_overlap, ord=2))
    ranked = sorted(
        (
            (
                sum(float(torch.abs(overlap[row, column])) for row, column in enumerate(order)),
                order,
            )
            for order in itertools.permutations(range(4))
        ),
        reverse=True,
    )

    result = track_states(overlap.unsqueeze(0), overlap_floor=0.0, assignment_margin_floor=0.0)

    assert result.steps[0].permutation == ranked[0][1]
    assert result.steps[0].assignment_margin == pytest.approx((ranked[0][0] - ranked[1][0]) / 4)


def test_low_overlap_raises_with_numeric_diagnostic() -> None:
    overlap = torch.diag(torch.tensor([0.4, 1.0])).unsqueeze(0)
    with pytest.raises(AmbiguousStateTrackingError, match="minimum principal overlap") as caught:
        track_states(overlap)
    assert caught.value.diagnostic.minimum_overlap == pytest.approx(0.4)


def test_degenerate_block_split_is_not_silently_tracked() -> None:
    overlap = torch.eye(3).unsqueeze(0)
    energies = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]])
    with pytest.raises(AmbiguousStateTrackingError, match="split or merge"):
        track_states(overlap, energies=energies)


def test_near_degeneracy_threshold_requires_explicit_review() -> None:
    overlap = torch.eye(2).unsqueeze(0)
    energies = torch.tensor([[0.0, 1e-4], [0.0, 2e-4]])
    with pytest.raises(AmbiguousStateTrackingError, match="near-degenerate") as caught:
        track_states(
            overlap,
            energies=energies,
            degeneracy_tolerance=1e-8,
            near_degeneracy_threshold=1e-3,
        )
    assert caught.value.diagnostic.near_degenerate_pairs[0][:2] == (0, 1)


def test_validation_rejects_invalid_frames_overlaps_and_thresholds() -> None:
    with pytest.raises(ValueError, match="frames must have shape"):
        adjacent_state_overlaps(torch.eye(2))
    with pytest.raises(ValueError, match="orthonormal"):
        adjacent_state_overlaps(torch.ones(2, 3, 2))
    with pytest.raises(ValueError, match="finite"):
        adjacent_state_overlaps(torch.tensor([[[1.0]], [[float("nan")]]]))
    with pytest.raises(ValueError, match="overlaps must have shape"):
        track_states(torch.eye(2))
    with pytest.raises(ValueError, match="contraction"):
        track_states((1.1 * torch.eye(2)).unsqueeze(0))
    with pytest.raises(ValueError, match="energies must have shape"):
        track_states(torch.eye(2).unsqueeze(0), energies=torch.zeros(3, 2))
    with pytest.raises(ValueError, match="near_degeneracy_threshold requires energies"):
        track_states(torch.eye(2).unsqueeze(0), near_degeneracy_threshold=0.1)
    with pytest.raises(ValueError, match="overlap_floor"):
        track_states(torch.eye(2).unsqueeze(0), overlap_floor=1.1)
    with pytest.raises(ValueError, match="assignment_margin_floor"):
        track_states(torch.eye(2).unsqueeze(0), assignment_margin_floor=-1.0)
    with pytest.raises(ValueError, match="degeneracy_tolerance"):
        track_states(torch.eye(2).unsqueeze(0), degeneracy_tolerance=-1.0)
    with pytest.raises(ValueError, match="cannot be smaller"):
        track_states(
            torch.eye(2).unsqueeze(0),
            energies=torch.zeros(2, 2),
            degeneracy_tolerance=0.2,
            near_degeneracy_threshold=0.1,
        )
    with pytest.raises(ValueError, match="on_ambiguous"):
        track_states(torch.eye(2).unsqueeze(0), on_ambiguous="ignore")  # type: ignore[arg-type]


def test_transformation_helpers_validate_shapes() -> None:
    transformations = torch.eye(2).expand(2, 2, 2)
    with pytest.raises(ValueError, match="rank-three"):
        align_state_frames(torch.eye(2), transformations)
    with pytest.raises(ValueError, match="incompatible"):
        align_state_frames(torch.eye(2).expand(3, 2, 2), transformations)
    with pytest.raises(ValueError, match="matrices must have shape"):
        transform_state_matrices(torch.eye(2), transformations)
    with pytest.raises(ValueError, match="incompatible"):
        transform_state_matrices(torch.eye(3).expand(2, 3, 3), transformations)
    nonunitary = torch.ones(2, 2, 2)
    with pytest.raises(ValueError, match="unitary"):
        align_state_frames(torch.eye(2).expand(2, 2, 2), nonunitary)
    with pytest.raises(ValueError, match="unitary"):
        transform_state_matrices(torch.eye(2).expand(2, 2, 2), nonunitary)


def test_transformation_helpers_promote_real_inputs_for_complex_gauges() -> None:
    phase_gauge = torch.diag(torch.tensor([1.0 + 0j, 1j])).expand(2, 2, 2)
    real_frames = torch.eye(2).expand(2, 2, 2)
    real_matrices = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).expand(2, 2, 2)

    aligned_frames = align_state_frames(real_frames, phase_gauge)
    transformed_matrices = transform_state_matrices(real_matrices, phase_gauge)

    expected = torch.tensor([[0.0, 1j], [-1j, 0.0]]).expand(2, 2, 2)
    assert aligned_frames.is_complex()
    assert transformed_matrices.is_complex()
    assert torch.equal(transformed_matrices, expected)


def test_float32_orthonormal_frames_use_dtype_aware_tolerances() -> None:
    torch.manual_seed(31)
    frames = torch.stack(
        tuple(torch.linalg.qr(torch.randn(6, 3, dtype=torch.float32)).Q for _ in range(3))
    )
    transformations = torch.stack(
        tuple(torch.linalg.qr(torch.randn(3, 3, dtype=torch.float32)).Q for _ in range(3))
    )

    overlaps = adjacent_state_overlaps(frames)
    aligned = align_state_frames(frames, transformations)

    assert overlaps.shape == (2, 3, 3)
    assert aligned.dtype == torch.float32
