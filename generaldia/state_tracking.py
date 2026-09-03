"""Connected-path electronic-state tracking with explicit gauge diagnostics.

The tracker consumes adjacent state-overlap matrices rather than assuming that raw
electronic-state vectors at different molecular geometries share an ambient basis.
This keeps the electronic-structure boundary explicit: model eigenvectors in one
fixed finite-state basis can be converted to overlaps locally, while external
backends should provide physically meaningful cross-geometry overlaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class SubspaceMatch:
    """One matched state or degenerate subspace at a path transition.

    ``previous_indices`` use the tracked ordering at the earlier geometry.
    ``candidate_indices`` use the raw ordering at the later geometry. Principal
    overlaps are singular values of the corresponding cross-overlap block. Inside a
    multi-state match, index-by-index pairing is bookkeeping rather than a physical
    assignment of individual degenerate roots.
    """

    previous_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    principal_overlaps: Tensor

    @property
    def is_degenerate(self) -> bool:
        """Whether this match represents a multi-state subspace."""

        return len(self.previous_indices) > 1


@dataclass(frozen=True)
class StateTrackingStep:
    """Diagnostics and transformation for one adjacent-geometry transition."""

    start_index: int
    permutation: tuple[int, ...]
    transformation: Tensor
    aligned_overlap: Tensor
    matches: tuple[SubspaceMatch, ...]
    minimum_overlap: float
    assignment_margin: float
    near_degenerate_pairs: tuple[tuple[int, int, float], ...]
    ambiguous: bool
    reasons: tuple[str, ...]

    @property
    def end_index(self) -> int:
        """Index of the later geometry in this transition."""

        return self.start_index + 1

    @property
    def degenerate_blocks(self) -> tuple[tuple[int, ...], ...]:
        """Tracked-index blocks aligned as subspaces rather than individual roots."""

        return tuple(match.previous_indices for match in self.matches if match.is_degenerate)


@dataclass(frozen=True)
class StateTrackingResult:
    """Path transformations and transition-level tracking diagnostics.

    For raw column frames ``U[k]``, the tracked frames are
    ``U[k] @ transformations[k]``. The first transformation is the identity, so
    the initial frame defines the path gauge.
    """

    transformations: Tensor
    steps: tuple[StateTrackingStep, ...]

    @property
    def aligned_overlaps(self) -> Tensor:
        """Adjacent overlaps after applying the tracked transformations."""

        return torch.stack(tuple(step.aligned_overlap for step in self.steps))

    @property
    def ambiguous_steps(self) -> tuple[int, ...]:
        """Starting indices of transitions that were recorded as ambiguous."""

        return tuple(step.start_index for step in self.steps if step.ambiguous)


class AmbiguousStateTrackingError(ValueError):
    """Raised when a path transition does not define stable state identities."""

    def __init__(self, diagnostic: StateTrackingStep) -> None:
        self.diagnostic = diagnostic
        reasons = "; ".join(diagnostic.reasons)
        super().__init__(
            f"ambiguous state tracking at transition {diagnostic.start_index}"
            f"->{diagnostic.end_index}: {reasons}"
        )


def adjacent_state_overlaps(frames: Tensor) -> Tensor:
    """Return overlaps between adjacent orthonormal column frames.

    Parameters
    ----------
    frames
        Tensor with shape ``(n_geometries, dimension, n_states)``. All frames must
        use one compatible ambient basis. For electronic-structure calculations
        whose orbital bases change with geometry, pass externally calculated
        cross-geometry overlaps directly to :func:`track_states` instead.
    """

    frames = torch.as_tensor(frames)
    if frames.ndim != 3 or frames.shape[0] < 2 or frames.shape[1] < frames.shape[2]:
        raise ValueError(
            "frames must have shape (n_geometries, dimension, n_states) with "
            "n_geometries >= 2 and dimension >= n_states"
        )
    if not (frames.is_floating_point() or frames.is_complex()):
        frames = frames.to(torch.get_default_dtype())
    if not torch.isfinite(frames).all():
        raise ValueError("frames must contain finite values")
    gram = frames.mH @ frames
    identity = torch.eye(frames.shape[-1], dtype=frames.dtype, device=frames.device).expand_as(gram)
    absolute_tolerance, relative_tolerance = _matrix_tolerances(frames)
    if not torch.allclose(gram, identity, atol=absolute_tolerance, rtol=relative_tolerance):
        raise ValueError("frame columns must be orthonormal at every geometry")
    return frames[:-1].mH @ frames[1:]


def align_state_frames(frames: Tensor, transformations: Tensor) -> Tensor:
    """Apply path-tracking transformations to raw column frames."""

    frames = torch.as_tensor(frames)
    transformations = torch.as_tensor(transformations, device=frames.device)
    if not (frames.is_floating_point() or frames.is_complex()):
        frames = frames.to(torch.get_default_dtype())
    if not (transformations.is_floating_point() or transformations.is_complex()):
        transformations = transformations.to(torch.get_default_dtype())
    if frames.ndim != 3 or transformations.ndim != 3:
        raise ValueError("frames and transformations must both be rank-three tensors")
    if (
        frames.shape[0] != transformations.shape[0]
        or frames.shape[-1] != transformations.shape[-1]
        or transformations.shape[-2] != transformations.shape[-1]
    ):
        raise ValueError("frames and transformations have incompatible path or state shapes")
    dtype = torch.promote_types(frames.dtype, transformations.dtype)
    frames = frames.to(dtype=dtype)
    transformations = transformations.to(dtype=dtype)
    if not torch.isfinite(frames).all():
        raise ValueError("frames must contain finite values")
    _require_unitary_transformations(transformations)
    return frames @ transformations


def transform_state_matrices(matrices: Tensor, transformations: Tensor) -> Tensor:
    """Transform path-indexed state matrices as ``W^dagger A W``.

    ``matrices`` must have shape ``(n_geometries, ..., n_states, n_states)``.
    Intermediate axes may represent atoms, Cartesian components, or observables.
    """

    matrices = torch.as_tensor(matrices)
    transformations = torch.as_tensor(transformations, device=matrices.device)
    if not (matrices.is_floating_point() or matrices.is_complex()):
        matrices = matrices.to(torch.get_default_dtype())
    if not (transformations.is_floating_point() or transformations.is_complex()):
        transformations = transformations.to(torch.get_default_dtype())
    if matrices.ndim < 3 or transformations.ndim != 3:
        raise ValueError(
            "matrices must have shape (n_geometries, ..., n_states, n_states) and "
            "transformations must have shape (n_geometries, n_states, n_states)"
        )
    n_geometries, n_states, final_states = transformations.shape
    if n_states != final_states:
        raise ValueError("transformations must be square on their final two axes")
    if matrices.shape[0] != n_geometries or matrices.shape[-2:] != (n_states, n_states):
        raise ValueError("matrices and transformations have incompatible path or state shapes")
    dtype = torch.promote_types(matrices.dtype, transformations.dtype)
    matrices = matrices.to(dtype=dtype)
    transformations = transformations.to(dtype=dtype)
    if not torch.isfinite(matrices).all():
        raise ValueError("matrices must contain finite values")
    _require_unitary_transformations(transformations)
    singleton_axes = (1,) * (matrices.ndim - 3)
    right = transformations.reshape((n_geometries, *singleton_axes, n_states, n_states))
    left = right.mH
    return left @ matrices @ right


def track_states(
    overlaps: Tensor,
    *,
    energies: Tensor | None = None,
    overlap_floor: float = 0.5,
    assignment_margin_floor: float = 1e-6,
    degeneracy_tolerance: float = 1e-8,
    near_degeneracy_threshold: float | None = None,
    on_ambiguous: Literal["raise", "record"] = "raise",
) -> StateTrackingResult:
    """Track electronic-state character along an ordered geometry path.

    ``overlaps[k, i, j]`` is the cross-geometry overlap between raw state ``i`` at
    geometry ``k`` and raw state ``j`` at geometry ``k + 1``. At every transition,
    the function performs a maximum-overlap assignment. Singleton states receive a
    complex phase correction. Energy-degenerate blocks present at both endpoints are
    matched by their principal overlaps and aligned with unitary Procrustes.
    Without ``energies``, every state is treated as a singleton and automatic
    degenerate-subspace handling is disabled.

    Ambiguous assignments raise by default. Set ``on_ambiguous="record"`` only when
    the caller will inspect ``result.ambiguous_steps`` and transition diagnostics.
    ``degeneracy_tolerance`` and ``near_degeneracy_threshold`` use the same absolute
    energy units as ``energies``.
    """

    overlaps = _validated_overlaps(overlaps)
    _validate_thresholds(
        overlap_floor,
        assignment_margin_floor,
        degeneracy_tolerance,
        near_degeneracy_threshold,
        on_ambiguous,
    )
    n_transitions, n_states, _ = overlaps.shape
    path_energies = _validated_energies(
        energies,
        n_geometries=n_transitions + 1,
        n_states=n_states,
        device=overlaps.device,
    )
    if near_degeneracy_threshold is not None and path_energies is None:
        raise ValueError("near_degeneracy_threshold requires energies")

    identity = torch.eye(n_states, dtype=overlaps.dtype, device=overlaps.device)
    transformations = [identity]
    diagnostics: list[StateTrackingStep] = []

    for transition, raw_overlap in enumerate(overlaps):
        previous_transformation = transformations[-1]
        effective_overlap = previous_transformation.mH @ raw_overlap
        reasons: list[str] = []

        if path_energies is None:
            previous_groups = _singleton_partition(n_states)
            candidate_groups = _singleton_partition(n_states)
            previous_tracked_energies = None
        else:
            previous_tracked_energies = _tracked_energies(
                path_energies[transition], previous_transformation
            )
            previous_groups = _energy_partition(previous_tracked_energies, degeneracy_tolerance)
            candidate_groups = _energy_partition(
                path_energies[transition + 1], degeneracy_tolerance
            )
            if sorted(map(len, previous_groups)) != sorted(map(len, candidate_groups)):
                reasons.append(
                    "degenerate-subspace dimensions split or merge across the transition"
                )
                previous_groups = _singleton_partition(n_states)
                candidate_groups = _singleton_partition(n_states)

        group_pairs, assignment_margin = _match_group_partitions(
            effective_overlap, previous_groups, candidate_groups
        )
        next_transformation = torch.zeros_like(identity)
        permutation = [-1] * n_states
        matches: list[SubspaceMatch] = []

        for previous_group, candidate_group in group_pairs:
            rows = torch.tensor(previous_group, dtype=torch.long, device=overlaps.device)
            columns = torch.tensor(candidate_group, dtype=torch.long, device=overlaps.device)
            block_overlap = effective_overlap[rows][:, columns]
            left, singular_values, right_h = torch.linalg.svd(block_overlap.mH, full_matrices=False)
            rotation = left @ right_h
            next_transformation[columns[:, None], rows[None, :]] = rotation
            for tracked_index, raw_index in zip(previous_group, candidate_group, strict=True):
                permutation[tracked_index] = raw_index
            matches.append(
                SubspaceMatch(
                    previous_indices=previous_group,
                    candidate_indices=candidate_group,
                    principal_overlaps=singular_values.detach().clone(),
                )
            )

        aligned_overlap = effective_overlap @ next_transformation
        minimum_overlap = min(float(match.principal_overlaps.min()) for match in matches)
        if minimum_overlap <= overlap_floor:
            reasons.append(
                f"minimum principal overlap {minimum_overlap:.6g} is not above "
                f"overlap_floor={overlap_floor:.6g}"
            )
        if assignment_margin <= assignment_margin_floor:
            reasons.append(
                f"assignment margin {assignment_margin:.6g} is not above "
                f"assignment_margin_floor={assignment_margin_floor:.6g}"
            )

        near_pairs: tuple[tuple[int, int, float], ...] = ()
        if near_degeneracy_threshold is not None and path_energies is not None:
            if previous_tracked_energies is None:
                raise RuntimeError("internal error: previous tracked energies were not calculated")
            candidate_tracked_energies = _tracked_energies(
                path_energies[transition + 1], next_transformation
            )
            near_pairs = _near_degenerate_pairs(
                previous_tracked_energies,
                candidate_tracked_energies,
                degeneracy_tolerance=degeneracy_tolerance,
                threshold=near_degeneracy_threshold,
            )
            if near_pairs:
                formatted = ", ".join(
                    f"({left}, {right}; gap={gap:.6g})" for left, right, gap in near_pairs
                )
                reasons.append(f"near-degenerate tracked pairs require review: {formatted}")

        diagnostic = StateTrackingStep(
            start_index=transition,
            permutation=tuple(permutation),
            transformation=next_transformation.detach().clone(),
            aligned_overlap=aligned_overlap.detach().clone(),
            matches=tuple(matches),
            minimum_overlap=minimum_overlap,
            assignment_margin=assignment_margin,
            near_degenerate_pairs=near_pairs,
            ambiguous=bool(reasons),
            reasons=tuple(reasons),
        )
        if diagnostic.ambiguous and on_ambiguous == "raise":
            raise AmbiguousStateTrackingError(diagnostic)
        transformations.append(next_transformation)
        diagnostics.append(diagnostic)

    return StateTrackingResult(torch.stack(transformations), tuple(diagnostics))


def _validated_overlaps(overlaps: Tensor) -> Tensor:
    overlaps = torch.as_tensor(overlaps)
    if (
        overlaps.ndim != 3
        or overlaps.shape[0] < 1
        or overlaps.shape[1] < 1
        or overlaps.shape[1] != overlaps.shape[2]
    ):
        raise ValueError(
            "overlaps must have shape (n_geometries - 1, n_states, n_states) with "
            "at least one transition"
        )
    if not (overlaps.is_floating_point() or overlaps.is_complex()):
        overlaps = overlaps.to(torch.get_default_dtype())
    if not torch.isfinite(overlaps).all():
        raise ValueError("overlaps must contain finite values")
    largest_singular_value = torch.linalg.svdvals(overlaps).amax()
    absolute_tolerance, _ = _matrix_tolerances(overlaps)
    if float(largest_singular_value) > 1.0 + absolute_tolerance:
        raise ValueError("each overlap matrix must be a contraction with singular values <= 1")
    return overlaps


def _validated_energies(
    energies: Tensor | None,
    *,
    n_geometries: int,
    n_states: int,
    device: torch.device,
) -> Tensor | None:
    if energies is None:
        return None
    energies = torch.as_tensor(energies, device=device)
    if energies.is_complex():
        raise ValueError("energies must be real")
    if not energies.is_floating_point():
        energies = energies.to(torch.get_default_dtype())
    if energies.shape != (n_geometries, n_states):
        raise ValueError(f"energies must have shape {(n_geometries, n_states)}")
    if not torch.isfinite(energies).all():
        raise ValueError("energies must contain finite values")
    return energies


def _validate_thresholds(
    overlap_floor: float,
    assignment_margin_floor: float,
    degeneracy_tolerance: float,
    near_degeneracy_threshold: float | None,
    on_ambiguous: str,
) -> None:
    if not 0 <= overlap_floor <= 1:
        raise ValueError("overlap_floor must lie between zero and one")
    if assignment_margin_floor < 0:
        raise ValueError("assignment_margin_floor cannot be negative")
    if degeneracy_tolerance < 0:
        raise ValueError("degeneracy_tolerance cannot be negative")
    if near_degeneracy_threshold is not None and near_degeneracy_threshold < degeneracy_tolerance:
        raise ValueError("near_degeneracy_threshold cannot be smaller than degeneracy_tolerance")
    if on_ambiguous not in {"raise", "record"}:
        raise ValueError('on_ambiguous must be either "raise" or "record"')


def _singleton_partition(n_states: int) -> tuple[tuple[int, ...], ...]:
    return tuple((index,) for index in range(n_states))


def _energy_partition(energies: Tensor, tolerance: float) -> tuple[tuple[int, ...], ...]:
    order = sorted(range(energies.numel()), key=lambda index: (float(energies[index]), index))
    groups: list[list[int]] = [[order[0]]]
    group_minimum = float(energies[order[0]])
    for index in order[1:]:
        if float(energies[index]) - group_minimum <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
            group_minimum = float(energies[index])
    return tuple(sorted((tuple(sorted(group)) for group in groups), key=lambda group: group[0]))


def _require_unitary_transformations(transformations: Tensor) -> None:
    if not torch.isfinite(transformations).all():
        raise ValueError("transformations must contain finite values")
    gram = transformations.mH @ transformations
    identity = torch.eye(
        transformations.shape[-1],
        dtype=transformations.dtype,
        device=transformations.device,
    ).expand_as(gram)
    absolute_tolerance, relative_tolerance = _matrix_tolerances(transformations)
    if not torch.allclose(gram, identity, atol=absolute_tolerance, rtol=relative_tolerance):
        raise ValueError("transformations must be unitary at every geometry")


def _matrix_tolerances(values: Tensor) -> tuple[float, float]:
    epsilon = torch.finfo(values.real.dtype).eps
    return max(1e-8, 10 * epsilon), max(1e-7, 10 * epsilon)


def _tracked_energies(raw_energies: Tensor, transformation: Tensor) -> Tensor:
    energy_matrix = torch.diag(raw_energies).to(dtype=transformation.dtype)
    tracked_matrix = transformation.mH @ energy_matrix @ transformation
    return torch.diagonal(tracked_matrix).real


def _match_group_partitions(
    effective_overlap: Tensor,
    previous_groups: tuple[tuple[int, ...], ...],
    candidate_groups: tuple[tuple[int, ...], ...],
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], float]:
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    margins: list[float] = []
    sizes = sorted({len(group) for group in previous_groups})
    for size in sizes:
        previous = tuple(group for group in previous_groups if len(group) == size)
        candidates = tuple(group for group in candidate_groups if len(group) == size)
        if len(previous) != len(candidates):
            raise RuntimeError("internal error: incompatible group partitions")
        scores = torch.empty(
            (len(previous), len(candidates)),
            dtype=effective_overlap.real.dtype,
            device=effective_overlap.device,
        )
        for row, previous_group in enumerate(previous):
            previous_indices = torch.tensor(
                previous_group, dtype=torch.long, device=effective_overlap.device
            )
            for column, candidate_group in enumerate(candidates):
                candidate_indices = torch.tensor(
                    candidate_group, dtype=torch.long, device=effective_overlap.device
                )
                block = effective_overlap[previous_indices][:, candidate_indices]
                scores[row, column] = torch.linalg.svdvals(block).mean()
        assignment, margin = _maximum_assignment(scores)
        margins.append(margin)
        pairs.extend((previous[row], candidates[column]) for row, column in enumerate(assignment))
    pairs.sort(key=lambda pair: pair[0][0])
    finite_margins = [margin for margin in margins if margin != float("inf")]
    return pairs, min(finite_margins, default=float("inf"))


def _maximum_assignment(scores: Tensor) -> tuple[tuple[int, ...], float]:
    assignment, best_score = _hungarian_maximize(scores)
    if scores.shape[0] == 1:
        return assignment, float("inf")
    alternatives = []
    for row, column in enumerate(assignment):
        _, alternative_score = _hungarian_maximize(scores, forbidden=(row, column))
        alternatives.append(alternative_score)
    second_score = max(alternatives)
    margin = max(0.0, (best_score - second_score) / scores.shape[0])
    return assignment, margin


def _hungarian_maximize(
    scores: Tensor, forbidden: tuple[int, int] | None = None
) -> tuple[tuple[int, ...], float]:
    """Return a deterministic maximum-weight square assignment."""

    values = scores.detach().to(dtype=torch.float64, device="cpu").tolist()
    n_rows = len(values)
    if n_rows == 0 or any(len(row) != n_rows for row in values):
        raise ValueError("assignment scores must form a nonempty square matrix")
    forbidden_cost = float(n_rows + 2)
    potentials_rows = [0.0] * (n_rows + 1)
    potentials_columns = [0.0] * (n_rows + 1)
    matched_row = [0] * (n_rows + 1)
    predecessor = [0] * (n_rows + 1)

    for row in range(1, n_rows + 1):
        matched_row[0] = row
        column = 0
        minimum = [float("inf")] * (n_rows + 1)
        used = [False] * (n_rows + 1)
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, n_rows + 1):
                if used[candidate_column]:
                    continue
                edge = (active_row - 1, candidate_column - 1)
                cost = (
                    forbidden_cost
                    if forbidden == edge
                    else -values[active_row - 1][candidate_column - 1]
                )
                reduced = cost - potentials_rows[active_row] - potentials_columns[candidate_column]
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(n_rows + 1):
                if used[candidate_column]:
                    potentials_rows[matched_row[candidate_column]] += delta
                    potentials_columns[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * n_rows
    for column in range(1, n_rows + 1):
        assignment[matched_row[column] - 1] = column - 1
    if forbidden is not None and assignment[forbidden[0]] == forbidden[1]:
        raise RuntimeError("internal error: forbidden assignment edge was selected")
    score = sum(values[row][column] for row, column in enumerate(assignment))
    return tuple(assignment), score


def _near_degenerate_pairs(
    previous_energies: Tensor,
    candidate_energies: Tensor,
    *,
    degeneracy_tolerance: float,
    threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    pairs = []
    for left in range(previous_energies.numel()):
        for right in range(left + 1, previous_energies.numel()):
            previous_gap = float(torch.abs(previous_energies[left] - previous_energies[right]))
            candidate_gap = float(torch.abs(candidate_energies[left] - candidate_energies[right]))
            minimum_gap = min(previous_gap, candidate_gap)
            both_degenerate = (
                previous_gap <= degeneracy_tolerance and candidate_gap <= degeneracy_tolerance
            )
            if not both_degenerate and minimum_gap <= threshold:
                pairs.append((left, right, minimum_gap))
    return tuple(pairs)
