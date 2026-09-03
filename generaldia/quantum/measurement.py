"""Qubit-wise commuting measurement grouping for finite-state Pauli Hamiltonians."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_VALID_PAULI = frozenset("IXYZ")


@dataclass(frozen=True)
class MeasurementGroup:
    """A mutually QWC set of Pauli terms measured in one tensor-product basis."""

    terms: tuple[tuple[str, complex], ...]
    basis: str

    @property
    def labels(self) -> tuple[str, ...]:
        """Return only the Pauli labels in this group."""

        return tuple(label for label, _coefficient in self.terms)

    @property
    def basis_changes(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Return local gates needed before computational-basis measurement.

        Factor indices follow GeneralDIA's left-to-right Pauli-label order. Backends
        with a different physical-qubit numbering convention must map these factor
        indices to their wire or qubit indices explicitly.
        """

        return measurement_basis_changes(self.basis)


@dataclass(frozen=True)
class MeasurementPlan:
    """A complete QWC measurement plan for a Pauli coefficient mapping."""

    method: str
    n_qubits: int
    groups: tuple[MeasurementGroup, ...]
    identity_term: tuple[str, complex] | None
    optimality_certified: bool

    @property
    def n_measurement_settings(self) -> int:
        """Return the number of nontrivial tensor-product measurement settings."""

        return len(self.groups)

    @property
    def n_terms(self) -> int:
        """Return the number of Pauli terms represented by the plan."""

        return sum(len(group.terms) for group in self.groups) + (self.identity_term is not None)


def _validate_label(label: str) -> None:
    invalid_character = isinstance(label, str) and any(
        character not in _VALID_PAULI for character in label
    )
    if not isinstance(label, str) or not label or invalid_character:
        raise ValueError("Pauli labels must contain one or more characters from I, X, Y, Z")


def _qwc_validated(left: str, right: str) -> bool:
    return all(
        left_factor == "I" or right_factor == "I" or left_factor == right_factor
        for left_factor, right_factor in zip(left, right, strict=True)
    )


def qubit_wise_commutes(left: str, right: str) -> bool:
    """Return whether two Pauli words commute in every one-qubit subspace.

    Two factors are compatible when they are equal or at least one of them is the
    identity. Qubit-wise commutativity is stronger than ordinary operator
    commutativity and is precisely the condition needed for simultaneous measurement
    with only single-qubit basis changes.
    """

    _validate_label(left)
    _validate_label(right)
    if len(left) != len(right):
        raise ValueError("Pauli labels must have the same length")
    return _qwc_validated(left, right)


def measurement_basis(labels: Sequence[str]) -> str:
    """Return the unique tensor-product basis compatible with a mutually QWC group."""

    if not labels:
        raise ValueError("labels cannot be empty")
    for label in labels:
        _validate_label(label)
    n_qubits = len(labels[0])
    if any(len(label) != n_qubits for label in labels):
        raise ValueError("Pauli labels must have a consistent length")

    basis = []
    for factor_index in range(n_qubits):
        non_identity = {label[factor_index] for label in labels if label[factor_index] != "I"}
        if len(non_identity) > 1:
            raise ValueError("labels are not mutually qubit-wise commuting")
        basis.append(next(iter(non_identity), "I"))
    return "".join(basis)


def measurement_basis_changes(basis: str) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Return local pre-measurement gates for a tensor-product Pauli basis.

    ``X`` uses a Hadamard gate, ``Y`` uses ``Sdg`` followed by Hadamard, and ``Z``
    requires no basis change. Identity factors are ignored because they are not
    measured. Returned indices refer to left-to-right positions in the Pauli label.
    """

    _validate_label(basis)
    changes: list[tuple[int, tuple[str, ...]]] = []
    for factor_index, character in enumerate(basis):
        if character == "X":
            changes.append((factor_index, ("H",)))
        elif character == "Y":
            changes.append((factor_index, ("Sdg", "H")))
    return tuple(changes)


def _conflict_adjacency(labels: Sequence[str]) -> tuple[frozenset[int], ...]:
    adjacency = [set() for _label in labels]
    for left_index, left in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            if not _qwc_validated(left, labels[right_index]):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def _largest_first_coloring(
    labels: Sequence[str], adjacency: Sequence[frozenset[int]]
) -> tuple[int, ...]:
    if not labels:
        return ()
    order = sorted(range(len(labels)), key=lambda index: (-len(adjacency[index]), labels[index]))
    colors = [-1] * len(labels)
    for vertex in order:
        forbidden = {colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
    return tuple(colors)


def _exact_coloring(labels: Sequence[str], adjacency: Sequence[frozenset[int]]) -> tuple[int, ...]:
    """Return an exact minimum coloring using branch-and-bound DSATUR search."""

    if not labels:
        return ()

    initial = _largest_first_coloring(labels, adjacency)
    best_colors = list(initial)
    best_count = max(initial) + 1
    colors = [-1] * len(labels)

    def saturation(vertex: int) -> int:
        return len({colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0})

    def search(colored_count: int, used_colors: int) -> None:
        nonlocal best_colors, best_count

        if colored_count == len(labels):
            if used_colors < best_count:
                best_count = used_colors
                best_colors = colors.copy()
            return
        if used_colors >= best_count:
            return

        uncolored = [index for index, color in enumerate(colors) if color < 0]
        vertex = max(
            uncolored,
            key=lambda index: (
                saturation(index),
                len(adjacency[index]),
                labels[index],
            ),
        )
        forbidden = {colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}

        for color in range(used_colors):
            if color in forbidden:
                continue
            colors[vertex] = color
            search(colored_count + 1, used_colors)
            colors[vertex] = -1

        new_color = used_colors
        if new_color + 1 < best_count:
            colors[vertex] = new_color
            search(colored_count + 1, used_colors + 1)
            colors[vertex] = -1

    search(0, 0)
    return tuple(best_colors)


def _validate_terms(terms: Mapping[str, complex]) -> tuple[int, tuple[tuple[str, complex], ...]]:
    if not terms:
        raise ValueError("terms cannot be empty")

    validated: list[tuple[str, complex]] = []
    n_qubits: int | None = None
    for label, coefficient in terms.items():
        _validate_label(label)
        if n_qubits is None:
            n_qubits = len(label)
        elif len(label) != n_qubits:
            raise ValueError("Pauli labels must have a consistent length")
        value = complex(coefficient)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Pauli coefficients must be finite")
        if value != 0:
            validated.append((label, value))

    if n_qubits is None:
        raise ValueError("terms cannot be empty")
    return n_qubits, tuple(validated)


def measurement_plan(
    terms: Mapping[str, complex],
    *,
    method: str = "largest_first",
    exact_max_terms: int = 20,
) -> MeasurementPlan:
    """Group Pauli terms into QWC tensor-product measurement settings.

    Parameters
    ----------
    terms
        Nonempty Pauli coefficient mapping, typically returned by
        :func:`generaldia.quantum.pauli.matrix_to_pauli`.
    method
        ``"largest_first"`` uses the paper's Largest-First greedy coloring heuristic
        on the QWC conflict graph. ``"exact"`` uses a branch-and-bound DSATUR search
        and certifies a minimum clique cover for sufficiently small term sets.
    exact_max_terms
        Maximum number of non-identity terms accepted by ``method="exact"``. The
        exact graph-coloring problem is exponential in the worst case, so this guard
        prevents accidental use on large Hamiltonians.

    Notes
    -----
    The all-identity Pauli term contributes a known constant and therefore requires no
    quantum measurement. It is returned separately as ``identity_term``.
    """

    if method not in {"largest_first", "exact"}:
        raise ValueError("method must be 'largest_first' or 'exact'")
    invalid_exact_limit = (
        not isinstance(exact_max_terms, int)
        or isinstance(exact_max_terms, bool)
        or exact_max_terms < 1
    )
    if invalid_exact_limit:
        raise ValueError("exact_max_terms must be a positive integer")

    n_qubits, validated_terms = _validate_terms(terms)
    identity_label = "I" * n_qubits
    identity_term = next((term for term in validated_terms if term[0] == identity_label), None)
    non_identity_terms = tuple(term for term in validated_terms if term[0] != identity_label)
    labels = tuple(label for label, _coefficient in non_identity_terms)

    if method == "exact" and len(labels) > exact_max_terms:
        raise ValueError(
            f"exact grouping is limited to at most {exact_max_terms} non-identity terms; "
            "use method='largest_first' for larger Hamiltonians"
        )

    adjacency = _conflict_adjacency(labels)
    if method == "largest_first":
        colors = _largest_first_coloring(labels, adjacency)
    else:
        colors = _exact_coloring(labels, adjacency)

    groups: list[MeasurementGroup] = []
    if colors:
        for color in range(max(colors) + 1):
            grouped_terms = tuple(
                non_identity_terms[index]
                for index, assigned in enumerate(colors)
                if assigned == color
            )
            group_labels = tuple(label for label, _coefficient in grouped_terms)
            groups.append(MeasurementGroup(grouped_terms, measurement_basis(group_labels)))

    return MeasurementPlan(
        method=method,
        n_qubits=n_qubits,
        groups=tuple(groups),
        identity_term=identity_term,
        optimality_certified=method == "exact" or len(labels) <= 1,
    )
