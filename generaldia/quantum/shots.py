"""Shot-based expectation reconstruction for grouped Pauli measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from .measurement import MeasurementGroup, MeasurementPlan, measurement_basis


def _validate_counts(counts: Mapping[str, int], n_qubits: int) -> tuple[dict[str, int], int]:
    if not counts:
        raise ValueError("counts cannot be empty")
    normalized: dict[str, int] = {}
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str) or len(bitstring) != n_qubits:
            raise ValueError("count bitstrings must match the Pauli-label length")
        if any(bit not in "01" for bit in bitstring):
            raise ValueError("count bitstrings must contain only 0 and 1")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("count values must be nonnegative integers")
        normalized[bitstring] = int(count)
    shots = sum(normalized.values())
    if shots < 1:
        raise ValueError("counts must contain at least one shot")
    return normalized, shots


def _expectation_from_normalized_counts(
    counts: Mapping[str, int], shots: int, label: str
) -> float:
    support = tuple(index for index, character in enumerate(label) if character != "I")
    weighted = 0
    for bitstring, count in counts.items():
        parity = sum(bitstring[index] == "1" for index in support) % 2
        weighted += (-1 if parity else 1) * count
    return weighted / shots


def expectation_from_counts(counts: Mapping[str, int], label: str) -> float:
    """Estimate one Pauli expectation value from counts in its tensor-product basis."""

    measurement_basis((label,))
    normalized, shots = _validate_counts(counts, len(label))
    return _expectation_from_normalized_counts(normalized, shots, label)


def group_expectations_from_counts(
    counts: Mapping[str, int], group: MeasurementGroup
) -> dict[str, float]:
    """Reconstruct every Pauli expectation in one QWC group from a shared shot set."""

    measurement_basis(group.labels)
    normalized, shots = _validate_counts(counts, len(group.basis))
    return {
        label: _expectation_from_normalized_counts(normalized, shots, label)
        for label in group.labels
    }


def grouped_energy_from_counts(
    plan: MeasurementPlan,
    group_counts: Sequence[Mapping[str, int]],
    *,
    coefficient_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Reconstruct a real Hamiltonian expectation from one count mapping per QWC group."""

    if coefficient_tolerance < 0:
        raise ValueError("coefficient_tolerance cannot be negative")
    if len(group_counts) != len(plan.groups):
        raise ValueError("group_counts must contain one count mapping per measurement group")

    expectations: dict[str, float] = {}
    energy = 0.0
    if plan.identity_term is not None:
        label, coefficient = plan.identity_term
        coefficient = complex(coefficient)
        if abs(coefficient.imag) > coefficient_tolerance:
            raise ValueError("shot-based energy estimation requires real Pauli coefficients")
        expectations[label] = 1.0
        energy += float(coefficient.real)

    shots_per_group: list[int] = []
    for group, counts in zip(plan.groups, group_counts, strict=True):
        measurement_basis(group.labels)
        normalized, shots = _validate_counts(counts, plan.n_qubits)
        shots_per_group.append(shots)
        group_expectations = {
            label: _expectation_from_normalized_counts(normalized, shots, label)
            for label in group.labels
        }
        expectations.update(group_expectations)
        for label, coefficient in group.terms:
            coefficient = complex(coefficient)
            if abs(coefficient.imag) > coefficient_tolerance:
                raise ValueError("shot-based energy estimation requires real Pauli coefficients")
            energy += float(coefficient.real) * group_expectations[label]

    return {
        "energy": float(energy),
        "expectations": expectations,
        "shots_per_group": tuple(shots_per_group),
    }
