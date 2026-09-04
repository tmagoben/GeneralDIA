import pytest

from generaldia.quantum.measurement import MeasurementGroup, MeasurementPlan
from generaldia.quantum.shots import (
    expectation_from_counts,
    group_expectations_from_counts,
    grouped_energy_from_counts,
)


def test_expectation_from_counts_uses_only_nonidentity_support() -> None:
    counts = {"000": 30, "011": 10, "110": 30, "101": 30}
    assert expectation_from_counts(counts, "ZIZ") == pytest.approx(0.2)
    assert expectation_from_counts(counts, "III") == pytest.approx(1.0)


def test_shared_counts_reconstruct_every_qwc_term() -> None:
    group = MeasurementGroup(
        terms=(("XI", 0.2 + 0j), ("IX", 0.3 + 0j), ("XX", 0.4 + 0j)),
        basis="XX",
    )
    expectations = group_expectations_from_counts({"00": 80, "11": 20}, group)
    assert expectations == pytest.approx({"XI": 0.6, "IX": 0.6, "XX": 1.0})


def test_grouped_energy_includes_identity_without_extra_shots() -> None:
    group = MeasurementGroup(
        terms=(("XI", 0.2 + 0j), ("IX", 0.3 + 0j), ("XX", 0.4 + 0j)),
        basis="XX",
    )
    plan = MeasurementPlan(
        method="largest_first",
        n_qubits=2,
        groups=(group,),
        identity_term=("II", 0.1 + 0j),
        optimality_certified=False,
    )
    result = grouped_energy_from_counts(plan, ({"00": 100},))
    assert result["energy"] == pytest.approx(1.0)
    assert result["expectations"] == pytest.approx(
        {"II": 1.0, "XI": 1.0, "IX": 1.0, "XX": 1.0}
    )
    assert result["shots_per_group"] == (100,)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: expectation_from_counts({}, "Z"),
        lambda: expectation_from_counts({"00": 1}, "Z"),
        lambda: expectation_from_counts({"2": 1}, "Z"),
        lambda: expectation_from_counts({"0": -1}, "Z"),
        lambda: expectation_from_counts({"0": 0}, "Z"),
    ],
)
def test_count_validation(operation) -> None:
    with pytest.raises(ValueError):
        operation()


def test_grouped_energy_rejects_group_count_mismatch() -> None:
    group = MeasurementGroup(terms=(("Z", 1.0 + 0j),), basis="Z")
    plan = MeasurementPlan("largest_first", 1, (group,), None, True)
    with pytest.raises(ValueError, match="one count mapping"):
        grouped_energy_from_counts(plan, ())


def test_grouped_energy_rejects_complex_coefficients() -> None:
    group = MeasurementGroup(terms=(("Z", 1.0 + 1e-3j),), basis="Z")
    plan = MeasurementPlan("largest_first", 1, (group,), None, True)
    with pytest.raises(ValueError, match="real Pauli coefficients"):
        grouped_energy_from_counts(plan, ({"0": 1},))
