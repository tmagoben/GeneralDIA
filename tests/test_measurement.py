import itertools

import numpy as np
import pytest

from generaldia.quantum.measurement import (
    measurement_basis,
    measurement_basis_changes,
    measurement_plan,
    qubit_wise_commutes,
)


def test_qwc_is_stronger_than_global_commutation() -> None:
    assert not qubit_wise_commutes("XX", "YY")
    assert qubit_wise_commutes("XZI", "IZY")
    assert qubit_wise_commutes("XZI", "XII")


def test_measurement_basi_and_local_basis_changes() -> None:
    labels = ("XZI", "IZY", "XZY")
    assert measurement_basis(labels) == "XZY"
    assert measurement_basis_changes("XZY") == (
        (0, ("H",)),
        (2, ("Sdg", "H")),
    )


def test_basis_change_gate_order_maps_x_and_y_to_z_measurements() -> None:
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    s_dagger = np.diag([1, -1j]).astype(np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)

    x_rotation = hadamard
    y_rotation = hadamard @ s_dagger  # chronological circuit order: Sdg, then H

    assert np.allclose(x_rotation.conj().T @ pauli_z @ x_rotation, pauli_x)
    assert np.allclose(y_rotation.conj().T @ pauli_z @ y_rotation, pauli_y)


def test_non_qwc_group_has_no_single_tensor_product_basis() -> None:
    with pytest.raises(ValueError, match="not mutually qubit-wise commuting"):
        measurement_basis(("XX", "YY"))


def test_paper_eq7_largest_first_reproduces_two_measurement_groups() -> None:
    terms = {
        "ZIII": 1.0,
        "ZZII": 1.0,
        "ZZZI": 1.0,
        "ZZZZ": 1.0,
        "IIXX": 1.0,
        "YIXX": 1.0,
        "YYXX": 1.0,
    }

    plan = measurement_plan(terms, method="largest_first")

    assert plan.n_measurement_settings == 2
    assert plan.n_terms == 7
    assert not plan.optimality_certified
    assert {frozenset(group.labels) for group in plan.groups} == {
        frozenset({"ZIII", "ZZII", "ZZZI", "ZZZZ"}),
        frozenset({"IIXX", "YIXX", "YYXX"}),
    }
    assert {group.basis for group in plan.groups} == {"ZZZZ", "YYXX"}


def test_paper_eq7_exact_mode_certifies_minimum_clique_cover() -> None:
    terms = {
        "ZIII": 1.0,
        "ZZII": 1.0,
        "ZZZI": 1.0,
        "ZZZZ": 1.0,
        "IIXX": 1.0,
        "YIXX": 1.0,
        "YYXX": 1.0,
    }

    plan = measurement_plan(terms, method="exact")

    assert plan.n_measurement_settings == 2
    assert plan.optimality_certified
    for group in plan.groups:
        assert all(
            qubit_wise_commutes(left, right)
            for left, right in itertools.combinations(group.labels, 2)
        )


def test_identity_term_requires_no_quantum_measurement() -> None:
    plan = measurement_plan({"II": 0.4})
    assert plan.groups == ()
    assert plan.identity_term == ("II", 0.4 + 0j)
    assert plan.n_measurement_settings == 0
    assert plan.n_terms == 1


def test_general_dia_four_state_example_has_three_nontrivial_settings() -> None:
    terms = {"II": 0.4, "XI": 0.2, "YZ": -0.15, "ZZ": 0.3}

    plan = measurement_plan(terms, method="exact")

    assert plan.identity_term == ("II", 0.4 + 0j)
    assert plan.n_measurement_settings == 3
    assert {group.basis for group in plan.groups} == {"XI", "YZ", "ZZ"}
    assert plan.optimality_certified


def test_zero_coefficients_do_not_create_measurement_settings() -> None:
    plan = measurement_plan({"II": 1.0, "XI": 0.0, "IZ": 0.5})
    assert plan.n_measurement_settings == 1
    assert plan.groups[0].labels == ("IZ",)
    assert plan.n_terms == 2


def test_exact_mode_has_an_explicit_exponential_size_guard() -> None:
    terms = {"X" * index + "Z" + "I" * (21 - index - 1): 1.0 for index in range(21)}
    with pytest.raises(ValueError, match="limited to at most 20"):
        measurement_plan(terms, method="exact")


@pytest.mark.parametrize(
    "operation",
    [
        lambda: qubit_wise_commutes("X", "II"),
        lambda: qubit_wise_commutes("A", "X"),
        lambda: measurement_basis(()),
        lambda: measurement_basis(("XI", "X")),
        lambda: measurement_basis_changes(""),
        lambda: measurement_plan({}),
        lambda: measurement_plan({"X": complex(float("nan"), 0)}),
        lambda: measurement_plan({"X": 1.0, "II": 2.0}),
        lambda: measurement_plan({"X": 1.0}, method="unknown"),
        lambda: measurement_plan({"X": 1.0}, exact_max_terms=0),
    ],
)
def test_measurement_validation(operation) -> None:
    with pytest.raises(ValueError):
        operation()
