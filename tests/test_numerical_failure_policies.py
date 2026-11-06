"""Numerical failure-policy regressions for scientific public interfaces.

These tests distinguish deterministic valid inputs from invalid NaN/Inf data.
No fallback may silently coerce complex geometry to real, truncate an electronic
state count, or substitute an unconverged derivative for physical evidence.
"""

import math

import numpy as np
import pytest
import torch

from generaldia.analytic import TwoStateAvoidedCrossing
from generaldia.dataset import MolecularDataset, MolecularSample
from generaldia.gauge import align_phase, unitary_procrustes
from generaldia.matrix import complex_hermitian_from_parts, unpack_real_symmetric
from generaldia.quantum.pauli import matrix_to_pauli, pauli_to_matrix
from generaldia.training import TrainingConfig


@pytest.mark.parametrize("n", [True, 1.0, 1.5, float("nan"), "2"])
def test_packed_matrix_does_not_truncate_or_coerce_state_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        unpack_real_symmetric(torch.ones(3), n)


def test_packed_matrix_accepts_numpy_integer_and_conserves_spectrum():
    upper_triangle = torch.tensor([2.0, -0.3, -1.0])
    matrix = unpack_real_symmetric(upper_triangle, np.int64(2))
    torch.testing.assert_close(matrix, matrix.mH)
    assert torch.trace(matrix).item() == pytest.approx(1.0)
    assert torch.linalg.eigvalsh(matrix).sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_real_symmetric_matrix_rejects_invalid_coefficients(value):
    with pytest.raises(ValueError, match="finite"):
        unpack_real_symmetric(torch.tensor([1.0, value, 2.0]), 2)


@pytest.mark.parametrize("component", ["diagonal", "real", "imag"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_complex_hermitian_constructor_rejects_invalid_coefficients(component, value):
    diagonal = torch.tensor([1.0, 2.0], dtype=torch.float64)
    real = torch.tensor([0.1], dtype=torch.float64)
    imag = torch.tensor([0.2], dtype=torch.float64)
    if component == "diagonal":
        diagonal[0] = value
    elif component == "real":
        real[0] = value
    else:
        imag[0] = value
    with pytest.raises(ValueError, match="finite"):
        complex_hermitian_from_parts(diagonal, real, imag)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slope": float("nan")},
        {"slope": float("inf")},
        {"coupling": float("nan")},
        {"coupling": float("inf")},
    ],
)
def test_analytic_reference_rejects_nonfinite_parameters(kwargs):
    with pytest.raises(ValueError, match="finite"):
        TwoStateAvoidedCrossing(**kwargs)


@pytest.mark.parametrize("method", ["hamiltonian", "exact_energies"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_analytic_reference_rejects_nonfinite_coordinate_in_both_paths(method, value):
    analytic = TwoStateAvoidedCrossing()
    with pytest.raises(ValueError, match="finite"):
        getattr(analytic, method)(torch.tensor([0.1, value], dtype=torch.float64))


@pytest.mark.parametrize("method", ["hamiltonian", "exact_energies"])
def test_analytic_reference_rejects_complex_coordinate_without_losing_imaginary_part(method):
    analytic = TwoStateAvoidedCrossing()
    with pytest.raises(ValueError, match="coordinate must be real"):
        getattr(analytic, method)(torch.tensor([0.1 + 0.2j], dtype=torch.complex128))


def test_analytic_reference_exact_eigenvalues_agree_at_negative_positive_and_zero_x():
    analytic = TwoStateAvoidedCrossing(slope=0.09, coupling=0.015)
    points = torch.tensor([-2.0, -0.3, 0.0, 0.3, 2.0], dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(analytic.hamiltonian(points))
    torch.testing.assert_close(eigenvalues, analytic.exact_energies(points), atol=1e-13, rtol=0)
    assert torch.all(eigenvalues[:, 1] - eigenvalues[:, 0] >= 0.03)


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), -float("inf"), -0.1, 1.0])
def test_phase_alignment_rejects_unusable_overlap_floor(floor):
    with pytest.raises(ValueError, match="overlap_floor"):
        align_phase(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), overlap_floor=floor)


@pytest.mark.parametrize(
    "reference,candidate",
    [
        ([float("nan"), 1.0], [1.0, 0.0]),
        ([1.0, float("inf")], [1.0, 0.0]),
        ([1.0, 0.0], [float("nan"), 0.0]),
        ([1.0, 0.0], [1.0, -float("inf")]),
    ],
)
def test_phase_alignment_fails_closed_on_nonfinite_physical_state(reference, candidate):
    with pytest.raises(ValueError, match="finite vectors"):
        align_phase(torch.tensor(reference), torch.tensor(candidate))


def test_phase_alignment_complex_state_preserves_observables_and_amplitude():
    reference = torch.tensor([0.2 + 0.3j, -0.8j], dtype=torch.complex128)
    candidate = reference * torch.exp(torch.tensor(1.1j, dtype=torch.complex128))
    result = align_phase(reference, candidate)
    torch.testing.assert_close(result, reference, atol=1e-13, rtol=0)
    torch.testing.assert_close(torch.vdot(result, result), torch.vdot(candidate, candidate))


@pytest.mark.parametrize("fractions", [(0.6, float("nan"), 0.4), (0.6, float("inf"), 0.4)])
def test_dataset_split_rejects_nonfinite_probabilities(fractions):
    sample = MolecularSample([1, 1], torch.zeros(2, 3), [0.0, 1.0])
    dataset = MolecularDataset([sample, sample, sample, sample])
    with pytest.raises(ValueError, match="finite"):
        dataset.split(fractions)


def test_dataset_split_stably_normalizes_large_finite_weights():
    sample = MolecularSample([1, 1], torch.zeros(2, 3), [0.0, 1.0])
    dataset = MolecularDataset([sample, sample, sample])
    partitions = dataset.split((1e308, 1e308, 1e308), seed=7)
    assert tuple(len(split) for split in partitions) == (1, 1, 1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"gradient_clip_norm": float("nan")}, "gradient_clip_norm"),
        ({"gradient_clip_norm": float("inf")}, "gradient_clip_norm"),
    ],
)
def test_training_rejects_nonfinite_optimizer_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_training_explicit_unclipped_fallback_remains_supported():
    options = TrainingConfig(gradient_clip_norm=None, learning_rate=1e-3, epochs=1)
    assert options.gradient_clip_norm is None
    assert options.learning_rate == pytest.approx(1e-3)


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), -0.1])
def test_pauli_expansion_rejects_nonfinite_or_negative_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        matrix_to_pauli(np.diag([1.0, 2.0]), tol=tol)


def test_pauli_decomposition_matches_exact_nontrivial_hermitian_matrix():
    source = np.array([[1.2, 0.3 - 0.4j], [0.3 + 0.4j, -0.6]], dtype=np.complex128)
    expanded = matrix_to_pauli(source, tol=1e-14)
    reconstructed = pauli_to_matrix(expanded)
    np.testing.assert_allclose(reconstructed, source, atol=1e-13, rtol=0)
    assert math.isclose(
        np.linalg.eigvalsh(reconstructed)[0],
        np.linalg.eigvalsh(source)[0],
        abs_tol=1e-13,
    )


def test_analytic_reference_uses_stable_hypot_for_extreme_finite_coordinates():
    model = TwoStateAvoidedCrossing(slope=0.09, coupling=0.015)
    coordinates = torch.tensor([-1e200, 1e200], dtype=torch.float64)
    energies = model.exact_energies(coordinates)
    assert torch.isfinite(energies).all()
    torch.testing.assert_close(
        energies[:, 1],
        torch.full((2,), 0.09e200, dtype=torch.float64),
        rtol=1e-14,
        atol=0,
    )


def test_analytic_reference_fails_closed_on_genuine_floating_point_overflow():
    model = TwoStateAvoidedCrossing(slope=1e200, coupling=1.0)
    with pytest.raises(ValueError, match="overflowed"):
        model.exact_energies(torch.tensor([1e200], dtype=torch.float64))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"epochs": 1.5}, "epochs"),
        ({"epochs": True}, "epochs"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.1}, "seed"),
        ({"report_every": 1.5}, "report_every"),
        ({"report_every": True}, "report_every"),
    ],
)
def test_training_rejects_invalid_integer_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_training_accepts_numpy_integer_reproducibility_controls():
    config = TrainingConfig(epochs=np.int64(2), seed=np.int64(12), report_every=np.int64(1))
    assert config.epochs == 2
    assert config.seed == 12
    assert config.report_every == 1
    assert type(config.epochs) is int
    assert type(config.seed) is int
    assert type(config.report_every) is int


def test_real_symmetric_packing_rejects_complex_entries_even_if_symmetric():
    with pytest.raises(ValueError, match="real values"):
        unpack_real_symmetric(
            torch.tensor([1.0 + 0j, 0.0 + 1.0j, 2.0 + 0j], dtype=torch.complex128),
            2,
        )


def test_analytic_hamiltonian_rejects_overflow_instead_of_returning_infinite_matrix():
    model = TwoStateAvoidedCrossing(slope=1e200, coupling=1.0)
    with pytest.raises(ValueError, match="overflowed"):
        model.hamiltonian(torch.tensor([1e200], dtype=torch.float64))


def test_phase_alignment_promotes_real_reference_to_complex_without_losing_phase():
    reference = torch.tensor([0.6, -0.8], dtype=torch.float64)
    candidate = reference.to(torch.complex128) * torch.exp(
        torch.tensor(0.51j, dtype=torch.complex128)
    )
    aligned = align_phase(reference, candidate)
    assert aligned.dtype == torch.complex128
    torch.testing.assert_close(aligned, reference.to(torch.complex128), atol=1e-13, rtol=0)


def test_unitary_procrustes_promotes_real_and_complex_subspace_frames():
    reference = torch.eye(3, dtype=torch.float64)[:, :2]
    phases = torch.diag(
        torch.exp(torch.tensor([0.3j, -0.2j], dtype=torch.complex128))
    )
    rotated = reference.to(torch.complex128) @ phases
    aligned = unitary_procrustes(reference, rotated)
    assert aligned.dtype == torch.complex128
    torch.testing.assert_close(aligned, reference.to(torch.complex128), atol=1e-13, rtol=0)


def test_training_seed_rejects_values_outside_numpy_rng_domain():
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(seed=2**32)
    upper = TrainingConfig(seed=2**32 - 1)
    assert upper.seed == 2**32 - 1


def test_training_canonical_seed_is_consumable_by_numpy_and_torch():
    config = TrainingConfig(seed=np.int64(19), epochs=np.int64(1), report_every=np.int64(1))
    state = np.random.get_state()
    try:
        np.random.seed(config.seed)
        torch.Generator().manual_seed(config.seed)
        assert np.random.randint(0, 1000) == 605
    finally:
        np.random.set_state(state)


def test_integer_real_symmetric_packing_is_promoted_to_float_for_eigensolver():
    matrix = unpack_real_symmetric(torch.tensor([2, 1, 3], dtype=torch.int64), 2)
    assert matrix.dtype == torch.get_default_dtype()
    torch.testing.assert_close(torch.linalg.eigvalsh(matrix), torch.linalg.eigvalsh(matrix.mH))


def test_hermitian_constructor_preserves_double_precision_offdiagonal_components():
    diagonal = torch.tensor([1.0, -2.0], dtype=torch.float32)
    tiny_complex_coupling = torch.tensor([0.12345678912345], dtype=torch.float64)
    imaginary = torch.tensor([0.25], dtype=torch.float32)
    matrix = complex_hermitian_from_parts(diagonal, tiny_complex_coupling, imaginary)
    assert matrix.dtype == torch.complex128
    assert matrix[0, 1].real.item() == pytest.approx(tiny_complex_coupling[0].item(), abs=1e-14)
    torch.testing.assert_close(matrix, matrix.mH, atol=0, rtol=0)
