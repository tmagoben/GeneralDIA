"""Analytic optical controls, independent singlet oracle and live two-basis H3+ checks."""

import numpy as np
import pytest

from generaldia.electronic_structure.h3plus_band import certify_transition_band
from generaldia.electronic_structure.singlet_oracle import singlet_spectrum
from generaldia.transition_band import transition_band


def test_split_band_requires_rotating_the_energy_operator_with_the_state_frame():
    d = np.array([[1, 2j], [2, 1], [0, 0.5j]])
    u = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    g = np.diag([0.3, 0.9])
    original = transition_band(d, g)
    rotated = transition_band(np.exp(0.4j) * d @ u, u.conj().T @ g @ u)
    np.testing.assert_allclose(rotated.tensor, original.tensor, atol=1e-12)
    expected = (2 / 3) * sum(g[j, j] * np.vdot(d[:, j], d[:, j]).real for j in (0, 1))
    assert original.oscillator == pytest.approx(expected)
    assert rotated.oscillator == pytest.approx(expected)
    assert transition_band(d @ u, g).oscillator != pytest.approx(expected)
    assert original.line_strength == pytest.approx(np.sum(abs(d) ** 2))


@pytest.mark.parametrize(
    "d,g,message",
    [
        (np.zeros((2, 2)), np.eye(2), "expected dipoles"),
        (np.zeros((3, 0)), np.zeros((0, 0)), "expected dipoles"),
        (np.ones((3, 2)), [[1, 2j], [2j, 3]], "Hermitian"),
        (np.ones((3, 2)), np.diag([0, 1]), "positive"),
        (np.full((3, 1), np.nan), [[1]], "finite"),
    ],
)
def test_bad_bands_fail_closed(d, g, message):
    with pytest.raises(ValueError, match=message):
        transition_band(d, g)


def test_dense_oracle_against_hand_solved_two_electron_singlets_and_ao_basis_change():
    # Independent particles in 0 and 0.5 Eh orbitals: singlets |00>, |01>+, |11>.
    s, h, eri = np.eye(2), np.diag([0.0, 0.5]), np.zeros((2, 2, 2, 2))
    r = np.zeros((3, 2, 2))
    r[0] = [[0, 1], [1, 0]]
    t = np.array([[1.2, -0.34], [0.2, 0.9]])
    for basis in (np.eye(2), t):
        e, mu = singlet_spectrum(
            basis.T @ s @ basis, basis.T @ h @ basis, eri, basis.T @ r @ basis, 1.0
        )
        np.testing.assert_allclose(e, [1, 1.5, 2], atol=1e-12)
        target = transition_band(mu[:, 0, 1:], np.diag(e[1:] - e[0]))
        np.testing.assert_allclose(target.tensor, np.diag([2, 0, 0]), atol=1e-12)
        assert target.oscillator == pytest.approx(2 / 3)


def test_eri_chemist_indices_give_direct_and_exchange_for_open_shell_singlet():
    v = np.zeros((2,) * 4)
    v[0, 0, 0, 0], v[1, 1, 1, 1] = 0.2, 0.3
    v[0, 0, 1, 1] = v[1, 1, 0, 0] = 0.07
    for a, b in ((1, 0), (0, 1)):
        for c, d in ((1, 0), (0, 1)):
            v[a, b, c, d] = 0.025
    e, _ = singlet_spectrum(np.eye(2), np.diag([0, 1]), v, np.zeros((3, 2, 2)), 0)
    closed = np.linalg.eigvalsh([[0.2, 0.025], [0.025, 2.3]])
    np.testing.assert_allclose(e, [closed[0], 1 + 0.07 + 0.025, closed[-1]], atol=1e-12)


def test_oracle_rejects_singular_and_malformed_integrals():
    args = [np.eye(2), np.diag([0, 1]), np.zeros((2,) * 4), np.zeros((3, 2, 2)), 0]
    for index, replacement in [
        (0, np.ones((2, 2))),
        (0, np.eye(9)),
        (2, np.eye(2)),
        (1, [[0, 2], [0, 1]]),
        (3, np.full((3, 2, 2), np.nan)),
    ]:
        bad = args.copy()
        bad[index] = replacement
        with pytest.raises(ValueError):
            singlet_spectrum(*bad)


@pytest.mark.optional
@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_live_h3plus_independent_transition_strengths(basis):
    pyscf = pytest.importorskip("pyscf")
    previous = pyscf.lib.num_threads()
    try:
        pyscf.lib.num_threads(1)
        evidence = certify_transition_band(basis)
    finally:
        pyscf.lib.num_threads(previous)
    assert evidence["pass"], evidence
    assert len(evidence["geometries"]) == 4
    for row in evidence["geometries"]:
        assert row["oscillator_sum"] > 0 and row["checks"]["energy_matched_zero_dipoles_rejected"]
        assert row["oracle_energy_error_hartree"] < 1e-8
