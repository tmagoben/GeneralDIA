"""Exact geometry and input-contract tests for H3+ symmetry-breaking modes."""

import math

import numpy as np
import pytest

from generaldia.electronic_structure.h3plus_ci import h3plus_ci_loop, h3plus_triangle


def _bond_lengths(geometry):
    return np.array(
        [
            np.linalg.norm(geometry[0] - geometry[1]),
            np.linalg.norm(geometry[0] - geometry[2]),
            np.linalg.norm(geometry[1] - geometry[2]),
        ]
    )


def test_equilateral_geometry_is_centered_and_has_correct_bonds():
    positions = h3plus_triangle(0.875)
    assert positions.shape == (3, 3)
    np.testing.assert_allclose(positions.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(_bond_lengths(positions), np.full(3, 0.875), atol=1e-14)
    assert np.all(positions[:, 2] == 0)


@pytest.mark.parametrize("x,y", [(0.035, 0.0), (0.0, -0.053), (0.08, 0.04)])
def test_two_branching_modes_reproduce_prescribed_distances(x, y):
    r = 0.875
    expected = np.array(
        [
            r + 2 * x / math.sqrt(6),
            r - x / math.sqrt(6) + y / math.sqrt(2),
            r - x / math.sqrt(6) - y / math.sqrt(2),
        ]
    )
    actual = _bond_lengths(h3plus_triangle(r, x, y))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    assert sum(actual) == pytest.approx(3 * r, abs=1e-13)


def test_loop_has_direct_exact_oriented_geometry_sequence():
    forward = h3plus_ci_loop(12, 0.05, bond_angstrom=0.875)
    reversed_loop = h3plus_ci_loop(12, 0.05, bond_angstrom=0.875, orientation=-1)
    assert forward.shape == (12, 3, 3)
    np.testing.assert_allclose(forward[0], reversed_loop[0], atol=1e-14)
    np.testing.assert_allclose(forward[1:], reversed_loop[:0:-1], atol=1e-14)
    assert not np.array_equal(forward[0], forward[-1])
    assert len({geometry.tobytes() for geometry in forward}) == 12


@pytest.mark.parametrize(
    "args, match",
    [
        ((0.0, 0.0, 0.0), "positive bond"),
        ((float("nan"), 0.0, 0.0), "finite"),
        ((0.875, float("inf"), 0.0), "finite"),
        ((0.875, 3.0, 0.0), "invalid triangle"),
    ],
)
def test_triangle_rejects_invalid_input(args, match):
    with pytest.raises(ValueError, match=match):
        h3plus_triangle(*args)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"points": 5, "radius_angstrom": 0.05}, "points"),
        ({"points": 8, "radius_angstrom": 0.0}, "positive radius"),
        ({"points": 8, "radius_angstrom": float("nan")}, "finite"),
        ({"points": 8, "radius_angstrom": 0.05, "orientation": 0}, "orientation"),
    ],
)
def test_loop_rejects_invalid_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        h3plus_ci_loop(**kwargs)
