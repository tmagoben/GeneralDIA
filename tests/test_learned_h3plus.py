"""Actual two-basis FCI references, held-out paths and wrong-topology rejection."""

import runpy
from pathlib import Path

import pytest

EXPERIMENT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/17_learned_h3plus_topology.py")
)


def test_geometry_split_excludes_permutation_equivalent_training_samples():
    training, paths = EXPERIMENT["split_geometries"]()
    keys = EXPERIMENT["geometry_keys"]
    assert len(training) == 108
    assert len(keys(training)) == 36
    for geometry, _ in paths.values():
        assert not (keys(training) & keys(geometry))
        assert keys(geometry[:, [2, 0, 1]]) == keys(geometry)


@pytest.mark.optional
@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_learned_hamiltonian_passes_heldout_physics_and_rejects_isospectral_control(basis):
    pyscf = pytest.importorskip("pyscf")
    old_threads = pyscf.lib.num_threads()
    pyscf.lib.num_threads(1)
    try:
        report = EXPERIMENT["run_benchmark"](basis)
    finally:
        pyscf.lib.num_threads(old_threads)
    assert report["pass"], [run["checks"] for run in report["runs"]]
    assert [r["seed"] for r in report["runs"]] == [0, 7, 23]
    for run in report["runs"]:
        assert run["heldout"]["pass"]
        wrong = run["isospectral_negative_control"]
        assert not wrong["pass"]
        assert wrong["checks"]["energy_mae"] and wrong["checks"]["energy_max_error"]
        assert not wrong["checks"]["heldout_topology"]
        assert wrong["energy_mae_hartree"] == pytest.approx(
            run["heldout"]["energy_mae_hartree"], abs=1e-14
        )
        paths = {p["name"]: p for p in run["heldout"]["paths"]}
        assert [paths[f"refinement-{n}"]["points"] for n in (8, 16, 32)] == [8, 16, 32]
        assert "nonenclosing" in paths and "reverse" in paths and "ellipse" in paths
