"""Independent, dependency-free evidence checks and deliberate corruptions."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_h3plus_report.py"
SPEC = importlib.util.spec_from_file_location("h3plus_checker", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


@pytest.fixture
def report():
    return CHECKER.load_report(ROOT / "tests/goldens/h3plus_basis_comparison_reference.json")


def single_report(report):
    result = copy.deepcopy(report["results"]["sto-3g"]["result"])
    result.pop("branching_gaps_hartree")
    return {
        **{
            key: report[key]
            for key in (
                "revision",
                "backend",
                "method",
                "geometry_units",
                "energies_and_gaps_units",
                "phase_convention",
            )
        },
        "kind": CHECKER.SINGLE_KIND,
        "stage": "v3.4.0.dev3",
        "result": result,
        "pass": True,
    }


def test_accepts_real_two_basis_and_legacy_schema_evidence(report):
    CHECKER.validate_report(report)
    CHECKER.validate_report(single_report(report))


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("kind",), "unknown", "unrecognized report kind"),
        (("pass",), 1, "report.pass"),
        (("schema_version",), True, "schema_version"),
        (("revision",), "", "revision"),
        (("geometry_units",), "bohr", "geometry_units"),
        (("phase_convention",), "opposite", "phase_convention"),
        (("settings", "conv_tol"), 1e-5, "conv_tol"),
        (("settings", "thresholds", "phase_tolerance_radians"), 0.5, "thresholds"),
        (("results", "6-31g", "settings", "radius_angstrom"), 0.07, "settings"),
        (("results", "6-31g", "result", "radius_angstrom"), 0.07, "shared geometry mismatch"),
        (("results", "6-31g", "result", "basis"), "sto-3g", "basis label mismatch"),
        (("results", "6-31g", "result", "charge"), True, "charge"),
        (("results", "6-31g", "pass"), False, "pass"),
        (("results", "6-31g", "result", "central_excited_gap_hartree"), 0.1, "inconsistent"),
        (("results", "6-31g", "result", "branching_gap_ratios"), [2, 5], "branching_gap_ratios"),
        (
            ("results", "6-31g", "result", "branching_gaps_hartree"),
            [0, 1, 1, 2],
            "must be positive",
        ),
        (("results", "6-31g", "result", "minimum_link_strength"), 0.2, "overlap link"),
        (("results", "6-31g", "result", "refinement_points"), [8, 16], "refinement_points"),
        (
            ("results", "6-31g", "result", "nonenclosing_phase_radians"),
            3.14,
            "physical_nonencircling_zero",
        ),
        (
            ("results", "6-31g", "result", "min_excited_gap_on_loop_hartree"),
            1e-8,
            "positive_excited_gap_on_loops",
        ),
        (
            ("results", "6-31g", "result", "gauge_phase_error_radians"),
            -1,
            "local_complex_gauge_covariance",
        ),
        (
            ("results", "6-31g", "result", "geometry_sha256", "nonenclosing"),
            "0" * 64,
            "identical_geometry_paths",
        ),
        (
            ("results", "6-31g", "result", "geometry_sha256", "nonenclosing"),
            "bad",
            "malformed SHA256",
        ),
        (("results", "6-31g", "result", "checks", "reversal_covariance"), "true", "JSON boolean"),
        (("checks",), {}, "missing or unknown required checks"),
    ],
)
def test_rejects_corrupt_or_inconsistent_reports(report, path, value, message):
    current = report
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(CHECKER.InvalidReport, match=message):
        CHECKER.validate_report(report)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("phases", [0, 0, 0], "inconsistent"),
        ("point_counts", [8, 16, 64], "mesh mismatch"),
        ("phase_tolerance", 0.5, "changed threshold"),
        ("minimum_link_singular_value", 0.1, "overlap link"),
        ("reference_phase", 0.0, "inconsistent"),
        ("reference_phase_error", 0.5, "inconsistent"),
        ("coarse_to_finest_phase_error", 0.5, "inconsistent"),
        ("successive_phase_errors", [0.5, 0.5], "inconsistent"),
        ("passed", False, "inconsistent"),
    ],
)
def test_recomputes_loop_evidence_instead_of_trusting_pass_flag(report, key, value, message):
    report["results"]["sto-3g"]["result"]["lower_excited_loop"][key] = value
    with pytest.raises(CHECKER.InvalidReport, match=message):
        CHECKER.validate_report(report)


def test_rejects_self_consistent_energy_only_wrong_topology(report):
    loop = report["results"]["sto-3g"]["result"]["lower_excited_loop"]
    loop.update(phases=[0, 0, 0], reference_phase_error=CHECKER.math.pi, passed=False)
    with pytest.raises(CHECKER.InvalidReport, match="failed pi-phase"):
        CHECKER.validate_report(report)


@pytest.mark.parametrize(
    "content",
    [
        '{"pass":true,"pass":false}',
        '{"extra":NaN}',
        '{"extra":Infinity}',
        '{"extra":1e999}',
        '{"extra":-Infinity}',
        "{broken",
    ],
)
def test_strict_json_rejects_duplicates_nonfinite_and_malformed_input(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        CHECKER.load_report(path)


def test_missing_required_check_is_rejected(report):
    del report["results"]["sto-3g"]["result"]["checks"]["energy_only_control_rejected"]
    with pytest.raises(CHECKER.InvalidReport, match="missing or unknown required checks"):
        CHECKER.validate_report(report)


def test_cli_without_site_packages_checks_both_formats_and_names_failing_control(report, tmp_path):
    comparison = tmp_path / "comparison.json"
    single = tmp_path / "single.json"
    comparison.write_text(json.dumps(report), encoding="utf-8")
    single.write_text(json.dumps(single_report(report)), encoding="utf-8")
    command = [sys.executable, "-I", "-S", str(SCRIPT)]
    good = subprocess.run([*command, str(single), str(comparison)], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert good.stdout.count("PASS") == 2
    bad_result = report["results"]["6-31g"]["result"]
    bad_result["min_ground_isolation_hartree"] = 0.0
    bad_result["checks"]["ground_and_higher_root_isolation"] = False
    report["results"]["6-31g"]["pass"] = False
    report["pass"] = False
    comparison.write_text(json.dumps(report), encoding="utf-8")
    bad = subprocess.run([*command, str(comparison)], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "6-31g.checks.ground_and_higher_root_isolation: failed required check" in bad.stderr


def test_cli_reports_missing_file_without_traceback(tmp_path):
    missing = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), str(tmp_path / "absent.json")],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "FAIL" in missing.stderr
    assert "Traceback" not in missing.stderr
