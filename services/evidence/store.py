"""Versioned, replayable H3+ report evaluations; no model or wavefunction execution."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from scripts import check_h3plus_report as checker

CHECKER_VERSION = "h3plus:" + sha256(Path(checker.__file__).read_bytes()).hexdigest()


def database_path() -> Path:
    return Path(os.environ.get("GENERALDIA_EVAL_DB", "generaldia-evaluations.sqlite3"))


def initialize(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS evaluations ("
            "id TEXT PRIMARY KEY, report TEXT NOT NULL, result TEXT NOT NULL)"
        )


def evaluate(path: Path, report: dict) -> tuple[dict, bool]:
    """Hash the stored canonical JSON; the same input and checker produce the same ID."""
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    input_hash = sha256(canonical.encode("ascii")).hexdigest()
    evaluation_id = sha256(f"{CHECKER_VERSION}:{input_hash}".encode("ascii")).hexdigest()
    metrics: list[dict] = []
    reason = None
    try:
        checker.validate_report(report, measurements=metrics)
    except checker.InvalidReport as exc:
        reason = str(exc)
    result = {
        "id": evaluation_id,
        "kind": report["kind"],
        "input_sha256": input_hash,
        "checker_version": CHECKER_VERSION,
        "verdict": "fail" if reason is not None else "pass",
        "first_failure": reason,
        "phase_measurements": metrics,
    }
    with closing(sqlite3.connect(path)) as db, db:
        cur = db.execute(
            "INSERT INTO evaluations(id, report, result) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO NOTHING",
            (evaluation_id, canonical, json.dumps(result, allow_nan=False)),
        )
        created = cur.rowcount == 1
        if not created:
            row = db.execute(
                "SELECT result FROM evaluations WHERE id=?", (evaluation_id,)
            ).fetchone()
            result = json.loads(row[0])
    return result, created


def read(path: Path, evaluation_id: str, *, report: bool = False) -> str | None:
    # MCP queries must never initialize a missing database or write to an existing one.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        sql = (
            "SELECT report FROM evaluations WHERE id=?"
            if report
            else "SELECT result FROM evaluations WHERE id=?"
        )
        row = db.execute(sql, (evaluation_id,)).fetchone()
    return None if row is None else row[0]


def get_result(path: Path, evaluation_id: str) -> dict:
    raw = read(path, evaluation_id)
    if raw is None:
        raise KeyError("Unknown evaluation ID")
    return json.loads(raw)


def compare(path: Path, left_id: str, right_id: str) -> dict:
    left, right = (get_result(path, key) for key in (left_id, right_id))
    if (left["checker_version"], left["kind"]) != (right["checker_version"], right["kind"]):
        raise ValueError("Compare reports of the same kind and checker version")
    a, b = ({m["check"]: m["error"] for m in r["phase_measurements"]} for r in (left, right))
    return {
        "left_id": left_id,
        "right_id": right_id,
        "verdicts": [left["verdict"], right["verdict"]],
        "right_first_failure": right["first_failure"],
        "shared_phase_error_delta_rad": {k: b[k] - a[k] for k in sorted(a.keys() & b.keys())},
        "not_measured_in_both": sorted(a.keys() ^ b.keys()),
    }
