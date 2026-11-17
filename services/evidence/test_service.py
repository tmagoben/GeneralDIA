"""HTTP and MCP integration tests; runnable with unittest or pytest, no Torch/PySCF needed."""

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from scripts import check_h3plus_report as checker
from services.evidence import agent, store
from services.evidence.app import FIXTURES, LIMIT_BYTES, create_app


class EvidenceServiceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "evidence.sqlite"
        self.client = TestClient(create_app(self.path))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.good = json.loads((FIXTURES / "physical_pass.json").read_bytes())
        self.bad = json.loads((FIXTURES / "false_pass.json").read_bytes())
        self.url = "/v1/evaluations"

    def test_archived_example_restart_idempotency_and_exact_replay(self):
        first = self.client.post(self.url, json=self.good)
        self.assertEqual(first.status_code, 201)
        result, url = first.json(), first.headers["location"]
        self.assertEqual(result["verdict"], "pass")
        self.assertIsNone(result["first_failure"])
        self.assertEqual(len(result["phase_measurements"]), 6)
        with TestClient(create_app(self.path)) as again:
            self.assertEqual(again.get(url).json(), result)
            raw = again.get(url + "/report").content
            self.assertEqual(hashlib.sha256(raw).hexdigest(), result["input_sha256"])
            self.assertEqual(json.loads(raw), self.good)
            second = again.post(self.url, content=raw, headers={"content-type": "application/json"})
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json(), result)

    def test_false_pass_uses_measured_error_and_stays_retrievable(self):
        response = self.client.post(self.url, json=self.bad)
        result = response.json()
        self.assertTrue(self.bad["pass"])
        self.assertEqual(response.status_code, 201)
        self.assertEqual(result["verdict"], "fail")
        self.assertIn("lower_excited_loop", result["first_failure"])
        measurement = next(m for m in result["phase_measurements"] if m["error"] > m["limit"])
        self.assertAlmostEqual(measurement["error"], 0.12)
        self.assertEqual((measurement["unit"], measurement["limit"]), ("rad", 0.06))
        self.assertEqual(self.client.get(response.headers["location"]).json(), result)

    def test_equivalent_phase_representations_and_full_comparison(self):
        for loop in ("lower_excited_loop", "upper_excited_loop"):
            item = self.good["result"][loop]
            item["phases"] = [math.pi, -math.pi, math.pi]
        self.assertEqual(self.client.post(self.url, json=self.good).json()["verdict"], "pass")
        base = Path(__file__).resolve().parents[2]
        gold = base / "tests/goldens/h3plus_basis_comparison_reference.json"
        reference = checker.load_report(gold)
        result = self.client.post(self.url, json=reference).json()
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(len(result["phase_measurements"]), 12)

    def test_rejects_malformed_http_input_without_a_database_record(self):
        h = {"content-type": "application/json"}
        for raw in (b"{", b'{"x":2,"x":3}', b'{"n":NaN}', b'{"n":1e999}', b"\xff", b"[]"):
            with self.subTest(raw=raw):
                response = self.client.post(self.url, content=raw, headers=h)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post(self.url, json={"kind": "new"}).status_code, 422)
        self.assertEqual(self.client.post(self.url, content=b"{}").status_code, 415)
        self.assertEqual(
            self.client.post(self.url, content=b"0" * (LIMIT_BYTES + 1), headers=h).status_code, 413
        )
        self.assertEqual(self.client.get(self.url + "/absent").status_code, 404)
        self.assertEqual(self.client.get(self.url + "/absent/report").status_code, 404)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0], 0)

    def test_structural_failure_and_interactive_examples(self):
        self.good["result"].pop("center_energies_hartree")
        result = self.client.post(self.url, json=self.good).json()
        self.assertEqual(result["verdict"], "fail")
        self.assertIn("center_energies_hartree", result["first_failure"])
        spec = self.client.get("/openapi.json").json()["paths"][self.url]["post"]
        demos = spec["requestBody"]["content"]["application/json"]["examples"]
        for k, expected in (("physical_pass", "pass"), ("false_pass", "fail")):
            response = self.client.post(self.url, json=demos[k]["value"])
            self.assertEqual(response.json()["verdict"], expected)

    def test_mcp_client_can_only_retrieve_and_compare(self):
        good, bad = (self.client.post(self.url, json=v).json() for v in (self.good, self.bad))

        async def exercise():
            async with create_connected_server_and_client_session(agent.server) as client:
                tools = (await client.list_tools()).tools
                self.assertEqual({t.name for t in tools}, {"get_evaluation", "compare_evaluations"})
                self.assertTrue(all(t.annotations.readOnlyHint for t in tools))
                found = await client.call_tool("get_evaluation", {"evaluation_id": good["id"]})
                self.assertFalse(found.isError)
                self.assertEqual(json.loads(found.content[0].text)["id"], good["id"])
                diff = await client.call_tool(
                    "compare_evaluations", {"left_id": good["id"], "right_id": bad["id"]}
                )
                self.assertFalse(diff.isError)
                self.assertEqual(json.loads(diff.content[0].text)["verdicts"], ["pass", "fail"])
                missing = await client.call_tool("get_evaluation", {"evaluation_id": "missing"})
                self.assertTrue(missing.isError)

        with patch.dict(os.environ, {"GENERALDIA_EVAL_DB": str(self.path)}):
            asyncio.run(exercise())
        missing = self.path.with_name("absent.sqlite")
        with self.assertRaises(sqlite3.OperationalError):
            store.get_result(missing, good["id"])
        self.assertFalse(missing.exists())
