"""Run from the checkout: python -m uvicorn services.evidence.app:app."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from scripts import check_h3plus_report as checker
from services.evidence import store

LIMIT_BYTES = 128 * 1024
FIXTURES = Path(__file__).with_name("fixtures")


class PhaseMeasurement(BaseModel):
    check: str
    error: float
    limit: float
    unit: Literal["rad"]


class Evaluation(BaseModel):
    id: str
    input_sha256: str
    checker_version: str
    kind: str
    verdict: Literal["pass", "fail"]
    first_failure: str | None
    phase_measurements: list[PhaseMeasurement]


def create_app(db_path: Path | None = None) -> FastAPI:
    path = store.database_path() if db_path is None else db_path

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await run_in_threadpool(store.initialize, path)
        yield

    api = FastAPI(
        title="GeneralDIA H3+ evidence verification",
        version="1.0",
        description="Recheck submitted numerical evidence. No FCI rerun or source authentication.",
        lifespan=lifespan,
    )
    examples = {
        k: {"value": json.loads((FIXTURES / f"{k}.json").read_text(encoding="utf-8"))}
        for k in ("physical_pass", "false_pass")
    }
    body = {"schema": {"type": "object"}, "examples": examples}

    @api.post(
        "/v1/evaluations",
        response_model=Evaluation,
        status_code=201,
        responses={200: {"model": Evaluation, "description": "Identical evaluation already stored"}},
        openapi_extra={"requestBody": {"required": True, "content": {"application/json": body}}},
    )
    async def submit(request: Request, response: Response):
        """Store scientific verdicts. Malformed or unsupported input is not stored."""
        media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type != "application/json":
            raise HTTPException(415, "Use application/json")
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > LIMIT_BYTES:
                raise HTTPException(413, "Report exceeds 128 KiB")
            payload.extend(chunk)
        try:
            report = checker.parse_report(payload.decode("utf-8"))
        except (ValueError, RecursionError) as exc:
            raise HTTPException(422, "Expected strict, finite UTF-8 JSON with unique keys") from exc
        if not isinstance(report, dict) or report.get("kind") not in (
            checker.SINGLE_KIND,
            checker.COMPARISON_KIND,
        ):
            raise HTTPException(422, "Unsupported H3+ report kind")
        result, created = await run_in_threadpool(store.evaluate, path, report)
        response.status_code = 201 if created else 200
        response.headers["Location"] = f"/v1/evaluations/{result['id']}"
        return result

    @api.get("/v1/evaluations/{evaluation_id}", response_model=Evaluation)
    def retrieve(evaluation_id: str):
        try:
            return store.get_result(path, evaluation_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown evaluation ID") from exc

    @api.get("/v1/evaluations/{evaluation_id}/report", response_class=Response)
    def replay_input(evaluation_id: str):
        original = store.read(path, evaluation_id, report=True)
        if original is None:
            raise HTTPException(404, "Unknown evaluation ID")
        return Response(original, media_type="application/json")

    return api


app = create_app()
