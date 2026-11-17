# Verify an H3+ conical-intersection report

This local service checks the same numerical evidence as `scripts/check_h3plus_report.py`.
It does not run electronic structure or trust a submitted `pass` flag. It accepts the
single STO-3G certificate and the two-basis comparison. The learned-topology and optical
examples have different schemas and are **not** accepted by this endpoint.

From the checkout, in a Python virtual environment:

```sh
python -m pip install -r services/evidence/requirements.txt
export GENERALDIA_EVAL_DB="$PWD/generaldia-evaluations.sqlite3"
python -m uvicorn services.evidence.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for interactive examples. Or, from a second shell:

```sh
curl -sS -H 'Content-Type: application/json' --data-binary @services/evidence/fixtures/physical_pass.json http://127.0.0.1:8000/v1/evaluations
curl -sS -H 'Content-Type: application/json' --data-binary @services/evidence/fixtures/false_pass.json http://127.0.0.1:8000/v1/evaluations
```

The first returns `verdict: pass`. The second returns `verdict: fail`: the lower
excited loop is 0.12 rad from either representation of pi, above the 0.06 rad gate.
Both are valid submissions and receive HTTP 201. Resending the same report with a
different key order returns the stored evaluation (200). Invalid JSON, duplicate
keys, or an unsupported kind returns 422; non-JSON content types return 415;
input over 128 KiB returns 413.

Use the response's `Location` to retrieve the decision and append `/report` for
the canonical JSON whose bytes hash to `input_sha256`. The ID incorporates the
checker source digest. Checks stop at the first failure: an absent metric is not
a measured zero. A pass proves internal numerical consistency, **not** who ran
the calculation or whether the submitted report is authentic.

## Inspect from an MCP client

```sh
python -m pip install -r services/evidence/requirements-test.txt
export GENERALDIA_EVAL_DB="$PWD/generaldia-evaluations.sqlite3"
python -m services.evidence.agent
```

Configure a stdio client with that command, this checkout as its working directory,
and the absolute database path created by the HTTP service. The only tools are
`get_evaluation(evaluation_id)` and `compare_evaluations(left_id, right_id)`.
They open SQLite in read-only mode. Comparison requires the same report kind and
checker version; it shows right-minus-left phase errors for checks measured in both.
The service is for a trusted local workstation; no public deployment, authentication,
or multi-tenant isolation is provided.

## Fixture provenance and test

`physical_pass.json` is the byte-identical `h3plus_ci_evidence.json` from
[Actions run 43304324620](https://github.com/tmagoben/GeneralDIA/actions/runs/43304324620),
artifact `13983826009`. File SHA-256:
`730ae1d3d6e2ea1b4cbfe40ef37e3dcae1a764a01e0df6de03cec1a0d8713d94`.
`false_pass.json` is a deliberately edited negative control, **not** a second physical run.

```sh
python -m unittest -v services.evidence.test_service
```
