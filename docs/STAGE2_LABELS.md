# Stage 2: prediction registry and delayed labels

Stage 2 adds a local SQLite index while retaining `data/stage1_streaming/predictions.jsonl` unchanged. The database lives at `data/stage2/prediction_labels.sqlite3` and is excluded from Git. Stage 1 writes its JSONL prediction first, then registers it in SQLite. If SQLite fails, JSONL is preserved, the failure is logged and added to the Stage 1 error JSONL, and prediction processing continues. Replay missed records after recovery with the idempotent CLI command below.

The existing Phase 9 collector remains unchanged. It links synthetic historical PaySim labels using `event_id`; those labels are not live outcomes and do not enter this Stage 2 store.

## SQLite tables and matching rules

- `predictions`: immutable original prediction, probability, score/level, timestamp, model version, origin, Kafka topic/partition/offset, and pending/resolved status. `transaction_id` is primary key; Kafka delivery metadata is also unique when present.
- `labels`: first accepted outcome per transaction, label source/type, confirmation time, status, and initial submission ID. Labels arriving before predictions remain pending and match on exact `transaction_id` when the prediction registers.
- `label_submissions`: label submission audit and dispositions (`pending_prediction`, accepted source class, duplicate, or conflict). A pending disposition changes when its prediction arrives; a rejected early trusted claim against a synthetic prediction is marked accordingly.
- `submission_replays`: audit of retries using an already-seen submission ID, including idempotent retries and reused-ID conflicts.
- `prediction_conflicts`: records an attempted different original prediction for an already registered transaction. The first record is never replaced.

The first label for a transaction is retained. An identical outcome under another submission ID is recorded as an idempotent duplicate. A different outcome is recorded as a conflict and does not change the first label. Reusing a submission ID with a different payload returns HTTP 409. Invalid outcomes return HTTP 422. An unmatched label remains pending until its prediction arrives.

Only `trusted_verified` labels from a source named in `FRAUD_TRUSTED_LABEL_SOURCES` and submitted with the configured `FRAUD_TRUSTED_LABEL_TOKEN` are eligible for the genuine verified count. Configure both explicitly in the API process environment; the token must be at least 32 characters. Bind the API to loopback as shown below. This is a local operator-controlled allowlist and shared credential; it is not a bank integration or an identity provider. Synthetic-origin predictions are always rejected for `trusted_verified`. Use `synthetic_test` for workflow-only labels; `untrusted`, pending, rejected, and conflicting records never contribute to genuine verified counts. There is no accuracy or performance metric in this stage.

With no trusted source and token configured (the default), the API reports genuine labels unavailable. No genuine label source is configured in this repository.

## Start the label API and import existing records

In PowerShell at the project root:

```powershell
.\venv\Scripts\Activate.ps1
python -m src.labels.cli init
python -m src.labels.cli replay
uvicorn src.labels.api:app --host 127.0.0.1 --port 8769
```

`replay` reads the Stage 1 JSONL without modifying it; run it again after any Stage 2 store outage. Repeated runs are safe. Existing data is deduplicated by transaction ID and Kafka topic/partition/offset. Changed prediction data for an existing ID is counted separately as a conflict and kept in `prediction_conflicts`; the first prediction remains unchanged.

After deploying the code, restart the existing Stage 1 Spark process gracefully once so it loads the SQLite registration hook. Keep the same checkpoint and Kafka topic; do not delete or reset either. The current Stage 1 process does not load code changes automatically.

Optional genuinely verified ingestion requires explicit local configuration before starting the API:

```powershell
$env:FRAUD_TRUSTED_LABEL_SOURCES = 'authorized_source_name'
$env:FRAUD_TRUSTED_LABEL_TOKEN = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
```

Keep the token private and provide it only to the authorized local submitter. Do not configure a source until the organization has actually authorized it. A label is still not genuine merely because it uses a trusted-looking source name; the source must be in the explicit allowlist and the caller must possess the token.

## Submit a synthetic workflow-test label

Use an ID copied from a Stage 1 prediction record. This example deliberately marks the submission synthetic; choose either outcome only to exercise the workflow, and do not use it as real truth:

```powershell
$body = @{
  transaction_id = 'SYNTH-REPLACE-WITH-AN-ID-FROM-JSONL'
  outcome = 'legitimate'
  label_source = 'manual_workflow_test'
  source_type = 'synthetic_test'
  confirmed_at = (Get-Date).ToUniversalTime().ToString('o')
  submission_id = [guid]::NewGuid().ToString('N')
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8769/labels' -Method Post -ContentType 'application/json' -Body $body
```

Check the result, status counts, and original immutable prediction:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8769/labels/SYNTH-REPLACE-WITH-AN-ID-FROM-JSONL'
Invoke-RestMethod 'http://127.0.0.1:8769/predictions/SYNTH-REPLACE-WITH-AN-ID-FROM-JSONL'
Invoke-RestMethod 'http://127.0.0.1:8769/stats'
python -m src.labels.cli stats
```

The same submission endpoint is `POST /labels`; it accepts JSON fields `transaction_id`, `outcome` (`fraud` or `legitimate`), `label_source`, `source_type`, `confirmed_at` (timezone-aware ISO-8601), and `submission_id`. Authorized `trusted_verified` requests must also send `X-Label-Token` with the configured secret.

The Streaming Monitor shows registry counts but does not present them as model accuracy. Keep the label API terminal open while submitting labels; Ctrl+C stops it. The Kafka and Spark terminals remain independent processes. The shared token and local SQLite file are single-machine development controls, not production-grade multi-user authentication or distributed storage.

## Verification — 2026-10-25

The existing SQLite database was inspected read-only. Its `prediction_conflicts` row is for `SYNTH-20261025112055-00000118`: Stage 1 JSONL lines 114 and 120 have the same Kafka partition/offset (`0/177`) and Spark `batch_id` (`110`). Both are V2 predictions of class `0` for the same synthetic transaction, but the probabilities and risk scores differ (`0.4303409159` / `43.0341` versus `0.42930456996` / `42.9305`). During JSONL replay, the first record was registered at `11:39:10.089820 UTC`; the conflict was observed at `11:39:10.166508 UTC`. `register_prediction` compares these values in its stable prediction fields, kept the first prediction, and wrote the second to `prediction_conflicts`. This evidence is consistent with at-least-once reprocessing of the same batch event. The retained records do not establish why the duplicate batch output was produced, so the specific Spark retry/restart trigger remains unknown.

This is expected conflict-audit behavior, not a label or submission-ID conflict and not evidence of a `register_prediction` defect. `prediction_conflicts` counts different prediction data for an already registered transaction, or a Kafka topic/partition/offset uniqueness collision. `conflicting_labels` counts label submissions whose outcome differs from the first label for that transaction. Reusing a submission ID with different fields is recorded separately in `submission_replays` as `submission_id_conflict`; those events are not included in either conflict counter. The database currently has one prediction conflict, zero conflicting labels, two submission-ID conflict retries, and one identical label retry. No database record or counter was changed to make the statistics appear clean.

Verification ran against temporary databases for automated tests; the existing project database was only queried read-only. Results:

- `.\venv\Scripts\python.exe -m unittest tests.test_stage2_labels -v` — 15 passed.
- `.\venv\Scripts\python.exe -m unittest tests.test_stage1_streaming -v` — 5 passed.
- `.\venv\Scripts\python.exe -m unittest discover -s tests -v` — 69 passed.
- `git diff --check` — passed.

The exact synthetic label retry returned `duplicate_submission` with `idempotent: true`; the canonical label remained one row and the duplicate-submission count became 1. Synthetic test labels remain separate from genuine verified labels. Current database statistics report 1 synthetic test label, 0 confirmed genuine labels, `genuine_source_configured: false`, and `genuine_labels_available: false`.
