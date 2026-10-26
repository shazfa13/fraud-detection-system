# Stage 7: End-to-end validation and final readiness

Stage 7 completes the repository's planned Phase 13 stability validation. It
checks the existing Stage 1–6 integration using the project virtual environment
and temporary test fixtures. It does not start Kafka or Spark services, run the
persistent Stage 3 monitor, train an operational model, promote a candidate, or
change serving artifacts.

## Reproduce the validation

Run these commands from the repository root in PowerShell:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage*.py" -v
.\venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow status --candidate models\candidates
```

The two test commands are read-only with respect to the live databases and
streaming state: tests that need writable data use temporary directories. The
Stage 6 unit test trains a small demonstration candidate from a temporary toy
fixture and verifies that Stage 5 blocks it; it does not train or promote an
operational candidate. The `pytest` package is absent from the project venv, so
the repository's `unittest` tests are used directly without installing or
changing dependencies.

## Results recorded on 2026-10-25

The focused Stage 1–6 test files passed **57 tests**. Full discovery passed
**107 tests**. Both runs used Python 3.12.7 from `venv`. The output includes one
expected mocked `database locked` error log from the Stage 1 resilience test and
a Starlette/httpx deprecation warning; both test runs completed successfully.

The shared active-model resolver selected **V2**, threshold **0.53**, with the
ordered **34-feature** contract and no active promotion manifest. FastAPI and
the streaming inference path use the shared resolver/import path. Stage 5's
read-only status command reported `candidate_artifacts_present_not_evaluated`,
active V2 at 0.53, and no promotion. Existing Stage 6 candidates are marked
`demonstration`, `eligible_for_genuine_promotion: false`, and
`candidate_trained_not_promoted`.

An isolated temporary copy of the six V1/V2 model files was used to verify that
the resolver rejects a missing V2 model, malformed model JSON, and malformed
activation manifest without touching the real artifacts. A direct `/health`
function check returned V2/0.53, and `TransactionRequest` rejected `isFraud`.
The Stage 5 unit test also checked `/health` and that serving and monitoring
share the resolver. A BOM-aware AST syntax sweep parsed all 50 Python files in
`src/` and `tests/` successfully; no Python files were changed.

Automated coverage includes: Stage 1 temporary-batch inference and validation;
Stage 2 provenance, synthetic-origin exclusion, duplicate and conflicting
label submissions; Stage 3 synthetic-only retraining-request blocking and
deduplication using test databases; Stage 4 insufficient-label blocking;
Stage 5 authorization, fail-closed manifest checks, dry-run, activation, and
rollback on temporary artifacts; and Stage 6 isolated demonstration training
and promotion rejection. Stage 3 persistent monitoring and live Kafka/Spark
were not run.

The V1/V2 model, metadata, and threshold hashes match the existing Phase 12
deployment report baseline. A before/after SHA-256 snapshot around Stage 7
tests also showed no changes to 495 files covering Stage 2 and Stage 3 SQLite
files (including WAL/SHM files), existing checkpoints, V1/V2 model artifacts,
and prediction JSONL files. The Stage 7 report contains the hash comparison and
the component check statuses. This snapshot only covers the listed local files;
it does not inspect an external Kafka broker or running service state.

## Limitations and next steps

This validation establishes automated repository behavior and local artifact
integrity, not production readiness or real-world fraud performance. The
dataset and Stage 6 candidate use synthetic PaySim labels. No live service or
broker integration was repeated because that can mutate offsets, logs,
checkpoints, or databases. The API `/health` behavior, request validation, and
shared model selection are covered through unit tests; a running HTTP server
was not started. `pytest` is unavailable in the venv, but all repository tests
are `unittest` discoverable and passed with the commands above.

Machine-readable results and a concise run summary are in
`reports/stage7_end_to_end_validation_20261025T150813Z.json` and
`reports/stage7_end_to_end_validation_20261025T150813Z.md`.
