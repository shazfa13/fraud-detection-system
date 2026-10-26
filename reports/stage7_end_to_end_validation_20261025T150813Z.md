# Stage 7 validation — 2026-10-25

**Status: pass with limitations.** The focused Stage 1–6 suite passed 57 tests;
full `unittest` discovery passed 107 tests. Active model resolution selected
V2 at threshold 0.53 with 34 ordered features. No operational model was trained
or promoted. The Stage 6 tests trained only temporary toy demonstration
candidates and confirmed promotion rejection.

SHA-256 and size snapshots matched for all 495 checked local persistence and
model files before and after the tests. The existing Phase 12 model artifact
hash baseline also matches. No persistent Stage 3 monitor or live API/Kafka/
Spark service was run. The `pytest` runner is absent from the venv; the
`unittest` suite completed successfully instead.

See [the Stage 7 procedure and limitations](../docs/STAGE7_END_TO_END_VALIDATION.md)
and the adjacent JSON report for check-level status.
