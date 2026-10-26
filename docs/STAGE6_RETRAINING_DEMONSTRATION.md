# Stage 6: Controlled retraining demonstration

Stage 6 demonstrates candidate training mechanics without changing Stage 4's
genuine-label gate. It reads the existing
data/processed/v2_retraining_dataset.parquet, which contains historical
PaySim transactions and simulated labels. It is suitable for a reproducible
portfolio demonstration because it has both classes and the exact V2 feature
schema. Its labels do not establish genuine ground truth or real-world
adaptation.

## Contract and data split

The input must have exactly [event_id, step] + V2's ordered 34 features +
[isFraud]. event_id and step are audit/order fields; isFraud is only the
target. Feature order is checked against V2 and frozen V1 metadata. Missing
feature values are left as NaN for XGBoost's native missing-value handling;
infinities, invalid labels, wrong types, and schema changes stop the workflow.

The full file is scanned in bounded Arrow batches. A seeded uniform reservoir
sample is drawn separately within each V2 chronological split and target class,
without replacement. Defaults cap train/validation/evaluation samples at
60,000/15,000/15,000 rows. V2 metadata supplies the boundaries: train through
step 422, validation through step 565, and held-out evaluation through step 743.
The candidate is fitted only on sampled training rows; early stopping and
threshold selection use validation rows only; final metrics use evaluation
rows only. Per-split source IDs are checked for duplicates and overlap.

The existing V2 trainer and evaluator are reused. For bounded runtime, this
demonstration caps the existing V2 configuration at 120 estimators, 20 early
stopping rounds, and two worker threads. The class weight is calculated from
the selected training sample. No preprocessing is fit on validation or
evaluation data; XGBoost accepts the numeric contract directly.

## Commands

Run from the project root. Preflight is read-only and is the default unless
--train is supplied.

    .\venv\Scripts\python.exe -m src.retraining.stage6_demo --preflight

Train a bounded candidate in a new, uniquely named directory.

    .\venv\Scripts\python.exe -m src.retraining.stage6_demo --train

Optional dataset, output root, seed, and sample-size flags are available via
--help. Each run writes to
models/candidates/stage6_demo/<UTC-run-id>/. Existing run directories are
never reused. A failed run may leave its newly created directory for diagnosis.
Artifacts include model.json, metadata.json, threshold.json, and
training_report.json; the report records dataset and feature fingerprints,
split counts, run configuration, measured metrics, hashes, and comparison
status.

## Results and comparison

The report includes precision, recall/fraud recall, F1, PR-AUC, ROC-AUC,
confusion counts, false positives/negatives, threshold, evaluation size, and
fraud count. If the shared active model is V2 and its ordered contract matches,
the V2 model and candidate are evaluated on the exact same sampled evaluation
rows. Otherwise only candidate metrics are reported. A same-row comparison
does not turn simulated PaySim labels into evidence of genuine performance.

Inspect the candidate/report directly without changing activation.

    Get-ChildItem models\candidates\stage6_demo -Directory
    Get-Content (Join-Path <candidate-directory> 'training_report.json') -Raw

Every candidate has training_mode=demonstration and
eligible_for_genuine_promotion=false. Stage 5's promotion operation rejects
these markers even if a report or approval file is supplied. Stage 6 never
calls activation or rollback, changes the active manifest, or alters V1/V2
artifacts. The active model therefore stays V2 at threshold 0.53.

## Relationship to genuine retraining

Genuine adaptive retraining remains: Stage 2 collects authorized resolved
labels with provenance; Stage 3 reports drift/performance evidence; Stage 4
requires its verified-label and fraud-count gates and trains only when eligible;
Stage 5 evaluates a genuine candidate and requires human approval before
promotion. Stage 6 does not call or override Stage 4 eligibility and its
historical labels are not eligible verified labels.

The sample is smaller and class-stratified, so its training class distribution
does not represent the source prevalence exactly. Metrics are measured on a
separate chronologically held-out sample retaining the source class proportion,
but the data remain synthetic and historical. A genuine verified cohort,
production-calibrated acceptance criteria, candidate-specific Stage 3 reference
window, and serving-process restart procedure remain necessary before
operational deployment.
