# Stage 4: Safe candidate retraining workflow

Stage 4 reads the Stage 2 label store in SQLite `mode=ro` and reads Stage 1's
prediction JSONL. It never changes labels, predictions, Stage 3 alerts or requests.
It does not consume a drift alert as permission to train. A candidate model is
written only when the explicit verified-label gates pass and the caller adds
`--train-candidate`; candidate files remain separate from `models/active`.
Stage 4 does not promote or deploy a candidate.

## Dataset and provenance rules

The model input contract is the ordered 34-feature list in V2 metadata, checked
against the frozen V1 contract. `step` is used only for temporal ordering;
`event_id` is an audit identity, and `isFraud` is the target. Identifiers,
names, prediction outputs, label lineage, and other audit fields are never model
inputs. The target remains `isFraud`.

Historical examples are taken only from the existing V2 parquet rows through
V2's recorded `train_end_step` (currently 422). They are the existing historical
PaySim training baseline, not new evidence of genuine outcomes. New examples must
join by stable `transaction_id` across a resolved Stage 2 label, its V2 prediction,
and exactly one Stage 1 feature-vector record. The label's source type must be
`trusted_verified`, its source must be in `FRAUD_TRUSTED_LABEL_SOURCES`, and the
authorization token must be configured. The stored label submission must have
`accepted_genuine` disposition. Synthetic-origin predictions, synthetic test or
untrusted labels, pending/rejected/unmatched labels, conflicting label or
prediction records, invalid/duplicate feature vectors, and labels without a
permitted lineage are excluded and counted. New transaction steps must be after
V2's validation boundary (currently 565), so new examples cannot overlap the
historical training window.

Stage 2 is queried read-only. No label values are imputed. `null` feature values
become NaN for XGBoost; nonnumeric values, booleans, infinities, missing features,
or incorrect feature ordering fail validation.

## Eligibility and temporal splits

The initial conservative requirements are at least **100 eligible new verified
labels**, including at least **10 verified fraud labels**. These are minimum
project gates, not statistically calibrated guarantees. Historical baseline rows
do not count toward the new-label minimum. Recent verified labels are ordered by
`step` and divided by whole step groups into 60% train, 20% validation, and 20%
test. The historical V2 training rows are added to the training partition only.
The validation and test partitions must each contain both classes and at least
one verified fraud example. Empty or overlapping splits stop the run. Ordering
fields are retained for audit but not passed to XGBoost.

If the label count, fraud count, split sizes, or class requirements fail, the
workflow reports `insufficient_eligible_labels` and exits before model training
or candidate creation. It does not lower thresholds to force a run.

## Run and outputs

From the project root, first produce a read-only eligibility report. The report
path must not already exist:

```powershell
$report = Join-Path $env:TEMP ('stage4-eligibility-' + [guid]::NewGuid().ToString('N') + '.json')
.\venv\Scripts\python.exe -m src.retraining.stage4_workflow --report $report
```

The report contains source/state breakdowns, exclusions, eligible counts,
historical training-row count and source hash, the ordered contract, dataset
fingerprint, and stop reason. The present local label store has one resolved
`synthetic_test` label and no configured trusted-source allowlist or token, so
the expected status is `insufficient_eligible_labels` with zero eligible new
labels. Do not use the historical Stage 3 retraining request to bypass this gate.

Candidate training is an explicit separate opt-in, and still stops unless every
eligibility and split gate passes:

```powershell
$report = Join-Path $env:TEMP ('stage4-candidate-' + [guid]::NewGuid().ToString('N') + '.json')
.\venv\Scripts\python.exe -m src.retraining.stage4_workflow --report $report --train-candidate
```

The candidate goes to a unique `models/candidates/stage4-*` directory by default,
with model, metadata, threshold, dataset fingerprint, source lineage summary,
split boundaries, training configuration, and test results. Existing candidate
files are never overwritten. No promotion step is called.

## Limitations

The V2 historical labels and baseline cohort are PaySim synthetic data. Genuine
Stage 2 labels are required for the new validation/test extension, but they do
not establish real-world representativeness. The initial 100/10 sample floors
and chronological split policy require future calibration with a sufficiently
large authorized verified cohort. Stage 4 does not implement monitoring,
approval, promotion, rollback, or deployment workflows.
