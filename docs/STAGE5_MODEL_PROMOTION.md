# Stage 5: Candidate validation, promotion, and rollback

Stage 5 never trains. Stage 4 remains the only training workflow and currently
reports that there are no eligible genuine verified labels. Synthetic fixtures
are for code-path tests only, not production performance evidence.

## Shared active-model resolution

FastAPI resolves `models/active_model_manifest.json` through
`src.retraining.active_model.resolve_active_model`. The Spark consumer imports
the same FastAPI module before inference, so it uses the same bundle. Stage 3
also resolves the shared configuration. With no manifest, the legacy
`FRAUD_ACTIVE_MODEL_VERSION` setting is honored and defaults to V2. A present
but malformed manifest fails closed; it does not fall back. The manifest binds
relative model, metadata, and threshold paths to SHA-256 values. The ordered 34
features still come from the frozen V1/V2 metadata contract.

Stage 3 currently has reference windows registered only for V1 and V2. It will
refuse an unknown candidate model version rather than silently use V2's
reference. Registering a candidate reference cohort is needed before monitoring
that candidate after any future activation.

## Candidate validation and evaluation

The candidate bundle must contain `model.json`, `metadata.json`, and
`threshold.json`, load as an XGBoost model, match all 34 feature names in order,
exclude audit/target columns, and have a threshold strictly between zero and
one. Stage 4 provenance keys are required: training dataset fingerprint,
label provenance summary, temporal split, training configuration, elapsed
training time, and training results. When metadata includes artifact hashes,
they must match.

Evaluation applies each model's saved threshold and compares both models on
the same supplied rows. The report includes precision, recall, F1, PR-AUC,
ROC-AUC, confusion counts, FP/FN, thresholds, row and fraud counts, feature
contract, and an ordered data fingerprint. It rejects duplicated transaction
IDs, missing/invalid labels, missing/non-finite features, and untrusted or
synthetic evaluation provenance. The caller must provide evidence that the
evaluation fingerprint is disjoint from candidate training and threshold
selection. Evaluation step values must also follow the candidate's reserved
Stage 4 test window. The command does not infer trust from a CSV/Parquet file alone.

Initial provisional criteria are at least 100 evaluation rows, 10 fraud rows,
fraud recall >= 0.80, and no more than 0.02 absolute regression in precision,
recall, F1, or PR-AUC against the active model. These are demonstration defaults,
not calibrated production standards. They require an authorized, sufficiently
large evaluation cohort and domain review before operational use. ROC-AUC is
reported as a ranking metric; the thresholded metrics use the stored thresholds.

Statuses include `candidate_unavailable`,
`candidate_artifacts_present_not_evaluated`, `evaluation_blocked`,
`candidate_rejected`, `candidate_approved_pending_authorization`,
`promotion_ready`, `promoted`, and `rollback_blocked`/`rolled_back`.

Read-only inventory:

```powershell
Set-Location D:\fraud-detection-system\fraud-detection-system
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow status
```

Evaluation requires an explicit cohort and provenance JSON. The report path
must be new. `--trusted-evaluation` is an explicit operator assertion and does
not replace review of the source records:

```powershell
$report = Join-Path $env:TEMP ('stage5-evaluation-' + [guid]::NewGuid().ToString('N') + '.json')
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow evaluate `
  --candidate models\candidates\<candidate-directory> `
  --data <held-out-parquet> --provenance <evaluation-provenance.json> `
  --report $report --trusted-evaluation
```

The provenance JSON should identify source and label provenance, explicitly
mark synthetic data, and provide evaluation and training fingerprints. Do not
set trusted provenance for synthetic test data.

## Human authorization and activation

Promotion requires an approval JSON bound to the exact candidate version,
three artifact hashes, candidate dataset fingerprint, and SHA-256 of the
evaluation report. It also requires an approver identity, timezone-aware
`approved_at`, and the literal statement:

`I approve promotion of this exact candidate and evaluation report.`

The human must inspect the report and independently create the approval file.
There is no standalone `--approve` switch. Dry-run is default and writes no
active manifest. Execution uses `--execute` only after rechecking all bindings.
The old active selection is saved as a manifest in a new directory under
`models/promotion_history`; model artifacts are never overwritten. The new
manifest is atomically replaced and resolved again. On failed post-activation
verification, the previous manifest is restored. A promotion record records
the before/after versions, hashes, report/approval references, time, and
outcome.

```powershell
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow promote `
  --candidate models\candidates\<candidate-directory> `
  --evaluation-report $report --approval <approval.json>
# After reviewing the printed promotion_ready result, explicitly execute:
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow promote `
  --candidate models\candidates\<candidate-directory> `
  --evaluation-report $report --approval <approval.json> --execute
```

Rollback is also dry-run by default, validates the prior bundle, and only
restores the most recent matching promotion once. It preserves the promotion
record and artifacts:

```powershell
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow rollback
.\venv\Scripts\python.exe -m src.retraining.stage5_workflow rollback --execute
```

## Limitations

No genuine eligible cohort or real Stage 4 candidate currently exists, so this
project cannot establish candidate readiness. In-memory API/Spark feature
history is process-local; changing an active model requires restarting both
serving processes, and Spark must not be duplicated on the same checkpoint.
The manifest does not reload an already-running process. Promotion tests use
temporary model artifacts and do not touch the active deployment. No model was
trained or activated as part of implementing Stage 5.
