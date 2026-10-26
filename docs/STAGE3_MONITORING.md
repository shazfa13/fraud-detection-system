# Stage 3: Streaming drift and performance monitoring

Stage 3 runs separately from the Spark prediction callback. It reads the append-only Stage 1
JSONL and Stage 2 SQLite store, computes a report, and persists monitoring runs, alerts, and
retraining requests in `data/stage3/monitoring.sqlite3`. It never starts training or changes a
model, threshold, Kafka topic, offset, or Spark checkpoint. Stage 1 stores the exact ordered
34-feature inference vector with each new output record; identifiers, `step`, and target labels
are not included in the drift vector. Older Stage 1 rows without vectors are skipped and can make
the current window insufficient until enough new rows arrive. Monitoring errors do not run in
the prediction process.

## Reference and drift method

The reference is the active model's chronological training window from
`data/processed/v2_retraining_dataset.parquet` (V2) or the corresponding V1 metadata and dataset
contract. It is fixed by model metadata `train_end_step`; the target, `event_id`, and `step` are
excluded from feature calculations. The parquet is scanned in bounded Arrow batches. Sampling
combines a deterministic uniform row reservoir targeted at 5,000 rows with per-feature finite-row
coverage reservoirs targeted at at least 30 observations (or the detector's configured minimum,
if larger). The final union can exceed 5,000 rows. Missing/non-finite feature values are retained
and counted rather than requiring all 34 features to be present in one row. This avoids loading the
multi-million-row dataset into memory and prevents sparse history features from silently receiving
only a few reference observations. Targeted rows modestly overrepresent sparse-feature cases and
may alter cross-feature correlations; the report records the method, source counts, sample counts,
per-feature finite/missing coverage, seed, feature list, and reference version. PaySim is synthetic
historical data, not a claim of real-world distribution representativeness.

Stage 1's latest 500 JSONL records are considered; valid rows are filtered to the configured
active model version and deduplicated by transaction ID. The existing `src.drift` PSI plus
KS/two-proportion tests and Benjamini-Hochberg correction are reused. PSI >= 0.20 or a corrected
p-value below 0.05 marks a feature; overall drift uses the existing 15% tested-feature decision
rule. These documented detector conventions are provisional monitoring rules, not calibrated
fraud-impact thresholds. At least 30 valid current observations are required, consistent with
the detector's per-feature minimum. Not enough records means `insufficient_data`, not no drift.

## Delayed-label performance and baseline

Only unique transactions joined to a resolved `trusted_verified` Stage 2 label and the matching
model-version prediction are candidates. The Stage 2 trusted-source allowlist and 32-character
authorization token must be configured. Synthetic-origin predictions, synthetic/untrusted labels,
pending/unmatched labels, any transaction with a conflicting label submission, and any prediction
with a prediction-conflict audit row are excluded. Duplicate submissions do not multiply rows;
Stage 2's `labels` table contains one retained label per transaction. The report separately counts
synthetic and pending records and identifies the label-source breakdown. Ineligible outcomes are
never imputed.

Classification uses the model's saved threshold (`model_v2_threshold.json`, 0.53 for V2), and
reports TP, FP, TN, FN, precision, recall, and F1. The baseline comes from the same model
metadata's chronological held-out test metrics, at that saved threshold (V2 test split: 181,624
rows, precision 0.99847, recall 0.99949, F1 0.99898). This baseline is a synthetic PaySim cohort;
comparability to verified live labels is unproven and is explicitly recorded. At least 100 trusted
labels and 10 fraud labels are required before metrics are actionable. The 0.10 absolute decline
for precision, recall, or F1 is configurable but **provisional**; it needs calibration with an
adequate verified cohort. No performance claim is made from the current one synthetic label.

Drift and performance alerts are separate. A feature shift cannot itself create a performance
alert. Insufficient labels yield an `insufficient_data` performance state. An alert is unique per
model, type, and observation window; repeated polling does not add a second alert. If drift or
performance produces an alert, a retraining request record is created with its reason, model,
alert identifiers, and evidence. The request is a Stage 4 input only; no training/activation code
is called.

## Run locally

From the project root with the project virtual environment:

```powershell
.\venv\Scripts\python.exe -m src.monitoring.cli
```

The CLI prints one JSON report and writes restart-safe records to the Stage 3 database. It does
not initialize or write the Stage 2 database. The dashboard's Streaming Monitor page reads the
latest persisted report and shows the model, threshold, drift/performance states, verified sample,
alerts, and request status. A missing report is normal until the command has run once.

## Tests and limitations

Focused tests use temporary parquet, JSONL, and SQLite paths:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_stage3_monitoring -v
```

They cover controlled distribution shifts, a same-distribution sample, insufficient/malformed
windows, repeated alert deduplication, trusted-label confusion counts, synthetic/conflict
exclusion, and the saved model threshold. These are deterministic unit tests, not live Kafka/Spark
integration tests. This system currently has no genuine trusted label source configured and its
Stage 1 historical JSONL predates feature-vector persistence. Consequently performance is
unavailable and live drift may remain insufficient until enough new Stage 1 rows are collected.
The saved reference and baseline are synthetic PaySim artifacts; operational thresholds still
need calibration against an authorized verified cohort before any retraining decision is acted on.

## Alert validation before Stage 4 — 2026-10-25

The persisted V2 alert reported 31/34 drifted features (0.9117647), using the configured
`features_drifted / features_tested >= 0.15` rule. That arithmetic and decision are correct for
the values presented to the detector. The alert is evidence of a difference between the sampled
feature frames; by itself it is not evidence that a deployed model has degraded.

**Verified from local artifacts and code:**

- V2 metadata uses training through step 422. The V2 retraining report says its training dataset
  combines Phase 3 rows at steps 1–323 with Phase 9 rows at steps 379–743. Its validation ends at
  step 565 and the held-out test starts after that.
- The Stage 1 producer previously selected events using the shared Phase 7/V1 metadata, whose
  test boundary is step 378. Its local synthetic loop selected 20 rows; the observed Stage 1
  vectors all had `step=379` and `data_origin=synthetic_paysim_demo`, then repeated the same
  `(type, amount)` sequence. Step 379 is inside the V2 training range.
- `src.api.app.build_feature_vector` is the shared builder used by historical `generate_feature_dataset`
  and Stage 1 inference. Ordered feature names match the 34-field V2 metadata contract. The static
  amount/balance calculations and one-hot categories matched for the first 20 logged events and
  corresponding step-379 parquet rows. Stateful account aggregates did not consistently match:
  destination history differed on 13/20 paired rows. Stage 1 starts with empty in-memory account
  state, while historical features were generated by replaying earlier transactions. The producer
  then repeats the same step without advancing the transaction timeline, so history counts/totals
  accrue in the synthetic window. This is a feature-context mismatch, not a changed feature name
  or a demonstrated formula change.
- The former reference sampler required every one of the 34 features to be finite in a row. The
  reported reference therefore had only 2,070 complete cases out of the V2 training range. Current
  drift calculations filter finite values independently per feature, so this complete-case
  reference selection could bias per-feature PSI and significance tests. The sampler now samples
  reference rows first and retains per-feature missing values; the existing detector handles each
  feature's finite observations and minimum count.
- PSI flags a feature at >=0.20; the existing detector ORs that with a Benjamini–Hochberg adjusted
  p-value <0.05 (KS for continuous, two-proportion test for binary). The overall 15% rule was
  applied as documented. These detector mechanics do not explain away the cohort mismatch.
- The old alert persisted the aggregate count/rate and drifted names but not PSI or p-values, so its
  per-feature trigger (PSI versus corrected significance) cannot be recovered. New monitoring
  reports persist per-feature finite counts, missing rates, PSI, raw and adjusted p-values, and the
  detector's reason string for auditability.
- The saved alert and request tables currently contain one alert and one linked request. Their
  identifiers and original evidence were read only and remain unchanged. Re-evaluating a fixed
  window in temporary tests records one alert and one request; the second run marks the alert as
  already recorded.

**Targeted corrections:** the live Stage 1 demo producer now selects the test window from the
configured model's metadata (V2 by default), avoiding a stream that overlaps V2 training. A
synthetic-only window can still produce an explicitly synthetic feature-drift alert, but cannot
create a Stage 4 retraining request. Existing alert/request rows are never cleared or rewritten.

After correcting reference sampling, a separate temporary-database evaluation of the then-current
Stage 1 log scanned all 5,015,193 V2 training-window rows and retained the configured 5,000-row
reservoir sample (including feature-level missing values). Its current window was synthetic-only;
30 features met per-feature sample minimums and 28 were flagged (0.9333). Performance remained
`insufficient_data`, and no retraining request was made. This confirms that the observed synthetic
feature vectors still differ substantially from the corrected reference; it does not establish a
genuine production-population drift or model-performance decline. A deterministic same-distribution
test sample did not trigger the alert, while the controlled shifted sample did.

The logged current feature data is not a complete replayable copy of the original alert window:
the alert recorded 369 input lines, while the local log available during this validation contained
fewer feature-bearing records in that original time interval. The exact original 31-feature list
and PSI/statistic table were not persisted with the alert. Therefore the precise fraction of the
original drift rate attributable to complete-case sampling versus state-context and repeated-demo
effects cannot be reconstructed exactly. The next diagnostic should collect a fresh, unique,
monotonic, held-out V2 sample with its feature vectors and origin markers, then compare it against
a deterministic per-feature reference sample using the same missing-value rule. Do not advance to
Stage 4 using the existing synthetic-only request.
