# Project results and limitations

This report separates V2's chronological test-set metrics from Stage 6's smaller demonstration evaluation. All underlying labels come from synthetic PaySim data; neither result establishes real-world bank performance.

## Dataset and model contract

The V2 training report identifies `data/processed/v2_retraining_dataset.parquet` as a 5,382,204-row PaySim-derived dataset, with 7,649 fraud labels and 5,374,555 legitimate labels. The chronological partitions are training through step 422 (5,015,193 rows), validation through step 565 (185,387 rows), and test through step 743 (181,624 rows). The test partition contains 1,956 fraud rows (about 1.077%). The 34-feature order is frozen in V1/V2 metadata. `event_id` is an audit identity, `step` is a temporal ordering field, and `isFraud` is the target; none is a model input.

Model V2's threshold `0.53` was selected on validation data by maximum F1. Its saved test report is separate from both validation results and the later Stage 6 demo evaluation.

## V2 held-out test results

The table reports the V2 test split and V1 scored on that same split, using each model's saved threshold (V1 `0.60`; V2 `0.53`). These values are copied from `reports/model_v2_training_report.json`.

| Model / threshold | Rows / fraud | Precision | Recall | F1 | PR-AUC | ROC-AUC | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 / 0.60 | 181,624 / 1,956 | 0.991820 | 0.991820 | 0.991820 | 0.998165 | 0.999927 | 1,940 / 179,652 / 16 / 16 |
| V2 / 0.53 | 181,624 / 1,956 | 0.998468 | 0.999489 | 0.998978 | 0.999490 | 0.999682 | 1,955 / 179,665 / 3 / 1 |

On this test split, V2's precision, recall, F1, and PR-AUC are higher than V1's; V1's ROC-AUC is slightly higher. The Phase 11 comparison report's stated promotion-candidate criteria—V2 F1, PR-AUC, and recall not below V1—were met. This was a historical offline evaluation on synthetic data, and the recorded decision was `V2_CANDIDATE_FOR_DEPLOYMENT`; it should not be interpreted as an external production approval.

## Stage 6 demonstration evaluation

The latest recorded Stage 6 candidate is [`20261025T145931231833Z-b4bc8a7c`](../models/candidates/stage6_demo/20261025T145931231833Z-b4bc8a7c/training_report.json). It used a bounded 60,000 / 15,000 / 15,000 train/validation/evaluation sample, chronological V2 boundaries, and synthetic PaySim outcomes. Its 15,000-row evaluation sample contained 162 fraud rows. These figures are **not** V2's 181,624-row test results.

| Model on the same Stage 6 evaluation rows | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Active V2 | 0.53 | 1.000000 | 0.993827 | 0.996904 | 0.993910 | 0.996161 | 161 / 14,838 / 0 / 1 |
| Stage 6 demo candidate | 0.76 | 0.993671 | 0.969136 | 0.981250 | 0.985706 | 0.998079 | 157 / 14,837 / 1 / 5 |

On this sample, the demo candidate's thresholded precision, recall, F1, and accuracy were below V2's; its ROC-AUC was higher and PR-AUC lower. It was not promoted: its provenance is synthetic demonstration data, the report sets `eligible_for_genuine_promotion` to false, and Stage 5 rejects that marker. This is evidence that the bounded candidate-training path ran and the safeguard held—not evidence of genuine adaptive retraining.

## Labels, monitoring, and retraining readiness

Stage 2 distinguishes `trusted_verified`, `untrusted`, and `synthetic_test` sources. In the current documented setup there is no configured genuine trusted-source allowlist/token and no eligible genuine verified label cohort; the local record described in the Stage 2 guide is a synthetic test label. Stage 4 requires at least 100 eligible verified labels, including 10 verified fraud labels. It therefore stops before genuine candidate creation. The historical Phase 9 PaySim labels are explicitly `SIMULATED_DELAYED_GROUND_TRUTH` and cannot satisfy this gate.

The saved Stage 3 run recorded 31 of 34 features drifted and a `feature_drift` retraining request for a synthetic-only window. Stage 3 correctly does not turn that request into training or promotion. The alert is statistical evidence for that observed synthetic comparison, not proof of live production drift. Phase 13 did not rerun the persistent monitor.

## What Phase 13 established

The Phase 13 Stage 7 report records 57 passing focused Stage 1–6 tests and 107 passing tests in full `unittest` discovery, syntax parsing of 50 Python files, matching V1/V2 artifact hashes, and an unchanged snapshot of 495 local files covering Stage 2/3 databases, checkpoints, model artifacts, and prediction JSONL logs. Tests exercised lifecycle guards with temporary fixtures. The project venv lacked `pytest`; it was not installed.

These checks support code-path and safety-contract behavior. They do not prove live service availability, load capacity, production topic behavior, performance on real investigations, cross-process account-state correctness, or that a genuine candidate can be trained and safely deployed. Historical Phase 12 evidence includes API and Kafka/Spark validation, but Phase 13 deliberately did not repeat those live runs.

## Limitations and work still needed

- PaySim is synthetic; the apparent class patterns and very high offline metrics may not transfer to actual adversarial behavior.
- No genuine, authorized delayed-label provider is configured. Real retraining must wait for verified labels with documented provenance and matching transactions.
- Stage 3's minimum sample rules and alert thresholds remain provisional; monitoring needs a comparable, sufficiently large verified population before operational decisions.
- `AccountState` is process-local. The current streaming configuration uses one Kafka partition; scaling or process restart requires a deliberate account-history strategy.
- The Stage 1 producer is an unbounded synthetic replay loop. It assigns fresh event IDs while replaying a finite held-out slice; event-ID uniqueness does not make underlying transactions new.
- Promotion is an operator-controlled offline procedure. A changed activation manifest is read on process startup; there is no live automatic drift-to-training-to-deployment connection.
- Phase 13 did not exercise current live Kafka/Spark services or the persistent monitor. Follow the runbook only in an environment where new Kafka messages and append-only outputs are intended.

## Source reports

- [V2 training and same-split V1 comparison](../reports/model_v2_training_report.json)
- [V2 deployment validation (historical Phase 12 evidence)](../reports/model_v2_deployment_report.json)
- [Phase 10 dataset construction](../reports/v2_retraining_dataset_report.json)
- [Phase 9 simulated label collection](../reports/recent_label_collection_report.json)
- [Stage 3 drift report](../reports/drift_report.json)
- [Stage 7 machine-readable validation](../reports/stage7_end_to_end_validation_20261025T150813Z.json)
