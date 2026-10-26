# Project presentation outline and viva preparation

This outline is designed for a 10–12 slide B.Tech CSE project defense. Present the system as an academic prototype validated on synthetic PaySim data. Keep “historically tested,” “unit tested,” “demonstrated,” and “not yet verified live” distinct.

## Slide outline

### 1. Title and introduction

- Adaptive Real-Time Financial Transaction Fraud Detection System.
- A Python/XGBoost prototype combining inference, streaming, labels, monitoring, and guarded offline retraining workflows.
- Active model: V2, threshold 0.53, 34 ordered inputs.

**Suggested visual:** Project title with a simple transaction-to-risk icon flow.

### 2. Problem statement and objectives

- Fraud classification is highly imbalanced; accuracy alone can hide missed fraud.
- Transaction history must be computed using only information available before each transaction.
- Goal: demonstrate low-latency scoring, auditable delayed labels, drift evidence, and safe model-lifecycle controls.
- PaySim is synthetic; results are educational, not a banking deployment claim.

**Suggested visual:** Fraud/legitimate imbalance graphic and a leakage “before vs. after” timeline.

### 3. Proposed system and architecture

- Kafka transports transaction events; Spark Structured Streaming processes micro-batches.
- A shared feature builder and active-model resolver feed XGBoost scoring.
- FastAPI exposes direct inference and health reporting.
- Delayed labels and monitoring are separate workflows; drift does not trigger training automatically.

**Suggested visual:** Use the [Mermaid architecture](PROJECT_WORKFLOW.md#architecture).

### 4. Technology stack

- Python, Pandas, NumPy, PyArrow for application and tabular data work.
- XGBoost for binary classification; scikit-learn for evaluation metrics; SciPy for statistical testing.
- FastAPI/Pydantic for validated REST requests.
- Kafka and Spark Structured Streaming for event ingestion and micro-batch inference.
- SQLite and JSONL for local registries and append-only records.

**Suggested visual:** Stack diagram by data, model, serving, streaming, and storage layers.

### 5. Fraud model and feature engineering

- XGBoost receives the frozen, metadata-ordered 34-feature contract.
- Inputs cover time, amount and ratios, pre-transaction balances, past sender/destination aggregates, destination velocity windows, and transaction type.
- `isFraud` is the training target only; `event_id` and `step` are audit/order fields.
- Account history is past-only and held in process memory.

**Suggested visual:** Feature-group blocks feeding a 34-column vector; explicitly cross out `isFraud` from inference inputs.

### 6. Real-time Kafka/Spark prediction workflow

- The synthetic producer emits raw events without `isFraud`.
- Spark validates and orders events, calls the shared feature builder, and scores with the configured model.
- Predictions/errors/status are recorded; Stage 1 also registers predictions with Stage 2.
- The project uses one Kafka partition and an existing checkpoint; streaming is at-least-once.

**Suggested visual:** Numbered event lifecycle with output and error sinks.

### 7. Delayed labels and monitoring

- Stage 2 joins delayed outcomes to stable transaction IDs and preserves the first label.
- Trusted labels require a configured source allowlist and authorization token; synthetic labels are excluded.
- Stage 3 calculates feature drift separately from performance on eligible verified labels.
- Alerts and retraining requests are persisted evidence; they do not train or promote models.

**Suggested visual:** Timeline from prediction to delayed outcome to monitoring report.

### 8. Guarded retraining and model lifecycle

- Stage 4 checks provenance, matching, conflicts, feature vectors, and minimum eligible-label counts.
- Stage 5 compares a separate candidate with the active model and binds human approval to exact artifacts and evaluation evidence.
- Promotion and rollback are explicit operations; running services need restart to load a new model.
- Stage 6 demonstration artifacts are blocked from genuine promotion.

**Suggested visual:** Eligibility → candidate → evaluation → human authorization → activation/rollback, with a gate at every transition.

### 9. V1/V2 evaluation results

- V2 test split: 181,624 PaySim rows, 1,956 fraud cases; V2 threshold 0.53.
- V2: precision 0.998468, recall 0.999489, F1 0.998978, PR-AUC 0.999490, ROC-AUC 0.999682.
- V1 on the same split: precision/recall/F1 0.991820, PR-AUC 0.998165, ROC-AUC 0.999927.
- V2 had fewer FP/FN (3/1 vs. 16/16); V1 ROC-AUC was slightly higher.

**Suggested visual:** Grouped metric chart with thresholds and the synthetic dataset note visible.

### 10. Stage 6 demonstration and promotion safeguard

- Stage 6 sampled 60k/15k/15k rows chronologically from historical PaySim.
- Candidate on 15k evaluation rows (162 fraud): precision 0.993671, recall 0.969136, F1 0.981250, threshold 0.76.
- V2 on the same rows: precision 1.000000, recall 0.993827, F1 0.996904, threshold 0.53.
- The candidate is labeled demonstration-only and ineligible for genuine promotion; no promotion occurred.

**Suggested visual:** Same-sample comparison table plus a locked “promotion blocked” badge.

### 11. Testing, limitations, and current status

- Phase 13: 57 focused Stage 1–6 tests and 107 full `unittest` tests passed.
- Syntax and local artifact/persistence integrity checks passed.
- Phase 13 did not repeat live Kafka/Spark/API services or run persistent Stage 3 monitoring.
- No genuine trusted label source or eligible verified-label cohort is available; the project is not production-ready.

**Suggested visual:** Checklist divided into tested, demonstrated, and pending live/real-label work.

### 12. Conclusion and future work

- The project demonstrates how to connect inference with auditable labels, monitoring evidence, and cautious offline model management.
- V2 remains active; Stage 6 did not replace it.
- Next: authorized genuine label integration, representative evaluation, calibrated alert/promotion policy, distributed account state, and controlled live testing.

**Suggested visual:** Current prototype → verified data → reviewed candidate → controlled deployment roadmap.

## Viva questions and answers

### 1. Why did you choose XGBoost?

XGBoost is a gradient-boosted decision-tree library that handles nonlinear interactions in tabular data and supports class weighting. It is the model used by this project. The reported results do not prove it is universally better than other algorithms; that would require a controlled comparison under the same data and evaluation protocol.

### 2. What is supervised learning?

Supervised learning fits a model from examples containing input features and known target labels. Here, the input is the 34-feature transaction vector and the training target is `isFraud`, with 1 meaning fraud and 0 meaning legitimate.

### 3. Why does the model use 34 features?

The 34 inputs are the project's frozen feature contract, preserved in V1/V2 metadata order. They cover transaction amount and ratios, pre-transaction balances, past account behavior, time/velocity, and one-hot transaction type. The contract was selected in the feature-engineering phases; changing it requires a coordinated model and serving update.

### 4. Why must `isFraud` be excluded from inference?

It is the answer the model is supposed to predict. Supplying it at prediction time would leak the target and invalidate the result. Pydantic rejects it as an unsupported extra field, and the Kafka/Spark event schema also omits it.

### 5. What are `event_id` and `step` used for?

`event_id` is an audit/matching identity in the historical dataset, while `step` orders simulated transactions in time (one PaySim step represents an hour). They support joins and chronological splits, but neither is part of the model input vector.

### 6. What do precision, recall, and F1 mean?

Precision is the fraction of predicted fraud cases that are actually fraud; recall is the fraction of actual fraud cases detected. F1 is their harmonic mean. High recall reduces missed fraud, while high precision limits false alerts. Their tradeoff matters more than accuracy in an imbalanced dataset.

### 7. How do PR-AUC and ROC-AUC differ?

PR-AUC summarizes the precision/recall curve and is informative when the positive class is rare. ROC-AUC summarizes true-positive versus false-positive rates across thresholds. Neither selects the operating threshold by itself, and both depend on the evaluation population and labels.

### 8. Why split chronologically instead of randomly?

Transactions later in time can have different behavior and must not influence training of a model evaluated on earlier data. Chronological partitions reduce future-information leakage and better mimic deployment. They do not alone guarantee that the evaluation set represents future real-world traffic.

### 9. What does Kafka do here?

Kafka is the event broker: the producer publishes transaction messages to the `fraud-transactions` topic and Spark consumes them. The project configures a single partition because its account history is process-local and ordered processing is required.

### 10. What does Spark Structured Streaming do?

Spark reads the topic as a stream, groups arriving events into micro-batches, validates and orders those records, and calls the prediction processing function. It is the streaming execution layer; it is not the classifier or the delayed-label database.

### 11. Why is only one partition configured?

The feature history (`AccountState`) is held in one process rather than partitioned by account or stored in a distributed state service. Multiple partitions could interleave account events across workers and make the past-only history inconsistent. Scaling state is future engineering work.

### 12. What are delayed labels?

They are outcomes learned after a prediction, for example after an investigation or chargeback. Stage 2 records these outcomes with source, authorization, timestamp, and submission audit data. This project has a local workflow but no genuine external investigation source configured.

### 13. How are PSI, KS, and multiple tests used?

PSI compares binned feature distributions. The Kolmogorov–Smirnov test compares two empirical continuous distributions; proportion tests are used for applicable categorical/binary data. Benjamini–Hochberg adjusts p-values across the feature tests to reduce false discoveries. The configured alert limits are provisional and need calibration on a representative population.

### 14. What is the difference between data drift and concept drift?

Data drift is a change in input feature distribution, often described as a change in P(X). Concept drift is a change in the relationship between features and target, P(Y|X). Feature-only monitoring can suggest data drift; establishing model-performance or concept change needs reliable, eligible labels.

### 15. Why are retraining eligibility gates needed?

They keep synthetic, unauthorized, unmatched, pending, conflicting, or insufficient labels from being treated as truth. Stage 4 currently requires at least 100 eligible verified labels including 10 fraud labels. A drift request is not permission to train.

### 16. How are candidates evaluated and promoted?

Stage 5 validates candidate metadata, hashes, schema, provenance, and held-out evaluation evidence, then compares the candidate with the active model on the same rows. Promotion requires an explicit human approval file bound to that candidate and report. The workflow does not autonomously approve or deploy a candidate.

### 17. How does rollback work?

The guarded Stage 5 workflow validates the recorded previous bundle and restores its activation manifest only when the promotion record and hashes match. It records the operation. Since services load the resolved model at startup, serving processes must restart to use the restored selection.

### 18. Why was the Stage 6 candidate not promoted?

It was trained as a demonstration from synthetic PaySim labels, not genuine verified outcomes. Its metadata explicitly sets `eligible_for_genuine_promotion` to false, and Stage 5 rejects demonstration candidates. On the same 15,000-row sample its thresholded precision, recall, and F1 were also below active V2's; this comparison is still synthetic.

### 19. What do the 107 passing tests establish?

They establish that the exercised unit and fixture-based paths passed, including validation, conflict handling, drift logic, eligibility gates, authorization checks, activation/rollback safeguards, and the demo guard. They do not establish production reliability, live-service behavior during Phase 13, real-world model quality, or performance under load.

### 20. What is needed for genuine adaptive retraining, and what are PaySim's limits?

The project needs an authorized source of correctly matched genuine outcomes, enough verified fraud and legitimate labels, representative time-separated evaluation data, calibrated acceptance/drift rules, and a reviewed candidate followed by controlled deployment and monitoring. PaySim is simulator-generated, lacks real institutional context, and cannot establish how human fraudsters or real customer populations will behave.

## Source material

Use the [results and limitations](PROJECT_RESULTS_AND_LIMITATIONS.md) for exact metric context, the [workflow guide](PROJECT_WORKFLOW.md) for architecture, and the [Stage 7 report](../reports/stage7_end_to_end_validation_20261025T150813Z.json) for Phase 13 test evidence.
