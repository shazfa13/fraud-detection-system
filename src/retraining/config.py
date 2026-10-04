"""
Phase 10 configuration — Build V2 Retraining Dataset.
==========================================================

Every tunable knob Phase 10's own code needs lives here, exactly once (same convention as
src/streaming/config.py, src/drift/config.py, src/labels/config.py). Phase 10 only ever READS
Model V1, Phase 8, and Phase 9 artifacts - it never writes to any of them.

Resolution note: resolves the project root from its own location (src/retraining/config.py ->
parents[2] is the project root), matching the other src/*/config.py modules.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Existing Phase 1-9 artifacts (READ ONLY - Phase 10 never writes to these) ----------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_transactions.parquet"   # Phase 3's full historical features (old V1 training source), if present
RECENT_LABELLED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "recent_labelled_data.parquet"  # Phase 9's authoritative output - NEVER regenerated here
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model_v1.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_v1_metadata.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "model_v1_threshold.json"

# ---- Frozen artifacts hashed before/after (Section: Integrity Checks) -------------------------
FROZEN_FILES = [
    "models/xgboost_model_v1.json", "models/model_v1_metadata.json", "models/model_v1_threshold.json",
    "notebooks/08_drift_detection.ipynb", "src/drift/config.py", "src/drift/drift_detector.py",
    "reports/drift_report.json", "reports/drift_feature_summary.csv",
    "notebooks/09_recent_label_collection.ipynb", "src/labels/config.py", "src/labels/label_collector.py",
    "reports/recent_label_collection_report.json",
    "src/api/app.py", "src/streaming/kafka_producer.py", "src/streaming/spark_streaming.py",
]
# Not in the task's own enumerated hash list, but "the Phase 9 dataset must remain unchanged" implies
# it too - hashed for the same before/after discipline, kept separate since it is data, not code.
FROZEN_DATA_FILES = ["data/processed/recent_labelled_data.parquet"]

# ---- Phase 10 outputs ------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "v2_retraining_dataset.parquet"
REPORT_PATH = REPORTS_DIR / "v2_retraining_dataset_report.json"

# ---- Temporal design (resolved dynamically from model_v1_metadata.json at runtime - these are
# the EXPECTED values documented here for readability only; the notebook verifies them against
# the live metadata rather than trusting these constants blindly). -------------------------------
EXPECTED_OLD_TRAIN_STEP_RANGE = (1, 323)          # Model V1's own training period
EXPECTED_VALIDATION_GAP = (324, 378)              # excluded entirely from the Phase 10 dataset
EXPECTED_RECENT_STEP_RANGE = (379, 743)           # Phase 9's recent labelled window

# ---- Columns explicitly excluded from the 34-feature model contract (never written as features) -
AUDIT_ONLY_COLUMNS = ["fraud_probability", "predicted_label", "risk_score", "risk_level"]
NON_FEATURE_IDENTITY_COLUMNS = ["event_id", "nameOrig", "nameDest"]   # identity/metadata, never features
TARGET_COLUMN = "isFraud"
TRANSACTION_ID_COLUMN = "event_id"   # reused unchanged from Phase 3/9 - see src/labels/label_collector.py docstring
