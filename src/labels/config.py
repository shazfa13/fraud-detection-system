"""
Phase 9 configuration — Recent Labelled Data Collection.
============================================================

Every tunable knob Phase 9's own code needs lives here, exactly once (same convention as
src/streaming/config.py and src/drift/config.py). Nothing here changes Model V1, the 34-feature
contract, the threshold, or any Phase 3/5/7/8 logic - those are read-only inputs to Phase 9.

Resolution note: resolves the project root from its own location (src/labels/config.py ->
parents[2] is the project root), matching the other src/*/config.py modules.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Existing Phase 1-8 artifacts (READ ONLY - Phase 9 never writes to these) -----------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_transactions.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model_v1.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_v1_metadata.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "model_v1_threshold.json"
FE_SUMMARY_PATH = PROJECT_ROOT / "reports" / "feature_engineering_summary.json"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.json"              # Phase 8 provenance (authoritative recent window)
DRIFT_FEATURE_SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift_feature_summary.csv"
DRIFT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_drift_detection.ipynb"
DRIFT_DETECTOR_PATH = PROJECT_ROOT / "src" / "drift" / "drift_detector.py"
DRIFT_CONFIG_PATH = PROJECT_ROOT / "src" / "drift" / "config.py"

# Phase 6 artifacts (inspected for provenance; NOT used as the Phase 9 matching key - see the
# notebook's "Resolve Transaction Identity" section for why: Phase 6's `simulation_id` is scoped
# to one small demonstration run, not the full recent window Phase 8's drift decision concerns).
PHASE6_DELAYED_GT_PATH = PROJECT_ROOT / "reports" / "delayed_ground_truth.csv"
PHASE6_SUMMARY_PATH = PROJECT_ROOT / "reports" / "live_phase6_summary.json"

# ---- Phase 7 artifacts (hashed for frozen-integrity verification only) ------------------------
STREAMING_PRODUCER_PATH = PROJECT_ROOT / "src" / "streaming" / "kafka_producer.py"
STREAMING_SPARK_PATH = PROJECT_ROOT / "src" / "streaming" / "spark_streaming.py"
APP_PATH = PROJECT_ROOT / "src" / "api" / "app.py"

# ---- Phase 9 outputs ------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RECENT_LABELLED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "recent_labelled_data.parquet"
REPORT_PATH = REPORTS_DIR / "recent_label_collection_report.json"
FIG_LABEL_DISTRIBUTION = FIGURES_DIR / "phase9_label_distribution.png"
FIG_LABEL_TIMELINE = FIGURES_DIR / "phase9_label_timeline.png"

# ---- Matching / validation ------------------------------------------------------------------
TRANSACTION_ID_COLUMN = "event_id"   # reuses Phase 3's existing, already-documented identifier
                                     # (META_COLS = ["event_id", "step"] in notebooks/03) - the row's
                                     # position after a STABLE sort of the full raw dataset by step.
                                     # Deterministic and collision-safe by construction (sequential
                                     # integers assigned once, over the whole historical file).
GROUND_TRUTH_SOURCE = "raw PaySim dataset (data/raw/*.csv), column 'isFraud' - SIMULATED DELAYED " \
                      "GROUND TRUTH (an academic stand-in: the label already exists in the frozen " \
                      "historical file, but is treated here as being revealed only now, for the " \
                      "recent window Phase 8's drift decision concerns). This is NOT real production " \
                      "label collection (no real investigations/chargebacks are involved)."
TIMELINE_BIN_COUNT = 40   # number of step-bins for the fraud-rate-over-time figure
