"""Phase 11 configuration for training and evaluating Model V2.

This module is intentionally separate from the Phase 10 configuration.  It only
defines Phase 11 inputs, outputs, and experiment constants; it never changes
any earlier-phase artifact.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V2_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "v2_retraining_dataset.parquet"
MODEL_V1_PATH = PROJECT_ROOT / "models" / "xgboost_model_v1.json"
METADATA_V1_PATH = PROJECT_ROOT / "models" / "model_v1_metadata.json"
THRESHOLD_V1_PATH = PROJECT_ROOT / "models" / "model_v1_threshold.json"

MODEL_V2_PATH = PROJECT_ROOT / "models" / "xgboost_model_v2.json"
METADATA_V2_PATH = PROJECT_ROOT / "models" / "model_v2_metadata.json"
THRESHOLD_V2_PATH = PROJECT_ROOT / "models" / "model_v2_threshold.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TRAINING_REPORT_PATH = REPORTS_DIR / "model_v2_training_report.json"
COMPARISON_JSON_PATH = REPORTS_DIR / "model_v1_vs_v2_comparison.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_v1_vs_v2_comparison.csv"

FIG_PR_CURVE = FIGURES_DIR / "phase11_v2_precision_recall_curve.png"
FIG_CONFUSION_MATRIX = FIGURES_DIR / "phase11_v2_confusion_matrix.png"
FIG_THRESHOLD_ANALYSIS = FIGURES_DIR / "phase11_v2_threshold_analysis.png"
FIG_PROBABILITY_DIST = FIGURES_DIR / "phase11_v2_probability_distribution.png"
FIG_V1_VS_V2 = FIGURES_DIR / "phase11_v1_vs_v2_metrics.png"

RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 30
RECENT_TRAIN_EXTENSION_FRACTION = 0.60
RECENT_VALIDATION_FRACTION = 0.20
RECENT_TEST_FRACTION = 0.20
THRESHOLD_GRID_STEP = 0.01
THRESHOLD_SELECTION_OBJECTIVE = "maximum F1 on the validation split only"

FROZEN_FILES = [
    "models/xgboost_model_v1.json", "models/model_v1_metadata.json", "models/model_v1_threshold.json",
    "notebooks/08_drift_detection.ipynb", "src/drift/config.py", "src/drift/drift_detector.py",
    "reports/drift_report.json", "reports/drift_feature_summary.csv",
    "notebooks/09_recent_label_collection.ipynb", "src/labels/config.py", "src/labels/label_collector.py",
    "reports/recent_label_collection_report.json",
    "notebooks/10_build_v2_retraining_dataset.ipynb", "src/retraining/config.py",
    "src/retraining/dataset_builder.py", "reports/v2_retraining_dataset_report.json",
    "src/api/app.py", "src/streaming/kafka_producer.py", "src/streaming/spark_streaming.py",
]
FROZEN_DATA_FILES = ["data/processed/recent_labelled_data.parquet", "data/processed/v2_retraining_dataset.parquet"]
