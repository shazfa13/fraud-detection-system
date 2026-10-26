"""Stage 3 monitor paths and deliberately configurable decision settings."""
from pathlib import Path

from src.drift import config as drift_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DB_PATH = PROJECT_ROOT / "data" / "stage3" / "monitoring.sqlite3"
STAGE1_JSONL_PATH = PROJECT_ROOT / "data" / "stage1_streaming" / "predictions.jsonl"
REFERENCE_DATA_PATHS = {
    "V1": PROJECT_ROOT / "data" / "processed" / "featured_transactions.parquet",
    "V2": PROJECT_ROOT / "data" / "processed" / "v2_retraining_dataset.parquet",
}
MODEL_METADATA_PATHS = {"V1": PROJECT_ROOT / "models" / "model_v1_metadata.json",
                        "V2": PROJECT_ROOT / "models" / "model_v2_metadata.json"}
MODEL_THRESHOLD_PATHS = {"V1": PROJECT_ROOT / "models" / "model_v1_threshold.json",
                         "V2": PROJECT_ROOT / "models" / "model_v2_threshold.json"}
BASELINE_REPORT_PATHS = {"V1": PROJECT_ROOT / "models" / "model_v1_metadata.json",
                         "V2": PROJECT_ROOT / "models" / "model_v2_metadata.json"}
ACTIVE_MODEL_VERSION = "V2"  # resolved from serving environment at runtime by monitor.get_active_model
RECENT_RECORD_LIMIT = 500
MIN_CURRENT_WINDOW = max(30, drift_config.MIN_SAMPLE_SIZE)
REFERENCE_SAMPLE_SIZE = 5000
REFERENCE_RANDOM_SEED = 3025
MIN_ELIGIBLE_LABELS = 100  # provisional floor; fraud recall needs >=10 positives too
MIN_ELIGIBLE_FRAUD_LABELS = 10
PERFORMANCE_DROP_THRESHOLD = 0.10  # provisional absolute metric drop; requires calibration on verified labels
PERFORMANCE_METRICS = ("recall", "f1", "precision")
BASELINE_SPLIT = "test"
