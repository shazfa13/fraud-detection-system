"""
Phase 7 configuration — Kafka + Spark Structured Streaming.
=============================================================

Every path/constant Phase 7's own code needs lives here, exactly once. Nothing here changes
anything about Model V1, the Phase 3 feature contract, or the Phase 5 API — those files
(models/*.json, src/api/app.py) are read, never written, by Phase 7.

Resolution note: this file resolves the project root from its own location
(src/streaming/config.py -> parents[2] is the project root), so it behaves the same whether
launched from the project root, from notebooks/, or via `spark-submit` from anywhere.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Existing Phase 1-6 artifacts (READ ONLY - Phase 7 never writes to these) -----------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model_v1.json"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "model_v1_threshold.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_v1_metadata.json"
FE_SUMMARY_PATH = PROJECT_ROOT / "reports" / "feature_engineering_summary.json"

# ---- Kafka ---------------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
KAFKA_TOPIC = "fraud-transactions"
KAFKA_NUM_PARTITIONS = 1     # see spark_streaming.py docstring: >1 partition would break the
                             # single global AccountState's chronological-order guarantee unless
                             # state were partitioned by account key - a documented Phase 7 limitation.

# ---- Local bridge (used ONLY when a live Kafka broker is not reachable - see README/notebook) --
# Same JSON payloads, same ordering guarantees, delivered as one file per message instead of over
# the Kafka wire protocol. This lets every stage AFTER "receive a transaction event" (JSON parsing,
# Spark Structured Streaming, feature engineering, Model V1, risk scoring) be exercised for real
# with genuine Spark Structured Streaming machinery, without requiring a reachable Kafka broker.
LOCAL_BRIDGE_INBOX_PATH = PROJECT_ROOT / "data" / "streaming_inbox"

# ---- Spark --------------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "spark_fraud_stream"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "phase7_streaming_predictions.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "phase7_streaming_summary.json"

# Maven coordinates for Spark's Kafka source connector. Fetched at runtime by `spark-submit
# --packages` (or `spark.jars.packages`) from Maven Central - requires that this host can reach
# Maven Central, which the sandboxed environment this code was authored/tested in could NOT
# (see the notebook's Environment section for the exact, reproducible test that established this).
# Pick the artifact matching the installed Spark/Scala build exactly. This project is using
# PySpark 4.2.0 with Scala 2.13.x, so the connector coordinate is:
#   org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0
SPARK_VERSION = "4.2.0"
SPARK_SCALA_VERSION = "2.13"
SPARK_KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_{SPARK_SCALA_VERSION}:{SPARK_VERSION}"
SPARK_KAFKA_PACKAGE_TEMPLATE = "org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}"

# ---- Producer pacing -------------------------------------------------------------------------
SIMULATION_DELAY = 0.5      # seconds between published transactions (mirrors Phase 6's configurable delay)
NUM_TRANSACTIONS = 20        # Phase 7's minimum required validation size; raise freely

MODEL_VERSION_EXPECTED = "V1"
