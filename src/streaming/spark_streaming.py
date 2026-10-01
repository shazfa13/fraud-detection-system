"""
Phase 7 — Spark Structured Streaming consumer for the fraud-transactions topic.
==================================================================================

Pipeline (exactly as required):

    Kafka (`fraud-transactions`)
        -> Spark Kafka DataFrame (format("kafka"))
        -> decode Kafka value (bytes -> UTF-8 JSON string)
        -> parse JSON with an explicit schema (`EVENT_SCHEMA`)
        -> foreachBatch: apply the EXISTING Phase 3/5 feature pipeline (imported, unchanged, from
           src.api.app) and the EXISTING, FROZEN Model V1 (models/xgboost_model_v1.json)
        -> fraud_probability / risk_score / risk_level via the EXISTING Phase 4 threshold/risk logic
        -> append to reports/phase7_streaming_predictions.csv

CRITICAL DESIGN CHOICE — feature/state reuse: this module does NOT reimplement feature
engineering. It imports `create_temporal_features`, `create_amount_features`,
`create_balance_features`, `encode_transaction_type`, `AccountState`, `build_feature_vector`,
`assign_risk_level`, `model`, `FINAL_FEATURES`, `CLASSIFICATION_THRESHOLD`, and `TransactionRequest`
directly from `src.api.app` (Phase 5). This guarantees zero drift from Phase 3/5's feature
definitions by construction — there is exactly one copy of the feature-engineering logic in this
project, and Phase 7 calls it rather than re-deriving it. Importing `src.api.app` also means this
process loads Model V1 exactly once at start-up, the same way the FastAPI process does.

Two ways to read the transaction stream (kept symmetric with kafka_producer.py):
  * `create_kafka_stream(spark)`        - real `format("kafka")`. Requires: (a) a reachable Kafka
                                          broker, (b) the `spark-sql-kafka-0-10` connector jar,
                                          fetched at runtime from Maven Central via
                                          `spark.jars.packages` / `--packages`. This is the primary,
                                          spec-compliant path.
  * `create_local_bridge_stream(spark)` - `format("json")` reading LOCAL_BRIDGE_INBOX_PATH (one file
                                          per message, written by kafka_producer.py's local-bridge
                                          transport). Used ONLY when (a) or (b) above is not
                                          available - see the Phase 7 notebook's Environment section
                                          for the concrete test that established this in the
                                          environment this code was authored and validated in.

Both paths call the exact same `process_batch` foreachBatch function, so everything downstream of
"a transaction event has arrived" (JSON parsing, ordering, feature engineering, Model V1, risk
scoring, output) is IDENTICAL regardless of which one is used.

IMPORTANT — single partition / single global account state: `AccountState` (Phase 5) keeps its
history in one Python dict inside one process. Spark's Kafka source can read multiple partitions
concurrently, which could deliver transactions to `process_batch` out of chronological order across
partitions. Phase 7 therefore requires the `fraud-transactions` topic to have exactly ONE partition
(config.KAFKA_NUM_PARTITIONS = 1) and additionally sorts every micro-batch by (step, sequence)
before touching AccountState, so historical features stay past-only even within a batch. Scaling to
multiple partitions would require partitioning AccountState by account key - a documented limitation,
not something this phase attempts to solve.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

from src.streaming import config
from src.streaming.kafka_producer import kafka_broker_reachable

# Reuse the EXISTING Phase 5 feature pipeline, state, and model - imported, not reimplemented.
from src.api.app import (TransactionRequest, build_feature_vector, assign_risk_level, model,
                         FINAL_FEATURES, CLASSIFICATION_THRESHOLD, MODEL_VERSION)

# -------------------------------------------------------------------------------------------
# Explicit schema for the Kafka JSON payload (matches kafka_producer.KAFKA_MESSAGE_FIELDS exactly;
# deliberately has NO `isFraud` field - the schema itself makes the label unrepresentable here).
# -------------------------------------------------------------------------------------------
EVENT_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("step", LongType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("amount", DoubleType(), nullable=False),
    StructField("nameOrig", StringType(), nullable=False),
    StructField("oldbalanceOrg", DoubleType(), nullable=False),
    StructField("newbalanceOrig", DoubleType(), nullable=True),
    StructField("nameDest", StringType(), nullable=False),
    StructField("oldbalanceDest", DoubleType(), nullable=False),
    StructField("newbalanceDest", DoubleType(), nullable=True),
])
assert "isFraud" not in EVENT_SCHEMA.fieldNames()


def _ensure_windows_hadoop_home() -> None:
    """PySpark on Windows expects HADOOP_HOME and a local winutils.exe helper."""
    if os.name != "nt":
        return
    hadoop_home = config.PROJECT_ROOT / "tools" / "hadoop"
    winutils = hadoop_home / "bin" / "winutils.exe"
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    if not winutils.exists():
        print(f"[INFO] HADOOP_HOME set to {hadoop_home}; winutils.exe is expected at {winutils}. Download it from the WinUtils project before running Kafka/Spark locally.")


def create_spark_session(app_name: str = "Phase7FraudStreaming") -> SparkSession:
    _ensure_windows_hadoop_home()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (SparkSession.builder
            .appName(app_name)
            .master("local[2]")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.jars.packages", config.SPARK_KAFKA_PACKAGE)
            .getOrCreate())


def _sequence_of(transaction_id: str) -> int:
    """SIM-000123 -> 123, used only to break ties within the same `step` in original send order."""
    try:
        return int(transaction_id.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def process_batch(batch_df: DataFrame, batch_id: int) -> None:
    """
    The one function both streaming sources call. Runs on the Spark driver: collects the
    (typically small) micro-batch to pandas, sorts it into chronological order, then processes
    transactions ONE AT A TIME so AccountState's past-only guarantee (Phase 5) is preserved.
    """
    if batch_df.rdd.isEmpty():
        return
    pdf = batch_df.toPandas()
    pdf["_seq"] = pdf["transaction_id"].map(_sequence_of)
    pdf = pdf.sort_values(["step", "_seq"], kind="stable").reset_index(drop=True)

    results = []
    for _, row in pdf.iterrows():
        txn = TransactionRequest(step=int(row["step"]), type=str(row["type"]), amount=float(row["amount"]),
                                 nameOrig=str(row["nameOrig"]), oldbalanceOrg=float(row["oldbalanceOrg"]),
                                 nameDest=str(row["nameDest"]), oldbalanceDest=float(row["oldbalanceDest"]))

        # --- Feature contract validation (per-transaction, not a one-off check) -----------------
        X = build_feature_vector(txn)                      # existing Phase 3/5 pipeline, unchanged
        if len(X.columns) != 34 or list(X.columns) != FINAL_FEATURES:
            raise RuntimeError(f"FEATURE CONTRACT VIOLATION on {row['transaction_id']}: "
                              f"got {len(X.columns)} feature(s) {list(X.columns)}, expected the "
                              f"existing 34-feature Model V1 contract {FINAL_FEATURES}. Stopping "
                              "rather than guessing; Model V1 itself was NOT modified.")

        fraud_probability = float(model.predict_proba(X)[:, 1][0])
        predicted_label = int(fraud_probability >= CLASSIFICATION_THRESHOLD)
        risk_score = round(fraud_probability * 100, 4)
        risk_level = assign_risk_level(risk_score)

        results.append({"transaction_id": row["transaction_id"], "step": int(row["step"]), "type": row["type"],
                        "amount": float(row["amount"]), "fraud_probability": fraud_probability,
                        "predicted_label": predicted_label, "risk_score": risk_score, "risk_level": risk_level,
                        "model_version": MODEL_VERSION, "prediction_timestamp": datetime.now(timezone.utc).isoformat()})
        print(f"[batch {batch_id}] {row['transaction_id']} step={int(row['step'])} type={row['type']:<9} "
             f"fraud_probability={fraud_probability:.4f} risk_level={risk_level}")

    out_df = pd.DataFrame(results)
    config.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not config.OUTPUT_PATH.is_file()
    out_df.to_csv(config.OUTPUT_PATH, mode="a", index=False, header=write_header)


def create_kafka_stream(spark: SparkSession):
    """Real Kafka source. Requires a reachable broker AND the spark-sql-kafka connector jar."""
    raw = (spark.readStream.format("kafka")
          .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
          .option("subscribe", config.KAFKA_TOPIC)
          .option("startingOffsets", "earliest")
          .load())
    parsed = raw.select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("event")).select("event.*")
    return (parsed.writeStream.foreachBatch(process_batch)
           .option("checkpointLocation", str(config.CHECKPOINT_PATH / "kafka"))
           .start())


def create_local_bridge_stream(spark: SparkSession):
    """Local-bridge source (see module docstring): format("json") over LOCAL_BRIDGE_INBOX_PATH."""
    config.LOCAL_BRIDGE_INBOX_PATH.mkdir(parents=True, exist_ok=True)
    raw = (spark.readStream.format("json").schema(EVENT_SCHEMA)
          .option("maxFilesPerTrigger", 5)
          .load(str(config.LOCAL_BRIDGE_INBOX_PATH)))
    return (raw.writeStream.foreachBatch(process_batch)
           .option("checkpointLocation", str(config.CHECKPOINT_PATH / "local_bridge"))
           .start())


def spark_kafka_connector_available(spark: SparkSession) -> bool:
    """Real check: try to actually construct a Kafka-format streaming DataFrame (fails fast if the
    connector jar is missing, WITHOUT starting a query)."""
    try:
        (spark.readStream.format("kafka")
         .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
         .option("subscribe", config.KAFKA_TOPIC).load())
        return True
    except Exception:
        return False


def run_streaming(mode: str = "auto", max_seconds: float = 60.0, idle_stop_after: float = 8.0):
    """
    Start the streaming query and let it run until either `max_seconds` elapses or no new batch
    has been processed for `idle_stop_after` seconds (this is a bounded VALIDATION run, not a
    daemon - a real deployment would simply not set these bounds).
    """
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    if mode == "auto":
        broker_ok = kafka_broker_reachable()
        connector_ok = spark_kafka_connector_available(spark)
        if broker_ok and connector_ok:
            mode = "kafka"
        else:
            reasons = []
            if not broker_ok:
                reasons.append(f"no Kafka broker reachable at {config.KAFKA_BOOTSTRAP_SERVERS}")
            if not connector_ok:
                reasons.append("spark-sql-kafka connector jar not available (Maven Central unreachable / not fetched)")
            print("WARNING: falling back to the local-bridge stream source. Reason(s): " + "; ".join(reasons))
            mode = "local_bridge"

    print(f"Streaming source: {mode}")
    query = create_kafka_stream(spark) if mode == "kafka" else create_local_bridge_stream(spark)

    start = time.time()
    last_batch_id = -1
    last_data_time = None          # set only once a micro-batch has actually processed rows
    while time.time() - start < max_seconds:
        time.sleep(1)
        progress = query.lastProgress
        if progress and progress.get("batchId") != last_batch_id:
            last_batch_id = progress.get("batchId")
            if progress.get("numInputRows", 0) > 0:
                last_data_time = time.time()
        # Stop early only after data has been seen AND the stream has then been idle for a while.
        # (An empty first batch before the producer starts must NOT trigger a stop.)
        if last_data_time is not None and time.time() - last_data_time > idle_stop_after:
            break
    query.stop()
    spark.stop()
    print(f"\nStreaming query stopped after {time.time() - start:.1f}s (mode={mode}).")
    return mode


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 7 Spark Structured Streaming consumer")
    parser.add_argument("--mode", choices=["auto", "kafka", "local_bridge"], default="auto")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args()
    run_streaming(mode=args.mode, max_seconds=args.max_seconds)
