"""Long-running Kafka -> Spark -> configured API-model inference pipeline (Stage 1)."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession

from src.streaming import config
from src.streaming.kafka_producer import kafka_broker_reachable

LOGGER = logging.getLogger("stage1_streaming")
REQUIRED = {"transaction_id", "step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"}
FORBIDDEN = {"isFraud"}
ALLOWED_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def validate_event(raw: str | bytes, kafka_partition: int | None = None,
                   kafka_offset: int | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Parse and validate an event; return (event, error), never coerce bad values to valid data."""
    try:
        event = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if not isinstance(event, dict):
            raise ValueError("JSON value must be an object")
        if FORBIDDEN.intersection(event):
            raise ValueError("target field is prohibited in prediction input")
        missing = sorted(REQUIRED - event.keys())
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        if not isinstance(event["transaction_id"], str) or not event["transaction_id"].strip():
            raise ValueError("transaction_id must be a non-empty string")
        if isinstance(event["step"], bool) or not isinstance(event["step"], int) or event["step"] < 1:
            raise ValueError("step must be an integer >= 1")
        if event["type"] not in ALLOWED_TYPES:
            raise ValueError(f"unsupported transaction type: {event['type']!r}")
        for name in ("nameOrig", "nameDest"):
            if not isinstance(event[name], str) or not event[name].strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("amount", "oldbalanceOrg", "oldbalanceDest"):
            value = event[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or value == float("inf") or value != value:
                raise ValueError(f"{name} must be a finite number >= 0")
        return event, None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raw_text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        try:
            parsed = json.loads(raw_text)
            # Never copy a target label into error artifacts, even when rejecting its message.
            raw_text = json.dumps({key: ("[REDACTED]" if key in FORBIDDEN else value)
                                   for key, value in parsed.items()}) if isinstance(parsed, dict) else raw_text
        except (json.JSONDecodeError, AttributeError):
            pass
        return None, {"timestamp_utc": _utc_now(), "error": str(exc), "raw_value": raw_text,
                      "kafka_partition": kafka_partition, "kafka_offset": kafka_offset}


def create_spark_session() -> SparkSession:
    from src.streaming.spark_streaming import _ensure_windows_hadoop_home
    import sys
    _ensure_windows_hadoop_home()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (SparkSession.builder.appName("Stage1FraudStreaming").master("local[2]")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.jars.packages", config.SPARK_KAFKA_PACKAGE).getOrCreate())


def process_batch(batch: DataFrame, batch_id: int) -> None:
    """Validate and infer on the driver, preserving the API's ordered stateful feature path."""
    from src.api import app

    if len(app.FINAL_FEATURES) != 34 or "isFraud" in app.FINAL_FEATURES:
        raise RuntimeError("Configured API model feature contract is not the expected 34 features")
    events: list[tuple[dict[str, Any], int | None, int | None]] = []
    for row in batch.select("value", "partition", "offset").toLocalIterator():
        event, error = validate_event(row["value"], row["partition"], row["offset"])
        if error:
            error["batch_id"] = batch_id
            _jsonl_append(config.STAGE1_ERROR_PATH, error)
            LOGGER.error("Invalid Kafka record partition=%s offset=%s: %s", row["partition"], row["offset"], error["error"])
        else:
            events.append((event, row["partition"], row["offset"]))
    events.sort(key=lambda item: (item[0]["step"], item[1] or 0, item[2] or 0))
    for event, partition, offset in events:
        try:
            txn = app.TransactionRequest(**{key: event[key] for key in REQUIRED if key != "transaction_id"})
            features = app.build_feature_vector(txn)
            if list(features.columns) != app.FINAL_FEATURES or len(features.columns) != 34:
                raise ValueError("inference feature columns differ from configured 34-feature model contract")
            probability = float(app.model.predict_proba(features)[:, 1][0])
            risk_score = round(probability * 100, 4)
            result = {"transaction_id": event["transaction_id"], "step": event["step"],
                      "type": event["type"], "amount": float(event["amount"]),
                      # Store the exact inference vector so Stage 3 can compare the live
                      # stream with the training reference. IDs and target fields stay out.
                      "feature_values": {name: (float(features.iloc[0][name])
                                                 if math.isfinite(float(features.iloc[0][name])) else None)
                                         for name in app.FINAL_FEATURES},
                      "processing_timestamp_utc": _utc_now(), "model_version": app.MODEL_VERSION,
                      "prediction": int(probability >= app.CLASSIFICATION_THRESHOLD),
                      "fraud_probability": probability, "risk_score": risk_score,
                      "risk_level": app.assign_risk_level(risk_score), "data_origin": event.get("data_origin", "unspecified"),
                      "kafka_partition": partition, "kafka_offset": offset, "batch_id": batch_id}
            _jsonl_append(config.STAGE1_OUTPUT_PATH, result)
            from src.labels.store import PredictionConflictError, register_prediction
            try:
                register_prediction(result, topic=config.KAFKA_TOPIC)
            except PredictionConflictError as exc:
                LOGGER.error("Stage 2 kept the first prediction for %s and recorded a replay conflict: %s",
                             event["transaction_id"], exc)
                try:
                    _jsonl_append(config.STAGE1_ERROR_PATH, {"timestamp_utc": _utc_now(),
                        "transaction_id": event["transaction_id"], "kafka_partition": partition,
                        "kafka_offset": offset, "batch_id": batch_id, "category": "stage2_prediction_conflict",
                        "error": f"{type(exc).__name__}: {exc}"})
                except OSError:
                    LOGGER.exception("Could not persist the Stage 2 prediction conflict to the Stage 1 error file")
            except Exception as exc:
                # JSONL remains the primary Stage 1 sink. Record Stage 2 failure clearly without
                # causing Spark to retry an otherwise successful model prediction batch.
                LOGGER.exception("Stage 2 SQLite registration failed for %s", event["transaction_id"])
                try:
                    _jsonl_append(config.STAGE1_ERROR_PATH, {"timestamp_utc": _utc_now(),
                        "transaction_id": event["transaction_id"], "kafka_partition": partition,
                        "kafka_offset": offset, "batch_id": batch_id, "category": "stage2_store_failure",
                        "error": f"{type(exc).__name__}: {exc}"})
                except OSError:
                    LOGGER.exception("Could not persist the Stage 2 registration failure to the Stage 1 error file")
        except Exception as exc:
            LOGGER.exception("Prediction failure for transaction %s", event.get("transaction_id"))
            _jsonl_append(config.STAGE1_ERROR_PATH, {"timestamp_utc": _utc_now(), "transaction_id": event.get("transaction_id"),
                         "kafka_partition": partition, "kafka_offset": offset, "batch_id": batch_id,
                         "error": f"{type(exc).__name__}: {exc}"})


def run_streaming() -> None:
    if not kafka_broker_reachable(config.KAFKA_BOOTSTRAP_SERVERS):
        raise RuntimeError(f"Kafka broker is not reachable at {config.KAFKA_BOOTSTRAP_SERVERS}")
    from kafka.admin import KafkaAdminClient
    admin = KafkaAdminClient(bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS)
    try:
        descriptions = admin.describe_topics([config.KAFKA_TOPIC])
        if not descriptions or len(descriptions[0].get("partitions", [])) != config.KAFKA_NUM_PARTITIONS:
            raise RuntimeError(f"Topic {config.KAFKA_TOPIC!r} must have exactly {config.KAFKA_NUM_PARTITIONS} partition(s) for ordered account state")
    finally:
        admin.close()
    from src.api import app
    if len(app.FINAL_FEATURES) != 34:
        raise RuntimeError("Configured model must use exactly 34 features")
    status_path = config.STAGE1_STATUS_PATH
    status_path.parent.mkdir(parents=True, exist_ok=True)
    stop_heartbeat = threading.Event()

    def write_status(state: str, error: str | None = None) -> None:
        status_path.write_text(json.dumps({"state": state, "updated_at_utc": _utc_now(), "model_version": app.MODEL_VERSION,
                               "topic": config.KAFKA_TOPIC, "pid": os.getpid(), "error": error}, indent=2), encoding="utf-8")

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    query = None
    write_status("starting")
    def heartbeat() -> None:
        while not stop_heartbeat.wait(10):
            write_status("running")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    failure: str | None = None
    try:
        query = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
                 .option("subscribe", config.KAFKA_TOPIC).option("startingOffsets", "latest")
                 .option("maxOffsetsPerTrigger", 1000).load().writeStream.foreachBatch(process_batch)
                 .option("checkpointLocation", str(config.STAGE1_CHECKPOINT_PATH)).start())
        write_status("running")
        query.awaitTermination()
    except KeyboardInterrupt:
        LOGGER.info("Stop requested; stopping the Spark query cleanly")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        write_status("error", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_heartbeat.set()
        if query is not None:
            query.stop()
        spark.stop()
        write_status("error" if failure else "stopped", failure)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_streaming()
