
import os
import sys
import csv
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)
os.environ["FRAUD_ACTIVE_MODEL_VERSION"] = "v2"

import pandas as pd
from kafka import KafkaProducer
from pyspark.sql.functions import col, from_json

from src.streaming import config, spark_streaming
from src.api import app

assert app.MODEL_VERSION == "V2", f"Expected V2, got {app.MODEL_VERSION}"
assert len(app.FINAL_FEATURES) == 34
assert abs(app.CLASSIFICATION_THRESHOLD - 0.53) < 1e-12

root = PROJECT_ROOT
run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
topic = f"phase12-v2-integration-{run_id}"
output_path = root / "reports" / f"phase12_v2_predictions_{run_id}.csv"
checkpoint_path = root / "checkpoints" / f"phase12_v2_{run_id}" / "kafka"
log_path = root / "reports" / f"phase12_v2_spark_{run_id}.log"
evidence_path = root / "reports" / f"phase12_v2_integration_evidence_{run_id}.json"

output_path.parent.mkdir(parents=True, exist_ok=True)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

event = {
    "transaction_id": f"PHASE12-{run_id}",
    "step": 379,
    "type": "PAYMENT",
    "amount": 4349.85,
    "nameOrig": f"C{run_id[-10:]}",
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 650.15,
    "nameDest": f"M{run_id[-10:]}",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 4349.85,
}
assert "isFraud" not in event

spark = None
query = None
producer = None
checks = {}


def process_batch(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return

    results = []
    for row in batch_df.collect():
        txn = app.TransactionRequest(
            step=int(row["step"]),
            type=str(row["type"]),
            amount=float(row["amount"]),
            nameOrig=str(row["nameOrig"]),
            oldbalanceOrg=float(row["oldbalanceOrg"]),
            nameDest=str(row["nameDest"]),
            oldbalanceDest=float(row["oldbalanceDest"]),
        )

        features = app.build_feature_vector(txn)
        if list(features.columns) != app.FINAL_FEATURES or len(features.columns) != 34:
            raise RuntimeError("34-feature contract validation failed")

        probability = float(app.model.predict_proba(features)[0, 1])
        label = int(probability >= app.CLASSIFICATION_THRESHOLD)
        score = round(probability * 100, 4)
        risk = app.assign_risk_level(score)

        results.append({
            "transaction_id": row["transaction_id"],
            "step": int(row["step"]),
            "type": row["type"],
            "amount": float(row["amount"]),
            "fraud_probability": probability,
            "predicted_label": label,
            "risk_score": score,
            "risk_level": risk,
            "model_version": app.MODEL_VERSION,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        })

    if results:
        pd.DataFrame(results).to_csv(
            output_path,
            mode="a",
            index=False,
            header=not output_path.exists(),
        )
        print(f"Processed Spark batch {batch_id}: {len(results)} prediction(s)")


try:
    print("Model:", app.MODEL_VERSION)
    print("Threshold:", app.CLASSIFICATION_THRESHOLD)
    print("Unique topic:", topic)

    # Publish one transaction to a unique test topic.
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        request_timeout_ms=20000,
    )
    metadata = producer.send(topic, value=event).get(timeout=30)
    producer.flush(timeout=15)
    producer.close()
    producer = None

    checks["kafka_publish"] = {
        "status": "PASS",
        "topic": metadata.topic,
        "partition": metadata.partition,
        "offset": metadata.offset,
    }
    print(
        "Kafka publish: PASS; partition =",
        metadata.partition,
        "offset =",
        metadata.offset,
    )

    # Create a separate Spark query; do not start the frozen Phase 7 stream.
    spark = spark_streaming.create_spark_session("Phase12IsolatedV2Integration")
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )

    parsed = (
        raw.select(
            from_json(
                col("value").cast("string"),
                spark_streaming.EVENT_SCHEMA,
            ).alias("event")
        )
        .select("event.*")
    )

    query = (
        parsed.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", str(checkpoint_path))
        .start()
    )

    deadline = time.time() + 90
    while time.time() < deadline:
        query.processAllAvailable()
        if output_path.exists():
            try:
                result_df = pd.read_csv(output_path)
                if not result_df.empty:
                    break
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
        time.sleep(1)
    else:
        raise TimeoutError("No prediction CSV row produced within 90 seconds")

    result_df = pd.read_csv(output_path)
    if len(result_df) != 1:
        raise AssertionError(f"Expected exactly one prediction, got {len(result_df)}")

    result = result_df.iloc[0].to_dict()
    probability = float(result["fraud_probability"])
    predicted_label = int(result["predicted_label"])
    score = float(result["risk_score"])
    expected_label = int(probability >= app.CLASSIFICATION_THRESHOLD)
    expected_risk = app.assign_risk_level(score)

    checks["prediction"] = {
        "status": "PASS" if result["model_version"] == "V2" else "FAIL",
        "model_version": result["model_version"],
        "transaction_id": result["transaction_id"],
        "fraud_probability": probability,
    }
    checks["threshold"] = {
        "status": "PASS" if predicted_label == expected_label else "FAIL",
        "threshold": app.CLASSIFICATION_THRESHOLD,
        "predicted_label": predicted_label,
        "expected_label": expected_label,
    }
    checks["risk_level"] = {
        "status": "PASS" if result["risk_level"] == expected_risk else "FAIL",
        "risk_score": score,
        "risk_level": result["risk_level"],
        "expected_risk_level": expected_risk,
    }
    checks["checkpoint"] = {
        "status": "PASS" if checkpoint_path.is_dir() else "FAIL",
        "path": str(checkpoint_path),
    }
    checks["feature_contract"] = {
        "status": "PASS",
        "feature_count": 34,
    }
    checks["output"] = {
        "status": "PASS",
        "path": str(output_path),
        "rows": len(result_df),
    }

    # Reject the run if any required validation failed.
    if any(item["status"] != "PASS" for item in checks.values()):
        raise AssertionError("One or more integration checks failed")
    if result["model_version"] != "V2":
        raise AssertionError("Streaming prediction did not report V2")
    if predicted_label != expected_label:
        raise AssertionError("Prediction label does not match threshold")
    if result["risk_level"] != expected_risk:
        raise AssertionError("Risk level does not match risk score")
    if not checkpoint_path.is_dir():
        raise AssertionError("Spark checkpoint directory was not created")
    if result["transaction_id"] != event["transaction_id"]:
        raise AssertionError("Output transaction ID does not match published event")

    print("\nPREDICTION:")
    print(json.dumps(result, indent=2, default=str))

    evidence = {
        "phase": 12,
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": app.MODEL_VERSION,
        "threshold": app.CLASSIFICATION_THRESHOLD,
        "topic": topic,
        "producer_transport": "kafka",
        "messages_sent": 1,
        "partition": metadata.partition,
        "offset": metadata.offset,
        "prediction_rows": len(result_df),
        "prediction": result,
        "checks": checks,
        "result_file": str(output_path.relative_to(root)),
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
    }

    # Write PASS evidence only after every check has succeeded.
    evidence_path.write_text(
        json.dumps(evidence, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nCHECKS:")
    print(json.dumps(checks, indent=2, default=str))
    print("Evidence report:", evidence_path)
    print("\nPHASE12_V2_KAFKA_SPARK_INTEGRATION: PASS")

except Exception as exc:
    print("\nPHASE12_V2_KAFKA_SPARK_INTEGRATION: FAIL")
    print(type(exc).__name__ + ":", str(exc))
    traceback.print_exc()
    print("Diagnostic paths:")
    print("Topic:", topic)
    print("Output:", output_path)
    print("Checkpoint:", checkpoint_path)
    print("Spark log:", log_path)
    raise

finally:
    if query is not None:
        try:
            query.stop()
        except Exception:
            pass

    if spark is not None:
        try:
            spark.stop()
        except Exception:
            pass

    if producer is not None:
        try:
            producer.close()
        except Exception:
            pass
