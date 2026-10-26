"""Continuous synthetic PaySim demo producer for the Stage 1 Kafka topic."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.streaming import config
import pandas as pd
from src.streaming.kafka_producer import KafkaTransport, build_kafka_message, load_transaction_stream


def active_model_metadata_path() -> Path:
    """Select the same model's held-out period that Stage 1 is configured to serve."""
    version = os.environ.get("FRAUD_ACTIVE_MODEL_VERSION", "v2").strip().lower()
    if version not in {"v1", "v2"}:
        raise ValueError("FRAUD_ACTIVE_MODEL_VERSION must be v1 or v2")
    return config.PROJECT_ROOT / "models" / f"model_{version}_metadata.json"


def load_stage1_transactions(limit: int = config.NUM_TRANSACTIONS):
    """Load held-out records for Stage 1's configured model, not legacy Phase 7's V1 window."""
    return load_transaction_stream(limit, metadata_path=active_model_metadata_path())


def select_validation_transactions(count: int, *, raw_data_path: Path = config.RAW_DATA_PATH,
                                   metadata_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Select distinct raw held-out transactions, one per strictly increasing step.

    This is an offline validation artifact, not a live Kafka producer. Source
    uniqueness is checked using the underlying PaySim fields, not generated IDs.
    """
    if count <= 0:
        raise ValueError("validation transaction count must be positive")
    metadata_path = metadata_path or active_model_metadata_path()
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    strategy = metadata["train_validation_test_strategy"]
    boundary = int(strategy["validation_end_step"])
    frame = pd.read_csv(raw_data_path, dtype={"step": "int32", "type": "category"})
    held_out = frame.loc[frame.step > boundary].sort_values("step", kind="stable")
    # Conservatively reject exact repeated raw transactions even across steps;
    # step ordering is checked separately and is not treated as source identity.
    identity_fields = ["type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                       "nameDest", "oldbalanceDest", "newbalanceDest"]
    unique_steps = held_out.drop_duplicates("step", keep="first")
    distinct_sources = unique_steps.drop_duplicates(identity_fields, keep="first")
    duplicate_source_rows_removed = len(unique_steps) - len(distinct_sources)
    selected = distinct_sources.head(count).copy()
    duplicate_count = int(selected.duplicated(identity_fields).sum())
    if len(selected) != count or duplicate_count or not selected.step.is_monotonic_increasing or selected.step.duplicated().any():
        raise RuntimeError(f"Could not construct {count} distinct held-out transactions with strictly increasing steps; "
                           f"constructed={len(selected)}, duplicates={duplicate_count}")
    selected = selected.reset_index(drop=True)
    summary = {"model_version": metadata["model_version"], "validation_end_step": boundary,
               "selected_row_count": len(selected), "distinct_transaction_count": len(selected) - duplicate_count,
               "minimum_step": int(selected.step.min()), "maximum_step": int(selected.step.max()),
               "duplicate_source_count": duplicate_count,
               "same_step_rows_removed_before_selection": int(len(held_out) - len(unique_steps)),
               "duplicate_source_rows_removed_before_selection": int(duplicate_source_rows_removed),
               "identity_fields": identity_fields,
               "note": "Underlying source uniqueness uses the raw transaction payload excluding step; generated event IDs and step alone are not the uniqueness proof."}
    return selected, summary


def write_validation_sample(count: int, output_path: Path, *, raw_data_path: Path = config.RAW_DATA_PATH,
                            metadata_path: Path | None = None) -> dict:
    """Write a bounded raw-event sample for offline comparison; never publishes to Kafka."""
    rows, summary = select_validation_transactions(count, raw_data_path=raw_data_path,
                                                     metadata_path=metadata_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            message_row = row.copy()
            message_row["transaction_id"] = f"VALIDATION-{summary['model_version']}-{int(row.step)}-{i:06d}"
            event = build_kafka_message(message_row)
            event["data_origin"] = "synthetic_validation_offline"
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    summary["output_path"] = str(output_path.resolve())
    print(json.dumps(summary, indent=2))
    return summary


def run_continuous(delay: float = config.SIMULATION_DELAY, topic: str = config.KAFKA_TOPIC) -> None:
    if delay < 0:
        raise ValueError("delay must be >= 0")
    # PaySim is a synthetic dataset. Replaying a small chronological slice continuously
    # demonstrates the pipeline without implying that these are genuine customer transactions.
    rows = load_stage1_transactions(config.NUM_TRANSACTIONS)
    if rows.empty:
        raise RuntimeError("No PaySim demo transactions are available.")
    producer = KafkaTransport()
    sequence = 0
    print(f"Publishing synthetic PaySim demo events continuously to {topic}; Ctrl+C to stop.")
    try:
        while True:
            for _, row in rows.iterrows():
                sequence += 1
                event = build_kafka_message(row)
                event["transaction_id"] = f"SYNTH-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{sequence:08d}"
                event["data_origin"] = "synthetic_paysim_demo"
                info = producer.send(topic, event["transaction_id"], event)
                print(f"Published synthetic {event['transaction_id']} offset={info['offset']}")
                time.sleep(delay)
    finally:
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=config.SIMULATION_DELAY,
                        help="Seconds between synthetic demo events")
    parser.add_argument("--topic", default=config.KAFKA_TOPIC)
    parser.add_argument("--validation-sample", type=int, metavar="N",
                        help="write N distinct held-out raw events to an offline JSONL file; does not publish")
    parser.add_argument("--output", type=Path,
                        help="required JSONL destination for --validation-sample")
    args = parser.parse_args()
    if args.validation_sample is not None:
        if args.output is None:
            parser.error("--output is required with --validation-sample")
        write_validation_sample(args.validation_sample, args.output)
    else:
        run_continuous(args.delay, args.topic)
