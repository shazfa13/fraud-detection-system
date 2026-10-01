"""
Phase 7 — Kafka producer for the fraud-transactions topic.
=============================================================

Reads real transactions from the existing PaySim raw dataset (never modifies it), replays them
in strict chronological order, and publishes each one as a JSON event — WITHOUT `isFraud` — to the
Kafka topic `fraud-transactions` (src/streaming/config.py: KAFKA_TOPIC).

Two delivery modes share the exact same message-construction code (`build_kafka_message`), so the
payload itself is identical either way; only the transport differs:

  * `mode="kafka"`        - a real `kafka-python` KafkaProducer, talking the real Kafka wire
                            protocol to a real broker at KAFKA_BOOTSTRAP_SERVERS. This is the
                            primary, spec-compliant path, meant to run on a machine with a real
                            Kafka broker reachable (see the Phase 7 notebook / README for the
                            Windows/PowerShell setup commands).
  * `mode="local_bridge"` - writes the identical JSON payload to one file per transaction under
                            LOCAL_BRIDGE_INBOX_PATH. Used ONLY when no Kafka broker is reachable
                            (see the Phase 7 notebook's Environment section for why this exists);
                            it lets everything downstream of "receive a transaction event" be
                            exercised with real Spark Structured Streaming.

`mode="auto"` (the default) tries a real Kafka broker first and falls back to the local bridge
with a clear, printed warning if none is reachable - it never silently pretends to have used Kafka
when it did not.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.streaming import config


# -------------------------------------------------------------------------------------------
# Chronological transaction source (mirrors notebooks/06_live_transaction_simulation.ipynb,
# Section 5: use the Phase 4 test period - transactions Model V1 never trained on - and the
# same chronological-cumulative-share fallback if that boundary was not recorded).
# -------------------------------------------------------------------------------------------
def resolve_test_period(raw_df: pd.DataFrame, metadata_path: Path) -> tuple[pd.DataFrame, str]:
    with open(metadata_path, encoding="utf-8") as fh:
        metadata = json.load(fh)
    strategy = metadata.get("train_validation_test_strategy", {})
    val_end_step = strategy.get("validation_end_step")
    if val_end_step is not None:
        test_df = raw_df[raw_df["step"] > val_end_step].copy()
        return test_df, f"Phase 4 recorded test period: step > {val_end_step}"

    train_frac = strategy.get("train_fraction_target", 0.70)
    val_frac = strategy.get("validation_fraction_target", 0.15)
    step_counts = raw_df["step"].value_counts().sort_index()
    cum_share = step_counts.cumsum() / step_counts.sum()
    val_end_step = int(cum_share[cum_share >= train_frac + val_frac].index[0])
    test_df = raw_df[raw_df["step"] > val_end_step].copy()
    return test_df, f"FALLBACK test period (recomputed): step > {val_end_step}"


def load_transaction_stream(num_transactions: int, raw_data_path: Path = config.RAW_DATA_PATH,
                            metadata_path: Path = config.METADATA_PATH) -> pd.DataFrame:
    """The first `num_transactions` rows of the test period, in chronological order (no shuffling)."""
    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Raw PaySim CSV not found at {raw_data_path}. Phase 1 data must be present locally "
                                "(it is intentionally git-ignored - see .gitignore).")
    raw_df = pd.read_csv(raw_data_path, dtype={"step": "int32", "type": "category"})
    test_df, description = resolve_test_period(raw_df, metadata_path)
    test_df = test_df.sort_values("step", kind="stable").reset_index(drop=True)
    num_transactions = min(num_transactions, len(test_df))
    stream_df = test_df.iloc[:num_transactions].reset_index(drop=True).copy()
    stream_df["transaction_id"] = [f"SIM-{i + 1:06d}" for i in range(len(stream_df))]
    print(f"Transaction source: {description}")
    print(f"Stream size: {len(stream_df)} transactions, step range "
         f"{int(stream_df['step'].min())}-{int(stream_df['step'].max())}")
    return stream_df


# -------------------------------------------------------------------------------------------
# Message construction — the ONE place that decides what leaves this process. `isFraud` is not
# read from `row` at all here, so it structurally cannot appear in the payload.
# -------------------------------------------------------------------------------------------
KAFKA_MESSAGE_FIELDS = ["transaction_id", "step", "type", "amount", "nameOrig", "oldbalanceOrg",
                        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]


def build_kafka_message(row: pd.Series) -> dict:
    """
    Build the exact JSON event published to `fraud-transactions`.

    NOTE on newbalanceOrig/newbalanceDest: these ARE included in the event (matching the raw
    PaySim row and the Phase 7 spec's own example schema), but the feature-engineering pipeline
    (src/api/app.py, reused unchanged by spark_streaming.py) NEVER reads them - Phase 3's leakage
    audit marked every feature derived from them UNSAFE. Carrying them on the wire is harmless;
    using them for a feature would not be, and nothing in this project's feature pipeline does.
    """
    message = {
        "transaction_id": str(row["transaction_id"]), "step": int(row["step"]), "type": str(row["type"]),
        "amount": float(row["amount"]), "nameOrig": str(row["nameOrig"]), "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]), "nameDest": str(row["nameDest"]),
        "oldbalanceDest": float(row["oldbalanceDest"]), "newbalanceDest": float(row["newbalanceDest"]),
    }
    assert "isFraud" not in message, "CRITICAL: isFraud must never be published to Kafka."
    assert set(message.keys()) == set(KAFKA_MESSAGE_FIELDS)
    return message


# -------------------------------------------------------------------------------------------
# Real Kafka transport
# -------------------------------------------------------------------------------------------
def kafka_broker_reachable(bootstrap_servers: str = config.KAFKA_BOOTSTRAP_SERVERS, timeout_s: float = 3.0) -> bool:
    """A cheap, real TCP-level reachability check (no kafka-python import needed just to check this)."""
    import socket
    host, _, port = bootstrap_servers.partition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


class KafkaTransport:
    """Thin wrapper around kafka-python's KafkaProducer. Real Kafka wire protocol, real broker required."""

    def __init__(self, bootstrap_servers: str = config.KAFKA_BOOTSTRAP_SERVERS):
        from kafka import KafkaProducer   # imported lazily so this module loads even without kafka-python installed
        self._producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                                       value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                                       key_serializer=lambda k: k.encode("utf-8") if k else None)

    def send(self, topic: str, key: str, message: dict) -> dict:
        future = self._producer.send(topic, key=key, value=message)
        record_metadata = future.get(timeout=10)
        return {"topic": record_metadata.topic, "partition": record_metadata.partition, "offset": record_metadata.offset}

    def close(self):
        self._producer.flush()
        self._producer.close()


# -------------------------------------------------------------------------------------------
# Local-bridge transport (used only when no Kafka broker is reachable - see module docstring)
# -------------------------------------------------------------------------------------------
class LocalBridgeTransport:
    """Writes one JSON file per message, in send order, to LOCAL_BRIDGE_INBOX_PATH."""

    def __init__(self, inbox_path: Path = config.LOCAL_BRIDGE_INBOX_PATH, reset: bool = True):
        self.inbox_path = inbox_path
        if reset and inbox_path.is_dir():
            for f in inbox_path.glob("*.json"):
                f.unlink()
        inbox_path.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def send(self, topic: str, key: str, message: dict) -> dict:
        self._seq += 1
        filename = f"{self._seq:08d}_{key}.json"
        path = self.inbox_path / filename
        path.write_text(json.dumps(message), encoding="utf-8")
        return {"topic": f"local_bridge:{topic}", "partition": 0, "offset": self._seq - 1}

    def close(self):
        pass


# -------------------------------------------------------------------------------------------
# Producer entry point
# -------------------------------------------------------------------------------------------
def run_producer(num_transactions: int = config.NUM_TRANSACTIONS, delay: float = config.SIMULATION_DELAY,
                 mode: str = "auto", topic: str = config.KAFKA_TOPIC) -> list[dict]:
    """
    Publish `num_transactions` real, chronologically-ordered transactions to `topic`.
    mode: "auto" (try Kafka, fall back to local bridge with a clear warning), "kafka", or "local_bridge".
    Returns the list of delivery-info dicts (one per message actually sent).
    """
    if mode == "auto":
        mode = "kafka" if kafka_broker_reachable() else "local_bridge"
        if mode == "local_bridge":
            print(f"WARNING: no Kafka broker reachable at {config.KAFKA_BOOTSTRAP_SERVERS} - "
                 f"falling back to the local bridge transport (see module docstring). "
                 f"This is NOT a real Kafka broker; start one and re-run with mode='kafka' for the real pipeline.")

    transport = KafkaTransport() if mode == "kafka" else LocalBridgeTransport()
    print(f"Producer transport: {mode}")

    stream_df = load_transaction_stream(num_transactions)
    deliveries = []
    for _, row in stream_df.iterrows():
        message = build_kafka_message(row)
        info = transport.send(topic, key=message["transaction_id"], message=message)
        deliveries.append({**info, "transaction_id": message["transaction_id"]})
        print(f"Sent {message['transaction_id']} | type={message['type']:<9} amount={message['amount']:>12,.2f} "
             f"-> topic={info['topic']} partition={info['partition']} offset={info['offset']}")
        time.sleep(delay)
    transport.close()
    print(f"\nProducer finished: {len(deliveries)} message(s) sent via '{mode}' transport.")
    return deliveries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 7 Kafka producer")
    parser.add_argument("--num-transactions", type=int, default=config.NUM_TRANSACTIONS)
    parser.add_argument("--delay", type=float, default=config.SIMULATION_DELAY)
    parser.add_argument("--mode", choices=["auto", "kafka", "local_bridge"], default="auto")
    args = parser.parse_args()
    run_producer(num_transactions=args.num_transactions, delay=args.delay, mode=args.mode)
