"""Focused tests for Stage 1 event validation and leakage protection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.streaming import config
from src.streaming import stage1_producer
from src.streaming.stage1_streaming import process_batch, validate_event


class Stage1ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = {"transaction_id": "SYNTH-1", "step": 1, "type": "PAYMENT", "amount": 10.0,
                      "nameOrig": "C1", "oldbalanceOrg": 20.0, "nameDest": "C2", "oldbalanceDest": 0.0,
                      "data_origin": "synthetic_paysim_demo"}

    def test_valid_event_keeps_stable_id_and_demo_marker(self) -> None:
        event, error = validate_event(json.dumps(self.event))
        self.assertIsNone(error)
        self.assertEqual(event["transaction_id"], "SYNTH-1")
        self.assertEqual(event["data_origin"], "synthetic_paysim_demo")

    def test_stage1_producer_uses_active_models_holdout_metadata(self) -> None:
        with patch.dict("os.environ", {"FRAUD_ACTIVE_MODEL_VERSION": "v2"}):
            self.assertEqual(stage1_producer.active_model_metadata_path(),
                config.PROJECT_ROOT / "models" / "model_v2_metadata.json")
        with patch.dict("os.environ", {"FRAUD_ACTIVE_MODEL_VERSION": "v1"}):
            self.assertEqual(stage1_producer.active_model_metadata_path(),
                config.PROJECT_ROOT / "models" / "model_v1_metadata.json")

    def test_rejects_malformed_missing_and_invalid_values(self) -> None:
        self.assertIsNotNone(validate_event("{")[1])
        self.assertIsNotNone(validate_event(json.dumps({"transaction_id": "x"}))[1])
        invalid = {**self.event, "amount": -1}
        self.assertIsNotNone(validate_event(json.dumps(invalid))[1])

    def test_rejects_target_and_redacts_it_from_error_record(self) -> None:
        event, error = validate_event(json.dumps({**self.event, "isFraud": 1}))
        self.assertIsNone(event)
        self.assertIn("[REDACTED]", error["raw_value"])
        self.assertNotIn('"isFraud": 1', error["raw_value"])

    def test_prediction_uses_configured_model_and_persists_contract_fields(self) -> None:
        class FakeBatch:
            def select(self, *_columns):
                return self

            def toLocalIterator(self):
                return iter([{**self_event_bytes, "partition": 0, "offset": 42}])

        self_event_bytes = {"value": json.dumps(self.event).encode()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            errors = Path(temporary) / "errors.jsonl"
            database = Path(temporary) / "stage2.sqlite3"
            with patch.object(config, "STAGE1_OUTPUT_PATH", output), patch.object(config, "STAGE1_ERROR_PATH", errors), \
                    patch("src.labels.config.STAGE2_DB_PATH", database):
                process_batch(FakeBatch(), 7)
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["transaction_id"], "SYNTH-1")
            self.assertEqual(record["model_version"], "V2")
            self.assertIn("fraud_probability", record)
            self.assertIn("processing_timestamp_utc", record)
            self.assertIn("kafka_offset", record)
            self.assertEqual(len(record["feature_values"]), 34)
            self.assertNotIn("isFraud", record["feature_values"])
            self.assertNotIn("isFraud", record)
            self.assertFalse(errors.exists())

    def test_sqlite_failure_does_not_lose_jsonl_prediction_or_stop_batch(self) -> None:
        class FakeBatch:
            def select(self, *_columns):
                return self

            def toLocalIterator(self):
                return iter([{"value": json.dumps(self_event).encode(), "partition": 0, "offset": 51}])

        self_event = {**self.event, "transaction_id": "SYNTH-STORE-FAIL-1"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            errors = Path(temporary) / "errors.jsonl"
            with patch.object(config, "STAGE1_OUTPUT_PATH", output), patch.object(config, "STAGE1_ERROR_PATH", errors), \
                    patch("src.labels.store.register_prediction", side_effect=OSError("database locked")):
                process_batch(FakeBatch(), 8)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
            error = json.loads(errors.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(error["category"], "stage2_store_failure")
            self.assertEqual(error["transaction_id"], "SYNTH-STORE-FAIL-1")


if __name__ == "__main__":
    unittest.main()
