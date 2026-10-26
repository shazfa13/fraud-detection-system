"""Stage 4 eligibility, split, and no-write safeguards using temporary inputs."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.monitoring import config as monitor_config
from src.retraining import stage4_workflow as stage4
from src.retraining.v2_trainer import FINAL_FEATURES, TARGET


class Stage4WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "stage2.sqlite3"
        db = sqlite3.connect(self.db_path)
        try:
            db.executescript("""
                CREATE TABLE predictions(transaction_id TEXT PRIMARY KEY,prediction INTEGER,
                    fraud_probability REAL,risk_score REAL,risk_level TEXT,prediction_timestamp TEXT,
                    model_version TEXT,data_origin TEXT,kafka_topic TEXT,kafka_partition INTEGER,
                    kafka_offset INTEGER,label_status TEXT,registered_at TEXT);
                CREATE TABLE labels(transaction_id TEXT PRIMARY KEY,outcome INTEGER,label_source TEXT,
                    source_type TEXT,confirmed_at TEXT,status TEXT,submission_id TEXT,created_at TEXT);
                CREATE TABLE label_submissions(submission_id TEXT PRIMARY KEY,transaction_id TEXT,
                    outcome INTEGER,label_source TEXT,source_type TEXT,confirmed_at TEXT,
                    disposition TEXT,submitted_at TEXT,duplicate_of TEXT);
                CREATE TABLE prediction_conflicts(conflict_id INTEGER PRIMARY KEY,transaction_id TEXT,
                    stored_record TEXT,incoming_record TEXT,observed_at TEXT);
            """)
        finally:
            db.close()
        self.features = stage4.validate_feature_contract()

    def tearDown(self):
        self.temp.cleanup()

    def _add(self, txn, *, source_type="trusted_verified", source="trusted_feed", status="resolved",
             disposition="accepted_genuine", origin="live", prediction=True, outcome=1, conflict=False):
        db = sqlite3.connect(self.db_path)
        try:
            if prediction:
                db.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (txn,0,.1,10,"LOW","2026-10-25T10:00:00+00:00","V2",origin,
                     "fraud-transactions",0,hash(txn)%100000,"resolved","2026-10-25T10:00:00+00:00"))
            db.execute("INSERT INTO labels VALUES(?,?,?,?,?,?,?,?)",
                (txn,outcome,source,source_type,"2026-10-25T11:00:00+00:00",status,"SUB-"+txn,"2026-10-25T11:00:00+00:00"))
            db.execute("INSERT INTO label_submissions VALUES(?,?,?,?,?,?,?,?,?)",
                ("SUB-"+txn,txn,outcome,source,source_type,"2026-10-25T11:00:00+00:00",disposition,
                 "2026-10-25T11:00:00+00:00",None))
            if conflict:
                db.execute("INSERT INTO label_submissions VALUES(?,?,?,?,?,?,?,?,?)",
                    ("CONFLICT-"+txn,txn,1-outcome,source,source_type,"2026-10-25T12:00:00+00:00",
                     "conflict","2026-10-25T12:00:00+00:00",None))
            db.commit()
        finally:
            db.close()

    def _log(self, txns):
        path=self.root/"predictions.jsonl"
        with path.open("w",encoding="utf-8") as out:
            for i,txn in enumerate(txns):
                out.write(json.dumps({"transaction_id":txn,"model_version":"V2","step":566+i,
                    "data_origin":"live","feature_values":{name:float(i) for name in self.features}})+"\n")
        return path

    def test_trusted_label_is_eligible_and_stage2_is_read_only(self):
        self._add("LIVE-1")
        log=self._log(["LIVE-1"])
        before=self.db_path.read_bytes()
        frame,summary=stage4.collect_eligible_labels(database_path=self.db_path,prediction_log_path=log,
            features=self.features,trusted_sources={"trusted_feed"},token_configured=True)
        self.assertEqual(len(frame),1)
        self.assertEqual(summary["eligible_fraud_count"],1)
        self.assertEqual(self.db_path.read_bytes(),before)
        with closing(stage4._readonly_db(self.db_path)) as db:
            with self.assertRaises(sqlite3.OperationalError):
                db.execute("INSERT INTO labels VALUES('NO',0,'x','untrusted','x','resolved','x','x')")

    def test_synthetic_pending_unmatched_conflicting_unauthorized_excluded(self):
        self._add("SYN",source_type="synthetic_test",source="manual",origin="synthetic_paysim_demo")
        self._add("PEND",status="pending",disposition="pending_prediction")
        self._add("UNMATCH",prediction=False,disposition="pending_prediction")
        self._add("CONFLICT",conflict=True)
        self._add("UNAUTH",source="not_allowlisted")
        self._add("SYNORIGIN",origin="synthetic_paysim_demo")
        log=self._log(["SYN","PEND","CONFLICT","UNAUTH","SYNORIGIN"])
        frame,summary=stage4.collect_eligible_labels(database_path=self.db_path,prediction_log_path=log,
            features=self.features,trusted_sources={"trusted_feed"},token_configured=True)
        self.assertTrue(frame.empty)
        self.assertEqual(summary["eligible_label_count"],0)
        self.assertGreaterEqual(summary["exclusions"]["synthetic_label"],1)
        self.assertGreaterEqual(summary["exclusions"]["pending_or_rejected"],1)
        self.assertGreaterEqual(summary["exclusions"]["unmatched_prediction"],1)
        self.assertGreaterEqual(summary["exclusions"]["label_conflict"],1)
        self.assertGreaterEqual(summary["exclusions"]["untrusted_or_unauthorized"],1)
        self.assertGreaterEqual(summary["exclusions"]["synthetic_prediction"],1)

    def test_feature_order_leakage_and_duplicate_vector_rejected(self):
        bad_metadata=self.root/"bad-v2-metadata.json"
        metadata=json.loads(monitor_config.MODEL_METADATA_PATHS["V2"].read_text(encoding="utf-8"))
        metadata["feature_names"][0]="step"
        bad_metadata.write_text(json.dumps(metadata),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"34 distinct safe"):
            stage4.validate_feature_contract(bad_metadata)
        bad=self._log(["X"])
        item=json.loads(bad.read_text().splitlines()[0]); item["feature_values"]["isFraud"]=0
        bad.write_text(json.dumps(item)+"\n",encoding="utf-8")
        self._add("X")
        frame,summary=stage4.collect_eligible_labels(database_path=self.db_path,prediction_log_path=bad,
            features=self.features,trusted_sources={"trusted_feed"},token_configured=True)
        self.assertTrue(frame.empty)
        self.assertEqual(summary["exclusions"]["invalid_feature_vector"],1)
        valid=self._log(["X"])
        with valid.open("a",encoding="utf-8") as out:
            out.write(valid.read_text(encoding="utf-8"))
        frame,summary=stage4.collect_eligible_labels(database_path=self.db_path,prediction_log_path=valid,
            features=self.features,trusted_sources={"trusted_feed"},token_configured=True)
        self.assertTrue(frame.empty)
        self.assertEqual(summary["exclusions"]["missing_or_duplicate_feature_vector"],1)
        example=pd.DataFrame([{"event_id":"X","step":566,**{n:0.0 for n in self.features},TARGET:1}])
        with self.assertRaisesRegex(ValueError,"Duplicate transaction"):
            stage4.validate_example_frame(pd.concat([example,example]),self.features)
        with self.assertRaisesRegex(ValueError,"Forbidden"):
            stage4.validate_example_frame(example.assign(step_feature=1),self.features+["step"])

    def test_chronological_split_is_deterministic_and_step_disjoint(self):
        hist=pd.DataFrame([{"event_id":f"h{i}","step":i+1,**{n:float(i) for n in self.features},TARGET:i%2}
                           for i in range(20)])
        recent=pd.DataFrame([{"event_id":f"r{i}","step":566+i,**{n:float(i) for n in self.features},TARGET:i%2}
                             for i in range(100)])
        one=stage4.chronological_split(hist,recent,self.features)
        two=stage4.chronological_split(hist,recent,self.features)
        self.assertEqual((one["train_end_step"],one["validation_end_step"],one["test_end_step"]),
                         (two["train_end_step"],two["validation_end_step"],two["test_end_step"]))
        parts=[one[k] for k in ("train_df","validation_df","test_df")]
        self.assertLess(parts[0].step.max(),parts[1].step.min())
        self.assertLess(parts[1].step.max(),parts[2].step.min())
        self.assertTrue(all(not (set(a.event_id)&set(b.event_id)) for i,a in enumerate(parts) for b in parts[i+1:]))

    def test_insufficient_labels_writes_report_and_does_not_create_candidate(self):
        hist=self.root/"historical.parquet"
        pd.DataFrame([{"event_id":i,"step":i+1,**{n:float(i) for n in self.features},TARGET:i%2}
                      for i in range(20)]).to_parquet(hist,index=False)
        report_path=self.root/"stage4-report.json"; candidate=self.root/"candidate"
        active_paths=[Path("models/active/model.json"),Path("models/active/metadata.json"),Path("models/active/threshold.json")]
        hashes_before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in active_paths}
        before=self.db_path.read_bytes()
        with patch.dict(os.environ,{"FRAUD_TRUSTED_LABEL_SOURCES":"","FRAUD_TRUSTED_LABEL_TOKEN":""}):
            report=stage4.run_stage4(label_db=self.db_path,prediction_log=self.root/"missing.jsonl",
                historical_dataset=hist,report_path=report_path,candidate_dir=candidate,train_candidate=True)
        self.assertEqual(report["status"],"insufficient_eligible_labels")
        self.assertFalse(report["training_started"])
        self.assertFalse(candidate.exists())
        self.assertEqual(self.db_path.read_bytes(),before)
        self.assertEqual(hashes_before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in active_paths})
        self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
