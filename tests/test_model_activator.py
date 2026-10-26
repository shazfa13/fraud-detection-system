"""Temporary dummy-artifact tests for model activation."""
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from src.retraining import model_activator as activator
PROJECT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((PROJECT/"models"/"model_v1_metadata.json").read_text(encoding="utf-8"))["feature_names"]
def bundle(path, version, features=None, threshold=.5):
    path.mkdir(parents=True, exist_ok=True)
    values = {
        "model.json": {"dummy": version},
        "metadata.json": {"model_name":"test","model_version":version,"feature_count":34,
                          "feature_names":features or FEATURES,
                          "selected_classification_threshold":threshold},
        "threshold.json": {"selected_threshold":threshold},
    }
    for name, value in values.items():
        (path/name).write_text(json.dumps(value), encoding="utf-8")
def sha(path):
    return {name:activator._hash(path/name) for name in activator.ARTIFACT_NAMES}
class ActivatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); root=Path(self.tmp.name)
        self.active,self.candidate,self.backup=root/"active",root/"candidate",root/"backup"
        bundle(self.active,"V2",threshold=.53); bundle(self.candidate,"V3",threshold=.47); self.backup.mkdir()
    def tearDown(self): self.tmp.cleanup()
    def test_valid_promotion_and_prior_backup(self):
        old,new=sha(self.active),sha(self.candidate)
        activator.promote_candidate(self.candidate,self.active,self.backup)
        self.assertEqual(sha(self.active),new); self.assertEqual(sha(self.backup),old)
    def test_missing_candidate_file_rejected(self):
        (self.candidate/"threshold.json").unlink(); before=sha(self.active)
        with self.assertRaises(activator.ModelActivationError): activator.promote_candidate(self.candidate,self.active,self.backup)
        self.assertEqual(sha(self.active),before)
    def test_reordered_features_rejected(self):
        changed=FEATURES.copy(); changed[0],changed[1]=changed[1],changed[0]
        bundle(self.candidate,"V3",changed,.47)
        with self.assertRaisesRegex(activator.ModelActivationError,"feature names/order"):
            activator.promote_candidate(self.candidate,self.active,self.backup)
    def test_bad_threshold_rejected(self):
        (self.candidate/"threshold.json").write_text('{"selected_threshold":3}')
        before=sha(self.active)
        with self.assertRaises(activator.ModelActivationError): activator.promote_candidate(self.candidate,self.active,self.backup)
        self.assertEqual(sha(self.active),before)
    def test_validation_failure_keeps_active(self):
        bundle(self.candidate,"V3",FEATURES[:-1],.47); before=sha(self.active)
        with self.assertRaises(activator.ModelActivationError): activator.promote_candidate(self.candidate,self.active,self.backup)
        self.assertEqual(sha(self.active),before)
    def test_rollback_swaps_and_is_reversible(self):
        old,new=sha(self.active),sha(self.candidate)
        activator.promote_candidate(self.candidate,self.active,self.backup); activator.rollback(self.active,self.backup)
        self.assertEqual(sha(self.active),old); self.assertEqual(sha(self.backup),new)
    def test_mid_promotion_failure_recovers_active(self):
        old=sha(self.active); replace,calls=activator.os.replace,0
        def fail_once(src,dst):
            nonlocal calls
            calls+=1
            if calls==4: raise OSError("simulated")
            return replace(src,dst)
        with patch.object(activator.os,"replace",side_effect=fail_once):
            with self.assertRaisesRegex(activator.ModelActivationError,"prior active restored"):
                activator.promote_candidate(self.candidate,self.active,self.backup)
        self.assertEqual(sha(self.active),old); self.assertEqual(activator.validate_bundle(self.active).model_version,"V2")
    def test_dry_run_changes_nothing(self):
        old,entries=sha(self.active),list(self.backup.iterdir())
        result=activator.promote_candidate(self.candidate,self.active,self.backup,dry_run=True)
        self.assertTrue(result.dry_run); self.assertEqual(sha(self.active),old); self.assertEqual(list(self.backup.iterdir()),entries)
    def test_second_promotion_preserves_last_backup(self):
        activator.promote_candidate(self.candidate,self.active,self.backup)
        backup,active=sha(self.backup),sha(self.active); bundle(self.candidate,"V4",threshold=.42)
        with self.assertRaisesRegex(activator.ModelActivationError,"rollback point"):
            activator.promote_candidate(self.candidate,self.active,self.backup)
        self.assertEqual(sha(self.backup),backup); self.assertEqual(sha(self.active),active)
    def test_nested_model_directories_are_rejected(self):
        nested=self.active/"candidate"; bundle(nested,"V3")
        before=sha(self.active)
        with self.assertRaisesRegex(activator.ModelActivationError,"non-nested"):
            activator.promote_candidate(nested,self.active,self.backup)
        self.assertEqual(sha(self.active),before); self.assertEqual(list(self.backup.iterdir()),[])
        with self.assertRaisesRegex(activator.ModelActivationError,"non-nested"):
            activator.rollback(self.active,self.active/"backup")
if __name__=="__main__": unittest.main(verbosity=2)
