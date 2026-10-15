"""Verify scientific coordinates and persistence, not only API call success."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tracking import ScalarRecorder, SwanLabMirror


class MemoryRun:
    def __init__(self):
        self.rows = []

    def log(self, data, step):
        self.rows.append((step, data))


class CoordinatesTest(unittest.TestCase):
    def test_eval_joins_train_and_next_lr_is_labeled(self):
        run, journal = MemoryRun(), io.StringIO()
        recorder = ScalarRecorder(run, journal, 65536)
        recorder.record({"optimizer_step": 93, "train_loss": 4.0, "learning_rate": 0.003})
        recorder.record({"optimizer_step": 93, "val_loss": 4.125,
                         "logging/rms/global/gradient": 0.02})
        recorder.record({"optimizer_step": 94, "train_loss": 3.99})
        recorder.flush()
        self.assertEqual([step for step, _ in run.rows], [94, 95])
        self.assertEqual(run.rows[0][1]["input_tokens_seen"], 94 * 65536)
        self.assertEqual(run.rows[0][1]["val_loss"], 4.125)
        self.assertEqual(run.rows[0][1]["next_learning_rate"], 0.003)
        self.assertNotIn("learning_rate", run.rows[0][1])
        self.assertEqual(len(journal.getvalue().splitlines()), 2)

    def test_divergence_is_preserved_without_a_fabricated_loss(self):
        run, journal = MemoryRun(), io.StringIO()
        recorder = ScalarRecorder(run, journal, 64)
        recorder.record({"optimizer_step": 0, "train_loss": float("nan")})
        recorder.flush()
        row = json.loads(journal.getvalue())
        self.assertTrue(recorder.nonfinite)
        self.assertIsNone(row["train_loss"])
        self.assertNotIn("train_loss", run.rows[0][1])
        self.assertEqual(run.rows[0][1]["diagnostics/nonfinite/train_loss"], 1)


class PersistenceTest(unittest.TestCase):
    def test_real_sdk_contains_train_eval_rms_at_matching_steps(self):
        import wandb
        from swanlab.proto.swanlab.record.v1.record_pb2 import Record
        from swanlab.sdk.internal.core_python.store import DataStoreReader
        from wandb.sdk.wandb_run import Run

        original_log = Run.log
        with tempfile.TemporaryDirectory(prefix="cs312-tracker-test-") as temp:
            folder = Path(temp)
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = temp
            with SwanLabMirror(folder, mode="offline", project="cs312-tracker-test",
                    workspace=None, experiment_name="coordinate-check", group="checks",
                    tags=["test"], config={"purpose": "integration_test"}, tokens_per_step=64):
                wandb.init(project="cs312-tracker-test", config={"batch_size": 4})
                wandb.define_metric("*", step_metric="optimizer_step")
                wandb.log({"optimizer_step": 0, "train_loss": 4.0,
                           "logging/rms/global/gradient": 0.125})
                wandb.log({"optimizer_step": 0, "val_loss": 4.25})
                wandb.log({"optimizer_step": 1, "train_loss": 3.5})
                wandb.log({"optimizer_step": 1, "val_loss": 3.75})
                wandb.finish()
            self.assertIs(Run.log, original_log)
            from import_wandb import read_history
            wb_files = list((folder / "wandb").glob("offline-run-*/*.wandb"))
            self.assertEqual(len(wb_files), 1)
            history = list(read_history(wb_files[0]))
            replay, replay_journal = MemoryRun(), io.StringIO()
            recorder = ScalarRecorder(replay, replay_journal, 64)
            for row in history:
                recorder.record(row)
            recorder.flush()
            self.assertEqual([s for s, _ in replay.rows], [1, 2])
            self.assertEqual(replay.rows[1][1]["val_loss"], 3.75)
            stored = {}
            files = list((folder / "swanlog").rglob("*.swanlab"))
            self.assertEqual(len(files), 1)
            reader = DataStoreReader()
            reader.open(files[0])
            try:
                for raw in reader:
                    record = Record()
                    record.ParseFromString(raw)
                    if record.HasField("scalar"):
                        stored[(record.scalar.key, record.scalar.step)] = record.scalar
            finally:
                reader.close()
            for key in ("train_loss", "val_loss"):
                self.assertIn((key, 1), stored)
                self.assertIn((key, 2), stored)
            self.assertIn(("logging/rms/global/gradient", 1), stored)
            self.assertEqual(stored[("val_loss", 2)].value.number, 3.75)
            self.assertEqual(len([key for key in stored if key[0] == "train_loss"]), 2)


if __name__ == "__main__":
    unittest.main()
