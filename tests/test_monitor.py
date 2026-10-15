"""Verify monitor completion accounting, resource exclusions and recovery boundaries."""
from __future__ import annotations

import sys
from pathlib import Path
import json
import os
import fcntl
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from monitor import analyze, effective_work, terminal_state
from install_monitor import MARKER, cron_text


POLICY = {"allowed_aliases": ["box2", "hy"], "excluded_aliases": ["box1", "h3"]}


def host(*, occupied=False, held=False, stopped=False):
    return {"gpus": [{"index": 0, "occupied": occupied, "course_lock_held": held}],
            "processes": [], "campaign_states": {"current": {"stopped": stopped}}}


def work(name, **extra):
    return {"id": name, "group": "test", "campaign": "current", "plan": "p.json", **extra}


class MonitorTest(unittest.TestCase):
    def analyze(self, jobs, claims=None, snapshots=None):
        return analyze(jobs, claims or {}, snapshots or {"hy_0": host()}, POLICY, 1000, 1800)

    def test_migration_counts_only_replacement(self):
        original = work("x")
        migrated = work("x-m80", supersedes={"campaign": "old", "run_id": "x"})
        jobs, replaced = effective_work({"old/x": original, "current/x-m80": migrated})
        self.assertEqual(set(jobs), {"current/x-m80"})
        self.assertEqual(replaced, {"old/x"})

    def test_running_diverged_failed_and_claimed_are_never_restarted(self):
        jobs = {"current/" + name: work(name) for name in ("running", "failed", "diverged", "pending")}
        claims = {"current/" + name: dict(state=name, status={"state": name}, host="hy_0", gpu=0,
            output="/run/" + name, progress_mtime=1000, result_recorded=True)
            for name in ("running", "failed", "diverged")}
        result = self.analyze(jobs, claims)
        self.assertEqual([r["key"] for r in result["recovery"]], ["current/pending"])
        self.assertEqual(result["states"]["diverged"], 1)

    def test_no_recovery_on_occupied_locked_or_reserved_gpu(self):
        jobs = {"current/x": work("x")}
        for name, snapshot in [("hy_0", host(occupied=True)), ("hy_0", host(held=True)),
                               ("box1_0", host()), ("h3_0", host()), ("hy_0", host(stopped=True))]:
            with self.subTest(name=name, snapshot=snapshot):
                self.assertFalse(self.analyze(jobs, snapshots={name: snapshot})["recovery"])

    def test_same_gpu_job_stays_on_its_host_and_gets_priority(self):
        jobs = {"current/free": work("free"),
                "current/fixed": work("fixed", required_host_index=4, required_gpu=0)}
        result = self.analyze(jobs, snapshots={"hy_0": host(), "hy_1": host()})
        assignments = {r["host"]: r["key"] for r in result["recovery"]}
        self.assertEqual(assignments, {"hy_0": "current/fixed", "hy_1": "current/free"})
        wrong = self.analyze({"current/fixed": jobs["current/fixed"]}, snapshots={"hy_1": host()})
        self.assertFalse(wrong["recovery"])

    def test_upload_in_progress_is_not_finished(self):
        self.assertEqual(terminal_state({"state": "complete", "result_recorded": False}), "finishing")
        self.assertEqual(terminal_state({"state": "complete", "result_recorded": True}), "complete")

    def test_new_claim_does_not_raise_missing_process_during_snapshot_race(self):
        claim = dict(state="running", status={"state": "running"}, host="hy_0", gpu=0,
                     output="/new-run", claimed_at=999, progress_mtime=999)
        result = self.analyze({"current/new": work("new")}, {"current/new": claim})
        self.assertFalse(any(p["kind"] == "process_missing" for p in result["problems"]))

    def test_cron_install_preserves_other_tasks_and_is_idempotent(self):
        previous = "# system agent\n* * * * * /system-agent\n0 * * * * old " + MARKER + "\n"
        line = "*/2 * * * * new " + MARKER
        actual = cron_text(previous, line)
        self.assertIn("* * * * * /system-agent\n", actual)
        self.assertNotIn("old ", actual)
        self.assertEqual(actual.count(MARKER), 1)
        self.assertEqual(cron_text(actual, line), actual)


class RecoveryWorkerTest(unittest.TestCase):
    def test_inherited_lock_and_task_allowlist(self):
        # Real process/file-lock behavior, with a CPU stub for the training task
        # and nvidia-smi so the test does not require or touch a GPU.
        script_dir = Path(__file__).resolve().parents[1] / "scripts"
        with tempfile.TemporaryDirectory(prefix="cs312-recovery-test-") as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            smi = root / "bin/nvidia-smi"
            smi.write_text("#!/bin/sh\ncase \"$*\" in *query-gpu=uuid*) echo GPU-test;; esac\n")
            smi.chmod(0o700)
            (root / "scripts/run.py").write_text('''import sys,json,os,fcntl
from pathlib import Path
path=Path(sys.argv[sys.argv.index('--output-dir')+1]);path.mkdir(parents=True)
with open(os.environ['TEST_LOCK'],'a')as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:held=True
 else:held=False
assert held,'Worker must hold the inherited lock throughout training'
(path/'status.json').write_text(json.dumps({'state':'complete'}))
''')
            (root / "plan.json").write_text(json.dumps({"runs": [work("x"), work("y")]}))
            campaign = root / "campaigns/test"
            lock_path = root / "lock"
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                spec = dict(plan="plan.json", campaign_root=str(campaign), host="hy_0", host_index=4,
                    campaign="test", remote_root=str(root), gpus=[0], sm_clocks=[1980], data_dir=str(root),
                    project="test", workspace="test", only_run_ids=["y"],
                    passed_gpu_lock_fds={"0": lock.fileno()})
                spec_path = root / "spec.json"
                spec_path.write_text(json.dumps(spec))
                bootstrap = (
                    "import sys; from pathlib import Path; "
                    f"sys.path.insert(0,{str(script_dir)!r}); import fleet_worker; "
                    f"fleet_worker.ROOT=Path({str(root)!r}); "
                    f"sys.argv=['fleet_worker','--spec',{str(spec_path)!r}]; fleet_worker.main()")
                env = dict(os.environ, PATH=str(root / "bin") + ":" + os.environ["PATH"], TEST_LOCK=str(lock_path))
                subprocess.run([sys.executable, "-c", bootstrap], env=env, pass_fds=(lock.fileno(),),
                               check=True, capture_output=True, text=True, timeout=15)
            self.assertFalse((campaign / "claims/x").exists())
            result = json.loads((campaign / "claims/y/result.json").read_text())
            self.assertEqual(result["state"], "complete")


if __name__ == "__main__":
    unittest.main()
