"""Mirror the course's W&B scalar records to SwanLab at optimizer-step coordinates.

The upstream emits two W&B records on evaluation steps. Its implicit W&B record
index is therefore not the optimizer step. Merge those records and explicitly use
completed optimizer steps on SwanLab. W&B remains an offline local backup.
"""
from __future__ import annotations

import json
import math
from numbers import Real
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def settings():
    path = ROOT / "reports/swanlab.local.json"
    return json.loads(path.read_text()) if path.exists() else {}


def api_key():
    value = os.environ.get("SWANLAB_API_KEY")
    if value:
        return value
    filename = os.environ.get("CS312_SWANLAB_KEY_FILE") or settings().get("api_key_file")
    return Path(filename).read_text().strip() if filename else None


def serializable(value):
    return json.loads(json.dumps(value, default=str))


def display_name(run_id, config):
    """Keep the first A1 sweep readable; execution identity belongs in config."""
    if not re.fullmatch(r"a1-(?:lr-[0-9.e+-]+-s\d+|baseline-s\d+)", run_id):
        return run_id
    model_seed, data_seed = config["model_seed"], config["data_seed"]
    seeds = (f"seed={model_seed}" if model_seed == data_seed
             else f"model_seed={model_seed} · data_seed={data_seed}")
    return f"A1 · lr={config['learning_rate']:g} · {seeds}"


class ScalarRecorder:
    """Combine train/validation records for the same completed optimizer step."""
    def __init__(self, run, journal, tokens_per_step):
        self.run = run
        self.journal = journal
        self.tokens_per_step = tokens_per_step
        self.step = None
        self.pending = {}
        self.nonfinite = False

    def record(self, data):
        if "optimizer_step" not in data:
            raise ValueError("Course metric record is missing optimizer_step.")
        step = int(data["optimizer_step"]) + 1
        if step < 1 or (self.step is not None and step < self.step):
            raise ValueError("Optimizer steps must be positive and monotonic.")
        if self.step is not None and step != self.step:
            self.flush()
        self.step = step
        for key, value in data.items():
            if key in {"step", "optimizer_step"} or key.startswith("_"):
                continue
            if not isinstance(value, Real):
                raise TypeError(f"Expected scalar for course metric {key}")
            value = float(value)
            # The upstream logs LR after scheduler.step(), so label it accurately.
            key = "next_learning_rate" if key == "learning_rate" else key
            if not math.isfinite(value):
                self.nonfinite = True
                self.pending[f"diagnostics/nonfinite/{key}"] = 1
                self.pending[key] = None
            else:
                self.pending[key] = value

    def flush(self):
        if self.step is None or not self.pending:
            return
        row = {"completed_steps": self.step,
               "input_tokens_seen": self.step * self.tokens_per_step, **self.pending}
        self.journal.write(json.dumps(row, allow_nan=False) + "\n")
        self.journal.flush()
        # Preserve missing/nonfinite values in the journal; don't invent a finite loss.
        self.run.log({key: value for key, value in row.items() if value is not None}, step=self.step)
        self.pending = {}


class SwanLabMirror:
    def __init__(self, folder, *, mode, project, workspace, experiment_name, group,
                 tags, config, tokens_per_step):
        self.folder = Path(folder)
        self.mode = mode
        self.init_args = dict(project=project, workspace=workspace,
            experiment_name=experiment_name, group=group, tags=list(tags),
            config=serializable(config), mode=mode, public=False,
            logdir=str(self.folder / "swanlog"))
        self.tokens_per_step = tokens_per_step
        self.run = None
        self.recorder = None
        self.journal = None
        self.failed = None

    def __enter__(self):
        import swanlab
        import wandb
        from wandb.sdk.wandb_run import Run

        if self.mode == "online":
            key = api_key()
            if not key:
                raise RuntimeError("SwanLab API key is missing; run scripts/configure_swanlab.py first.")
            swanlab.login(api_key=key, save=False)
        self.run = swanlab.init(**self.init_args, settings=swanlab.Settings(
            interactive=False,
            terminal=swanlab.Settings.Terminal(proxy_type="none"),
            probe=swanlab.Settings.Probe(git=False, runtime=False, requirements=False),
        ))
        receipt = dict(id=self.run.id, url=self.run.url if self.mode == "online" else None,
                       project=self.init_args["project"], workspace=self.init_args["workspace"],
                       mode=self.mode, log_dir=str(self.run.dir))
        (self.folder / "swanlab.json").write_text(json.dumps(receipt, indent=2) + "\n")
        self.journal = (self.folder / "metrics.jsonl").open("a", buffering=1)
        self.recorder = ScalarRecorder(self.run, self.journal, self.tokens_per_step)
        self.original_init, self.original_log = wandb.init, Run.log

        def init(*args, **kwargs):
            result = self.original_init(*args, **kwargs)
            self.run.config.update({"resolved_train_config": serializable(dict(result.config))})
            return result

        def log(wb_run, data, *args, **kwargs):
            result = self.original_log(wb_run, data, *args, **kwargs)
            self.recorder.record(data)
            return result

        wandb.init, Run.log = init, log
        return self

    def mark_failed(self, message):
        self.failed = message

    def __exit__(self, exc_type, exc, tb):
        import wandb
        from wandb.sdk.wandb_run import Run
        wandb.init, Run.log = self.original_init, self.original_log
        try:
            self.recorder.flush()
            reason = str(exc) if exc is not None else self.failed
            if self.recorder.nonfinite:
                reason = reason or "Non-finite metric observed during training."
            status_path = self.folder / "status.json"
            if status_path.exists():
                self.run.config.update({"outcome": serializable(json.loads(status_path.read_text()))})
            self.run.finish(state="crashed" if reason else "success", error=reason)
        finally:
            self.journal.close()
        return False
