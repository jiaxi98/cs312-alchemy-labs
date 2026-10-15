"""Run one pinned course configuration on one GPU with SwanLab and local diagnostics."""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "a1"


def verify_source(expected_commit):
    deployment = ROOT / "DEPLOYMENT.json"
    if deployment.exists():
        snapshot = json.loads(deployment.read_text())
        if snapshot["upstream_commit"] != expected_commit:
            raise RuntimeError("Deployed upstream commit differs from the experiment plan.")
        for name, digest in snapshot["files"].items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Deployed source changed: {name}")
        return snapshot["lab_commit"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if actual_commit != expected_commit or dirty:
        raise RuntimeError("Upstream differs from the pinned clean checkout.")
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


class Transcript:
    """Mirror stdout and preserve the upstream's printed validation observations."""
    pattern = re.compile(r"Step (\d+)/(\d+), .*?Loss: ([^,]+), .*?, val Loss: (\S+)")

    def __init__(self, folder):
        self.console = sys.stdout
        self.log = (folder / "train.log").open("a", buffering=1)
        self.metrics = (folder / "validation.jsonl").open("a", buffering=1)
        self.buffer = ""
        self.rows = []

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            match = self.pattern.search(line)
            if match:
                step, total, loss, val = match.groups()
                row = dict(completed_steps=int(step), total_steps=int(total),
                           train_loss=float(loss), val_loss=float(val),
                           source="upstream_stdout_6_decimal_places")
                self.rows.append(row)
                self.metrics.write(json.dumps(row) + "\n")
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()
        self.metrics.flush()

    def close(self):
        self.flush()
        self.log.close()
        self.metrics.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/a1/initial.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh directory for this attempt; never overwritten.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tracker", choices=("swanlab-online", "swanlab-offline", "wandb-offline"),
                        default="swanlab-online")
    parser.add_argument("--swanlab-project", default="cs312-alchemy-labs")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--cpu-check", action="store_true",
                        help="Tiny synthetic integration check; not a course result.")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    selected = next((r for r in plan["runs"] if r["id"] == args.run_id), None)
    if selected is None:
        parser.error(f"Unknown run id: {args.run_id}")
    folder = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    if args.dry_run:
        print(json.dumps({**selected, "data_dir": str(data_dir), "output_dir": str(folder),
                          "logging": args.tracker + " + local W&B/JSONL backup",
                          "single_visible_gpu_required": not args.cpu_check}, indent=2))
        return

    # Resolve paths before importing the upstream's module-level default factories.
    os.environ["DL_ALCHEMY_DATA_DIR"] = str(data_dir)
    os.environ["DL_ALCHEMY_MODEL_DIR"] = str(folder / "checkpoints")
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(folder)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.tracker == "swanlab-online":
        from tracking import api_key
        if not api_key():
            raise RuntimeError("SwanLab API key is missing; run scripts/configure_swanlab.py first.")
    # Shared filesystems can fail when dlopen reads newly compiled extensions.
    # Configure both caches before importing torch/Triton; keep logs on shared storage.
    cache = Path(tempfile.mkdtemp(prefix="cs312-compile-",
                                 dir=os.environ.get("CS312_CACHE_ROOT", "/tmp")))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    if args.cpu_check:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(UPSTREAM))
    import torch
    from train import TrainConfig, train
    from metric_logging import AFTER_TRAIN_STEP, MetricLogger

    lab_commit = verify_source(plan["upstream_commit"])
    if not args.cpu_check and torch.cuda.device_count() != 1:
        raise RuntimeError("Each experiment requires exactly one visible CUDA GPU.")
    if not args.cpu_check and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The course mixed-precision recipe requires BF16 support.")
    sources = json.loads((ROOT / "sources.lock.json").read_text())
    if not args.cpu_check:
        ready = json.loads((data_dir / "dclm_9p6m_ctx1024/READY.json").read_text())
        if ready["revision"] != sources["dataset"]["revision"]:
            raise RuntimeError("Data revision differs from sources.lock.json.")
        for entry in ready["files"]:
            if (data_dir / "dclm_9p6m_ctx1024" / entry["path"]).stat().st_size != entry["size_bytes"]:
                raise RuntimeError("Data file size differs from the verified manifest.")
    folder.mkdir(parents=True, exist_ok=False)
    scalar_file = (folder / "train_metrics.jsonl").open("a", buffering=1)

    def log_local(ctx):
        # AFTER_TRAIN_STEP is before zero_grad; this LR is for the NEXT step.
        scalar_file.write(json.dumps({
            "completed_steps": ctx.step + 1,
            "train_loss": ctx.loss.detach().item(),
            "next_learning_rate": ctx.scheduler.get_last_lr()[0],
        }) + "\n")
        return {}

    config = TrainConfig(
        **selected["config"], run_name_suffix=args.run_id, model_dir=folder / "checkpoints",
        wandb_online=True, wandb_entity="", wandb_project="cs312-alchemy",
        wandb_tags=("a1", selected["group"], args.run_id), resume_from_checkpoint=False,
    )
    config = replace(config, metric_loggers=config.metric_loggers + (
        MetricLogger(event=AFTER_TRAIN_STEP, fn=log_local),))
    if args.cpu_check:
        from data import token_dataset_at
        from model_config import LMConfig
        config = replace(config,
            model_config=LMConfig(name="cpu-pipeline-check", vocab_size=32,
                context_length=16, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16),
            train_dataset=token_dataset_at(data_dir / "train", context_length=16),
            val_dataset=token_dataset_at(data_dir / "val", context_length=16),
            num_train_sequences=16, batch_size=4, num_evals=2, data_seed=None,
            precision="fp32", latest_checkpoint_frequency=None, save_model=False,
        )
        torch.set_num_threads(2)
    metadata = asdict(replace(config, metric_loggers=()))
    environment = dict(
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        wandb=version("wandb"), numpy=version("numpy"),
        gpu=None if args.cpu_check else torch.cuda.get_device_name(0),
        capability=None if args.cpu_check else torch.cuda.get_device_capability(0),
        upstream_commit=plan["upstream_commit"], lab_commit=lab_commit,
        cpu_check=args.cpu_check, compile_cache=str(cache),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        sources=sources,
        tracker=args.tracker,
    )
    if args.tracker.startswith("swanlab"):
        environment["swanlab"] = version("swanlab")
    if not args.cpu_check:
        environment["verified_data"] = ready
    write_json(folder / "config.json", metadata)
    write_json(folder / "environment.json", environment)
    start = time.perf_counter()
    write_json(folder / "status.json", dict(state="running", run_id=args.run_id,
        started_at=datetime.now(timezone.utc).isoformat(), cpu_check=args.cpu_check))
    transcript = Transcript(folder)
    if args.tracker.startswith("swanlab"):
        from tracking import SwanLabMirror
        tracking = SwanLabMirror(folder,
            mode=args.tracker.removeprefix("swanlab-"), project=args.swanlab_project,
            workspace=args.swanlab_workspace, experiment_name=f"{args.run_id}--{folder.name}",
            group="a1-" + selected["group"], tags=config.wandb_tags,
            config=dict(train=metadata, provenance=environment, scope=selected["purpose"]),
            tokens_per_step=config.batch_size * config.train_dataset.context_length)
    else:
        tracking = nullcontext()
    try:
        with tracking as mirror:
            with redirect_stdout(transcript):
                train(config)
            if not transcript.rows or transcript.rows[-1]["completed_steps"] != transcript.rows[-1]["total_steps"]:
                raise RuntimeError("Training returned without a terminal validation observation.")
            import math
            finite = all(math.isfinite(row["val_loss"]) and math.isfinite(row["train_loss"])
                         for row in transcript.rows)
            if mirror is not None:
                finite = finite and not mirror.recorder.nonfinite
                if not finite:
                    mirror.mark_failed("Non-finite training or validation metric")
            write_json(folder / "status.json", dict(state="complete" if finite else "diverged",
                run_id=args.run_id, seconds=time.perf_counter() - start, cpu_check=args.cpu_check,
                final=transcript.rows[-1], finite=finite,
                peak_cuda_memory_gib=None if args.cpu_check else torch.cuda.max_memory_allocated() / 1024**3))
    except BaseException:
        write_json(folder / "status.json", dict(state="failed", run_id=args.run_id,
            seconds=time.perf_counter() - start, cpu_check=args.cpu_check, error=traceback.format_exc()))
        raise
    finally:
        transcript.close()
        scalar_file.close()


if __name__ == "__main__":
    main()
