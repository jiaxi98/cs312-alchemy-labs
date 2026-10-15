"""Import one existing course run into SwanLab, optionally following its live log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tracking import ScalarRecorder, api_key, serializable, settings


def read_history(filename, *, allow_partial=False):
    from swanlab.converter.wb.wbdatastore import WBDataStoreReader, DataStoreTruncatedError
    from wandb.proto.wandb_internal_pb2 import Record
    with WBDataStoreReader() as reader:
        reader.open(filename)
        while True:
            try:
                raw = reader.scan()
            except DataStoreTruncatedError:
                if allow_partial:
                    return
                raise
            if raw is None:
                return
            record = Record()
            record.ParseFromString(raw)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                if key and not key.startswith("_"):
                    row[key] = json.loads(item.value_json)
            if row:
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--workspace")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    files = list((root / "wandb").glob("offline-run-*/*.wandb"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one offline W&B file for this attempt.")
    status = json.loads((root / "status.json").read_text())
    if status["state"] == "running" and not args.follow:
        raise RuntimeError("Use --follow for an active run; do not publish it as completed.")
    output = root / "swanlab-import"
    output.mkdir(exist_ok=False)
    import swanlab
    if args.mode == "online":
        key = api_key()
        if not key:
            raise RuntimeError("Configure SwanLab credentials first.")
        swanlab.login(api_key=key, save=False)
    config = json.loads((root / "config.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    tracking = settings()
    run = swanlab.init(project=args.project or tracking.get("project", "cs312-alchemy-labs"),
        workspace=args.workspace or tracking.get("workspace"),
        experiment_name=f"{status['run_id']}--{root.name}", group="a1-lr",
        tags=["a1", "lr", "imported-wandb"], mode=args.mode, public=False,
        logdir=str(output / "swanlog"),
        config={"train": config, "provenance": environment,
                "import": {"source": "W&B binary records", "x_axis": "optimizer_step + 1"}},
        settings=swanlab.Settings(interactive=False,
            terminal=swanlab.Settings.Terminal(proxy_type="none"),
            probe=swanlab.Settings.Probe(hardware=False,runtime=False,requirements=False,
                                        git=False,monitor=False)))
    receipt = {"id": run.id, "url": run.url if args.mode == "online" else None}
    (output / "swanlab.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "metrics.jsonl").open("w", buffering=1) as journal:
        recorder = ScalarRecorder(run, journal, config["batch_size"] * config["train_dataset"]["context_length"])
        try:
            while True:
                status = json.loads((root / "status.json").read_text())
                running = status["state"] == "running"
                # Replay the last unflushed step: its evaluation may not have been
                # written yet. Earlier steps are already immutable and are skipped.
                floor = (recorder.step - 1) if recorder.step is not None else 0
                recorder.pending = {}
                for row in read_history(files[0], allow_partial=running):
                    if int(row["optimizer_step"]) + 1 > floor:
                        recorder.record(row)
                if not running:
                    recorder.flush()
                    break
                (output / "progress.json").write_text(json.dumps({
                    "last_seen_step": recorder.step, "source_state": status["state"],
                    **receipt}, indent=2) + "\n")
                time.sleep(15)
            expected = status.get("final", {}).get("completed_steps")
            if expected is not None and recorder.step != expected:
                raise RuntimeError(f"Imported {recorder.step} steps, expected {expected}.")
            run.config.update({"outcome": serializable(status)})
            success = status["state"] == "complete" and not recorder.nonfinite
            run.finish(state="success" if success else "crashed",
                       error=None if success else f"Source run ended as {status['state']}")
            (output / "progress.json").write_text(json.dumps({
                "last_seen_step": recorder.step, "source_state": status["state"],
                "import_state": "complete", **receipt}, indent=2) + "\n")
        except BaseException as exc:
            recorder.flush()
            run.finish(state="crashed", error=str(exc))
            raise
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
