"""Run one independent course job per GPU, claiming finite work from a shared queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    plan_path = ROOT / spec["plan"]
    plan = json.loads(plan_path.read_text())
    campaign = Path(spec["campaign_root"])
    claims = campaign / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    node = campaign / "hosts" / spec["host"]
    node.mkdir(parents=True, exist_ok=True)
    status_path = node / "status.json"
    states = {}
    guard = threading.Lock()

    def save(gpu, row):
        with guard:
            states[str(gpu)] = row
            atomic(status_path, dict(host=spec["host"], updated_at=time.time(), workers=states))

    def lane(gpu):
        slot_lock = open(f"/tmp/cs312_gpu_{gpu}.lock", "a")
        fcntl.flock(slot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        for row in plan["runs"]:
            if (campaign / "STOP").exists():
                break
            reserved = row.get("required_host_index")
            if reserved is not None and reserved != spec["host_index"]:
                continue
            claim = claims / row["id"]
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            attempt = f"{spec['campaign']}-{row['id']}"
            output = Path(spec["remote_root"]) / "runs" / attempt
            owner = dict(run_id=row["id"], host=spec["host"], gpu=gpu, output=str(output),
                         attempt=attempt, claimed_at=time.time(), state="claimed")
            atomic(claim / "owner.json", owner)
            save(gpu, owner)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
                       CS312_HOST=spec["host"], CS312_MAX_SM_CLOCK_MHZ=str(spec["sm_clocks"][gpu]),
                       TORCHINDUCTOR_COMPILE_THREADS="4")
            for key in list(env):
                if key.startswith(("OMPI_", "PMI_", "PMIX_")) or key in (
                    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
                    env.pop(key, None)
            command = [sys.executable, "-u", str(ROOT / "scripts/run.py"),
                "--plan", str(plan_path), "--run-id", row["id"], "--data-dir", spec["data_dir"],
                "--output-dir", str(output), "--tracker", "swanlab-online",
                "--swanlab-project", spec["project"], "--swanlab-workspace", spec["workspace"]]
            try:
                with (claim / "train.log").open("ab") as log:
                    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=log, start_new_session=True, pass_fds=(slot_lock.fileno(),))
                    owner.update(state="running", pid=child.pid)
                    atomic(claim / "owner.json", owner)
                    save(gpu, owner)
                    exit_code = child.wait()
                result_path = output / "status.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else None
                success = exit_code == 0 and result and result["state"] == "complete"
                owner.update(state="complete" if success else "failed", exit_code=exit_code,
                             finished_at=time.time(), result=result)
            except Exception as exc:
                owner.update(state="failed", error=repr(exc), finished_at=time.time())
            atomic(claim / "result.json", owner)
            completed.append(row["id"])
            save(gpu, owner)
        save(gpu, dict(state="idle", finished_tasks=completed, finished_at=time.time()))
        slot_lock.close()

    errors = []
    with ThreadPoolExecutor(max_workers=len(spec["gpus"])) as pool:
        futures = {pool.submit(lane, gpu): gpu for gpu in spec["gpus"]}
        for future, gpu in futures.items():
            try:
                future.result()
            except Exception as exc:
                error = dict(state="failed", error=repr(exc))
                errors.append(dict(gpu=gpu, **error))
                save(gpu, error)
    atomic(node / "COMPLETE.json", dict(host=spec["host"], errors=errors, finished_at=time.time()))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
