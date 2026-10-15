"""Audit declared A1 work and restore unclaimed work on idle, allowed GPU lanes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

from cluster import ROOT, allocation_policy, local_settings, remote, require_allowed
from fleet import canonical_host_index, hosts
from monitor_remote import RECOVER, SNAPSHOT
from tracking import api_key, settings

STATE = ROOT / "runtime/monitor"


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def clock_text(seconds=None):
    return datetime.fromtimestamp(seconds if seconds is not None else time.time(), ZoneInfo("Asia/Shanghai")).isoformat()


def definitions(config):
    work = {}
    for campaign in config["campaigns"]:
        plan = json.loads((ROOT / campaign["plan"]).read_text())
        for row in plan["runs"]:
            work[campaign["name"] + "/" + row["id"]] = {
                **row, "campaign": campaign["name"], "plan": campaign["plan"]}
    return work


def effective_work(work):
    replaced = set()
    for row in work.values():
        previous = row.get("supersedes")
        if previous:
            replaced.add(previous["campaign"] + "/" + previous["run_id"])
    return {key: row for key, row in work.items() if key not in replaced}, replaced


def terminal_state(claim):
    if claim is None:
        return "pending"
    status = claim.get("status") or {}
    if status.get("state") == "diverged":
        return "diverged"
    state = claim["state"]
    # The trainer writes complete before finishing its cloud upload. Treat it as
    # draining until the supervisor has actually observed the process exit.
    if state == "complete" and not claim.get("result_recorded"):
        return "finishing"
    return state


def eligible(row, host, gpu, policy):
    alias = host.rsplit("_", 1)[0]
    if alias not in policy["allowed_aliases"] or alias in policy["excluded_aliases"]:
        return False
    return (row.get("required_host_index") in (None, canonical_host_index(host))
            and row.get("required_gpu") in (None, gpu))


def work_digest(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for campaign in config["campaigns"]:
        digest.update((ROOT / campaign["plan"]).read_bytes())
    return digest.hexdigest()


def analyze(work, claims, snapshots, policy, now, stale_after):
    effective, replaced = effective_work(work)
    states, rows, problems = Counter(), [], []
    active_processes = {p.get("output") for host in snapshots.values()
                        for p in host["processes"] if p.get("output")}
    by_host = {host: {g["index"]: g for g in row["gpus"]} for host, row in snapshots.items()}
    for key, row in effective.items():
        claim = claims.get(key)
        state = terminal_state(claim)
        states[state] += 1
        entry = {"key": key, "display_name": row.get("display_name", row["id"]), "state": state,
                 "group": row["group"], "steps": (claim or {}).get("steps", 0)}
        if claim:
            entry.update(host=claim["host"], gpu=claim["gpu"],
                         final=(claim.get("status") or {}).get("final"))
        rows.append(entry)
        if state in ("failed", "diverged", "interrupted"):
            problems.append({"kind": state, "key": key, "error": (claim or {}).get("error", "")})
        if state in ("running", "claimed", "finishing") and claim:
            host = snapshots.get(claim["host"])
            if (host and claim["output"] not in active_processes
                    and now - float(claim.get("claimed_at") or now) > 120):
                # Claims can be created after /proc is sampled. Give fresh jobs
                # a grace period and confirm missing processes in two cycles.
                problems.append({"kind": "process_missing", "key": key})
            if now - float(claim.get("progress_mtime") or now) > stale_after:
                problems.append({"kind": "no_recent_training_progress", "key": key,
                                 "minutes": round((now - claim["progress_mtime"]) / 60, 1)})
    pending = [(key, row) for key, row in effective.items() if key not in claims]
    # Put fixed-GPU work first when matching free slots, otherwise an unbound job
    # could unnecessarily occupy the only GPU able to finish a serial comparison.
    pending.sort(key=lambda item: item[1].get("required_gpu") is None)
    recovery = []
    chosen = set()
    for host, gpu_map in sorted(by_host.items()):
        for gpu, info in sorted(gpu_map.items()):
            if info["occupied"] or info["course_lock_held"]:
                continue
            for key, row in pending:
                if key in chosen or not eligible(row, host, gpu, policy):
                    continue
                if snapshots[host]["campaign_states"].get(row["campaign"], {}).get("stopped"):
                    continue
                chosen.add(key)
                recovery.append({"host": host, "gpu": gpu, "key": key, "row": row})
                break
    return dict(total=len(effective), states=dict(states), rows=rows, problems=problems,
                recovery=recovery, replaced_count=len(replaced))


def snapshot(config):
    discovered, connect = hosts()
    if not discovered:
        raise RuntimeError("No allowed hosts are available")
    root = local_settings()["remote_root"]
    names = [c["name"] for c in config["campaigns"]]
    results, errors = {}, {}

    def probe(name, host, include_claims):
        client = connect(host)
        try:
            return remote(client, SNAPSHOT, dict(root=root, host=name, campaigns=names,
                          include_claims=include_claims), timeout=55)
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(probe, name, host, index == 0): name
                   for index, (name, host) in enumerate(discovered)}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                errors[name] = str(exc)
    if discovered[0][0] not in results:
        raise RuntimeError("Cannot read authoritative shared claims: " + str(errors))
    claims = results[discovered[0][0]]["claims"]
    return results, errors, claims, dict(discovered), connect


def publish_policy(host, connect, policy):
    client = connect(host)
    try:
        remote(client, '''import json,sys,os
from pathlib import Path
s=json.load(sys.stdin);p=Path(s['root'])/'monitor';p.mkdir(parents=True,exist_ok=True)
temp=p/'allocation_policy.tmp';temp.write_text(json.dumps(s['policy']));os.replace(temp,p/'allocation_policy.json')
print(json.dumps({'published':True}))''', dict(root=local_settings()["remote_root"], policy=policy))
    finally:
        client.close()


def recover(candidate, stage, snapshot, host, connect, run_number):
    row = candidate["row"]
    require_allowed(candidate["host"].rsplit("_", 1)[0])
    tracking = settings()
    key = api_key()
    if not key:
        raise RuntimeError("SwanLab credential is missing")
    old_spec = snapshot.get("ready") or {}
    if not old_spec:
        raise RuntimeError("GPU host runtime/data readiness file is missing")
    # Clock values are from the measured host specification, avoiding another
    # maximum-clock nvidia-smi query while neighboring GPUs are training.
    client = connect(host)
    try:
        clocks = remote(client, '''import pathlib,json,sys
s=json.load(sys.stdin);r=pathlib.Path(s['root'])/'campaigns'
for name in s['campaigns']:
 p=r/name/'hosts'/s['host']/'spec.json'
 if p.exists():print(json.dumps(json.loads(p.read_text())['sm_clocks']));break
else:raise RuntimeError('No measured clock metadata for this host')''',
            dict(root=stage["remote_root"], host=candidate["host"],
                 campaigns=[c["name"] for c in load(ROOT / "monitor_config.json")["campaigns"]]))
        spec = dict(host=candidate["host"], host_index=canonical_host_index(candidate["host"]),
            campaign=row["campaign"], plan=row["plan"],
            campaign_root=f"{stage['remote_root']}/campaigns/{row['campaign']}",
            remote_root=stage["remote_root"], code=stage["code"], python=stage["python"],
            data_dir=old_spec["data"], gpus=[candidate["gpu"]], sm_clocks=clocks,
            project=tracking["project"], workspace=tracking["workspace"], api_key=key,
            proxy=local_settings().get("http_proxy"),
            recovery_id=f"monitor-{int(time.time())}-{run_number}",
            expected_boot=snapshot["boot_id"], only_run_ids=[row["id"]],
            plan_sha256=hashlib.sha256((ROOT / row["plan"]).read_bytes()).hexdigest())
        return remote(client, RECOVER, spec, timeout=55)
    finally:
        client.close()


def report(summary):
    lines = ["# A1 持续监控", "", f"更新时间：{summary['checked_at_beijing']}", "",
        f"状态：**{summary['phase']}**。有效目标 {summary['total']} 项；当前计数：`{summary['states']}`。", "",
        f"授权资源：box2 + hy，最多80卡。当前CUDA进程占用 {summary['cuda_cards']} 卡。", "",
        "已有GPU worker在任务结束后立即领取下一项；本监控每2分钟检查一次并恢复空闲卡上的漏领任务。", "",
        f"本轮恢复动作：{len(summary['actions'])}。当前异常：{len(summary['problems'])}。", "",
        "| 类型 | 任务 / 主机 | 说明 |", "|---|---|---|"]
    for row in summary["problems"]:
        text = row.get("error", "")[-350:].replace("\n", " ").replace("|", "/")
        lines.append(f"| {row['kind']} | {row.get('key', row.get('host', ''))} | {text} |")
    lines += ["", "原来的box1/h3中断任务由迁移任务取代，只统计有效目标一次。",
        "固定GPU实验继续遵守其绑定关系；数值发散和已开始但失败的运行不会自动重复启动。",
        "后台事件保存在 runtime/monitor/events.jsonl；本报告为本机实时文件，不自动提交Git。",
        "当前环境不支持聊天自动唤醒；本报告与SwanLab会持续更新。", ""]
    path = ROOT / "reports/monitor_live.local.md"
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(lines))
    temp.replace(path)


def run_cycle(apply=False):
    config = load(ROOT / "monitor_config.json")
    policy = allocation_policy()
    if (STATE / "STOP").exists():
        return {"phase": "stopped_by_user"}
    target_digest = work_digest(config)
    completed = load(STATE / "COMPLETE.json", {})
    if completed.get("work_digest") == target_digest:
        return completed
    stage = load(STATE / "stage.local.json")
    if apply and not stage:
        raise RuntimeError("Install monitor stage metadata before enabling recovery")
    snapshots, errors, claims, host_map, connect = snapshot(config)
    checked_at = time.time()
    analysis = analyze(definitions(config), claims, snapshots, policy, checked_at, config["stale_after_seconds"])
    old = load(STATE / "status.json", {})
    suspects = [p["key"] for p in analysis["problems"] if p["kind"] == "process_missing"]
    previous_suspects = set(old.get("missing_process_observations", []))
    problems = [p for p in analysis["problems"]
                if p["kind"] != "process_missing" or p["key"] in previous_suspects]
    problems += [{"kind": "host_unreachable", "host": k, "error": v} for k, v in errors.items()]
    actions = []
    if apply:
        # Local policy is authoritative; publishing it also makes new recovery
        # workers recheck the current permitted allocations before every job.
        publish_policy(next(iter(host_map.values())), connect, policy)
        for number, candidate in enumerate(analysis["recovery"][:config["max_recovery_launches_per_cycle"]]):
            try:
                result = recover(candidate, stage, snapshots[candidate["host"]],
                                 host_map[candidate["host"]], connect, number)
                actions.append(result)
            except Exception as exc:
                problems.append({"kind": "recovery_deferred", "key": candidate["key"], "error": str(exc)})
    states = analysis["states"]
    done = states.get("complete", 0) + states.get("diverged", 0) == analysis["total"]
    phase = "complete" if done else "attention" if problems else "running"
    result = dict(checked_at_beijing=clock_text(checked_at), checked_at=checked_at, phase=phase,
        work_digest=target_digest,
        total=analysis["total"], states=states, replaced_count=analysis["replaced_count"],
        cuda_cards=sum(g["occupied"] for h in snapshots.values() for g in h["gpus"]),
        course_lock_cards=sum(g["course_lock_held"] for h in snapshots.values() for g in h["gpus"]),
        snapshot_host_count=len(snapshots), max_parallel_gpus=policy["max_parallel_gpus"],
        missing_process_observations=suspects,
        problems=problems, actions=actions, rows=analysis["rows"])
    old_problems = {(p["kind"], p.get("key", p.get("host"))) for p in old.get("problems", [])}
    new_problems = [p for p in problems if (p["kind"], p.get("key", p.get("host"))) not in old_problems]
    completed_before = {r["key"] for r in old.get("rows", []) if r["state"] in ("complete", "diverged")}
    newly_finished = [r["key"] for r in analysis["rows"] if r["state"] in ("complete", "diverged") and r["key"] not in completed_before]
    event = dict(time=clock_text(), completed=newly_finished, problems=new_problems,
                 actions=actions, phase=phase)
    if not old or newly_finished or new_problems or actions or old.get("phase") != phase:
        with (STATE / "events.jsonl").open("a") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in event.items() if k != "completed"}, ensure_ascii=False), flush=True)
    atomic(STATE / "snapshot.local.json", dict(hosts=snapshots, claims=claims, errors=errors))
    atomic(STATE / "status.json", result)
    atomic(ROOT / "reports/monitor_status.local.json", result)
    report(result)
    if done:
        atomic(STATE / "COMPLETE.json", result)
        atomic(ROOT / "reports/monitor_final.local.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            run_cycle(apply=args.apply)
        except Exception as exc:
            error = dict(time=clock_text(), error=str(exc))
            previous = load(STATE / "last_error.json", {})
            if previous.get("error") != error["error"]:
                with (STATE / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps({"time": error["time"], "phase": "monitor_error",
                                             "error": error["error"]}, ensure_ascii=False) + "\n")
            atomic(STATE / "last_error.json", error)
            last = load(STATE / "status.json")
            if last:
                report({**last, "phase": "monitor_error; last successful snapshot shown",
                        "problems": [{"kind": "monitor_error", "error": error["error"]}],
                        "actions": []})
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
