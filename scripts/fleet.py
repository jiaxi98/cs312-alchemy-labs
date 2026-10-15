"""Prepare and operate the finite CS312 campaign across the verified H20 allocation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from cluster import ROOT, allocation_policy, local_settings, remote
from tracking import api_key, settings


def hosts():
    sys.path.insert(0, local_settings()["fleet_scripts"])
    import fleet_occupancy_watch as fleet
    from star_nocfg_fleet import connect
    rows = []
    for alias in allocation_policy()["allowed_aliases"]:
        _, found = fleet.hosts_for(alias)
        for index, host in enumerate(found):
            rows.append((f"{alias}_{index}", host))
    return rows, connect


def canonical_host_index(name):
    # Preserve the indices used by frozen plans even after allocations are removed.
    alias, index = name.rsplit("_", 1)
    return {"box1": 0, "box2": 2, "hy": 4, "h3": 12}[alias] + int(index)


def call_all(source, specs, *, selected=None):
    rows, connect = hosts()
    if selected and set(selected) - {name for name, _ in rows}:
        raise RuntimeError("Requested host is unavailable or reserved by the user")
    result = {}
    def call(index, name, host):
        c = connect(host)
        try:
            spec = specs(canonical_host_index(name), name)
            return name, remote(c, source, spec, timeout=55)
        finally:
            c.close()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(call, i, name, host): name for i, (name, host) in enumerate(rows)
                   if selected is None or name in selected}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, value = future.result()
                result[name] = value
                print(name, json.dumps(value, ensure_ascii=False)[:500], flush=True)
            except Exception as exc:
                result[name] = {"error": str(exc)}
                print(name, "ERROR", str(exc)[:300], flush=True)
    return result


PROBE = r'''
import json,pathlib,sys,subprocess
s=json.load(sys.stdin);p=pathlib.Path('/tmp/cs312-swanlab-v0102');r=p/'FLEET_READY.json';out={}
out['ready']=json.loads(r.read_text())if r.exists()else None
out['tail']=(p/'prepare_fleet.log').read_text()[-1000:]if(p/'prepare_fleet.log').exists()and not r.exists()else''
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
out['apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
print(json.dumps(out))
'''

START = r'''
import pathlib,json,sys,os,subprocess,fcntl,time
s=json.load(sys.stdin);root=pathlib.Path(s['campaign_root']);node=root/'hosts'/s['host'];node.mkdir(parents=True,exist_ok=True)
receipt=node/'launch.json'
if receipt.exists():print(receipt.read_text());sys.exit(0)
lock=open('/tmp/scaling_extension_node.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ready=pathlib.Path('/tmp/cs312-swanlab-v0102/FLEET_READY.json');assert ready.exists(),'Host not prepared'
clocks=subprocess.check_output(['nvidia-smi','--query-gpu=clocks.max.sm','--format=csv,noheader,nounits'],text=True,timeout=25).splitlines()
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=25).splitlines()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=25)
occupied={line.split(',')[0].strip()for line in apps.splitlines()}
assert len(clocks)==8
if not s.get('wait_for_slot'):
 assert all(uuids[i].strip()not in occupied for i in s['gpus']),'Target GPU busy'
spec={k:v for k,v in s.items()if k not in ['api_key','proxy']};spec['sm_clocks']=[int(x.strip())for x in clocks]
(node/'spec.json').write_text(json.dumps(spec))
env=dict(os.environ,SWANLAB_API_KEY=s['api_key'],http_proxy=s['proxy'],https_proxy=s['proxy'],HTTP_PROXY=s['proxy'],HTTPS_PROXY=s['proxy'],PYTHONUNBUFFERED='1')
with(node/'supervisor.log').open('ab')as log:
 child=subprocess.Popen([s['python'],'-u',s['code']+'/scripts/fleet_worker.py','--spec',str(node/'spec.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
stat=pathlib.Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
result={'host':s['host'],'pid':child.pid,'start':stat[19],'gpus':s['gpus'],'campaign':s['campaign'],'time':time.time()}
receipt.write_text(json.dumps(result));print(json.dumps(result))
'''

STATUS = r'''
import pathlib,json,sys,subprocess,time
s=json.load(sys.stdin);p=pathlib.Path(s['campaign_root'])/'hosts'/s['host'];out={'host':s['host'],'time':time.time()}
for name in ['status.json','COMPLETE.json']:
 f=p/name;out[name]=json.loads(f.read_text())if f.exists()else None
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
out['apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
print(json.dumps(out))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("probe", "launch", "status"))
    p.add_argument("--campaign", default="a1-r2-20261015")
    p.add_argument("--plan", default="experiments/a1/round2.json")
    p.add_argument("--hosts", nargs="+")
    p.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--wait-for-slot", action="store_true",
                   help="Queue behind an existing course GPU lock without interrupting it")
    args = p.parse_args()
    stage = json.loads((ROOT / "reports/stage.local.json").read_text())
    tracking = settings()
    def spec(i, name):
        return dict(host=name, host_index=i, campaign=args.campaign, plan=args.plan,
            campaign_root=f"{stage['remote_root']}/campaigns/{args.campaign}",
            code=stage["code"], python="/tmp/cs312-swanlab-v0102/.venv/bin/python",
            data_dir="/tmp/cs312-swanlab-v0102/data", remote_root=stage["remote_root"],
            wait_for_slot=args.wait_for_slot,
            gpus=args.gpus, project=tracking["project"], workspace=tracking["workspace"],
            api_key=api_key(), proxy=local_settings().get("http_proxy"))
    result = call_all({"probe": PROBE,"launch": START,"status": STATUS}[args.action],
                      spec, selected=args.hosts)
    filename = ROOT / f"reports/{args.campaign}_{args.action}.local.json"
    filename.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", filename)
    if any("error" in row for row in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
