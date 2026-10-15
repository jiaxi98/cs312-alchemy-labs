"""Deploy and inspect CS312 on an existing allocation via the local fleet adapter.

The adapter provides read-only instance discovery and authenticated connections;
its credentials and host records are never serialized into the course repository.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def allocation_policy():
    return json.loads((ROOT / "allocation_policy.json").read_text())


def require_allowed(alias):
    policy = allocation_policy()
    if alias not in policy["allowed_aliases"] or alias in policy["excluded_aliases"]:
        raise RuntimeError(f"Course jobs are disabled on allocation {alias}: reserved by the user.")


def local_settings():
    path = ROOT / "reports/cluster.local.json"
    return json.loads(path.read_text()) if path.exists() else {}


def connection(alias, index):
    require_allowed(alias)
    adapter = os.environ.get("CS312_FLEET_SCRIPTS") or local_settings().get("fleet_scripts")
    if not adapter:
        raise RuntimeError("Set CS312_FLEET_SCRIPTS to the existing fleet scripts directory.")
    sys.path.insert(0, adapter)
    import fleet_occupancy_watch as fleet
    from star_nocfg_fleet import connect
    _, hosts = fleet.hosts_for(alias)
    return connect(hosts[index])


def remote(client, source, spec, timeout=55):
    # Consume the SSH payload before running short probes, which might otherwise
    # exit and close their channel while the caller is still writing stdin.
    preamble = "import sys,io; _cs312_payload=sys.stdin.read(); sys.stdin=io.StringIO(_cs312_payload)\n"
    stdin, stdout, stderr = client.exec_command("python -c " + shlex.quote(preamble + source), timeout=timeout)
    stdin.write(json.dumps(spec))
    stdin.channel.shutdown_write()
    out, err = stdout.read().decode(), stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    stdin.close(); stdout.close(); stderr.close()
    if code:
        raise RuntimeError(err[-4000:] or out[-4000:])
    return json.loads(out)


def stage(client):
    settings = local_settings()
    remote_root = os.environ.get("CS312_REMOTE_ROOT") or settings.get("remote_root")
    remote_python = os.environ.get("CS312_REMOTE_PYTHON") or settings.get("remote_python")
    if not remote_root or not remote_python:
        raise RuntimeError("Configure CS312_REMOTE_ROOT and CS312_REMOTE_PYTHON.")
    upstream = ROOT / "upstream/a1"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    if commit != lock["upstream"]["commit"] or subprocess.check_output(
            ["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Only the pinned, clean upstream can be deployed.")
    lab_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit the course changes before deploying a frozen snapshot.")
    paths = []
    for folder in ("scripts", "experiments"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and ".local." not in p.name)
    # git ls-files works both for embedded repositories and normal submodules.
    # Export source only: submodule .git pointers are not portable to GPU nodes.
    paths.extend(upstream / name for name in subprocess.check_output(
        ["git", "-C", str(upstream), "ls-files", "-z"], text=True).split("\0") if name)
    paths.append(ROOT / "sources.lock.json")
    paths.append(ROOT / "requirements-tracking.txt")
    paths.append(ROOT / "allocation_policy.json")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = json.dumps(dict(files=hashes, upstream_commit=commit, lab_commit=lab_commit), sort_keys=True).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        info = tarfile.TarInfo("DEPLOYMENT.json")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    payload = buffer.getvalue()
    spec = dict(code=f"{remote_root}/code/{digest[:16]}", sha=hashlib.sha256(payload).hexdigest())
    source = '''import sys,pathlib,json,tarfile,io,hashlib
s=json.loads(sys.argv[1]);raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==s['sha']
root=pathlib.Path(s['code']);root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw))as archive:archive.extractall(root,filter='data')
files=json.loads((root/'DEPLOYMENT.json').read_text())['files']
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in files.items())
print(json.dumps({'code':str(root),'file_count':len(files)}))'''
    stdin, stdout, stderr = client.exec_command(
        "python -c " + shlex.quote(source) + " " + shlex.quote(json.dumps(spec)), timeout=55)
    stdin.write(payload); stdin.channel.shutdown_write()
    out, err = stdout.read().decode(), stderr.read().decode()
    status = stdout.channel.recv_exit_status()
    stdin.close(); stdout.close(); stderr.close()
    if status:
        raise RuntimeError(err[-3000:])
    return {**json.loads(out), "manifest_sha256": digest, "remote_root": remote_root,
            "python": remote_python, "lab_commit": lab_commit}


DOWNLOAD = r'''
import pathlib,sys,json,subprocess,os,time
s=json.load(sys.stdin);root=pathlib.Path(s['remote_root']);root.mkdir(parents=True,exist_ok=True)
folder=root/'data';folder.mkdir(exist_ok=True)
receipt=root/'download.json'
if receipt.exists():
 old=json.loads(receipt.read_text())
 try:
  st=pathlib.Path('/proc',str(old['pid']),'stat').read_text().rsplit(')',1)[1].split()
  alive=st[19]==old['start'] and st[0]!='Z'
 except OSError:alive=False
 if alive:print(json.dumps(old));sys.exit(0)
lock=json.loads((pathlib.Path(s['code'])/'sources.lock.json').read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DL_ALCHEMY_DATA_DIR=str(folder),PYTHONUNBUFFERED='1')
if s.get('http_proxy'):
 env.update(http_proxy=s['http_proxy'],https_proxy=s['http_proxy'],HTTP_PROXY=s['http_proxy'],HTTPS_PROXY=s['http_proxy'])
cmd=[s['python'],'-u',s['code']+'/scripts/prepare_data.py','--output-dir',str(folder/'dclm_9p6m_ctx1024')]
with(root/'download.log').open('ab')as log:
 child=subprocess.Popen(cmd,cwd=s['code']+'/upstream/a1',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
st=pathlib.Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
result={'pid':child.pid,'start':st[19],'time':time.time(),'revision':lock['dataset']['revision'],'command':cmd}
receipt.write_text(json.dumps(result));print(json.dumps(result))
'''

START = r'''
import pathlib,sys,json,subprocess,os,time,fcntl
s=json.load(sys.stdin);node_lock=open('/tmp/scaling_extension_node.lock','a')
fcntl.flock(node_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=s['gpu']; gpu_lock=open(f'/tmp/cs312_gpu_{gpu}.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(line.split(',')[0].strip()==uuids[gpu].strip()for line in apps.splitlines()),'Target GPU is busy'
root=pathlib.Path(s['remote_root']);output=root/'runs'/s['attempt']
assert not output.exists(),'Attempt already exists; inspect instead of relaunching'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
if s.get('swanlab_api_key'):
 env['SWANLAB_API_KEY']=s['swanlab_api_key']
if s.get('http_proxy'):
 env.update(http_proxy=s['http_proxy'],https_proxy=s['http_proxy'],HTTP_PROXY=s['http_proxy'],HTTPS_PROXY=s['http_proxy'])
for key in list(env):
 if key.startswith(('OMPI_','PMI_','PMIX_'))or key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'):
  env.pop(key,None)
data=root/'data'
if s.get('cpu_check'):
 data=root/'synthetic-check';data.mkdir(exist_ok=True)
 build="""import numpy as np,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);rng=np.random.default_rng(123)
for split,n in [('train',16),('val',8)]:
 p=root/split;p.mkdir(exist_ok=True);rng.integers(0,32,size=(n,16),dtype=np.uint16).tofile(p/'tokens.bin')
 (p/'metadata.json').write_text(json.dumps({'dtype':'uint16','num_sequences':n,'seq_len':16,'token_file':'tokens.bin'}))
"""
 subprocess.run([s['python'],'-c',build,str(data)],check=True,env=dict(env,CUDA_VISIBLE_DEVICES=''))
cmd=[s['python'],'-u',s['code']+'/scripts/run.py','--run-id',s['run_id'],'--data-dir',str(data),'--output-dir',str(output),'--tracker',s['tracker'],'--swanlab-project',s['swanlab_project']]
if s.get('swanlab_workspace'):cmd.extend(['--swanlab-workspace',s['swanlab_workspace']])
if s.get('cpu_check'):cmd.append('--cpu-check')
logs=root/'launches';logs.mkdir(exist_ok=True)
with(logs/(s['attempt']+'.log')).open('ab')as log:
 child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,pass_fds=(gpu_lock.fileno(),))
st=pathlib.Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
receipt={'pid':child.pid,'start':st[19],'gpu':gpu,'attempt':s['attempt'],'output':str(output),'command':cmd,'time':time.time()}
(logs/(s['attempt']+'.json')).write_text(json.dumps(receipt));print(json.dumps(receipt))
'''

STATUS = r'''
import json,pathlib,sys,subprocess
s=json.load(sys.stdin);root=pathlib.Path(s['remote_root']);out={'runs':{},'data':{}}
for p in(root/'data').rglob('*'):
 if p.is_file():out['data'][str(p.relative_to(root/'data'))]=p.stat().st_size
p=root/'download.log';out['download_tail']=p.read_text(errors='replace')[-2000:]if p.exists()else''
for p in(root/'launches').glob('*.json'):
 r=json.loads(p.read_text());d=pathlib.Path(r['output']);q=d/'status.json'
 try:st=pathlib.Path('/proc',str(r['pid']),'stat').read_text().rsplit(')',1)[1].split();alive=st[19]==r['start'] and st[0]!='Z'
 except OSError:alive=False
 log=p.with_suffix('.log');row={'alive':alive,'status':json.loads(q.read_text())if q.exists()else None,'log_tail':log.read_text(errors='replace')[-3500:]if log.exists()else''}
 out['runs'][r['attempt']]=row
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "download", "run", "status"))
    parser.add_argument("--alias", default=allocation_policy()["default_alias"])
    parser.add_argument("--host", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-id", default="a1-smoke")
    parser.add_argument("--attempt")
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--tracker", choices=("swanlab-online", "swanlab-offline", "wandb-offline"),
                        default="swanlab-online")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-workspace")
    args = parser.parse_args()
    state_file = ROOT / "reports/stage.local.json"
    client = connection(args.alias, args.host)
    try:
        if args.action == "stage":
            result = stage(client)
            state_file.write_text(json.dumps(result, indent=2) + "\n")
        else:
            spec = json.loads(state_file.read_text())
            if args.action == "download":
                spec["http_proxy"] = os.environ.get("CS312_HTTP_PROXY") or local_settings().get("http_proxy")
            if args.action == "run":
                if not args.attempt:
                    parser.error("--attempt is required for a run")
                spec.update(gpu=args.gpu,run_id=args.run_id,attempt=args.attempt,cpu_check=args.cpu_check)
                from tracking import settings, api_key
                tracking = settings()
                spec.update(tracker=args.tracker,
                    swanlab_project=args.swanlab_project or tracking.get("project", "cs312-alchemy-labs"),
                    swanlab_workspace=args.swanlab_workspace or tracking.get("workspace"),
                    http_proxy=os.environ.get("CS312_HTTP_PROXY") or local_settings().get("http_proxy"))
                if args.tracker == "swanlab-online":
                    spec["swanlab_api_key"] = api_key()
                    if not spec["swanlab_api_key"]:
                        raise RuntimeError("Configure SwanLab with scripts/configure_swanlab.py before launching.")
            result = remote(client, {"download": DOWNLOAD,"run": START,"status": STATUS}[args.action], spec)
            if args.action == "status":
                (ROOT / "reports/status.local.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
