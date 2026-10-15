"""Small readback and guarded recovery programs sent to existing allowed hosts."""

SNAPSHOT = r'''
import json,sys,time,os,fcntl,subprocess
from pathlib import Path
s=json.load(sys.stdin);root=Path(s['root']);processes={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=(p/'cmdline').read_bytes().decode(errors='replace').split('\0');st=(p/'stat').read_text().rsplit(')',1)[1].split()
  if st[0]=='Z':continue
  scripts=[x for x in a if x.startswith(str(root)+'/code/')and x.endswith('.py')]
  if not scripts:continue
  row={'pid':int(p.name),'start':st[19],'script':scripts[0]}
  for flag,key in [('--spec','spec'),('--output-dir','output')]:
   if flag in a:row[key]=a[a.index(flag)+1]
  processes[int(p.name)]=row
 except (OSError,IndexError):pass
gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
apps_text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
apps={line.split(',')[0].strip() for line in apps_text.splitlines()if line.strip()}
gpus=[]
for line in gpu_text.splitlines():
 index,uuid,util,memory=[v.strip()for v in line.split(',')];index=int(index)
 held=False
 with open(f'/tmp/cs312_gpu_{index}.lock','a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:held=True
 gpus.append({'index':index,'uuid':uuid,'utilization':int(util),'memory_mib':int(memory),'occupied':uuid in apps,'course_lock_held':held})
def load(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
def tail_row(p):
 try:
  with p.open('rb')as f:
   f.seek(0,2);size=f.tell();f.seek(max(0,size-65536));raw=f.read()
  lines=raw.splitlines()
  if raw and not raw.endswith(b'\n'):lines=lines[:-1]
  return json.loads(lines[-1])if lines else None
 except (FileNotFoundError,json.JSONDecodeError):return None
claims={};campaign_states={}
for campaign in s['campaigns']:
 base=root/'campaigns'/campaign
 campaign_states[campaign]={'stopped':(base/'STOP').exists()}
 if not s.get('include_claims'):continue
 for p in(base/'claims').glob('*'):
  owner=load(p/'owner.json')
  if owner is None:continue
  d=Path(owner['output']);status=load(d/'status.json');result=load(p/'result.json');m=d/'train_metrics.jsonl';last=tail_row(m)
  state=(result or{}).get('state')or(status or{}).get('state')or owner['state']
  log=p/'train.log'
  error=(status or{}).get('error')or(result or{}).get('error')or''
  if state=='failed'and not error and log.exists():
   with log.open('rb')as f:f.seek(0,2);n=f.tell();f.seek(max(0,n-2000));error=f.read().decode(errors='replace')
  claims[campaign+'/'+p.name]={'campaign':campaign,'run_id':p.name,'host':owner['host'],'gpu':owner['gpu'],'output':str(d),'pid':owner.get('pid'),'state':state,'result_recorded':result is not None,'claimed_at':owner.get('claimed_at'),'status':status,'steps':(last or{}).get('completed_steps',0),'progress_mtime':m.stat().st_mtime if m.exists()else owner.get('claimed_at',0),'last_validation':tail_row(d/'validation.jsonl'),'error':error[-1500:]}
ready=load(Path('/tmp/cs312-swanlab-v0102/FLEET_READY.json'))
print(json.dumps({'host':s['host'],'checked_at':time.time(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'gpus':gpus,'processes':list(processes.values()),'campaign_states':campaign_states,'claims':claims,'ready':ready}))
'''

RECOVER = r'''
import json,sys,os,time,fcntl,subprocess,hashlib
from pathlib import Path
s=json.load(sys.stdin);root=Path(s['remote_root']);code=Path(s['code']);node_lock=open('/tmp/scaling_extension_node.lock','a')
fcntl.flock(node_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
policy=json.loads((root/'monitor/allocation_policy.json').read_text());alias=s['host'].rsplit('_',1)[0]
assert alias in policy['allowed_aliases']and alias not in policy['excluded_aliases'],'Allocation reserved by user'
assert not(root/'monitor/STOP').exists(),'Monitor stopped'
assert not(root/'campaigns'/s['campaign']/'STOP').exists(),'Campaign stopped'
assert hashlib.sha256((code/s['plan']).read_bytes()).hexdigest()==s['plan_sha256'],'Plan changed'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==s['expected_boot'],'Host rebooted; refresh inventory'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).splitlines()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
occupied={line.split(',')[0].strip()for line in apps.splitlines()if line.strip()}
locks=[]
for gpu in s['gpus']:
 assert uuids[gpu].strip()not in occupied,'GPU occupied since snapshot'
 lock=open(f'/tmp/cs312_gpu_{gpu}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
plan=json.loads((code/s['plan']).read_text());valid_ids=set()
for row in plan['runs']:
 if row.get('required_host_index')not in (None,s['host_index']):continue
 if row.get('required_gpu')is not None and row['required_gpu']not in s['gpus']:continue
 if not(root/'campaigns'/s['campaign']/'claims'/row['id']).exists():valid_ids.add(row['id'])
assert set(s['only_run_ids'])<=valid_ids and valid_ids,'No eligible unclaimed work'
ready=Path('/tmp/cs312-swanlab-v0102/FLEET_READY.json');assert ready.exists()
status_dir=root/'campaigns'/s['campaign']/'hosts'/s['host']/s['recovery_id'];status_dir.mkdir(parents=True,exist_ok=False)
spec={k:v for k,v in s.items()if k not in ('api_key','proxy','expected_boot','plan_sha256')};spec['status_dir']=str(status_dir);spec['policy_path']=str(root/'monitor/allocation_policy.json');spec['passed_gpu_lock_fds']={str(gpu):lock.fileno()for gpu,lock in zip(s['gpus'],locks)}
(status_dir/'spec.json').write_text(json.dumps(spec))
env=dict(os.environ,SWANLAB_API_KEY=s['api_key'],PYTHONUNBUFFERED='1')
if s.get('proxy'):env.update(http_proxy=s['proxy'],https_proxy=s['proxy'],HTTP_PROXY=s['proxy'],HTTPS_PROXY=s['proxy'])
with(status_dir/'supervisor.log').open('ab')as log:
 child=subprocess.Popen([s['python'],'-u',str(code/'scripts/fleet_worker.py'),'--spec',str(status_dir/'spec.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,pass_fds=tuple(lock.fileno()for lock in locks))
st=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
result={'pid':child.pid,'start':st[19],'host':s['host'],'gpus':s['gpus'],'only_run_ids':s['only_run_ids'],'campaign':s['campaign'],'time':time.time(),'spec':str(status_dir/'spec.json')}
(status_dir/'launch.json').write_text(json.dumps(result));print(json.dumps(result))
'''
