"""Independent read-only monitor; never launches/restarts workers."""
import json,subprocess,shlex,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];c=json.loads((R/'ROOT.json').read_text())
code=r'''import json,sys,subprocess,time,os
from pathlib import Path
r=Path(sys.argv[1]);v=dict(monotonic=time.monotonic(),units={},processes=[])
for n in ['CONTROLLER_STARTED.json','SESSION_RESULT.json','FIRST_FAILURE.json']:
 p=r/'execution'/n
 if p.exists():v[n]=json.loads(p.read_text())
for p in (r/'execution').glob('*') if (r/'execution').exists() else []:
 if not p.is_dir():continue
 x=dict(committed_tokens=len(list(p.glob('TOKEN_*.json'))))
 for n in ['RESULT.json','FIRST_FAILURE.json','FIRST_FAILURE_B0.json','FIRST_FAILURE_B1.json','FIRST_FAILURE_B2.json']:
  q=p/n
  if q.exists():
   a=json.loads(q.read_text());x[n]={k:a[k] for k in ['status','error','tokens','method','TPS','seconds'] if k in a}
   if 'conditions' in a:x[n]['conditions']={k:z['status'] for k,z in a['conditions'].items()}
 q=p/'worker.log'
 if q.exists():x['log_tail']=q.read_text(errors='replace')[-800:]
 v['units'][p.name]=x
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
  if any(x.startswith(str(r)+'/scripts/gpu_') for x in cmd):v['processes'].append(dict(pid=int(p.name),starttick=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=cmd))
 except (OSError,UnicodeError):pass
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);v['NVML_apps']=p.stdout.strip()
v['controller_log_tail']=(r/'GPU_CONTROLLER.log').read_text(errors='replace')[-1500:]
s=os.statvfs(r);v['disk_available_bytes']=s.f_bavail*s.f_frsize
print(json.dumps(v))'''
p=subprocess.run(['ssh','-S',c['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join([c['remote_python'],'-B','-c',code,c['remote_root']])],capture_output=True,text=True,timeout=30)
assert p.returncode==0,p.stderr;v=json.loads(p.stdout);o=R/'monitoring';o.mkdir(exist_ok=True)
(o/('SNAPSHOT_'+str(int(v['monotonic']*1000))+'.json')).write_text(json.dumps(v,indent=2)+'\n')
(o/'LATEST_SNAPSHOT.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(dict(monotonic=v['monotonic'],units={k:dict(tokens=x['committed_tokens'],result=x.get('RESULT.json'),failure=x.get('FIRST_FAILURE.json')) for k,x in v['units'].items()},processes=[dict(pid=x['pid'],unit=x['cmdline'][-1]) for x in v['processes']],NVML_apps=v['NVML_apps'],protective_seconds_remaining=v.get('CONTROLLER_STARTED.json',{}).get('protective_deadline',v['monotonic'])-v['monotonic'],session=v.get('SESSION_RESULT.json'),failure=v.get('FIRST_FAILURE.json'),controller_log_tail=v['controller_log_tail'])))
