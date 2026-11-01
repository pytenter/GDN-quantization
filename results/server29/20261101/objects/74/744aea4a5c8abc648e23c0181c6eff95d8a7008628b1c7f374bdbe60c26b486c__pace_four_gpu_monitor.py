"""Bounded blind observer and exclusively claimed CPU terminal delivery.

No Torch, scoring, model loading, scientific dispatch, signal, or reset occurs.
Predecessor helper code supplies immutable path/SHA/transfer primitives only.
"""
import argparse,base64,datetime,fcntl,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('four_gpu_delivery_primitives',HERE/'pace_independent_delivery.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
OLD_COMMON=O.REMOTE_COMMON;OLD_AUDIT55=O.REMOTE_AUDIT55;OLD_AUDIT29=O.REMOTE_AUDIT29
REVISION='QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_LIFECYCLE_REPAIR_20261026'
REMOTE='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_lifecycle_repair_20261026'
ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/'+REVISION
OLD_REMOTE='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_exponent_repair_20261025'
OLD_ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_EXPONENT_REPAIR_20261025'
OLD_PINS=dict(root=OLD_REMOTE,archive=OLD_ARCHIVE,source_sha256='8c4bfa5076906bd6edbc355e4ce3373cb9e0550eb8ce285dee6f07afbd13cb9d',permit_sha256='1f6aadf001d6650143f54bfe55231c1b4ae743d1b17a2d3875a7889f956216b4',manifest_sha256='2e23b9c7e240ba793ed21d4c4f5eba9be4448047be3e8f3252dad82fa306aff3')
COMMON=r'''
import base64,ctypes,hashlib,json,os,subprocess,sys,time
from pathlib import Path
TASK='QWEN_GDN_PACE48_AIME26_SEED1_60_V1'
REVISION='QWEN_GDN_PACE48_AIME26_SEED1_60_V1_4GPU_LIFECYCLE_REPAIR_20261026'
REMOTE='/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_4gpu_lifecycle_repair_20261026'
ARCHIVE='/data/zypan/qwen_gdn_pace48_archive/'+REVISION
def req(ok,message):
 if not ok:raise RuntimeError(message)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def raw(p):return Path(p).read_bytes()
def read(p):return json.loads(raw(p))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def safe(root,name):
 p=Path(name);req(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name and name!='.' and not any(c in name for c in '\x00\r\n'),'Unsafe path')
 q=Path(root)/p;z=q
 while z!=Path(root):
  req(not z.is_symlink(),'Symlink forbidden');z=z.parent
 return q
def rows_check(root,rows,external=False):
 seen=set();total=0
 for x in rows:
  req(x['path'] not in seen,'Duplicate inventory');seen.add(x['path']);p=Path(x['path']) if external else safe(root,x['path']);a=p.stat()
  req(p.is_file() and not p.is_symlink() and a.st_size==x['bytes'] and sha(p)==x['sha256'],'Fresh SHA/bytes mismatch: '+str(p));z=p.stat()
  req((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'File changed during audit')
  if external:
   for k,attr in [('inode','st_ino'),('mtime_ns','st_mtime_ns'),('device','st_dev')]:
    if k in x:req(x[k]==getattr(z,attr),'External stat differs: '+str(p))
  total+=z.st_size
 return dict(status='PASS_FRESH_SHA_BYTES',count=len(rows),bytes=total)
def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().split(b'\0')
 return dict(pid=int(pid),starttick=int(s[19]),cmdline=[v.decode() for v in cmd if v])
def kernel_snapshot(pid):
 fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 return dict(state=fields[0],ppid=int(fields[1]),starttick=int(fields[19]))
def compare(x):
 try:y=identity(x['pid'])
 except FileNotFoundError:return dict(expected=x,status='GONE',owned_alive=False)
 if y['starttick']!=x['starttick']:return dict(expected=x,actual=y,status='PID_REUSED_FOREIGN',owned_alive=False)
 try:
  kernel=kernel_snapshot(x['pid'])
 except (FileNotFoundError,ProcessLookupError):return compare(x)
 if kernel['starttick']!=x['starttick']:return dict(expected=x,actual=y,status='PID_REUSED_FOREIGN',owned_alive=False)
 pending=kernel['state'] in ('Z','X','x') or not y['cmdline']
 return dict(expected=x,actual=y,kernel=kernel,status='EXIT_PENDING' if pending else 'EXACT_ALIVE' if y==x else 'SAME_TICK_EXEC_UNRESOLVED',owned_alive=True)
def snapshot(p):
 r=Path(REMOTE);req(p['root']==REMOTE and p['archive']==ARCHIVE and sha(r/'source_manifest.json')==p['source_sha256'] and sha(r/'GPU_PERMIT.json')==p['permit_sha256'],'Root/SOURCE/permit differs')
 metadata=read(r/'ROOT.json');req(metadata['execution_revision']==REVISION and metadata['remote_root']==REMOTE and metadata['archive_root']==ARCHIVE,'Root metadata differs')
 ledger=read(r/'RESOURCE_LEDGER.json') if (r/'RESOURCE_LEDGER.json').exists() else {};ids=[]
 for n,key in [('OFFLINE_CHAIN_STARTED.json','bash'),('OFFLINE_CHAIN_ACTUAL_CHILD.json','parent'),('OFFLINE_PARENT_STARTED.json','identity'),('SESSION_STARTED.json','identity'),('FINISHER_STARTED.json','identity')]:
  if (r/n).is_file():ids.append(read(r/n)[key])
 ids += [x['identity'] for x in ledger.get('intervals',[]) if x.get('identity')]
 for q in r.glob('FROZEN_V4_FRESH_PASS_*.PROCESS_STARTED.json'):ids.append(read(q)['identity'])
 jobs={j['id']:j for j in read(r/'FORMAL_JOB_MANIFEST.json')['jobs']};active=[]
 for unit,row in ledger.get('active_units',{}).items():
  x=row['identity'];ids.append(x);gpu=row['gpu'];j=jobs.get(unit);t=r/(j['condition']+'_PACE48_DP'+str(j['decode_period']))/'trajectories'/unit if j else r/'execution'/unit
  d=read(t/'DISPATCH.json');req(d['unit']==unit and d['gpu']==gpu and d['gpu_uuid']==row['gpu_uuid'] and d['source_manifest_sha256']==p['source_sha256'] and d['permit_sha256']==p['permit_sha256'],'Active dispatch identity differs')
  state=compare(x);v=dict(unit=unit,gpu=gpu,gpu_uuid=row['gpu_uuid'],identity=state)
  if state['owned_alive']:
   try:rawenv=(Path('/proc')/str(x['pid'])/'environ').read_bytes()
   except (FileNotFoundError,ProcessLookupError):rawenv=b'';state=compare(x);v['identity']=state
   env=dict(v.split(b'=',1) for v in rawenv.split(b'\0') if b'=' in v)
   if state['status']=='EXACT_ALIVE' and rawenv:
    req(env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()==row['gpu_uuid'],'Actual UUID CVD differs');v['CUDA_VISIBLE_DEVICES']=row['gpu_uuid']
   else:
    state=compare(x);v['identity']=state
    req(state['status'] in ('EXIT_PENDING','GONE','PID_REUSED_FOREIGN'),'Actual live identity/environment mismatch')
    v['CUDA_VISIBLE_DEVICES_observation']='UNAVAILABLE_DURING_OWN_EXIT_BOUNDARY'
  if (t/'PROGRESS.json').exists():
   progress=read(t/'PROGRESS.json');v['progress']={k:progress[k] for k in ('phase','condition','generated_tokens','input_forwards','token','tokens','total','case','layer') if k in progress}
  if (t/'DRIVER_IDENTITY.json').exists():
   driver=read(t/'DRIVER_IDENTITY.json');req(driver['status']=='PASS' and driver['actual_uuid']==driver['expected_uuid']==driver['CUDA_VISIBLE_DEVICES']==row['gpu_uuid'] and driver['logical_ordinal']==0,'Driver ordinal identity differs');v['driver_uuid']=driver['actual_uuid']
  active.append(v)
 states=[compare(x) for x in {canonical(x):x for x in ids}.values()];scoped=[]
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:
   x=identity(int(q.name))
   if any(REMOTE in a for a in x['cmdline']):scoped.append(x)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 nvml=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);req(nvml.returncode==0,'NVML process query failed')
 apps=[]
 for line in nvml.stdout.splitlines():
  a=[s.strip() for s in line.split(',')]
  if len(a)==3:apps.append(dict(pid=int(a[0]),uuid=a[1],memory_MiB=float(a[2])))
 own=[a for a in apps if any(a['pid']==x['pid'] and compare(x)['owned_alive'] for x in ids)]
 exists={n:(r/n).is_file() for n in ['DELIVERY_READY.json','OFFLINE_CHAIN_OBSERVED_EXIT.json','OFFLINE_PARENT_RESULT.json','PARENT_CONTROLLER_EXIT.json','PARENT_FINISHER_EXIT.json','FINISHER_EXIT.json','final_verdict.json']}
 closed=not any(x['owned_alive'] for x in states) and not scoped and not own
 ready=closed and all(exists.values())
 if ready:
  chain=read(r/'OFFLINE_CHAIN_OBSERVED_EXIT.json');fin=read(r/'PARENT_FINISHER_EXIT.json');parent=read(r/'OFFLINE_PARENT_RESULT.json')
  req(chain['status']=='GENUINE_BASH_OWNING_WAIT' and type(chain['actual_shell_wait_returncode']) is int and chain['actual_shell_wait_returncode']==0 and chain['source_manifest_sha256']==p['source_sha256'] and chain['permit_sha256']==p['permit_sha256'],'Missing/failed genuine owning Bash wait is a CPU delivery failure')
  req(fin['actual_exited'] and fin['identity_gone'] and fin['returncode']==0 and parent['failure'] is None,'Actual owningPopen finisher termination failed')
  req(all(x['released'] and x['returncode'] is not None for x in ledger['intervals']) and not ledger['active_units'],'Ledger not truly released')
 failed=(r/'FIRST_FAILURE.json').exists() or (r/'FINISHER_FAILURE.json').exists()
 return dict(status='READY_FOR_INDEPENDENT_AUDIT' if ready else 'CPU_DELIVERY_NEEDS_ATTENTION' if closed and (r/'SESSION_STARTED.json').exists() else 'ACTIVE_FAILURE_STOPPING' if failed else 'PENDING_NO_ACTION',blind=True,formal_completed=ledger.get('formal_total_unique',len(read(r/'REUSE_ADMISSION.json')['reused_ids'])),formal_current_completed=ledger.get('current_formal_count',0),formal_reused=len(read(r/'REUSE_ADMISSION.json')['reused_ids']),active=active,identities=states,scoped_processes=scoped,NVML=dict(own_apps=own,foreign_apps=[a for a in apps if a not in own]),terminal_exists=exists,source_manifest_sha256=p['source_sha256'],permit_sha256=p['permit_sha256'],unix=time.time(),scientific_dispatches=0,signals=0)
'''
AUDIT55=r'''
r=Path(REMOTE);req(snapshot(P)['status']=='READY_FOR_INDEPENDENT_AUDIT','Terminal state changed')
sys.path.insert(0,str(r/'src'));import pace_control as C
from pace_four_gpu_results import verify_reuse,completed_rows,verify_lifecycle
source=C.authorize(r);checks=dict(source=source,external_model17=C.verify_external(r,full_model=True),reuse=verify_reuse(r,full_sources=True),primary=rows_check(r,read(r/'DELIVERY_MANIFEST.json')['files']))
invalid=[];complete=completed_rows(r,invalid);ledger=read(r/'RESOURCE_LEDGER.json');by={(x['stage'],x['unit']):x for x in ledger['intervals']}
for x in ledger['completed_units']:
 t=C.unit_target(r,x['stage'],x['unit']);v=C.verify_result(r,x['stage'],x['unit']);interval=read(t/'RESOURCE_INTERVAL.json');verify_lifecycle(r,t,interval);req(interval==by[x['stage'],x['unit']] and interval['returncode']==0 and interval['released'],'Legal completion interval differs')
 receipt=read(t/'DURABLE29_RECEIPT.json');req(receipt['manifest_sha256']==x['durable_manifest_sha256'] and receipt['actual_exit_returncode']==0,'Durable completed receipt differs')
native=[]
for x in ledger['intervals']:
 t=C.unit_target(r,x['stage'],x['unit'])
 for q in list(t.glob('LOADED_LIBRARY_*.json'))+list(t.glob('POSTFLIGHT_LIBRARY.json')):
  v=read(q);req(v['status']=='PASS_ACTUAL_LOADED_NATIVE_SHA_INODE','Mapped native closure proof missing');native.append(dict(path=q.relative_to(r).as_posix(),verification=rows_check(r,v['selected'],True)))
checks['actual_native_artifacts']=native
verdict=read(r/'final_verdict.json');req(verdict['formal_completed']==len(complete),'Final legal completion count differs')
manifest=read(r/'DELIVERY_MANIFEST.json');digest=sha(r/'DELIVERY_MANIFEST.json');ready=read(r/'DELIVERY_READY.json');archive=read(r/'ARCHIVE_VERIFICATION.json')
req(manifest['source_manifest_sha256']==P['source_sha256'] and manifest['permit_sha256']==P['permit_sha256'] and ready['manifest_sha256']==archive['manifest_sha256']==digest,'Final manifest/ready binding differs')
late=[]
for n in ['FINISHER_EXIT.json','PARENT_FINISHER_EXIT.json','OFFLINE_PARENT_RESULT.json','DELIVERY_READY.json','ARCHIVE_VERIFICATION.json','OFFLINE_CHAIN_STARTED.json','OFFLINE_CHAIN_ACTUAL_CHILD.json','OFFLINE_CHAIN_OBSERVED_EXIT.json','finisher.log']:
 q=r/n;late.append(dict(path=n,bytes=q.stat().st_size,sha256=sha(q)))
req(snapshot(P)['status']=='READY_FOR_INDEPENDENT_AUDIT','Terminal identity changed during audit')
print(json.dumps(dict(status='PASS_FOUR_GPU_INDEPENDENT55',checks=checks,manifest_sha256=digest,manifest_base64=base64.b64encode(raw(r/'DELIVERY_MANIFEST.json')).decode(),formal_completed=len(complete),final_status=verdict['status'],invalid=invalid,late_files=late,source_manifest_sha256=P['source_sha256'],permit_sha256=P['permit_sha256'],GPU_kernel_calls=0)))
'''
AUDIT29=r'''
r=Path(ARCHIVE);req(sha(r/'DELIVERY_MANIFEST.json')==P['manifest_sha256'],'Archive29 manifest differs');m=read(r/'DELIVERY_MANIFEST.json');req(m['source_manifest_sha256']==P['source_sha256'] and m['permit_sha256']==P['permit_sha256'],'Archive29 pins differ');proof=rows_check(r,m['files'])
expected={x['path'] for x in m['files']}|{'DELIVERY_MANIFEST.json','ARCHIVE_VERIFICATION.json'};req({p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file()}==expected,'Unexpected archive29 files')
live=Path(ARCHIVE.replace('/'+REVISION,'/LIVE_'+REVISION))/'units';units=[]
req(live.exists() or not read(r/'RESOURCE_LEDGER.json')['intervals'],'Missing LIVE units for actual intervals')
for p in (sorted(live.iterdir()) if live.exists() else []):
 req(p.is_dir() and not p.is_symlink() and '.staging' not in p.name,'Unclosed LIVE unit');manifest=read(p/'DURABLE29_MANIFEST.json');units.append(dict(unit=p.name,manifest_sha256=sha(p/'DURABLE29_MANIFEST.json'),verification=rows_check(p,manifest['files'])));receipt=read(p/'DURABLE29_RECEIPT.json');req(receipt['manifest_sha256']==sha(p/'DURABLE29_MANIFEST.json') and receipt['released'],'LIVE original receipt differs')
bootstrap=Path(ARCHIVE.replace('/'+REVISION,'/LIVE_'+REVISION))/'SOURCE_BOOTSTRAP';bootstrap_proof=rows_check(bootstrap,read(bootstrap/'SOURCE_BOOTSTRAP_MANIFEST.json')['files'])
print(json.dumps(dict(SOURCE_BOOTSTRAP=bootstrap_proof,status='PASS_FOUR_GPU_INDEPENDENT29',verification=proof,LIVE_units=units,archive_verification_sha256=sha(r/'ARCHIVE_VERIFICATION.json'),GPU_queries=0)))
'''

ANCESTOR_AUDIT=r'''
origins=[]
for spec in P['lineage']['roots']:
 r=Path(spec['remote'] if P['role']=='55' else spec['archive'])
 req(sha(r/'source_manifest.json')==spec['source'] and sha(r/'GPU_PERMIT.json')==spec['permit'] and sha(r/'DELIVERY_MANIFEST.json')==spec['manifest'],'Closed lineage anchor differs')
 origins.append(dict(root=str(r),source=rows_check(r,read(r/'source_manifest.json')['files']),science=rows_check(r,read(r/'DELIVERY_MANIFEST.json')['files'])))
supplements=[];live=[]
if P['role']=='29':
 for x in P['lineage']['supplements']:
  r=Path(x['root']);matches=[p for p in r.iterdir() if p.is_file() and p.suffix=='.json' and 'manifest' in p.name.lower() and sha(p)==x['manifest']]
  req(len(matches)==1,'Closed supplement manifest differs');supplements.append(dict(root=str(r),verification=rows_check(r,read(matches[0])['files'])))
 for label in P['lineage']['live_revisions']:
  r=Path('/data/zypan/qwen_gdn_pace48_archive')/('LIVE_'+label);s=r/'SOURCE_BOOTSTRAP';live.append(dict(root=str(s),verification=rows_check(s,read(s/'SOURCE_BOOTSTRAP_MANIFEST.json')['files'])))
  for u in sorted((r/'units').iterdir()) if (r/'units').exists() else []:
   req(u.is_dir() and '.staging' not in u.name,'Unclosed origin LIVE unit');d=read(u/'DURABLE29_RECEIPT.json');req(d['released'] and d['manifest_sha256']==sha(u/'DURABLE29_MANIFEST.json'),'Closed origin LIVE receipt differs');live.append(dict(root=str(u),verification=rows_check(u,read(u/'DURABLE29_MANIFEST.json')['files'])))
print(json.dumps(dict(status='PASS_ALL_FROZEN_LINEAGE_SOURCES_SCIENCE_AND_ARCHIVES',role=P['role'],origins=origins,supplements=supplements,LIVE=live,GPU_calls=0,mutations=0)))
'''

def lineage_audit(transport,role):
    return O.call(transport,ANCESTOR_AUDIT,dict(role=role,lineage=json.loads((ROOT/'provenance/CLOSED_LINEAGE_PINS.json').read_bytes())),role=role,timeout=7200)

def configure():
    O.REMOTE_COMMON=COMMON;O.REMOTE=REMOTE;O.ARCHIVE=ARCHIVE;O.REVISION=REVISION

def predecessor_audit(transport,role):
    code='import json\nP=json.loads('+repr(json.dumps(OLD_PINS))+')\n'+OLD_COMMON+(OLD_AUDIT55 if role=='55' else OLD_AUDIT29)
    result=transport.run(code,role=role,python='/data/ydai/miniconda3/envs/bitdecode/bin/python' if role=='55' else '/usr/bin/python3',timeout=7200)
    result.pop('manifest_base64',None);return result

def self_test():
    import tempfile
    for body in (AUDIT55,AUDIT29,ANCESTOR_AUDIT,O.REMOTE_FETCH,O.REMOTE_SUPPLEMENT):compile(COMMON+body,'four_gpu_CPU_remote','exec')
    ns={};exec(COMMON,ns)
    with tempfile.TemporaryDirectory(prefix='PACE4_monitor_CPU_') as d:
        r=Path(d);O.write_new(r/'a.txt',b'abc');row=dict(path='a.txt',bytes=3,sha256=O.digest(r/'a.txt'));assert ns['rows_check'](r,[row])['count']==1
        for name in ('../escape','/absolute','a\n.txt'):
            try:ns['safe'](r,name)
            except RuntimeError:pass
            else:raise AssertionError('Unsafe path admitted')
        lock=O.acquire_reader_lock(r);assert lock and O.acquire_reader_lock(r) is None;lock.close()
    identity=dict(pid=1,starttick=5,cmdline=['owned']);ns['identity']=lambda p:dict(identity,starttick=6);assert not ns['compare'](identity)['owned_alive']
    ns['kernel_snapshot']=lambda p:dict(state='R',ppid=2,starttick=5)
    ns['identity']=lambda p:dict(identity,cmdline=['exec_transition']);assert ns['compare'](identity)['owned_alive']
    ns['identity']=lambda p:dict(identity,cmdline=[]);assert ns['compare'](identity)['status']=='EXIT_PENDING'
    ns['identity']=lambda p:dict(identity);ns['kernel_snapshot']=lambda p:dict(state='Z',ppid=2,starttick=5)
    assert ns['compare'](identity)['status']=='EXIT_PENDING'
    assert O.connection_failure(TimeoutError()) is False and not O.connection_failure(RuntimeError('Source pin differs'))
    return dict(status='PASS_FOUR_GPU_CPU_MONITOR',cases=12,GPU_calls=0,model_loads=0)

def run(root,deliver=False):
    root=Path(root).resolve();configure();lock=O.acquire_reader_lock(root)
    if lock is None:return dict(status='EXISTING_READER_NO_REPEAT')
    try:
        source=O.digest(root/'source_manifest.json');permit=O.digest(root/'GPU_PERMIT.json');prior=O.existing_delivery(root,source,permit)
        if prior:return prior
        m=json.loads((root/'source_manifest.json').read_bytes());row=next(x for x in m['files'] if x['path']=='operations/pace_four_gpu_monitor.py');O.require(O.digest(__file__)==row['sha256'],'Monitor source differs')
        transport_path=root/'operations/remote.py';t=O.load_transport(transport_path,'db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40')
        pins=dict(root=REMOTE,archive=ARCHIVE,source_sha256=source,permit_sha256=permit)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        try:observation=O.call(t,'print(json.dumps(snapshot(P)))',pins,timeout=60)
        except BaseException as e:
            value=dict(status='WAIT_CONNECTION_NO_CLAIM' if O.connection_failure(e) else 'READONLY_FAILURE_NEEDS_ATTENTION',error=repr(e),source_manifest_sha256=source,permit_sha256=permit)
            O.write_new(root/'monitoring'/('OBSERVATION_ERROR_'+stamp+'.json'),O.canonical(value)+b'\n');return value
        obs=root/'monitoring'/('INDEPENDENT_BLIND_OBSERVATION_'+stamp+'.json');O.write_new(obs,O.canonical(observation)+b'\n')
        if observation['status']!='READY_FOR_INDEPENDENT_AUDIT' or not deliver:return dict(observation,observation=str(obs))
        O.write_new(root/'operations/INDEPENDENT_TERMINAL_DELIVERY_CLAIM.json',O.canonical(dict(status='CLAIMED_ONCE',observation_sha256=O.digest(obs),source_manifest_sha256=source,permit_sha256=permit,unix=time.time()))+b'\n')
        out=root/'independent_delivery';out.mkdir(exist_ok=False)
        try:
            a55=O.call(t,AUDIT55,pins,timeout=7200);manifest_raw=base64.b64decode(a55.pop('manifest_base64'),validate=True);O.require(hashlib.sha256(manifest_raw).hexdigest()==a55['manifest_sha256'],'Manifest transfer differs')
            O.write_new(out/'AUDIT55.json',O.canonical(a55)+b'\n');old55=predecessor_audit(t,'55');O.write_new(out/'PREDECESSOR_AUDIT55.json',O.canonical(old55)+b'\n')
            lineage55=lineage_audit(t,'55');O.write_new(out/'ALL_LINEAGE55.json',O.canonical(lineage55)+b'\n')
            a29=O.call(t,AUDIT29,dict(pins,manifest_sha256=a55['manifest_sha256']),role='29',timeout=7200);O.write_new(out/'AUDIT29.json',O.canonical(a29)+b'\n');old29=predecessor_audit(t,'29');O.write_new(out/'PREDECESSOR_AUDIT29.json',O.canonical(old29)+b'\n')
            lineage29=lineage_audit(t,'29');O.write_new(out/'ALL_LINEAGE29.json',O.canonical(lineage29)+b'\n')
            O.require(a29['archive_verification_sha256']==next(x['sha256'] for x in a55['late_files'] if x['path']=='ARCHIVE_VERIFICATION.json'),'Archive receipt mirror differs')
            late=out/'late_exit_receipts';late.mkdir();late_rows=O.fetch(t,REMOTE,a55['late_files'],late,role='55',maximum_total=33554432)
            manifest=json.loads(manifest_raw);core=out/'primary_archive_small_files';core.mkdir();small=[x for x in manifest['files'] if x.get('download_small') is True and not x['path'].lower().endswith(O.FORBIDDEN_SUFFIXES)]
            copied=O.fetch(t,ARCHIVE,small,core,role='29',maximum_total=2147483648);O.write_new(core/'DELIVERY_MANIFEST.json',manifest_raw)
            proof=dict(status='PASS_MAC_SMALL_FILES_ALL_SHA_BYTES',files=copied,late=late_rows,no_model_or_Tensor_or_binary_download=True);O.write_new(out/'MAC_VERIFICATION.json',O.canonical(proof)+b'\n')
            bundle=[]
            for p in sorted(out.rglob('*')):
                if p.is_file() and core not in p.parents:
                    b=p.read_bytes();bundle.append(dict(path=p.relative_to(out).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
            digest=hashlib.sha256(O.canonical([{k:v for k,v in x.items() if k!='base64'} for x in bundle])).hexdigest()
            supplement=O.call(t,O.REMOTE_SUPPLEMENT,dict(final=ARCHIVE+'.independent_terminal_supplement_'+digest,core_manifest_sha256=a55['manifest_sha256'],files=bundle),role='29',timeout=300);O.write_new(out/'SUPPLEMENT_VERIFICATION.json',O.canonical(supplement)+b'\n')
            result=dict(status='PASS_INDEPENDENT_TERMINAL_DELIVERY_AND_STOP',execution_revision=REVISION,method_variant='PACE48_FP16_EXPONENT_V1',state_scale_format='FP16_MANTISSA_PLUS_OPTIONAL_INT8_EXPONENT',source_manifest_sha256=source,permit_sha256=permit,formal_completed=a55['formal_completed'],final_status=a55['final_status'],core_archive=ARCHIVE,core_manifest_sha256=a55['manifest_sha256'],supplement=supplement,Mac_files=len(copied),scientific_dispatches=0,signals=0,GPU_kernel_calls=0,extra_independent_review_required_before_monitor_deletion=True);O.write_new(out/'RESULT.json',O.canonical(result)+b'\n');return result
        except BaseException as e:
            failure=dict(status='CPU_DELIVERY_FAILURE_NO_AUTOMATIC_RETRY',error=repr(e),traceback=traceback.format_exc(),claim_preserved=True,GPU_calls=0);O.write_new(out/'FIRST_FAILURE.json',O.canonical(failure)+b'\n');raise
    finally:lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--local-root',default=str(ROOT));p.add_argument('--deliver-if-ready',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(self_test() if a.self_test else run(a.local_root,a.deliver_if_ready),ensure_ascii=False))

if __name__=='__main__':main()
