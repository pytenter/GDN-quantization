"""One bounded fresh read-only PACE provenance audit; no GPU/API/model/scoring."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
import pace_continuation as K
import pace_control as C
TRANSPORT_SHA='db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'
GRANT_SHA='f9cfbada775455554bf3d33a4d22522e48ad0a75fab03ed2caa3611b72232ef3'
TRITON_PATH=ROOT.parent/'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_CUPTI_REPAIR_20261025/analysis/TRITON_RUNTIME_FRESH_SHA.json'
TRITON_SHA='bc695cc73e549272d9231cfe1f6b833dbf3519c9964e3820746acfee92dcafe5'
OLD_LOCAL=ROOT.parent/'QWEN_GDN_PACE48_AIME26_SEED1_60_V1'
PRIMARY=OLD_LOCAL/'independent_delivery/primary_archive_small_files'
SUPPLEMENT=K.OLD_ARCHIVE+'.independent_terminal_supplement_529389b2de188a0662c86b46a7185c56ecab478fa393146f95a38fbad8168b26'
SUPPLEMENT_MANIFEST_SHA='da6e44754828439e1a0a0798d14ea17d6715ef9176c1003f5658512546fa39b9'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(value,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')

REMOTE=r'''
import hashlib,json,os,signal,time
from pathlib import Path
signal.alarm(590)
START=time.time()
def req(x,msg):
 if not x:raise RuntimeError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def safe(root,name):
 p=Path(name);req(not p.is_absolute() and '..'not in p.parts and str(p)==name and name!='.','Unsafe scientific path')
 q=Path(root)/p;c=q
 while c!=Path(root):req(not c.is_symlink(),'Scientific symlink forbidden');c=c.parent
 return q
def raw(p):
 p=Path(p);a=p.stat();b=p.read_bytes();z=p.stat();req((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Mutated metadata while read:'+str(p));return b
def read(p):return json.loads(raw(p))
def digest(p):
 p=Path(p);a=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 z=p.stat();req((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Changed while hash:'+str(p))
 return h.hexdigest(),z
def check(root,rows,external=False):
 records=[];seen=set()
 for x in rows:
  name=x['path'];req(name not in seen,'Duplicate SHA inventory:'+name);seen.add(name)
  p=Path(name)if external else safe(root,name)
  h,s=digest(p);req(s.st_size==x['bytes'] and h==x['sha256'],'SHA/bytes differ:'+name)
  if external:
   for key,actual in [('inode',s.st_ino),('mtime_ns',s.st_mtime_ns),('device',s.st_dev)]:
    if key in x:req(actual==x[key],'Original external stat changed:'+name+':'+key)
  records.append(dict(path=name,bytes=s.st_size,sha256=h,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns))
 return dict(status='PASS_FRESH_ALL_SHA_BYTES_AND_STABLE_STATS',count=len(records),bytes=sum(x['bytes']for x in records),
             actual_rows_sha256=hashlib.sha256(canonical(records)).hexdigest(),actual_rows=records)
r=Path(P['root']);req(str(r)in{P['old55'],P['old29']},'Only two fixed old roots allowed')
checks={};controls={}
for name,x in P['controls'].items():
 v=check(r,[dict(path=name,**x)]);controls[name]=v['actual_rows'][0]
source=read(r/'source_manifest.json');req(source['task_id']==P['task'] and source['count']==len(source['files'])==133,'Original source133 schema differs')
checks['source133']=check(r,source['files'])
m=read(r/'DELIVERY_MANIFEST.json');req(m['task_id']==P['task'] and m['count']==len(m['files'])==317 and m['total_bytes']==18360743 and m['source_manifest_sha256']==P['source_sha'] and m['permit_sha256']==P['permit_sha'],'Original primary manifest differs')
checks['primary317']=check(r,m['files'])
s=read(r/'SCIENCE_FREEZE.json');req(s['task_id']==P['task'],'Science task differs');checks['science_freeze_files']=check(r,s['files'])
req(digest(r/'calibration/head_mask.json')[0]==P['mask_sha'],'Original once-mask differs')
metadata=[]
for stage,unit in [('math_gpu_gates','GATES_ABC'),('calibration','CALIBRATION_ONCE'),('gate_d','FULL_MODEL_D')]+[('formal',u)for u in P['reused']]:
 target=(r/'execution'/unit if stage!='formal'else r/('R2_PACE48_DP8'if '_R2_'in unit else'R3_PACE48_DP1')/'trajectories'/unit)
 dispatch=read(target/'DISPATCH.json');identity=read(target/'ACTUAL_CHILD_IDENTITY.json');started=read(target/'STARTED.json');interval=read(target/'RESOURCE_INTERVAL.json');wrapper=read(target/'RESULT.json');driver=read(target/'DRIVER_IDENTITY.json')
 req(dispatch['execution_root']==P['old55'] and dispatch['gpu']==0 and dispatch['gpu_uuid']==P['uuid'] and dispatch['source_manifest_sha256']==P['source_sha'] and dispatch['permit_sha256']==P['permit_sha'] and dispatch['stage']==stage and dispatch['unit']==unit,'Original dispatch identity differs:'+unit)
 req(identity==started['identity']==interval['identity']==wrapper['identity'] and identity['cmdline']==[P['python55'],P['old55']+'/scripts/pace_worker.py','--root',P['old55'],'--stage',stage,'--unit',unit] and type(identity['starttick'])is int and identity['starttick']>0,'Actual argv/starttick differs:'+unit)
 req(started['dispatch_sha256']==digest(target/'DISPATCH.json')[0] and interval['returncode']==0 and interval['released']is True and interval['accounting_final']is True and interval['failure']is None,'Actual legal exit/release differs:'+unit)
 req(wrapper['status']=='PASS' and wrapper['gpu_uuid']==P['uuid'] and wrapper['source_manifest_sha256']==P['source_sha'] and wrapper['permit_sha256']==P['permit_sha'],'Original RESULT binding differs:'+unit)
 payload=dict(wrapper);payloadsha=payload.pop('payload_sha256');req(hashlib.sha256(canonical(payload)).hexdigest()==payloadsha,'Payload canonical SHA differs:'+unit)
 req(driver['status']=='PASS' and driver['actual_uuid']==P['uuid'] and driver['CUDA_VISIBLE_DEVICES']==P['uuid'] and driver['visible_device_count']==1 and driver['logical_ordinal']==0,'Original driver UUID differs:'+unit)
 metadata.append(dict(stage=stage,unit=unit,identity=identity,actual_returncode=0,released=True,result_sha256=digest(target/'RESULT.json')[0],interval_sha256=digest(target/'RESOURCE_INTERVAL.json')[0],payload_sha256=payloadsha))
failed=r/'R3_PACE48_DP1/trajectories'/P['failed'];failure=read(failed/'RESOURCE_INTERVAL.json')
req(failure['returncode']==1 and failure['released']is True and not(failed/'RESULT.json').exists(),'Old failed partial zero-credit identity changed')
session=read(r/'SESSION_RESULT.json');req(session['status']=='FAIL_CLOSED_STOP' and session['formal_completed']==3 and session['own_workers_exit_and_NVML_release']is True,'Original trueSTOP differs')
if P['host']=='55':
 for filename in ['EXTERNAL_NATIVE_BINDINGS.json','EXTERNAL_PYTHON_BINDINGS.json','MODEL_TOKENIZER_FRESH_SHA.json']:
  bind=read(r/'analysis'/filename);checks[filename]=check(None,bind['files'],True)
 checks['shared_Triton322_CPU_inventory']=check(None,P['triton_rows'],True)
 checks['late_exit_receipts']=check(r,P['late55_rows'])
else:
 sup=Path(P['supplement']);req(digest(sup/'SUPPLEMENT_MANIFEST.json')[0]==P['supplement_manifest_sha'],'Original sibling supplement manifest differs')
 sm=read(sup/'SUPPLEMENT_MANIFEST.json');req(sm['count']==len(sm['files'])==9 and sm['total_bytes']==58435 and sm['core_archive']==P['old29'] and sm['core_manifest_sha256']==P['controls']['DELIVERY_MANIFEST.json']['sha256'],'Original sibling supplement binding differs')
 checks['late9_supplement']=check(sup,sm['files'])
req(digest(r/'source_manifest.json')[0]==P['source_sha'] and digest(r/'GPU_PERMIT.json')[0]==P['permit_sha'] and digest(r/'SCIENCE_FREEZE.json')[0]==P['science_sha'],'Original top controls changed during audit')
print(json.dumps(dict(status='PASS',host=P['host'],root=str(r),UTC_unix=time.time(),elapsed_seconds=time.time()-START,
 source_count=133,source_sha256=P['source_sha'],permit_sha256=P['permit_sha'],science_freeze_sha256=P['science_sha'],head_mask_sha256=P['mask_sha'],formal_count=3,
 full_model17_SHA_PASS=P['host']=='55',no_absolute_model_or_GPU_requirement_on29=True,
 checks=checks,controls=controls,reused_units=metadata,old_failed_partial=dict(unit=P['failed'],actual_returncode=1,released=True,completion_credit=0),
 imports=['stdlib_only'],CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_queries=0,GPU_contexts=0,kernel_calls=0,model_loads=0,scoring_calls=0,signals=0,remote_mutations=0),sort_keys=True))
'''

def main():
    result=ROOT/K.FRESH_REVIEW
    if result.exists():raise RuntimeError('Review exists: never overwrite or silently re-audit')
    if sha(ROOT/'operations/remote.py')!=TRANSPORT_SHA or sha(ROOT/'operations/HUMAN_PACE_SCALE_REPAIR_AUTHORIZATION.json')!=GRANT_SHA:
        raise RuntimeError('Pinned transport/grant mismatch')
    if sha(TRITON_PATH)!=TRITON_SHA:raise RuntimeError('Shared read-only Triton inventory changed')
    tr=json.loads(TRITON_PATH.read_text());trrows=tr['files']+tr['distribution_metadata']+tr['host_c_compiler']
    if len(trrows)!=322:raise RuntimeError('Exact shared311+6+5 Triton/compiler rows required')
    eq=K.verify_equivalence(ROOT,PRIMARY)
    spec=importlib.util.spec_from_file_location('pace_fixed_cpu_transport',ROOT/'operations/remote.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    late=[]
    for p in (OLD_LOCAL/'independent_delivery/late_exit_receipts').iterdir():
        if p.is_file():late.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    common=dict(task=C.TASK,old55=K.OLD_REMOTE,old29=K.OLD_ARCHIVE,python55='/data/ydai/miniconda3/envs/bitdecode/bin/python',
        source_sha=K.OLD_SOURCE_SHA,permit_sha=K.OLD_PERMIT_SHA,science_sha=K.OLD_SCIENCE_SHA,mask_sha=K.MASK_SHA,
        uuid=C.UUID,controls=K.ORIGIN_CONTROL_PINS,reused=list(K.REUSED_IDS),failed=K.FAILED_ID,
        supplement=SUPPLEMENT,supplement_manifest_sha=SUPPLEMENT_MANIFEST_SHA,triton_rows=trrows,late55_rows=late)
    def host(role):
        p=dict(common,host=role,root=K.OLD_REMOTE if role=='55'else K.OLD_ARCHIVE)
        code='P='+repr(p)+'\n'+REMOTE
        value=m.run(code,role=role,python=common['python55']if role=='55'else'/usr/bin/python3',timeout=600)
        file=ROOT/'analysis'/('PACE_REUSE3_FRESH_HOST'+role+'.json');save(file,value)
        return role,value,dict(path=file.relative_to(ROOT).as_posix(),bytes=file.stat().st_size,sha256=sha(file))
    hosts={};files=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        futures={pool.submit(host,role):role for role in ('55','29')}
        for f in concurrent.futures.as_completed(futures):
            role=futures[f]
            try:
                role,value,metadata=f.result();hosts[role]=value;files.append(metadata)
                print(json.dumps(dict(host=role,status=value['status'],elapsed_seconds=value['elapsed_seconds'])),flush=True)
            except BaseException as e:
                errors.append(dict(host=role,error=repr(e),traceback=traceback.format_exc()))
    if errors:
        p=ROOT/'analysis'/('PACE_REUSE3_FRESH_AUDIT_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
        save(p,dict(status='FAIL_PREFREEZE_READONLY_AUDIT',errors=errors,successful_hosts=list(hosts),GPU_calls=0,remote_mutations=0))
        raise RuntimeError('True CPU provenance audit failure preserved: '+str(p))
    value=dict(status='PASS_FRESH_DUAL_HOST_PACE_REUSE3_SOURCE_SCIENCE_MODEL',UTC=datetime.now(timezone.utc).isoformat(),
        canonical_task_id=C.TASK,execution_revision=K.REVISION,human_authorization_sha256=GRANT_SHA,
        source_sha256=K.OLD_SOURCE_SHA,permit_sha256=K.OLD_PERMIT_SHA,science_freeze_sha256=K.OLD_SCIENCE_SHA,
        head_mask_sha256=K.MASK_SHA,formal_count=3,hosts=hosts,raw_metadata_files=files,
        accepted_domain_equivalence=dict(status='PASS',**{k:v for k,v in eq.items()if k!='status'}),
        shared_environment_inventory=dict(path=str(TRITON_PATH),sha256=TRITON_SHA,files311_dist6_compiler5=322,
             scope='Shared existing environment byte inventory only; no WScale execution, process call, qualification or science credit'),
        GPU_calls=0,NVML_calls=0,model_loads=0,scoring_calls=0,remote_mutations=0,signals=0)
    save(result,value)
    print(json.dumps(dict(status=value['status'],path=str(result),sha256=sha(result),raw_metadata_files=files)),flush=True)

if __name__=='__main__':main()
