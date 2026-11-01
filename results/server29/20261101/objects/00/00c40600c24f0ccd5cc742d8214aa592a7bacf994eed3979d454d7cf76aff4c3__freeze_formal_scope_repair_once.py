"""Single CPU admission/freeze, no GPU work. New explicit human authorization."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n'
PROGRAM=r'''
import ast,pathlib,json,sys,os,time,hashlib
r=pathlib.Path(sys.argv[1]).resolve();f=json.load(sys.stdin)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(os.path.lexists(r/n)for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','CONTROLLER_STARTED.json','OFFLINE_STARTED.json'])
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n'
def write(p,v):
 with pathlib.Path(p).open('xb')as s:s.write(enc(v));s.flush();os.fsync(s.fileno())
cfg=read(r/'ROOT.json');auth=read(r/'HUMAN_FORMAL_SCOPE_REPAIR_AUTHORIZATION.json');scope=read(r/'FORMAL_SCOPE_REPAIR_SOURCE_SCOPE.json');old=pathlib.Path(scope['remote_prior_root'])
assert cfg['remote_root']==str(r)and cfg['remote_execution_slug']==r.name
assert auth['human_request']=='帮我修复，然后开展实验'and auth['reused_gate_count']==58 and auth['remaining_gate_count']==0 and auth['GPU0_P1_untouched']and not auth['GPU0_authorized']
assert f['human_formal_scope_repair_authorization_sha256']==sha(r/'HUMAN_FORMAL_SCOPE_REPAIR_AUTHORIZATION.json')
assert f['count']==len(f['files'])==len({x['path']for x in f['files']})
for x in f['files']:
 n=pathlib.Path(x['path']);p=r/n
 assert not n.is_absolute()and '..'not in n.parts and p.resolve().is_relative_to(r)and p.is_file()and not p.is_symlink()
 assert p.stat().st_size==x['bytes']and sha(p)==x['sha256'],x['path']
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
actual={str(p.relative_to(r))for p in r.rglob('*')if p.is_file()}
assert actual=={x['path']for x in f['files']},'Source inventory omits an actual input'
assert sha(old/'SOURCE_FREEZE.json')==scope['prior_source_sha256']=='5de5da05403af067c389ce7da1c12a3e5fa4afcd8600cd45313cd5a01a023b11'
assert sha(old/'FINAL_SEAL_RECEIPT.json')==scope['prior_seal_sha256']
changed=[]
for x in scope['prior_sources']:
 p=old/x['path'];assert p.stat().st_size==x['bytes']and sha(p)==x['sha256'],'Parent changed:'+x['path']
 if sha(r/x['path'])!=x['sha256']:
  assert x['path']in scope['allowed_changed_existing'],'Unexpected scientific change:'+x['path']
  changed.append(dict(path=x['path'],old_sha256=x['sha256'],new_sha256=sha(r/x['path'])))
for n,d in [('src/resident_rotation.py','4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95'),('scripts/qualify_gpu.py','a41b84fbecfe262b37211ba27804329955f23577ef46babade06a115ec88fff2'),('src/qualification_guards.py','39a691a77f0ba09c6052e9f1674550677df99bd8eba67a8df28227cd099265cf'),('src/resident_execution_control.py','3fd0b6d0d843ed1adf9b27281021bb5b0f4fcb0c37b8f9eb2feedee1800d8e42')]:assert sha(r/n)==d
assert sha(r/'scripts/resident_formal_worker.py')==sha(old/'scripts/resident_formal_worker.py')and sha(r/'scripts/resident_finalize_offline.py')==sha(old/'scripts/resident_finalize_offline.py')
oldtext=(old/'src/profiler_detach_lifecycle.py').read_text();newtext=(r/'src/profiler_detach_lifecycle.py').read_text();needle="    manifest = json.loads((root / 'QUALIFICATION_JOB_MANIFEST.json').read_text())"
assert oldtext[oldtext.index(needle):]==newtext[newtext.index(needle):]
before=ast.parse(oldtext);after=ast.parse(newtext)
for name in ['_detach_exact_pinned_handle','detach_completed_profiler','detach_qualification_profile_if_completed']:
 a=next(x for x in before.body if isinstance(x,ast.FunctionDef)and x.name==name);b=next(x for x in after.body if isinstance(x,ast.FunctionDef)and x.name==name);assert ast.dump(a)==ast.dump(b)
# Only the event count became dynamic; dispatch, process ownership, watchdog and
# all scientific worker calls are otherwise exactly the previous controller.
assert (r/'scripts/resident_controller.py').read_text()==(old/'scripts/resident_controller.py').read_text().replace('count=48,',"count=len(self.completed['qualification']),")
suite=read(r/'analysis/FORMAL_SCOPE_TWO_FRESH_TARGET_CPU_SUITE.json');assert len(suite['runs'])==12
for name,count in [('SCOPE',83),('DETACH',21),('REUSE',19),('CONTROL',79),('FINISHER',45),('REUSED_FINISHER',5)]:
 rows=[x for x in suite['runs']if x['case']==name];assert len(rows)==2
 for x in rows:
  result=x['result'];assert x['actual_wait_returncode']==0 and result.get('check_count',result.get('count'))==count
  for p,digest in result.get('source_pins',{}).items():assert sha(r/p)==digest,p
  if 'finisher_sha256'in result:assert result['finisher_sha256']==sha(r/'scripts/resident_finalize_offline.py')
for i in range(2):
 v=read(r/('analysis/FORMAL_SCOPE_READONLY_FINAL_MAC_R'+str(i)+'.json'));assert v['check_count']==88 and not v['Torch_imported']
 for p,d in v['source_pins'].items():assert sha(r/p)==d,p
ind=read(r/'analysis/FORMAL_SCOPE_INDEPENDENT_READONLY58_CPU.json');assert ind['check_count']==4 and ind['tool_sha256']==sha(r/'operations/download_verified_delivery.py')and ind['audit']['reused_count']==58
dual=read(r/'analysis/FORMAL_SCOPE_ORIGIN58_DUAL_HOST_FRESH_REVIEW.json');assert {x['role']for x in dual['reports']}=={'55','29'}
assert all(sum(o['units']for o in x['origins'])==58 and sum(o['original_profile_trace_fresh_SHA']for o in x['origins'])==6 for x in dual['reports'])
sys.path[:0]=[str(r/'src'),str(r/'scripts'),str(r/'operations')]
import resident_execution_control as C,qualification_guards as G,qualification_reuse as Q,resident_controller as controller,resident_archive as archive,engineering_gpu_evidence as E,profiler_engineering_evidence as P
reused=Q.seed(C,r);assert len(reused)==58
units=read(r/'QUALIFICATION_JOB_MANIFEST.json')['units'];jobs=read(r/'FORMAL_JOB_MANIFEST.json')['jobs'];assert len(units)==58 and set(reused)=={u['id']for u in units}
assert jobs==read(old/'FORMAL_JOB_MANIFEST.json')['jobs']and len(jobs)==60 and {j['condition']for j in jobs}=={'C1','S3'}
results={u['id']:Q.validate(C,r,u)for u in units};performance=C.performance_audit(r,units,results);assert performance['status']=='PASS_OBJECTIVE_EFFICIENT_RESIDENT_EXECUTION_AND_ACTUAL_COST_DISCLOSED'
assert all(not os.path.lexists(r/'qualification'/u['id'])for u in units)and all(not os.path.lexists(r/'formal'/j['id'])for j in jobs)
parent=read(old/'SESSION_RESULT.json');assert parent['qualification_completed']==58 and parent['formal_completed']==0 and parent['own_release_all_PASS']and not C.same_process(parent['controller_identity'])
assert read(old/'FINISHER_RESULT.json')['status']=='PASS_REAL_TERMINAL_SCIENCE_ARCHIVED_ON29'and not C.same_process(read(old/'FINISHER_STARTED.json')['identity'])
for p in (old/'formal').glob('*/RESOURCE_INTERVAL.json'):
 interval=read(p);assert interval['release']=='PASS'and not C.same_process(interval['identity'])
assert E.validate(r)['total_GPU_work_dispatches']==12 and P.validate(r,C)==read(r/'analysis/PROFILER_ENGINEERING_EVIDENCE.json')
external=C.verify_model_and_external(r,fresh_external=True);assert len(external)==61 and len(G.load_native_closure(r)['files'])==293
prior_p1=C.validate_prior_p1_stop(r);prepared=controller.prepare_execution_directories(r)
obs=C.query_gpu();assert not any(x['uuid']in C.UUIDS.values()for x in obs['apps'])
for gpu,uuid in C.UUIDS.items():
 d=next(x for x in obs['devices']if x['physical_index']==gpu);assert d['uuid']==uuid and '3090'in d['name']and d['memory_free_MiB']>=23000
ram=os.statvfs('/dev/shm');data=os.statvfs('/data');assert ram.f_bavail*ram.f_frsize>=100*1024**3 and data.f_bavail*data.f_frsize>=8*1024**3
cat,ex=archive.catalog(r,{});assert 'torch'not in sys.modules
source=hashlib.sha256(enc(f)).hexdigest()
admission=dict(status='PASS_HUMAN_FORMAL_SCOPE_REPAIR_REUSE58_FINAL_CPU_ADMISSION_NO_NEW_QUAL_GPU',source_freeze_sha256=source,source_count=f['count'],qualification_reused=58,qualification_remaining=0,current_qualification_GPU_dispatches=0,current_engineering_GPU_dispatches=0,inherited_engineering_GPU_dispatches=27,formal_generations=0,controller_sha256=sha(r/'scripts/resident_controller.py'),lifecycle_sha256=sha(r/'src/profiler_detach_lifecycle.py'),human_formal_scope_repair_authorization_sha256=sha(r/'HUMAN_FORMAL_SCOPE_REPAIR_AUTHORIZATION.json'),qualification_reuse_admission_sha256=sha(r/'QUALIFICATION_REUSE_ADMISSION.json'),CPU_checks_each=dict(formal_scope=83,detach=21,reuse=19,control=79,finisher=45,reused_finisher=5,Mac_readonly=88),independent_actual_readonly_reuse_checks=4,actual_external_fresh_SHA_count=61,native_ELF_fresh_SHA_stat_count=293,model_stat_count=17,old_science_sources_freshSHA_immutable=True,only_exact_allowed_support_changes=changed,qualification_suffix_byte_identical=True,scientific_worker_and_finalizer_unchanged=True,actual_directory_preparation=prepared,NVML=obs,prior_P1_stop=prior_p1,free_RAM_bytes=ram.f_bavail*ram.f_frsize,free_data_bytes=data.f_bavail*data.f_frsize,real_archive_catalog_count=len(cat),real_archive_catalog_no_symlinks=True,all_old_failure_terminations_unchanged=True,Torch_imported=False,monotonic=time.monotonic())
write(r/'operations/FINAL_CPU_ADMISSION.json',admission);write(r/'SOURCE_FREEZE.json',f)
permit=dict(task_id=C.TASK,execution_attempt=r.name,ALLOW_GPU='QUALIFICATION_ONLY',source_freeze_sha256=source,devices=list(C.UUIDS.values()),human_authorization_sha256=sha(r/'HUMAN_CONTINUATION_AUTHORIZATION.json'),human_formal_scope_repair_authorization_sha256=sha(r/'HUMAN_FORMAL_SCOPE_REPAIR_AUTHORIZATION.json'),human_qualification_reuse_authorization_sha256=sha(r/'HUMAN_QUALIFICATION_REUSE_AUTHORIZATION.json'),qualification_reuse_admission_sha256=sha(r/'QUALIFICATION_REUSE_ADMISSION.json'),prior_p1_stop_admission_sha256=sha(r/'PRIOR_P1_USER_STOP_ADMISSION.json'),qualification_plan_sha256=sha(r/'QUALIFICATION_JOB_MANIFEST.json'),execution_plan_sha256=sha(r/'GPU_EXECUTION_PLAN.json'),runtime_native_closure_sha256=sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),final_CPU_admission_sha256=sha(r/'operations/FINAL_CPU_ADMISSION.json'),formal_permission=False,qualification_reused=58,qualification_remaining=0,conditional_formal_separate_permit_required=True,GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,no_progress_watchdog_seconds=1800,own_cleanup_observation_seconds=120,no_GPU_retry_after_STARTED=True)
write(r/'GPU_PERMIT.json',permit);assert C.verify_sources(r)==source
print(json.dumps(dict(status=admission['status'],source_count=f['count'],source_sha256=source,permit_sha256=sha(r/'GPU_PERMIT.json'),admission_sha256=sha(r/'operations/FINAL_CPU_ADMISSION.json'))))
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();assert a.execute
 cfg=json.loads((R/'ROOT.json').read_text());assert not(R/'SOURCE_FREEZE.json').exists()
 ssh=['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55']
 inventory="import pathlib,json,hashlib;r=pathlib.Path("+repr(cfg['remote_root'])+");print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(r.rglob('*'))if p.is_file()]))"
 x=subprocess.run(ssh+[shlex.join([cfg['remote_python'],'-B','-c',inventory])],capture_output=True,text=True,check=True)
 files=json.loads(x.stdout)
 for p in R.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():
   row=next(x for x in files if x['path']==str(p.relative_to(R)));assert row['bytes']==p.stat().st_size and row['sha256']==sha(p),row['path']
 f=dict(task_id=cfg['task_id'],execution_attempt=cfg['remote_execution_slug'],status='ONCE_FROZEN_FORMAL_SCOPE_REPAIR_REUSE58_NO_QUALIFICATION_REDISPATCH',human_formal_scope_repair_authorization_sha256=sha(R/'HUMAN_FORMAL_SCOPE_REPAIR_AUTHORIZATION.json'),prior_source_freeze_sha256='5de5da05403af067c389ce7da1c12a3e5fa4afcd8600cd45313cd5a01a023b11',count=len(files),files=files,qualification_reused=58,qualification_current_remaining=0,current_GPU_dispatches_at_freeze=0,formal_generations_at_freeze=0,current_engineering_GPU_dispatches_at_freeze=0,inherited_engineering_dispatches=27)
 x=subprocess.run(ssh+[shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',cfg['remote_python'],'-B','-c',PROGRAM,cfg['remote_root']])],input=json.dumps(f),capture_output=True,text=True,timeout=300)
 with (R/'operations/FORMAL_SCOPE_FREEZE_TRANSPORT.json').open('x')as s:json.dump(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),s,indent=2)
 assert x.returncode==0,x.stderr;print(x.stdout)
 for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','operations/FINAL_CPU_ADMISSION.json']:
  b=subprocess.run(ssh+[shlex.join(['cat',cfg['remote_root']+'/'+n])],capture_output=True,check=True).stdout
  with (R/n).open('xb')as s:s.write(b)
if __name__=='__main__':main()
