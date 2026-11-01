"""Local once-only driver for the newly authorized pre-GPU CPU freeze. No GPU launch."""
import argparse,hashlib,json,subprocess,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def encoded(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
PROGRAM=r'''
from pathlib import Path
import sys,os,json,hashlib,time
r=Path(sys.argv[1]).resolve();payload=json.load(sys.stdin)
def enc(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2).encode()+b'\n'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,b):
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(os.path.lexists(r/n) for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','CONTROLLER_STARTED.json','OFFLINE_STARTED.json'])
cfg=read(r/'ROOT.json');auth=read(r/'HUMAN_RERUN_AUTHORIZATION.json');scope=read(r/'RESTART_SOURCE_SCOPE.json');f=payload['freeze'];source_sha=hashlib.sha256(enc(f)).hexdigest()
assert cfg['remote_root']==str(r) and auth['human_request']=='你给我重新运行' and auth['GPU0_P1_must_continue_unmodified'] and auth['GPU0_authorized'] is False
assert cfg['latest_human_rerun_authorization_sha256']==sha(r/'HUMAN_RERUN_AUTHORIZATION.json')==f['human_rerun_authorization_sha256']
assert f['count']==len(f['files'])==len({x['path'] for x in f['files']}) and f['scientific_qualification_GPU_dispatches_at_freeze']==f['formal_generations_at_freeze']==0
for row in f['files']:
 p=r/row['path'];assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(r) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
old=Path(scope['remote_prior_root']);assert sha(old/'SOURCE_FREEZE.json')==auth['prior_source_freeze_sha256']=='6694376521ff687c2cd827c154cc048f3bbf17c0c378b7ac375e03388b5c104d'
assert sha(old/'FINAL_SEAL_RECEIPT.json')==auth['prior_stop_seal_sha256']
for row in scope['prior_sources']:
 p=old/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Original science modified:'+row['path']
 if row['path'] not in scope['allowed_changes']:
  p=r/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Unexpected non-control change:'+row['path']
assert sha(r/'src/resident_rotation.py')=='4d505a51b1aeeb8235acfa6feee1caba38328d6ec654628559190b5e0a8f1f95'
sys.path[:0]=[str(r/'src'),str(r/'operations'),str(r/'scripts')]
import resident_execution_control as C,qualification_guards as G,resident_controller as controller,resident_archive as archive,engineering_gpu_evidence as E
suite=read(r/'analysis/REPAIRED_CONTROL_AND_FINISHER_TWO_FRESH_V2.json');assert len(suite['runs'])==6
for name,count in [('EXIT',30),('CONTROL',79),('FINISHER',45)]:
 rows=[x for x in suite['runs'] if x['case']==name];assert len(rows)==2 and len({(x['actual_identity']['pid'],x['actual_identity']['starttick']) for x in rows})==2
 for row in rows:
  v=row['result'];assert row['actual_wait_returncode']==0 and v.get('check_count',v.get('count'))==count and not C.same_process(row['actual_identity'])
  if name=='EXIT':assert v['controller_sha256']==sha(r/'scripts/resident_controller.py') and v['GPU_dispatches']==0 and v['real_fresh_CPU_child_processes']==12
  if name=='CONTROL':
   assert not v['real_model_loaded'] and v['GPU_units_dispatched']==0
   for p,h in v['source_pins'].items():assert sha(r/p)==h,p
  if name=='FINISHER':assert v['finisher_sha256']==sha(r/'scripts/resident_finalize_offline.py')
for i in range(2):
 v=read(r/('analysis/READONLY_MAC_REPAIR_V2_FRESH_%d.json'%i));assert v['check_count']==88 and not v['Torch_imported']
 for p,h in v['source_pins'].items():assert sha(r/p)==h,p
engineering=E.validate(r);assert engineering['total_GPU_work_dispatches']==12
assert all(not C.same_process(x['worker_identity']) and not C.same_process(x['supervisor_identity']) for x in engineering['successful_completions'])
prior=C.read(old/'SESSION_RESULT.json');assert prior['qualification_completed']==16 and prior['formal_completed']==0 and prior['own_release_all_PASS']
assert not C.same_process(prior['controller_identity'])
external=C.verify_model_and_external(r,True);assert len(external)==61 and len(G.load_native_closure(r)['files'])==293
prior_P1=C.validate_prior_p1_stop(r)
units=C.read(r/'QUALIFICATION_JOB_MANIFEST.json')['units'];jobs=C.read(r/'FORMAL_JOB_MANIFEST.json')['jobs']
assert len(units)==len({x['id'] for x in units})==58 and len(jobs)==len({x['id'] for x in jobs})==60 and {x['condition'] for x in jobs}=={'C1','S3'}
assert all(x['seed']==1 and x['max_new_tokens']==81920 and x['gpu_uuid']==C.UUIDS[x['gpu']] for x in jobs)
assert all(not os.path.lexists(r/'qualification'/x['id']) and not os.path.lexists(r/'dispatch/qualification'/(x['id']+'.json')) for x in units)
assert all(not os.path.lexists(r/'formal'/x['id']) for x in jobs)
prepared=controller.prepare_execution_directories(r);obs=C.query_gpu();assert not any(x['uuid'] in C.UUIDS.values() for x in obs['apps'])
for gpu,uuid in C.UUIDS.items():
 d=next(x for x in obs['devices'] if x['physical_index']==gpu);assert d['uuid']==uuid and '3090' in d['name'] and d['memory_free_MiB']>=23000
ram=os.statvfs('/dev/shm');data=os.statvfs('/data');assert ram.f_bavail*ram.f_frsize>=100*1024**3 and data.f_bavail*data.f_frsize>=8*1024**3
cat,exclusions=archive.catalog(r,{})
catalog=dict(status='PASS_ACTUAL_REAL_ROOT_ARCHIVE_CATALOG_NO_SYMLINKS',count=len(cat),bytes=sum(x['bytes'] for x in cat),files=cat,excluded_exact_paths=exclusions)
write(r/'operations/FINAL_PREFREEZE_REAL_ARCHIVE_CATALOG.json',enc(catalog))
assert 'torch' not in sys.modules
admission=dict(status='PASS_NEW_HUMAN_AUTHORIZED_EXIT_REPAIR_FINAL_CPU_ADMISSION_NOT_GPU_PROOF',source_freeze_sha256=source_sha,source_count=f['count'],human_rerun_authorization_sha256=sha(r/'HUMAN_RERUN_AUTHORIZATION.json'),prior295_science_immutable_fresh_SHA=True,mathematical_and_scientific_source_unchanged_except_explicit_control_scope=True,controller_sha256=sha(r/'scripts/resident_controller.py'),CPU_exit_checks_each=30,CPU_checks_each=79,CPU_finisher_checks_each=45,CPU_Mac_readonly_checks_each=88,two_fresh_target_CPU_runs_each=True,actual_external_dependency_fresh_SHA_count=61,actual_fresh_native_closure_count=293,model_stat_count=17,actual_required_parent_directory_preparation=prepared,prior_P1_stop=prior_P1,NVML=obs,free_RAM_bytes=ram.f_bavail*ram.f_frsize,free_data_bytes=data.f_bavail*data.f_frsize,prior_engineering12_inherited_only=True,new_engineering_GPU_dispatches=0,scientific_qualification_GPU_dispatches=0,formal_generations=0,prior16_receipts_reused_as_new_PASS=False,Torch_imported=False,archive_catalog_sha256=sha(r/'operations/FINAL_PREFREEZE_REAL_ARCHIVE_CATALOG.json'),monotonic=time.monotonic())
write(r/'operations/FINAL_CPU_ADMISSION.json',enc(admission));write(r/'SOURCE_FREEZE.json',enc(f))
permit=dict(task_id=C.TASK,execution_attempt=r.name,ALLOW_GPU='QUALIFICATION_ONLY',source_freeze_sha256=source_sha,devices=list(C.UUIDS.values()),human_rerun_authorization_sha256=sha(r/'HUMAN_RERUN_AUTHORIZATION.json'),human_authorization_sha256=sha(r/'HUMAN_CONTINUATION_AUTHORIZATION.json'),repair_authorization_sha256=sha(r/'HUMAN_REPAIR_AUTHORIZATION.json'),native_exit_repair_authorization_sha256=sha(r/'HUMAN_NATIVE_EXIT_REPAIR_AUTHORIZATION.json'),prior_p1_stop_admission_sha256=sha(r/'PRIOR_P1_USER_STOP_ADMISSION.json'),runtime_native_closure_sha256=sha(r/'analysis/RUNTIME_NATIVE_CLOSURE.json'),qualification_plan_sha256=sha(r/'QUALIFICATION_JOB_MANIFEST.json'),execution_plan_sha256=sha(r/'GPU_EXECUTION_PLAN.json'),final_CPU_admission_sha256=sha(r/'operations/FINAL_CPU_ADMISSION.json'),engineering_gpu_evidence_binding_sha256=sha(r/'operations/ENGINEERING_GPU_EVIDENCE_BINDING.json'),engineering_probe_completion_sha256=sha(engineering['active_completion']),formal_permission=False,conditional_formal_separate_permit_required=True,GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,no_progress_watchdog_seconds=1800,own_cleanup_observation_seconds=120,no_GPU_retry_after_STARTED=True)
write(r/'GPU_PERMIT.json',enc(permit));assert C.verify_sources(r)==source_sha
print(json.dumps(dict(status=admission['status'],source_sha256=source_sha,source_count=f['count'],permit_sha256=sha(r/'GPU_PERMIT.json'),admission_sha256=sha(r/'operations/FINAL_CPU_ADMISSION.json'),controller_sha256=admission['controller_sha256'],scientific_GPU_dispatches=0)))
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();assert a.execute
 assert not (ROOT/'SOURCE_FREEZE.json').exists() and not (ROOT/'GPU_PERMIT.json').exists()
 cfg=json.loads((ROOT/'ROOT.json').read_text());scope=json.loads((ROOT/'RESTART_SOURCE_SCOPE.json').read_text());rows=[]
 for p in sorted(ROOT.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():rows.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 rows+=scope['remote_only25'];assert len(rows)==len({x['path'] for x in rows})
 freeze=dict(task_id=cfg['task_id'],execution_attempt=Path(cfg['remote_root']).name,status='ONCE_NEW_HUMAN_AUTHORIZED_EXIT_REPAIR_SOURCE_FROZEN_QUALIFICATION_ONLY',human_rerun_authorization_sha256=sha(ROOT/'HUMAN_RERUN_AUTHORIZATION.json'),prior_source_freeze_sha256=scope['prior_source_freeze_sha256'],count=len(rows),files=sorted(rows,key=lambda x:x['path']),scientific_qualification_GPU_dispatches_at_freeze=0,formal_generations_at_freeze=0,new_engineering_GPU_dispatches_at_freeze=0,inherited_prior_engineering_dispatches=12,prior16_qualifications_count_as_new_PASS=False)
 cmd=['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',cfg['remote_python'],'-B','-c',PROGRAM,cfg['remote_root']])]
 p=subprocess.run(cmd,input=json.dumps(dict(freeze=freeze)),capture_output=True,text=True,timeout=240)
 (ROOT/'operations/FREEZE_NEW_EXIT_REPAIR_TRANSPORT.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
 assert p.returncode==0,p.stderr
 print(p.stdout)
 for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','operations/FINAL_CPU_ADMISSION.json','operations/FINAL_PREFREEZE_REAL_ARCHIVE_CATALOG.json']:
  q=subprocess.run(['ssh','-S',cfg['ssh_control_path'],'-o','BatchMode=yes','zypan@202.38.247.55',shlex.join(['cat',cfg['remote_root']+'/'+n])],capture_output=True);assert q.returncode==0,q.stderr
  with (ROOT/n).open('xb') as f:f.write(q.stdout)
 assert sha(ROOT/'SOURCE_FREEZE.json')==hashlib.sha256(encoded(freeze)).hexdigest()
if __name__=='__main__':main()
