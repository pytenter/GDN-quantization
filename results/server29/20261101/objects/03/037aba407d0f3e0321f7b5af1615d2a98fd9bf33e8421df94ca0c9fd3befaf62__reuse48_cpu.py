"""Meaningful fixed-origin reuse negatives and actual controller CPU fixtures.

GPU/device/model/runtime/remote output endpoints are explicit mocks. Real48
historical bytes are read only; mocks are never scientific GPU or AIME credit.
"""
import argparse,copy,hashlib,importlib.util,json,os,shutil,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
_ORIGIN_NAME='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_BYTEVIEW_REPAIR_20261025'
O=Path('/tmp')/_ORIGIN_NAME if sys.platform.startswith('linux') and (Path('/tmp')/_ORIGIN_NAME/'SOURCE_FREEZE.json').is_file() else R.parent/_ORIGIN_NAME
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(b,m):
 if not b:raise AssertionError(m)
def load(name,p):s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def link(source,target):
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  try:os.link(source,target)
  except OSError as exc:
   if exc.errno!=18:raise
   shutil.copyfile(source,target)  # CPU fixture endpoint may be a different filesystem.

def origin_fixture(base):
 origin=base/'origin_readonly_byteview';origin.mkdir();mappingfile=O/'DELIVERY_FROM55/LOCAL_PATH_MAP.json';mapping={x['path']:O/'DELIVERY_FROM55/server55'/x['local_path'] for x in read(mappingfile)['server55']} if mappingfile.is_file() else {};source=read(O/'SOURCE_FREEZE.json');audit=read(R/'provenance/REUSED_QUALIFICATION48/CPU_ORIGIN48_READONLY_AUDIT.json')
 names={x['path'] for x in source['files']}|set(audit['origin_controls'])|{'analysis/TRITON_RUNTIME_FRESH_SHA.json'}|{x['path'] for u in audit['units'] for x in u['files']}
 for name in names:
  p=O/name
  if not p.is_file():p=mapping[name]
  link(p,origin/name)
 return origin

def reuse_checks(base):
 sys.path.insert(0,str(R/'src'));import qualification_reuse as Q;import execution_control as C
 f=base/'reuse_input';f.mkdir();wrapper=read(R/'QUALIFICATION_REUSE_ADMISSION.json')
 names=[Q.AUDIT_PATH,Q.REVIEW_PATH,'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json','GPU_EXECUTION_PLAN.json']+[x['path'] for x in wrapper['source_math_unchanged']]
 for name in names:link(R/name,f/name)
 fresh=dict(status='PASS_FRESH_DUAL_HOST_REUSE48_BYTEVIEW_SOURCE_RAW',CPU_MOCK_ONLY=True,GPU_calls=0,not_scientific_admission=True,hosts={h:dict(status='PASS',source_count=334,unit_count=48,source_sha256=Q.ORIGIN_SOURCE_SHA256,proof_files_fresh_SHAbytes=555,science_count=5397,science_bytes=1731153792,CPU_MOCK_ONLY=True) for h in ('55','29')})
 dump(f/Q.FRESH_PATH,fresh);wrapper['dual_host_fresh_review_sha256']=sha(f/Q.FRESH_PATH);dump(f/'QUALIFICATION_REUSE_ADMISSION.json',wrapper)
 actualorigin=origin_fixture(base);realpath=Q.Path
 def map_path(value):return actualorigin if str(value)==Q.ORIGIN_ROOT else Path(value)
 Q.Path=map_path
 try:
  plan=read(f/'GPU_EXECUTION_PLAN.json');positive=Q.admit_reused48(f,plan);require(positive['count']==48 and positive['proof_files_fresh_SHAbytes']==555,'Real48 origin proof positive failed');rows=[]
  def negative(name,action,restore):
   try:
    action()
    try:Q.admit_reused48(f,plan)
    except (RuntimeError,KeyError,FileNotFoundError) as exc:rows.append(dict(case=name,status='PASS_REJECTED',error=str(exc)))
    else:raise AssertionError('Negative accepted: '+name)
   finally:restore()
  for key,value in [('status','PASS_FOREIGN_GENERIC'),('qualification_reused',49),('qualification_new',3),('formal_reused',1),('origin_audit_path','/tmp/foreign.json'),('origin_audit_sha256','0'*64),('origin_final_review_sha256','0'*64),('human_authorization_sha256','0'*64),('old49_failed_exit_not_reused',False),('no_current_legacy_started_result_interval',False),('dual_host_fresh_review_sha256','0'*64),('source_math_unchanged',[])]:
   mutated=copy.deepcopy(wrapper);mutated[key]=value
   negative('admission_'+key,lambda x=mutated:dump(f/'QUALIFICATION_REUSE_ADMISSION.json',x),lambda:dump(f/'QUALIFICATION_REUSE_ADMISSION.json',wrapper))
  for host,key,value in [('55','unit_count',49),('29','source_sha256','f'*64),('29','proof_files_fresh_SHAbytes',554),('55','status','UNRESOLVED')]:
   mutated=copy.deepcopy(fresh);mutated['hosts'][host][key]=value
   def do(x=mutated):dump(f/Q.FRESH_PATH,x);w=copy.deepcopy(wrapper);w['dual_host_fresh_review_sha256']=sha(f/Q.FRESH_PATH);dump(f/'QUALIFICATION_REUSE_ADMISSION.json',w)
   def restore():dump(f/Q.FRESH_PATH,fresh);dump(f/'QUALIFICATION_REUSE_ADMISSION.json',wrapper)
   negative('fresh_'+host+'_'+key,do,restore)
  legacy=f/'execution'/plan['qualification_units'][0]['id'];negative('old_RESULT_created_in_current',lambda:legacy.mkdir(parents=True),lambda:legacy.rmdir())
  badplan=copy.deepcopy(plan);badplan['qualification_units'][-1]['gpu']=2
  try:Q.remaining_units(badplan)
  except RuntimeError:rows.append(dict(case='rewritten_remaining_GPU',status='PASS_REJECTED'))
  else:raise AssertionError('Rewritten remaining accepted')
  for unit in Q.reused_units(plan):
   try:Q.require_current_dispatch(f,unit)
   except RuntimeError:pass
   else:raise AssertionError('Old48 dispatch accepted:'+unit['id'])
  rows.append(dict(case='all48_dispatch_rejected_before_STARTED',status='PASS_REJECTED',count=48))
  for unit in Q.remaining_units(plan):require(Q.require_current_dispatch(f,unit)==unit,'Originalremaining4 rejected')
  require(not (f/'GPU_RUN_STARTED.json').exists(),'CPU reuseaudit startedGPU')
  report=dict(status='PASS_REAL48_HISTORICAL_BYTES_FIXED_ORIGIN_NEGATIVES_CPU',positive=positive,checks=rows,check_count=len(rows),explicit_mocks=['dualhost fresh endpoint receipt','fixed historical root path mapping to local read-only copy'],GPU_calls=0,AIME_samples=0,not_GPU_qualification=True)
  dump(base/'REUSE48_CPU_REPORT.json',report);return report
 finally:Q.Path=realpath

REUSE_SHIM='''
import importlib.util,json,os
from pathlib import Path
import execution_control as C
R=Path(os.environ['WSCALE_CPU_FIXTURE_ROOT'])
s=importlib.util.spec_from_file_location('_actual_qualification_reuse',R/'production/qualification_reuse.py');real=importlib.util.module_from_spec(s);s.loader.exec_module(real)
remaining_units=real.remaining_units
reused_units=real.reused_units
require_current_dispatch=real.require_current_dispatch
assert_no_legacy_current_evidence=real.assert_no_legacy_current_evidence
def admit_reused48(root,plan):
 real.remaining_units(plan);real.assert_no_legacy_current_evidence(root,plan)
 value=C.read(R/'MOCKED_ORIGIN_REUSE_RECEIPT.json');C.require(value['count']==48,'CPU fixture oldorigin receipt differs');return value
'''
DURABLE_MOCK='''
import execution_control as C

def publish_unit(root,target,*,unit,formal):
 C.require(C.read(target/'RESOURCE_INTERVAL.json')['release']=='PASS','CPU durable fixture ran before trueownPopen release')
 cfg=C.read(root/'CASE.json')
 if cfg['case']=='durable_failure' and not formal:raise RuntimeError('EXPECTED_CPU_MOCK_DURABLE29_FAILURE')
 value=dict(status='PASS_DURABLE29_CLOSED_UNIT',unit=unit,formal=formal,source_freeze_sha256=C.sha(root/'SOURCE_FREEZE.json'),permit_sha256=C.sha(root/'GPU_PERMIT.json'),CPU_MOCK_ONLY=True,remote_writes=0)
 C.atomic_new(target/'DURABLE29_RECEIPT.json',value);return value
'''

def controller_checks(base,reuse_report):
 old=load('_original_control_cpu',R/'tests/control_integration_cpu.py');old.R=R;results=[]
 for case in ('success','qualification_failure','positive_RESULT_nonzero_exit','durable_failure','formal_failure'):
  f,pins,foreign=old.stage_fixture(base,case)
  for name in ('src/qualification_reuse.py',):link(R/name,f/'production'/Path(name).name)
  (f/'src/qualification_reuse.py').write_text(REUSE_SHIM);(f/'src/durable_output29.py').write_text(DURABLE_MOCK)
  # Only historical filesystem endpoint is mocked here; the actual strict
  # 4current loop and actual controller/worker fresh-Popen barrier run.
  receipt=copy.deepcopy(reuse_report['positive']);receipt['CPU_MOCK_ONLY']=True;dump(f/'MOCKED_ORIGIN_REUSE_RECEIPT.json',receipt)
  p=f/'src/execution_control.py';s=p.read_text();needle="real.authorize=authorize";replacement="""def fixture_authorize(root,permit):
 real.require(Path(root).resolve()==ROOT.resolve(),'CPU fixture scope mismatch')
 p=real.read(permit);real.require(p['source_freeze_sha256']==real.verify_sources(Path(root)),'CPU fixture frozen source differs');return p
real.authorize=fixture_authorize""";require(needle in s,'Fixture auth location differs');s=s.replace(needle,replacement)
  s=s.replace("value=original_atomic(path,obj)","value=original_atomic(path,obj)\n if Path(path).name=='RESULT.json' and Path(path).parent.name=='GPU1_base_MATCHED4096' and json.loads((ROOT/'CASE.json').read_text())['case']=='positive_RESULT_nonzero_exit':os._exit(11)")
  p.write_text(s)
  p=f/'src/runtime_engine.py';s=p.read_text();s=s.replace("cfg=C.read(self.root/'CASE.json')\n  if cfg['case']=='external_sigterm'", "cfg=C.read(self.root/'CASE.json')\n  if cfg['case']=='qualification_failure' and unit['id']=='GPU1_base_MATCHED4096':raise RuntimeError('EXPECTED_CPU_MOCK_NEW4_QUALIFICATION_FAILURE')\n  if cfg['case']=='external_sigterm'");p.write_text(s)
  paths=[p for p in sorted(f.rglob('*')) if p.is_file() and p.name not in ('SOURCE_FREEZE.json','GPU_PERMIT.json')];dump(f/'SOURCE_FREEZE.json',dict(CPU_FIXTURE_ONLY=True,not_GPU_qualification=True,files=[dict(path=str(p.relative_to(f)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]));permit=read(f/'GPU_PERMIT.json');permit['source_freeze_sha256']=sha(f/'SOURCE_FREEZE.json');dump(f/'GPU_PERMIT.json',permit)
  rc,_=old.run_controller(f,case);terminal=read(f/'SESSION_RESULT.json');ledger=read(f/'RESOURCE_LEDGER.json');freshqual=list(f.glob('execution/*/DISPATCH.json'));formal=list(f.glob('formal/*/DISPATCH.json'));gate=(f/'AUTHORIZATION_FORMAL.json').exists()
  if case=='success':
   require(rc==0 and terminal['status']=='GPU_COMPLETE_REUSE48_PLUS4_QUALIFICATION_90_FORMAL','CPU continuation success failed:'+str(terminal));require([p.parent.name for p in sorted(freshqual)]==sorted(['GPU1_'+q+'_MATCHED4096' for q in ('base','W','R','RW')]),'Original48 were redispatched');require(len(formal)==90 and len(ledger['intervals'])==94,'Continuation4+90 truePopen ledger differs');q=read(f/'GPU_CODEC_QUALIFICATION.json');require(q['inspection']['count']==52 and q['inspection']['current_count']==4 and q['inspection']['reused_count']==48,'Combined strict barrier wrong')
   count=len(freshqual)+len(formal);rc2,_=old.run_controller(f,'restart');require(rc2!=0 and len(list(f.glob('execution/*/DISPATCH.json')))+len(list(f.glob('formal/*/DISPATCH.json')))==count,'Controller restart creatednewdispatch')
  else:
   require(rc!=0 and terminal['status']=='FAIL_CLOSED_STOP' and (f/'FIRST_FAILURE.json').exists(),'Failure path did not trueSTOP')
   if case in ('qualification_failure','positive_RESULT_nonzero_exit','durable_failure'):require(not gate and not formal and len(freshqual)==1,'Failedqualification/durable29 wrongly admittedformal/repeateddispatch')
   if case=='positive_RESULT_nonzero_exit':require(read(f/'execution/GPU1_base_MATCHED4096/RESULT.json')['status']=='PASS' and ledger['intervals'][0]['exit_returncode']==11 and terminal['fresh_qualification_count']==0,'PositiveRESULT wrongly credited despiteactualnonzeroexit')
   if case=='formal_failure':require(gate and len(formal)<90,'Formal failure notpreserved')
  require(all(x['release']=='PASS' for x in ledger['intervals']),'OwnCPUchild release unresolved');old.assert_no_owned_children(f)
  row=dict(case=case,status='PASS',actual_controller_returncode=rc,new_qualification_dispatches=len(freshqual),new_formal_dispatches=len(formal),actual_Popen_intervals=len(ledger['intervals']),formal_gate=gate,CPU_MOCK_ONLY=True,GPU_qualification_credit=0,AIME_samples=0,remote_writes=0,fixture_root=str(f),production_controller_sha256=sha(f/'scripts/gpu_controller.py'),production_worker_sha256=sha(f/'scripts/gpu_unit_worker.py'),production_strict_inspector_sha256=sha(f/'src/control_qualification.py'));results.append(row);print(json.dumps(row),flush=True)
 report=dict(status='PASS_ACTUAL_CONTROLLER_REUSE48_4NEW_90FORMAL_CPU_BARRIER',cases=results,explicit_mocks=['historical48 endpoint already separately validated from actual bytes','model/runtime/device/native','durable29 endpoint','fixture authorization root and source manifest','poll acceleration'],actual_components=['controller','worker','exactplan+dispatch refusal','strict4current qualification loop','formalpayloadvalidator','real fresh Popen processes and owncleanup','formal barrier+restart refusal'],GPU_calls=0,scientific_qualification_credit=0,AIME_samples=0,remote_writes=0)
 dump(base/'REUSE48_CONTROLLER_CPU_REPORT.json',report);return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence-root',type=Path);a=ap.parse_args();base=a.evidence_root or Path(tempfile.mkdtemp(prefix='wscale_reuse48_cpu_'));base.mkdir(parents=True,exist_ok=True);one=reuse_checks(base);two=controller_checks(base,one);print(json.dumps(dict(status='PASS_REUSE48_CPU',origin_negative_checks=one['check_count'],controller_cases=len(two['cases']),evidence_root=str(base),GPU_calls=0)),flush=True)
if __name__=='__main__':main()
