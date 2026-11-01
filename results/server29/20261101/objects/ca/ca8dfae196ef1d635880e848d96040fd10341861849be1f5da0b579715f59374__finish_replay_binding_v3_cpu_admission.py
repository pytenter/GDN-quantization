"""Finish reviewable CPU repair admission even when an authorized GPU is occupied. Never dispatch."""
import json
import remote
L=remote.ROOT/'revisions/replay_binding_v3'
code=r'''
import sys,json,os,time,ast
from pathlib import Path
R=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1/revisions/replay_binding_v3');O=R.parent/'native_admission_v2';P=R.parents[1];sys.path.insert(0,str(R/'src'));import execution_control as C
assert not(R/'SOURCE_FREEZE.json').exists()and not(R/'GPU_RUN_STARTED.json').exists()
a=C.read(R/'analysis/REPAIRED_REVISION_CPU_VALIDATION.json');assert a['status']=='PASS_TWO_FRESH_REPLAY_BINDING_CPU_SUITES'and len(a['runs'])==12 and all(x['exit']==0 for x in a['runs'])
assert sorted((x['script'],x['result']['count'])for x in a['runs'])==sorted((s,n)for s,n in [('cpu_qualification.py',113),('cpu_control_review.py',45),('cpu_native_admission_review.py',5),('cpu_production_loader_review.py',34),('cpu_worker_entrypoint_review.py',160),('cpu_input_admission_review.py',24)]for _ in(0,1))
C.atomic_new(R/'operations/PRE_FREEZE_FOREIGN_GPU_GATE_PRESERVED.json',dict(status='RESOURCE_BLOCKED_NO_DISPATCH',error="RuntimeError('Foreign process on assigned GPU; never kill it')",scope='First CPU source admission completed its external hash pass, then failed idle admission before freeze/permit/GPU launch. No science FIRST_FAILURE/STARTED, no scientific trajectory, no GPU signals.',GPU_dispatches=0,known_selected_foreign_pid=1242196,known_selected_foreign_starttick=121916506))
start=time.monotonic();C.model_guard(R);external=C.verify_external(R)
assert external==9313 and not C.reference_root().exists()
free=os.statvfs('/dev/shm');assert free.f_bavail*free.f_frsize>=24*1024**3
C.storage_guard(R);C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs'])
from qualification_input_admission import validate;input_admission=validate(R)
expected={'src/execution_control.py','src/qualification_input_admission.py','scripts/gpu_controller.py','scripts/cpu_worker_entrypoint_review.py','scripts/cpu_input_admission_review.py','protocol/non_aime_fixture.json'}
changed=[];unchanged=[]
for folder in ('src','scripts','protocol'):
 for p in sorted((R/folder).rglob('*')):
  if not p.is_file():continue
  rel=str(p.relative_to(R));old=O/rel;newsha=C.sha(p);oldsha=C.sha(old)if old.is_file()else None;row=dict(path=rel,old_sha256=oldsha,new_sha256=newsha)
  (unchanged if oldsha==newsha else changed).append(row)
assert {x['path']for x in changed}==expected
for n in ['src/runtime_engine.py','src/canonical_h128.py','src/four_codec_adapter.py','src/persistent_adapter.py','src/group_codecs.py','src/independent_reference.py','src/protocol_binding.py','scripts/gpu_unit_worker.py','scripts/finalize_offline.py','scripts/score_pass_cpu.py','FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','protocol/sampler.py','protocol/aime26_scorer_v4.py','AIME26_DATASET_FREEZE.json']:assert(R/n).read_bytes()==(O/n).read_bytes(),n
oldcontrol=(O/'scripts/gpu_controller.py').read_text();newcontrol=(R/'scripts/gpu_controller.py').read_text();added=" from qualification_input_admission import validate as validate_qualification_inputs\n C.atomic_new(R/'analysis/EXECUTION_QUALIFICATION_INPUT_ADMISSION.json',validate_qualification_inputs(R))\n";assert newcontrol.replace(added,'',1)==oldcontrol
oldexec=(O/'src/execution_control.py').read_text();newexec=(R/'src/execution_control.py').read_text();assert newexec.replace('qwen_group64_four_codec_aime26_seed1_v1_replay_binding_v3','qwen_group64_four_codec_aime26_seed1_v1_native_admission_v2')==oldexec
assert C.sha(O/'SOURCE_FREEZE.json')=='740d66ce7bacd5c3f45a79ca927579788a1d33f7854ece3ad73e217035514547'and C.sha(O/'FINAL_SEAL_RECEIPT.json')=='3436f902a60e97f30005de2ab6933a2d434096875287bd0a3615c0261e3810d5'
obs=C.query_gpu();owners=[]
for app in obs['apps']:
 if app['uuid']in C.UUIDS.values():owners.append(dict(app=app,identity=C.process_identity(app['pid'])))
proof=dict(status='PASS_REPAIRED_SOURCE_SCOPE_CPU_ONLY',changed_production_and_test_inputs=changed,unchanged_files=unchanged,model_worker_runtime_math_sampler_scorer_formal_roster_byte_identical=True,controller_diff='Only CPU input admission before GPU; queue/dispatch/exit/watchdog unchanged',control_diff='Only new temporary reference namespace',fixture_diff='Full original32768 replay, exact prefix64/512/384, no padding or repeats',old_sealed_sources_untouched=True,actual_external_fresh_SHA_count=external,model_stat_count=17,input_admission=input_admission,NVML=obs,selected_owner_identities=owners,GPU_resource_status='RESOURCE_BLOCKED_NO_DISPATCH'if owners else'OBSERVED_IDLE_RECHECK_BEFORE_LAUNCH',same_host_validation_seconds=time.monotonic()-start,GPU_dispatches=0)
C.atomic_new(R/'analysis/FINAL_REPAIR_SOURCE_SCOPE.json',proof)
root=C.read(R/'ROOT.json');root['status']='CPU_REPAIRED_RESOURCE_BLOCKED_NO_DISPATCH'if owners else'CPU_REPAIRED_READY_FOR_ADMITTED_LAUNCH';C.status(R/'ROOT.json',root)
print(json.dumps(proof))
'''
compile(code,'<CPU-only-repair-admission>','exec')
r=remote.run(code,timeout=600)
with(L/'analysis/FINAL_REPAIR_SOURCE_SCOPE.json').open('x')as f:json.dump(r,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:r[k]for k in ('status','actual_external_fresh_SHA_count','model_stat_count','GPU_resource_status','same_host_validation_seconds','GPU_dispatches')}))
