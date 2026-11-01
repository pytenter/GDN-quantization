"""New explicit authorization: freeze once, launch once, no restart of either sealed attempt."""
import json,sys
from pathlib import Path
import remote
P=remote.ROOT;L=P/'revisions/replay_binding_v3'
if '--launch'not in sys.argv:raise SystemExit('Explicit --launch required')
code=r'''
import ast,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
P=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1');O=P/'revisions/native_admission_v2';R=P/'revisions/replay_binding_v3';sys.path.insert(0,str(R/'src'));import execution_control as C
assert not(R/'SOURCE_FREEZE.json').exists()and not(R/'GPU_PERMIT.json').exists()and not(R/'GPU_RUN_STARTED.json').exists(),'Already frozen/started: do not reset or relaunch'
assert C.sha(P/'SOURCE_FREEZE.json')=='dd3c3cf9ba95a83d956e024025173e0e1e9e835428d495d2ce0af1c32098a6be'and C.sha(P/'FINAL_SEAL_RECEIPT.json')=='6a803bc583a790e858cdff6c7438473d08333e28ba7b560ea745ca30627fa2ee'
assert C.sha(O/'SOURCE_FREEZE.json')=='740d66ce7bacd5c3f45a79ca927579788a1d33f7854ece3ad73e217035514547'and C.sha(O/'FINAL_SEAL_RECEIPT.json')=='3436f902a60e97f30005de2ab6933a2d434096875287bd0a3615c0261e3810d5'
assert C.read(R/'HUMAN_REPAIR_AUTHORIZATION.json')['human_request']=='继续帮我修复啊，然后正式开始跑AIME26'
v=C.read(R/'analysis/REPAIRED_REVISION_CPU_VALIDATION.json');assert v['status']=='PASS_TWO_FRESH_REPLAY_BINDING_CPU_SUITES'and len(v['runs'])==12 and all(x['exit']==0 for x in v['runs'])
assert sum(x['result']['count']for x in v['runs'])==762
scope=C.read(R/'analysis/FINAL_REPAIR_SOURCE_SCOPE.json');assert scope['status']=='PASS_REPAIRED_SOURCE_SCOPE_CPU_ONLY'and scope['actual_external_fresh_SHA_count']==9313
for name,h in next(x for x in v['runs']if x['script']=='cpu_qualification.py')['result']['source_sha256'].items():assert C.sha(R/name)==h
for n in ['FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','protocol/sampler.py','protocol/aime26_scorer_v4.py','AIME26_DATASET_FREEZE.json','src/runtime_engine.py','src/canonical_h128.py','scripts/gpu_unit_worker.py','scripts/finalize_offline.py']:assert(R/n).read_bytes()==(O/n).read_bytes()
from qualification_input_admission import validate;inputs=validate(R)
C.validate_jobs(R,C.read(R/'FORMAL_JOB_MANIFEST.json')['jobs']);C.model_guard(R)
for x in C.read(R/'analysis/EXTERNAL_NATIVE_BINDINGS.json')['files']+C.read(R/'analysis/EXTERNAL_PYTHON_BINDINGS.json')['files']:
 s=Path(x['path']).stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns']),x['path']
for g in C.UUIDS:C.device(g)
assert not C.reference_root().exists();shm=os.statvfs('/dev/shm');assert shm.f_bavail*shm.f_frsize>=24*1024**3;C.storage_guard(R)
archive=Path(C.read(R/'ROOT.json')['archive_root']);assert not archive.exists();free=os.statvfs(archive.parent);assert free.f_bavail*free.f_frsize>=32*1024**3
root=C.read(R/'ROOT.json');root['status']='CPU_ADMITTED_REPLAY_INPUT_REPAIR_BEFORE_UNIQUE_START';C.status(R/'ROOT.json',root)
note='''# Replay input binding repair, same Group64 project

Latest human authorization: 继续帮我修复啊，然后正式开始跑AIME26. Current execution revision is replay_binding_v3. Both previous executions remain true STOP with their original frozen sources, seals, logs, costs, exits and qualification records. Original failed attempt has0 qualification credit; native_admission_v2 has2 true micro PASS but no persistent-model/AIME credit. Both are historical only; this execution reruns all finite38 before original conditional120, with no trajectory joining.

Repair: current protocol/non_aime_fixture.json restores the complete32768 token IDs of the exact original nonAIME doc1 replay (rawSHA18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d). Its first64 are identical to the wrongly selected short fixture. No repeated/padded/resampled tokens. New CPU admission in controller verifies full identity and38 input bounds before model/GPU leaves. Queues, watchdog, process ownership, worker, model loader, all mathematical/runtime modules, sampler, scorer, prompt/input IDs, four conditions, seed1,81920 cap, question15/15UUID bundle assignment and order remain unchanged. Only other control change isolates temporary RAM reference namespace. Native inventory still contains the individually pinned NVML and same-driver compute helpers.

Two fresh CPU processes each passed381 checks:113 math/TinyQwen/sampler,45 control,5 actual saved-native-map strict checker replay,34 production loader/real31tokenizer records,24 production input-admission positives/negatives,160 real worker-entrypoint paths covering38 qualification and120formal IDs plus short-input regressions. Model/device/native identity and generation in worker-entrypoint tests are explicitly CPU mocks; they give zero GPU qualification or AIME credit. Actual GPU38 remains mandatory. CPU preparation failures and corrected test harness AST assumption remain in provenance/operations, not classified as scientific GPU failures.

Only server29 GPU5 GPU-07d9c381-888d-1111-a944-56abcc583573 and GPU6 GPU-de53f243-b271-521c-d875-d96611ed8574, real UUID CVD→logicalcuda0. Other GPUs and server55 C1/S3 are untouched. Every sample is a fresh process; no reset, re-dispatch or automatic patch/retry after STARTED. No GPUh/wall/unit elapsed cap;1800 true phase/token no-progress watchdog and120 own release observation only. Exact physicalGPUh UNRESOLVED is never0. Full120 legal completions and releases are required before two-fresh CPU FrozenV4 scoring and paired historicalGroup32 analysis. Otherwise preserve real counts/first cause and fully deliver true STOP. Old Group32 assets/results remain read-only. Models/environments/tensors are not downloaded toMac.
'''
C.atomic_new(R/'SOURCE_MODEL_RUNTIME_FREEZE.json',dict(status='FROZEN_REPLAY_INPUT_REPAIR_CPU_PASSED_GPU_PENDING',execution_source_pins=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=C.sha(p))for p in sorted((R/'src').glob('*.py'))],model_identity_sha256=C.sha(R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json'),native_bindings_sha256=C.sha(R/'analysis/EXTERNAL_NATIVE_BINDINGS.json'),python_bindings_sha256=C.sha(R/'analysis/EXTERNAL_PYTHON_BINDINGS.json'),repair_validation_sha256=C.sha(R/'analysis/REPAIRED_REVISION_CPU_VALIDATION.json'),repair_scope_sha256=C.sha(R/'analysis/FINAL_REPAIR_SOURCE_SCOPE.json'),prior_attempt_source_sha256=C.sha(O/'SOURCE_FREEZE.json'),prior_attempt_seal_sha256=C.sha(O/'FINAL_SEAL_RECEIPT.json'),original_attempt_source_sha256=C.sha(P/'SOURCE_FREEZE.json'),original_attempt_seal_sha256=C.sha(P/'FINAL_SEAL_RECEIPT.json'),model='Qwen3.5-9B',torch='2.5.1+cu121',cuda='12.1',driver='580.178.04',mathematics_unchanged=True,GPU_qualification_claim=False))
with(R/'REPAIR_IMPLEMENTATION_AND_VALIDATION.md').open('x')as f:f.write(note)
C.atomic_new(R/'GPU_UUID_AND_PERMISSION_AUDIT.json',dict(status='PASS_ONLY_AUTHORIZED_GPU5_GPU6_IDLE_BEFORE_UNIQUE_START',only_physical_indices=[5,6],only_uuids=list(C.UUIDS.values()),observation=C.query_gpu(),human_repair_authorization_sha256=C.sha(R/'HUMAN_REPAIR_AUTHORIZATION.json'),other_tasks_untouched=True))
rows=[]
for p in sorted(R.rglob('*')):
 assert not p.is_symlink(),str(p)
 rel=str(p.relative_to(R))
 if p.is_file()and p.relative_to(R).parts[0]not in ('operations','monitoring')and rel not in ('SOURCE_FREEZE.json','GPU_PERMIT.json'):
  rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=C.sha(p)))
C.atomic_new(R/'SOURCE_FREEZE.json',dict(task_id=C.TASK,execution_revision='replay_binding_v3',status='FROZEN_BEFORE_SINGLE_NEW_EXPLICIT_REPAIR_EXECUTION',files=rows,count=len(rows),human_repair_authorization_sha256=C.sha(R/'HUMAN_REPAIR_AUTHORIZATION.json'),previous_source_sha256=C.sha(O/'SOURCE_FREEZE.json'),previous_seal_sha256=C.sha(O/'FINAL_SEAL_RECEIPT.json'),old_failures_not_rejudged=True,no_future_hotpatch_or_GPU_retry=True))
for g in C.UUIDS:C.device(g)
C.atomic_new(R/'GPU_PERMIT.json',dict(task_id=C.TASK,ALLOW_GPU='YES',scope='EXPLICIT_REPLAY_INPUT_REPAIR_QUALIFICATION38_THEN_ORIGINAL_CONDITIONAL120_ONCE',devices=list(C.UUIDS.values()),physical_indices=[5,6],source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),plan_sha256=C.sha(R/'GPU_EXECUTION_PLAN.json'),human_grant_sha256=C.sha(R/'HUMAN_GPU_AUTHORIZATION.json'),human_repair_authorization_sha256=C.sha(R/'HUMAN_REPAIR_AUTHORIZATION.json'),GPUh_hard_cap=None,total_wall_cap=None,unit_elapsed_cap=None,watchdog_seconds=1800,cleanup_seconds=120))
C.authorize(R,R/'GPU_PERMIT.json');tmux='qwen-group64-replay-binding-v3';assert subprocess.run(['tmux','has-session','-t',tmux],capture_output=True).returncode!=0
cmd=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1','PYTHONPATH=/data/zypan/envs/qwen-gdn-group64-historical-user-site-v1','/data/zypan/envs/qwen-gdn-group64-historical-port-v1/bin/python','-u','-B',str(R/'scripts/run_offline_once.py'),str(R)]
subprocess.run(['tmux','new-session','-d','-s',tmux,shlex.join(cmd)],check=True)
receipt=dict(status='UNIQUE_EXPLICIT_REPLAY_INPUT_REPAIR_EXECUTION_ACTUALLY_LAUNCHED',tmux=tmux,source_count=len(rows),source_sha256=C.sha(R/'SOURCE_FREEZE.json'),permit_sha256=C.sha(R/'GPU_PERMIT.json'),human_repair_authorization_sha256=C.sha(R/'HUMAN_REPAIR_AUTHORIZATION.json'),revision_root=str(R),qualification=38,conditional_formal=120,monotonic=time.monotonic(),old_sources_seals_untouched=True)
C.atomic_new(R/'operations/UNIQUE_LAUNCH_RECEIPT.json',receipt);C.atomic_new(P/'operations/REPLAY_BINDING_V3_UNIQUE_LAUNCH_RECEIPT.json',receipt)
print(json.dumps(receipt))
'''
compile(code,'<one-shot-replay-input-launch>','exec')
r=remote.run(code,timeout=300)
with(L/'operations/UNIQUE_LAUNCH_RECEIPT.json').open('x')as f:json.dump(r,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(r,ensure_ascii=False))
