"""One newly human-authorized revision in the existing project; no old source mutation."""
import base64
import json
import sys
from pathlib import Path
import remote

L = remote.ROOT / 'revisions/replay_binding_v3'
if L.exists():
    raise RuntimeError('New revision already exists; inspect instead of applying twice')
code = r'''
import ast,hashlib,json,os,shutil,time
from pathlib import Path
P=Path('/data/zypan/analyses/qwen_gdn_group64_four_codec_aime26_seed1_v1')
O=P/'revisions/native_admission_v2';R=P/'revisions/replay_binding_v3'
assert not R.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:
  if isinstance(v,bytes):raise TypeError('use binary')
  if isinstance(v,str):f.write(v)
  else:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
assert sha(O/'SOURCE_FREEZE.json')=='740d66ce7bacd5c3f45a79ca927579788a1d33f7854ece3ad73e217035514547'
assert sha(O/'FINAL_SEAL_RECEIPT.json')=='3436f902a60e97f30005de2ab6933a2d434096875287bd0a3615c0261e3810d5'
assert read(O/'operations/OFFLINE_ACTUAL_EXIT.json')==dict(controller_exit=1,finisher_exit=0,monotonic=1217055.478903494)
old=read(O/'SOURCE_FREEZE.json');R.mkdir(parents=True)
rebuild={'ROOT.json','HUMAN_REPAIR_AUTHORIZATION.json','SOURCE_MODEL_RUNTIME_FREEZE.json','GPU_UUID_AND_PERMISSION_AUDIT.json','GROUP64_CPU_CODEC_QUALIFICATION.json','REPAIR_IMPLEMENTATION_AND_VALIDATION.md'}
external={'analysis/EXTERNAL_NATIVE_BINDINGS.json','analysis/EXTERNAL_PYTHON_BINDINGS.json','analysis/MODEL_TOKENIZER_FRESH_SHA.json'}
for x in old['files']:
 name=x['path'];p=O/name
 assert p.is_file()and not p.is_symlink()and p.stat().st_size==x['bytes']and sha(p)==x['sha256'],name
 preserve=name in rebuild or(name.startswith('analysis/')and name not in external)
 q=R/('provenance/NATIVE_ADMISSION_V2/SOURCE_CONTROL/'+name if preserve else name);q.parent.mkdir(parents=True,exist_ok=True)
 if name.startswith(('runtime/','history/','fixtures/')):os.link(p,q)
 else:shutil.copy2(p,q)
for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','FIRST_FAILURE.json','SESSION_RESULT.json','PARTIAL_COUNTS.json','RESOURCE_LEDGER.json','GPU_RESOURCE_LEDGER.jsonl','operations/OFFLINE_ACTUAL_EXIT.json','operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json','operations/POST_SEAL_INPUT_BINDING_FIRST_CAUSE_REVIEW.json']:
 q=R/'provenance/NATIVE_ADMISSION_V2'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/n,q)
for p in (O/'execution').rglob('*'):
 if p.is_file():
  assert p.suffix.lower()not in('.bin','.npz','.pt','.safetensors')
  q=R/'provenance/NATIVE_ADMISSION_V2/execution'/p.relative_to(O/'execution');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for p in (O/'operations/REPLAY_BINDING_CPU_CANDIDATE').rglob('*'):
 if p.is_file():
  q=R/'provenance/CPU_INPUT_BINDING_CANDIDATE'/p.relative_to(O/'operations/REPLAY_BINDING_CPU_CANDIDATE');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for n in ['operations','monitoring','analysis']:(R/n).mkdir(exist_ok=True)
# Apply the exact previously reviewed full-input candidate, not a padded/repeated stream.
p=R/'protocol/non_aime_fixture.json';p.write_bytes((O/'operations/REPLAY_BINDING_CPU_CANDIDATE/protocol/non_aime_fixture.json').read_bytes())
assert sha(p)=='0b4c355f01b00d5958879218138944a9a71db43dec1ddfe1ca3d596f6eb605b3'
assert len(read(p)['tokens'])==32768
for n in ['analysis/EXTERNAL_NATIVE_BINDINGS.json','analysis/EXTERNAL_PYTHON_BINDINGS.json']:
 v=read(R/n)
 for x in v['files']:
  if x['path'].startswith(str(O/'runtime')+'/'):
   x['previous_revision_path']=x['path'];x['path']=str(R)+x['path'][len(str(O)):]
 (R/n).write_text(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
for name,a,b in [('src/canonical_h128.py',str(O/'runtime/transformers-qwen35/src'),str(R/'runtime/transformers-qwen35/src')),('src/execution_control.py','qwen_group64_four_codec_aime26_seed1_v1_native_admission_v2','qwen_group64_four_codec_aime26_seed1_v1_replay_binding_v3')]:
 p=R/name;s=p.read_text();assert s.count(a)==1;s=s.replace(a,b);p.write_text(s)
root=read(O/'ROOT.json');root.update(local_root='/Users/pytenter/Documents/临时项目1/QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/revisions/replay_binding_v3',remote_root=str(R),archive_root='/data/zypan/qwen_group64_four_codec_aime26_archive/revisions/QWEN_GDN_GROUP64_FOUR_CODEC_AIME26_SEED1_V1/replay_binding_v3',execution_revision='replay_binding_v3',status='INPUT_BINDING_REPAIRED_CPU_PENDING',source_freeze=None,permit=None,created_client_time=CLIENT_TIME,parent_repaired_attempt_source_sha256=sha(O/'SOURCE_FREEZE.json'),parent_repaired_attempt_seal_sha256=sha(O/'FINAL_SEAL_RECEIPT.json'))
write(R/'ROOT.json',root)
write(R/'HUMAN_REPAIR_AUTHORIZATION.json',dict(human_request='继续帮我修复啊，然后正式开始跑AIME26',scope='Apply already reviewed exact full32768 historical nonAIME replay input, add CPU input admission before GPU leaves, requalify finite38 then original conditional120. This is one new explicit repair authorization; preserve both prior STOPs/source/seals/true costs; no automatic hotpatch/retry after this unique start.',only_GPU_indices=[5,6],only_GPU_uuids=root['selected_uuids'],prior_repaired_source_sha256=sha(O/'SOURCE_FREEZE.json'),prior_repaired_seal_sha256=sha(O/'FINAL_SEAL_RECEIPT.json'),original_full_task_sha256=sha(P/'USER_REQUEST.txt'),mathematics_and_sampler_unchanged=True,no_GPU_retry_after_STARTED=True))
history=[]
for base in [P,O]:history.append(dict(root=str(base),source_sha256=sha(base/'SOURCE_FREEZE.json'),seal_sha256=sha(base/'FINAL_SEAL_RECEIPT.json'),legal_qualification=0 if base==P else 2,formal_completed=0,qualification_credit_in_new_attempt=0,resource_ledger=read(base/'RESOURCE_LEDGER.json')))
write(R/'PRIOR_EXECUTION_HISTORY_AND_COST.json',dict(status='TWO_PRIOR_TRUE_STOPS_READONLY_NO_NEW_QUALIFICATION_CREDIT',attempts=history,current_attempt_ledger_separate=True,EXACT_physical_GPUh='UNRESOLVED'))
assert all((R/'src'/n).read_bytes()==(O/'src'/n).read_bytes()for n in ['group_codecs.py','independent_reference.py','persistent_adapter.py','four_codec_adapter.py','runtime_engine.py','protocol_binding.py'])
fn=lambda p:{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert fn(R/'src/canonical_h128.py')==fn(O/'src/canonical_h128.py')
assert all((R/n).read_bytes()==(O/n).read_bytes()for n in ['scripts/gpu_unit_worker.py','scripts/finalize_offline.py','scripts/score_pass_cpu.py','FORMAL_JOB_MANIFEST.json','QUESTION_GPU_ASSIGNMENT.json','protocol/sampler.py','protocol/aime26_scorer_v4.py','AIME26_DATASET_FREEZE.json'])
assert sha(O/'SOURCE_FREEZE.json')=='740d66ce7bacd5c3f45a79ca927579788a1d33f7854ece3ad73e217035514547'and sha(O/'FINAL_SEAL_RECEIPT.json')=='3436f902a60e97f30005de2ab6933a2d434096875287bd0a3615c0261e3810d5'
names=[str(p.relative_to(R))for p in R.rglob('*')if p.is_file()and(p.relative_to(R).parts[0]in ('src','scripts','protocol','analysis')or len(p.relative_to(R).parts)==1)]
print(json.dumps(dict(status='PASS_INPUT_BINDING_APPLIED_PRE_GPU',root=str(R),immutable_hardlink_scope=['runtime','history','fixtures'],prior_sources_seals_untouched=True,files={n:(R/n).read_text()for n in names})))
'''.replace('CLIENT_TIME',repr(__import__('datetime').datetime.now(__import__('datetime').timezone(__import__('datetime').timedelta(hours=8))).isoformat()))
result = remote.run(code, timeout=240)
L.mkdir(parents=True)
for name,value in result.pop('files').items():
    p=L/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
(L/'operations').mkdir(exist_ok=True);(L/'monitoring').mkdir(exist_ok=True)
(L/'operations/remote.py').write_bytes((remote.ROOT/'operations/remote.py').read_bytes())
remote.save('operations/REPLAY_BINDING_V3_APPLIED.json',result)
print(json.dumps(result))
