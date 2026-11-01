"""Bind audited historical assets and inspect native checkpoints on CPU."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1');D=R/'execution_03';OLD=R/'execution_02'
W=Path('/data/zypan/worktrees/ling-kda-prospective-qualified-completion');WOLD=Path('/data/zypan/worktrees/ling-kda-prospective-engineering-recovery')
os.environ['LING_PROSPECTIVE_RESULTS']=str(D);os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True;sys.path.insert(0,str(W))
from recovery_state import sha,atomic_json
def read(p):return json.loads(Path(p).read_text())
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())

assert read(D/'PHASE0_HISTORICAL_ASSET_AUDIT.json')['status']=='PASS'
assert read(D/'NEW_RECOVERY_TEST_REPORT.json')['status']=='PASS'
for p in [D/'baseline_cpu_reverification_02/GPU_ADMISSION_TEST_REPORT.json',D/'baseline_cpu_reverification_02/CONTROLLER_RESUME_TEST_REPORT.json']:assert read(p)['status']=='PASS'
source_math={}
for filename in ['functional_worker.py','formal_worker.py']:
    aa,bb=ast.parse((WOLD/filename).read_text()),ast.parse((W/filename).read_text())
    def nodes(t):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['transfer','object_fingerprint','audit_state','restore_audit','HiddenObserver','checkpoint','original_state_diagnostic','observe_qdq','collect_metrics']}
    a,b=nodes(aa),nodes(bb);assert a==b
    for n,v in a.items():source_math[filename+':'+n]=hashlib.sha256(v.encode()).hexdigest()
    loops=lambda t:[ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='position']
    assert loops(aa)==loops(bb);source_math[filename+':full_scientific_position_loop']=hashlib.sha256('\n'.join(loops(aa)).encode()).hexdigest()
unchanged=read(OLD/'NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json')['unchanged_files']
for name,h in unchanged.items():assert sha(W/name)==h
atomic_json(D/'NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json',dict(status='PASS_UNCHANGED_MATH_BOUNDARY',unchanged_AST=source_math,unchanged_files=unchanged,source_base_commit='dfcbb7e8497a585f8ef94c0eec6d4146ade860b0',actual_GPU_qualification='PENDING',changes='new execution paths/device routing, per-UUID tickets, explicit graphics visibility, two qualification lanes, historical checkpoint reuse bindings, CPU delivery only; full scientific forward loops/helper AST unchanged'))
for name in ['FROZEN_SPEC.json','FROZEN_4DOC_SELECTION.json','SELECTION_FREEZE_RECEIPT.json','SHORT_HORIZON_SELECTION.csv','SHORT_HORIZON_SELECTION_STATUS.json','INPUT_ELIGIBILITY_REPRODUCTION.json','PREREGISTRATION_RECONCILIATION.json','SECONDARY_SPEC.json','MODEL_WEIGHT_SHA256.txt','ANALYSIS_PLAN.json']:
    copy(OLD/name,D/name)
for p in (OLD/'preregistration').iterdir():
    if p.is_file():copy(p,D/'preregistration'/p.name)
for p in (OLD/'screening').glob('*.json'):
    if p.name.endswith('.SELECTION_DECISION.json') or p.name.count('.')==1:copy(p,D/'screening'/p.name)
spec=read(D/'FROZEN_SPEC.json');assert sha(D/'FROZEN_SPEC.json')=='8ae8927ed37209efccbf19c672b9baeb46c0d7f272601f8f54d1f9609b64f3d1'
pf=read(D/'GPU_PREFLIGHT_INTERPRETATION.json');assert pf['status']=='PASS_TWO_AUTHORIZED_IDLE_COMPUTE_DEVICES'
g4='GPU-de1f9152-32e5-a033-d2bf-ea7abecb606f';g5='GPU-07d9c381-888d-1111-a944-56abcc583573'
mapping={d['document']:g4 if i<3 else g5 for i,d in enumerate(spec['documents'])}
atomic_json(D/'EXECUTION_ROUTING.json',dict(execution=str(D),source_worktree=str(W),original_execution=str(OLD),original_checkpoint_contract_sha256=sha(D/'FROZEN_SPEC.json'),reference_experiment=str(D),authorized_devices={g4:4,g5:5},document_devices=mapping,baseline_graphics={u:pf['baseline_graphics'][u] for u in [g4,g5]},user_authorization='我下面给你的指令，你可以选用两个空闲的GPU来进行',formal_global_order=read(D/'FROZEN_4DOC_SELECTION.json')['execution_order'],formal_serial=True,qualification_two_lanes=True,original_scientific_spec_operational_fields='HISTORICAL_PROVENANCE_ONLY; THIS_ROUTING_IS_CURRENT_EXECUTION',preflight_window_sha256=pf['devices'][0]['window_sha256']))
cfg=read(OLD/'CALIBRATION_CONFIG.json');cfg['gpu_uuids'].update({str(i):u for u,i in [(g4,4),(g5,5)]});cfg['document_gpu'].update({d:4 if u==g4 else 5 for d,u in mapping.items()})
atomic_json(D/'CALIBRATION_CONFIG.json',cfg)
atomic_json(D/'FP_PAIRING_CONTRACT_GATE.json',dict(status='PASS',spec_sha256=sha(D/'FROZEN_SPEC.json'),mode='NEW_EXECUTION_OWN_FP_PAYLOAD_SAME_PER_DOCUMENT_UUID_RUNTIME',reference_experiment=str(D),FP_execution_and_payload_integrity='PENDING_FORMAL',new_device_qualification='PENDING'))
bindings={};(D/'runs').mkdir();(D/'formal').mkdir();cpchecks=[]
import torch
assert not torch.cuda.is_initialized()
import functional_worker as f
for item in read(OLD/'CHECKPOINT_MANIFEST.json')['checkpoints']:
    doc,rot=item['document'],item['rotation'];key=doc+'__'+rot;p=Path(item['parent_manifest_path']);cp=Path(item['path']);n=p.with_name(p.name.replace('.json','.NATURAL_EXIT.json'))
    assert sha(p)==item['parent_manifest_sha256'] and sha(cp)==item['sha256']
    x=torch.load(cp,map_location='cpu',weights_only=False)
    assert x['schema']=='full_native_hybrid_cache_v1' and x['next_position']==8192 and x['last_position']==8191
    assert x['frozen_spec_sha256']==sha(D/'FROZEN_SPEC.json') and x['token_sha256']==item['token_file_sha256']
    assert x['quantizer_state']['rotation_sha256']==item['rotation_sha256']
    assert x['quantizer_state']['every_token'] and x['quantizer_state']['no_high_precision_shadow']
    assert f.object_fingerprint(torch,x['cache'])==x['cache_fingerprint']
    assert x['recurrent_layers']==cfg['layers'];del x
    copy(p,D/'runs'/p.name);(D/'runs'/cp.name).symlink_to(cp)
    bindings[key]=dict(parent_sha256=sha(p),checkpoint_sha256=sha(cp),original_manifest_path=str(OLD/'RESULT_AND_DELIVERY_HASHES.json'),original_manifest_sha256=sha(OLD/'RESULT_AND_DELIVERY_HASHES.json'),original_natural_exit_path=str(n),original_natural_exit_sha256=sha(n),original_gpu_uuid=item['GPU_UUID'],checkpoint_contract_sha256=sha(D/'FROZEN_SPEC.json'))
    cpchecks.append(dict(document=doc,rotation=rot,checkpoint_sha256=sha(cp),native_cache_fingerprint_exact=True,serialized_spec_token_rotation_history_identity='PASS',target_device_restore='PENDING_REAL_FRESH64'))
    print(json.dumps(dict(event='CPU_CHECKPOINT_IDENTITY_PASS',document=doc,rotation=rot)),flush=True)
atomic_json(D/'REUSED_PARENT_BINDINGS.json',dict(status='PASS_EXACT9_PARENTS_NO_RECOMPUTATION',units=bindings))
atomic_json(D/'CHECKPOINT_REUSE_CPU_AUDIT.json',dict(status='PASS_ACTUAL9_SERIALIZED_NATIVE_CHECKPOINTS',units=cpchecks,CUDA_initialized=torch.cuda.is_initialized(),new_target_device_numerical_qualification='PENDING_REAL_FRESH64',epoch=time.time()))
print('PREPARATION_COMPLETE_NO_GPU_WORKERS',flush=True)
