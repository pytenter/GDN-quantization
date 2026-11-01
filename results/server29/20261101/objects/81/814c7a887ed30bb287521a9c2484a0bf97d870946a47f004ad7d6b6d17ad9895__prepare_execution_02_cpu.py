"""Read-only historical verification, exact score import and new source freeze."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

D=Path('/data/zypan/experiment_runs/LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1/execution_02')
ORIG=D.parent/'execution_01'
W=Path('/data/zypan/worktrees/ling-kda-prospective-engineering-recovery')
sys.path.insert(0,str(W));sys.dont_write_bytecode=True
from recovery_state import sha,atomic_json
from gpu_admission import UUID,classify_window,classify,choose
from gpu_telemetry import observe

def read(p):return json.loads(p.read_text())
def put(p,j):atomic_json(p,j)

assert sha(ORIG/'RESULT_AND_DELIVERY_HASHES.json')=='202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b'
original=read(ORIG/'RESULT_AND_DELIVERY_HASHES.json')
for n,info in original['files'].items():assert sha(ORIG/n)==info['sha256'],n
for n in ['GPU_ADMISSION_TEST_REPORT.json','CONTROLLER_RESUME_TEST_REPORT.json']:assert read(D/n)['status']=='PASS'
assert not (D/'SOURCE_CODE_FREEZE.json').exists()
assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
rootaudit=read(ORIG/'PREREGISTRATION_RECONCILIATION.json');assert rootaudit['status']=='PASS'
for row in rootaudit['source_assets']:
    st=Path(row['path']).stat();assert (st.st_size,st.st_mtime_ns)==(row['bytes'],row['mtime_ns']),row['path']
    if row['verification']=='FRESH_SHA':assert sha(Path(row['path']))==row['sha256']
pool=read(ORIG/'preregistration/PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json')
raw_checked=token_checked=0
for row in pool['inputs']:
    assert sha(Path(row['raw_text_path']))==row['raw_sha256'];raw_checked+=1
    assert sha(Path(row['token_file_path']))==row['token_file_sha256'];token_checked+=1
assert raw_checked==token_checked==100
for n in ['PREREGISTRATION_RECONCILIATION.json','INPUT_ELIGIBILITY_REPRODUCTION.json','CALIBRATION_CONFIG.json','ANALYSIS_PLAN.json','SECONDARY_SPEC.json','MODEL_WEIGHT_SHA256.txt']:
    shutil.copyfile(ORIG/n,D/n)
shutil.copytree(ORIG/'preregistration',D/'preregistration')
(D/'screening').mkdir();(D/'runs').mkdir();(D/'formal').mkdir()
copied={}
for n,info in original['files'].items():
    if n.startswith('screening/'):
        shutil.copyfile(ORIG/n,D/n);assert sha(D/n)==info['sha256'];copied[n]=info['sha256']
ds=sorted((read(p) for p in (D/'screening').glob('*.SELECTION_DECISION.json')),key=lambda r:r['scan_index'])
assert [d['document'] for d in ds]==read(D/'INPUT_ELIGIBILITY_REPRODUCTION.json')['eligible_order'][:3]
rule={};exec((W/'next_prospective_pair_rule.py').read_text(),rule)
for d in ds:
    r=read(D/'screening'/f"{d['document']}.json")
    assert r['status']=='PASS' and len(r['results'])==16 and sha(D/'screening'/f"{d['document']}.json")==d['short_result_sha256']
    assert {x['condition']:{k:x['aggregate'][k] for k in ['P1','L7_FUNCTIONAL_ORIGINAL','L7_STATE_ORIGINAL','L7_TOTAL_ORIGINAL']} for x in r['results']}==d['scores']
    a=[d['scores'][c]['P1'] for c in d['roster']];b=[d['scores'][c]['L7_TOTAL_ORIGINAL'] for c in d['roster']]
    assert dict(zip(d['roster'],rule['midranks'](a)))==d['P1_midranks']
    assert dict(zip(d['roster'],rule['midranks'](b)))==d['L7_midranks']
    assert rule['opposite_pairs'](d['roster'],a,b,3)==d['eligible_pairs'] and d['selected_pair']==d['eligible_pairs'][0]
put(D/'CONTROLLER_RESUME_AUDIT.json',dict(status='PASS_EXACT_48_RESTORED',original_execution=str(ORIG),original_manifest_sha256=sha(ORIG/'RESULT_AND_DELIVERY_HASHES.json'),
    original_source_commit='7713fd20068479ccd7594b47b56fe540a93bde3f',new_execution=str(D),new_source_commit=head,
    imported_files_sha256=copied,scan_cursor=3,short_score_count=48,eligible_documents=[d['document'] for d in ds],
    next_frozen_document=read(D/'INPUT_ELIGIBILITY_REPRODUCTION.json')['eligible_order'][3],new_GPU_work_used_for_import=0,epoch=time.time()))
critical={}
for name,names in {'functional_worker.py':['transfer','object_fingerprint','audit_state','restore_audit','HiddenObserver','checkpoint'],
                   'formal_worker.py':['original_state_diagnostic','observe_qdq','collect_metrics']}.items():
    old=ast.parse((ORIG/name).read_text());new=ast.parse((W/name).read_text())
    for fun in names:
        x=next(n for n in ast.walk(old) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==fun)
        y=next(n for n in ast.walk(new) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==fun)
        assert ast.dump(x)==ast.dump(y),(name,fun);critical[name+':'+fun]=hashlib.sha256(ast.dump(x).encode()).hexdigest()
same={}
for n in ['persistent_engine_v1.py','objective.py','run_matched_l7.py','next_prospective_pair_rule.py','v2_runtime.py','v3_guard.py']:
    assert sha(W/n)==sha(ORIG/n);same[n]=sha(W/n)
put(D/'NUMERICAL_SOURCE_COMPATIBILITY_AUDIT.json',dict(status='PASS_UNCHANGED_MATH_BOUNDARY',unchanged_files=same,unchanged_AST=critical,
    changes='GPU admission/route, orchestration/recovery, worker exit receipts and CPU delivery only; no numeric helper, quantizer, scale/clipping, KL reducer, scoring windows or roster change',
    own12_parent_fresh64_qualification='PENDING_BEFORE_LONG_HORIZON'))
files={p.name:sha(p) for p in W.glob('*.py')}
if (D/'gpu_telemetry.py').exists():(D/'gpu_telemetry.py').rename(D/'initial_GPU_diagnostic_script.py')
for n in files:(D/n).symlink_to(W/n)
put(D/'SOURCE_CODE_FREEZE.json',dict(status='FROZEN_BEFORE_FIRST_EXECUTION_02_GPU_WORK',source_commit=head,source_worktree=str(W),
    base_commit='7713fd20068479ccd7594b47b56fe540a93bde3f',config_sha256=sha(D/'CALIBRATION_CONFIG.json'),files=files,frozen_epoch=time.time(),
    original48_source_commit_preserved=True,scientific_identity_unchanged=True))
code=read(ORIG/'CODE_FREEZE.json');code['functional_worker.py']=files['functional_worker.py'];put(D/'CODE_FREEZE.json',code)
formal=read(ORIG/'FORMAL_CODE_FREEZE.json');formal['files']=files;put(D/'FORMAL_CODE_FREEZE.json',formal)
(D/'SOURCE_DIFF.patch').write_text(subprocess.check_output(['git','diff','7713fd20068479ccd7594b47b56fe540a93bde3f','HEAD'],cwd=W,text=True))
put(D/'RECOVERY_ASSET_AUDIT.json',dict(status='PASS',verified_original_manifest_SHA=sha(ORIG/'RESULT_AND_DELIVERY_HASHES.json'),
    verified_original_small_files=len(original['files']),raw_input_SHA_fresh=raw_checked,token_file_SHA_fresh=token_checked,
    original100_retokenization_reused_with_exact_bytes=True,source_weight_SHA_scope='original qualified SHA plus unchanged size/mtime; other designated sources fresh SHA',
    disk_free_bytes=shutil.disk_usage(D).free,new_source_commit=head,new_GPU_work=0))
rows=observe([UUID],D/'GPU_V2_QUALIFICATION_WINDOW.jsonl')
state=classify_window(rows)
initial=[json.loads(l) for l in (D/'GPU_DIAGNOSTIC_WINDOW.jsonl').read_text().splitlines()]
legacy=read(ORIG/'STOP.json');last=json.loads((ORIG/'CONTROLLER_GPU_ADMISSION.jsonl').read_text().splitlines()[-1])
put(D/'GPU_ADMISSION_DIAGNOSIS.json',dict(status=state,historical_root_cause='UNKNOWN',historical_failure=legacy,
    historical_sample=last,verified_same_UUID=UUID,GPU_changed=False,current_window_samples=len(rows),current_window_states=[classify(r) for r in rows],
    direct_NVML_memory_v2=True,system_reserved_memory_separately_accounted=True,
    driver_Xid_log_status='UNKNOWN_PERMISSION_LIMITED',PID_namespace_relation='UNKNOWN_PROC1_PERMISSION_LIMITED',
    initial_device_observation_count=len(initial),no_reset_or_other_process_kill=True,numerical_history_contract='PRIOR_SAME_UUID_REUSED_12_OWN_HISTORIES_STILL_REQUIRED'))
assert state=='AVAILABLE_VERIFIED',('GPU_ADMISSION_UNRESOLVED_STOP',state)
assert choose({UUID:state},{UUID:True})==UUID
put(D/'INITIAL_GPU_SELECTION.json',dict(status='AVAILABLE_VERIFIED',uuid=UUID,index=rows[-1]['target'][0]['index'],
    selection='preferred original qualified UUID currently admitted; alternatives have no numerical PASS and are not used',
    own12_long_qualification='PENDING',no_cross_GPU_checkpoint_reuse=True))
assert sha(ORIG/'RESULT_AND_DELIVERY_HASHES.json')=='202bf8f64d7a2bfeab93451a607d575f7cc7052ab96692d88de02176f536516b'
print(json.dumps(dict(status='EXECUTION_02_READY',source_commit=head,restored_scores=48,next_document=read(D/'CONTROLLER_RESUME_AUDIT.json')['next_frozen_document'],GPU_admission=state)))
