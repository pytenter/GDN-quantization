"""Conditional V4 CPU preparation after completed five-process exact qualification."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess
R=Path.cwd();C=R/'experiments/LING_FUSED_RMSNORM_GATED_DETERMINISM_CLOSURE_V1'
V=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V3_CONTEMPORARY'
D=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V4_DETERMINISTIC'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def emit(p,x):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
assert load(C/'reports/FINAL_VERDICT.json')['DETERMINISTIC_TRAINING_RUNTIME_QUALIFICATION']=='PASS'
assert not D.exists() and not subprocess.check_output(['git','status','--porcelain']).strip()
parents=load(C/'provenance/protected_parent_inventory.json')['files']
for p in C.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:parents.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
for r in parents:assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes'],r['path']
subprocess.check_call(['git','switch','-c','exp/ling-l7r-l8-layer-aware-matched-training-v4-deterministic'])
D.mkdir();emit(Path('provenance/protected_parent_inventory.json'),dict(gate='PASS',files=parents,parent_closure_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
for folder in ['configs','protocol']:
    dest=D/folder;dest.mkdir(exist_ok=True)
    for p in (V/folder).glob('*.json'):shutil.copyfile(p,dest/p.name)
cfgs={k:load(D/'configs'/k) for k in ['l7r.json','l8.json']}
for k,v in cfgs.items():assert v==load(V/'configs'/k) and sha(D/'configs'/k)==sha(V/'configs'/k)
identity=load(V/'protocol/fresh_b_v3_identity.json');assert identity['FRESH_B_PRIOR_TREATMENT_FORWARD']==0
for row in identity['tokens']:assert sha(row['destination'])==row['sha256']
emit(Path('protocol/fresh_b_v4_identity.json'),dict(FRESH_B_V4_IDENTITY_GATE='PASS',prior_treatment_forwards=0,same_selection_manifest=sha(D/'protocol/fresh_b_selection_manifest.json')==identity['selection_manifest_sha256'],same_tokens=True,reselection=False,parent_identity_sha256=sha(V/'protocol/fresh_b_v3_identity.json'),tokens=identity['tokens']))
emit(Path('protocol/v3_v4_scientific_identity.json'),dict(V3_V4_SCIENTIFIC_PROTOCOL_IDENTITY_GATE='PASS',configs_byte_identical=True,train_manifest_byte_identical=True,validation_manifest_byte_identical=True,checkpoint_rule_byte_identical=True,optimizer_and_draw_order_unchanged=True,only_intervention='early learnability L7R all18 versus L8 early6 fixedH',only_runtime_change='qualified deterministic launch profile',training_data_root=str(V/'inputs/traces'),legacy_source_root=str(V/'inputs/runtime_repo'),historical_trainer=str(V/'inputs/historical_trainer.py'),historical_trainer_sha256=sha(V/'inputs/historical_trainer.py')))
launch=load(C/'runtime/diagnostic_launch_manifest_attempt2.json');hm=load(C/'runtime/historical_fixed_forward_manifest.json')
bound=[load(C/('analysis/whole_boundaries_P'+str(i)+'.json')) for i in range(1,6)]
# Freeze an ACTUAL qualified selection, not a hand-written or searched kernel config.
selections=[]
for b in bound:
    q=b['norm_backward_launches'][-1];assert q['kernel']=='layer_norm_gated_bwd_kernel'
    chosen=dict(kwargs={k:q['launch'][k] for k in ['BT']},**{k:q['compiled'][k] for k in ['num_warps','num_stages','num_ctas']})
    selections.append(chosen)
assert all(x==selections[0] for x in selections)
runtime=D/'runtime';runtime.mkdir(exist_ok=True);shutil.copytree(C/'runtime/historical',runtime/'historical')
shutil.copyfile(C/'runtime/historical_fixed_forward_manifest.json',runtime/'historical_fixed_forward_manifest.json')
emit(Path('runtime/qualified_backward_selection.json'),dict(kernel='layer_norm_gated_bwd_kernel',configuration=selections[0],source='last actual compiled launch in each of the five bitwise-identical whole updates',all_five_same=True,historical_backward_fix_claim=False,prospective_runtime_freeze=True,configuration_search_performed=False,sources=[dict(path=str(C/('analysis/whole_boundaries_P'+str(i)+'.json')),sha256=sha(C/('analysis/whole_boundaries_P'+str(i)+'.json'))) for i in range(1,6)]))
emit(Path('runtime/v4_deterministic_runtime_manifest.json'),dict(gate='PASS',qualification='PASS',qualification_manifest_sha256=sha(C/'manifests/final_sha256.json'),qualification_verdict_sha256=sha(C/'reports/FINAL_VERDICT.json'),package_environment=load(C/'runtime/environment_identity.json'),source_freeze=load(C/'runtime/source_freeze_attempt2.json'),historical_fixed_forward=hm,qualified_backward_selection=selections[0],launch_profile=launch['env'],python=launch['executable'],autotune_allowed_to_change_formal_config=False,training_hardware='GPU0_ONLY_SEQUENTIAL',training_order=['L7R','L8'],Fresh_B_hardware='2xRTX4090 same document all conditions same GPU',AIME_endpoint_restore='historical canonical manifest only; no assumed training-kernel injection'))
for name in ['v4_common.py','v4_readiness.py','v4_run_readiness.py']:
    p=Path('/tmp/rmsnorm_closure_staging')/name;dest=D/'scripts'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
shutil.copyfile('/tmp/rmsnorm_closure_staging/v4_bootstrap.py',D/'scripts/v4_bootstrap.py')
files=[]
for p in sorted((D/'scripts').glob('*.py')):compile(p.read_text(),str(p),'exec');files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
for r in load(C/'runtime/source_freeze_attempt2.json')['files']:
    p=Path(r['path']);assert sha(p)==r['sha256'];files.append(r)
emit(Path('runtime/readiness_source_freeze.json'),dict(gate='PASS',files=files,GPU_measurements_before=0))
print('V4_CPU_PREPARATION_PASS',selections[0],flush=True)
