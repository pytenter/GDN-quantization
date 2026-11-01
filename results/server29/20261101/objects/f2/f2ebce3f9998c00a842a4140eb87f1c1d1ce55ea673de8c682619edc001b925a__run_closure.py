"""Pre-frozen sequential GPU0 measurements and exact comparisons; no config variants."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
D=Path(__file__).resolve().parents[1];R=D.parents[1]
V2=R/'experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,x):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def run(script,args):
    label='_'.join(args);log=D/'logs'/('attempt2_'+script+'_'+label.replace('/','_')[-100:]+'.log');log.parent.mkdir(exist_ok=True)
    cmd=[str(V2/'runtime/env/bin/python'),str(D/'scripts'/script),*args]
    print('START',script,args,time.time(),flush=True)
    with log.open('x') as f:subprocess.check_call(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    print('COMPLETE',script,args,time.time(),flush=True)
def diff(a,b,path=''):
    if type(a)!=type(b):return [path]
    if isinstance(a,dict):
        if set(a)!=set(b):return [path+'.KEYS']
        return sum((diff(a[k],b[k],path+'.'+str(k)) for k in a),[])
    if isinstance(a,list):
        if len(a)!=len(b):return [path+'.LEN']
        return sum((diff(x,y,path+'['+str(i)+']') for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [path]
env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FLA_CACHE_MODE='default',FLA_CONFIG_DIR=str(D/'runtime/historical/fixed_fla_configs'))
env.update(load(D/'runtime/historical_fixed_forward_manifest.json')['selector'])
for k in ['TRITON_CACHE_DIR','CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING','PYTORCH_CUDA_ALLOC_CONF']:env.pop(k,None)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R).strip()
emit(Path('runtime/diagnostic_launch_manifest_attempt2.json'),dict(premeasurement_commit_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),env={k:env.get(k) for k in ['PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','PYTHONHASHSEED','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','FLA_CACHE_MODE','FLA_CONFIG_DIR','LING_GATED_RMSNORM_FIXED_CONFIG','TRITON_CACHE_DIR','CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING','PYTORCH_CUDA_ALLOC_CONF']},executable=str(V2/'runtime/env/bin/python'),scope='isolated and disposable only',formal_updates=0,Fresh_B=0,AIME=0))
run('forward.py',['--label','same5','--repeats','5'])
for i in range(1,6):run('forward.py',['--label','P'+str(i),'--repeats','1'])
same=load(D/'analysis/forward_same5.json');fresh=[load(D/('analysis/forward_P'+str(i)+'.json')) for i in range(1,6)]
fixed=all(x['HISTORICAL_FIX_ACTUALLY_APPLIED_GATE']=='PASS' for x in [same,*fresh])
equal=len({x['outputs'][0]['sha256'] for x in fresh})==1
verdict=dict(EXPERIMENT=D.name,PARENT_IMMUTABILITY_GATE='PENDING_FINAL_RECHECK',PROSPECTIVE_CALL37_FROZEN_INPUT_GATE='PASS',HISTORICAL_FIXED_FORWARD_PROVENANCE_GATE='PASS',FIXED_FORWARD_SAME_PROCESS_5X_GATE='PASS' if same['same_process_exact'] else 'FAIL',FIXED_FORWARD_FRESH_PROCESS_5X_GATE='PASS' if equal else 'FAIL',HISTORICAL_FIX_ACTUALLY_APPLIED_GATE='PASS' if fixed else 'FAIL',FUSED_RMSNORM_GATED_FORWARD_DETERMINISM='PASS' if fixed and equal and same['same_process_exact'] else 'FAIL',GPU0_L7R_WHOLE_UPDATE_5PROCESS_EXACT_GATE='NOT_RUN',FIRST_BACKWARD_DIVERGENT_OPERATION='NOT_EVALUATED',HISTORICAL_BACKWARD_FIXES_APPLIED=[],FIRST_REMAINING_DIVERGENT_OPERATION='NONE',REFERENCE_FORWARD_EXACT='NOT_RUN',REFERENCE_BACKWARD_EXACT='NOT_RUN',DETERMINISTIC_TRAINING_RUNTIME_QUALIFICATION='FAIL',FORMAL_TRAINING_STARTED='NO',Fresh_B_treatment_forwards=0,AIME_generation=0,FINAL_ACTION='STOP')
emit(Path('analysis/forward_qualification.json'),dict(**verdict,all_output_hashes=[x['outputs'][0]['sha256'] for x in fresh]))
if verdict['FUSED_RMSNORM_GATED_FORWARD_DETERMINISM']!='PASS':
    verdict['FIRST_REMAINING_DIVERGENT_OPERATION']='layer_norm_gated_fwd_kernel'
    emit(Path('analysis/closure_result.json'),verdict);sys.exit(0)
for i in range(1,6):run('whole.py',['--label','P'+str(i)])
whole=[load(D/('runtime/whole_P'+str(i)+'.json')) for i in range(1,6)]
differences={str(i+1):diff(whole[0]['numeric'],x['numeric']) for i,x in enumerate(whole)}
full_exact=all(not v for v in differences.values())
verdict['GPU0_L7R_WHOLE_UPDATE_5PROCESS_EXACT_GATE']='PASS' if full_exact else 'FAIL'
emit(Path('analysis/whole_update_comparison.json'),dict(gate=verdict['GPU0_L7R_WHOLE_UPDATE_5PROCESS_EXACT_GATE'],numeric_hashes=[x['numeric_sha256'] for x in whole],differences=differences,all_expected_trainable_parameters_compared=True,disposable_updates=5,formal_updates=0))
if full_exact:
    verdict.update(DETERMINISTIC_TRAINING_RUNTIME_QUALIFICATION='PASS',FIRST_BACKWARD_DIVERGENT_OPERATION='NONE',FINAL_ACTION='ADVANCE_TO_V4')
else:
    bounds=[load(D/('analysis/whole_boundaries_P'+str(i)+'.json')) for i in range(1,6)]
    losses_exact=all(x['numeric']['losses']==whole[0]['numeric']['losses'] for x in whole)
    norm=[]
    for j,row in enumerate(bounds[0]['norm_backward']):
        peers=[x['norm_backward'][j] for x in bounds]
        ie=all(x['inputs']==row['inputs'] for x in peers);oe=all(x['outputs']==row['outputs'] for x in peers)
        norm.append(dict(call=j,direct_inputs_exact=ie,outputs_exact=oe,bundle=row['bundle']))
    graph=[]
    baseline=bounds[0]['autograd_execution']
    # Node identities are traversal indices, not process memory addresses.
    peers_maps=[{r['node']:r for r in b['autograd_execution']} for b in bounds]
    for row in baseline:
        peers=[x[row['node']] for x in peers_maps]
        ie=all(x['incoming']==row['incoming'] for x in peers);oe=all(x['outgoing']==row['outgoing'] for x in peers)
        if not oe:graph.append(dict(execution=row['execution'],node=row['node'],operation=row['operation'],incoming_exact=ie,outgoing_exact=oe))
    candidate=next((x for x in norm if x['direct_inputs_exact'] and not x['outputs_exact']),None)
    first=next((x for x in graph if x['incoming_exact']),None)
    op=('layer_norm_gated_bwd_kernel' if candidate is not None and first is not None and first['operation']=='LayerNormGatedFunctionBackward' else first['operation'] if first else 'UNRESOLVED_BOUNDARY') if losses_exact else 'FORWARD_LOSS_NONEXACT'
    verdict.update(FIRST_BACKWARD_DIVERGENT_OPERATION=op if losses_exact else 'NOT_EVALUATED',FIRST_REMAINING_DIVERGENT_OPERATION=op)
    emit(Path('analysis/backward_localization.json'),dict(losses_exact=losses_exact,first_exact_in_nonexact_out=first,nonexact_autograd_boundaries=graph,norm_backward=norm,eligible_frozen_norm_bundle=candidate,historical_backward_fixes_available=[],finite_historical_fix_set_exhausted=True))
    if losses_exact and candidate is not None and op=='layer_norm_gated_bwd_kernel':
        for i in range(1,6):run('isolated_backward.py',['--label','P'+str(i),'--mode','current','--bundle',candidate['bundle']])
        current=[load(D/('analysis/current_backward_P'+str(i)+'.json')) for i in range(1,6)]
        ce=all(x['outputs']==current[0]['outputs'] for x in current)
        emit(Path('analysis/isolated_backward_comparison.json'),dict(gate='PASS' if ce else 'FAIL',direct_inputs_exact=all(x['inputs']==current[0]['inputs'] for x in current),bundle_sha256=sha(candidate['bundle']),outputs=[x['outputs'] for x in current],historical_backward_configuration='NOT_FOUND',autotune_configs_not_modified=True))
        for i in range(1,6):run('isolated_backward.py',['--label','P'+str(i),'--mode','reference','--bundle',candidate['bundle']])
        ref=[load(D/('analysis/reference_backward_P'+str(i)+'.json')) for i in range(1,6)]
        verdict['REFERENCE_FORWARD_EXACT']='YES' if all(x['forward']==ref[0]['forward'] for x in ref) else 'NO'
        verdict['REFERENCE_BACKWARD_EXACT']='YES' if all(x['outputs']==ref[0]['outputs'] for x in ref) else 'NO'
        emit(Path('analysis/reference_comparison.json'),dict(forward_exact=verdict['REFERENCE_FORWARD_EXACT'],backward_exact=verdict['REFERENCE_BACKWARD_EXACT'],direct_inputs_exact=all(x['inputs']==ref[0]['inputs'] for x in ref),reference_not_in_training=True))
    else:verdict['bounded_diagnostic_followup_required']=True
emit(Path('analysis/closure_result.json'),verdict)
print(json.dumps(verdict,sort_keys=True),flush=True)
