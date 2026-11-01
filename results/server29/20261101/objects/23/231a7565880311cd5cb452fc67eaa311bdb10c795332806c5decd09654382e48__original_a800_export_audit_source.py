#!/usr/bin/env python3
"""Reuse the archived scientific runner verbatim; amend identities and audit destinations only."""
import csv
import hashlib
import inspect
import io
import json
from pathlib import Path
import traceback
import posttrain_closure_audit as A

ROOT=A.ROOT

def effective(E):
    p=A.read('analysis/posttrain_cpu_prerequisites.json');A.require(p['status']=='PASS','CPU prerequisites')
    guard=ROOT/'qualification/posttraining_closure_immutable_guards.jsonl'
    rows=[json.loads(l) for l in guard.read_text().splitlines()]
    A.require(rows[-1]['point']=='FRESH_FULL_GPU_INTEGRITY' and rows[-1]['status']=='PASS' and rows[-1]['tensor_count']==429,'fresh full GPU parameter integrity')
    A.require(E.torch.version.cuda=='12.1','actual CUDA runtime differs')
    indexing=A.read('readiness/final_hidden_indexing_audit.json');fp=A.read('qualification/fp_runtime_stability.json')
    A.require(indexing['status']=='PASS' and len(indexing['records'])==192,'Qwen hidden indexing coverage')
    A.require(all(r['exact_lm_head_input'] and r['input_token_index']==r['hidden_position']==r['logit_position']==r['next_token_target_index']-1 for r in indexing['records']),'Qwen final normalized lm_head indexing')
    A.require(fp['status']=='PASS' and fp['LM_HEAD_FORWARD_REFERENCE_PARITY']=='PASS','LM head direct reference gate')
    A.require(all(s['reference_exact'] for run in fp['runs'] for s in run),'lm_head captured-hidden/logit exact reference')
    A.save('provenance/posttrain_effective_provenance.json',dict(p,status='PASS',POSTTRAIN_EFFECTIVE_PROVENANCE_GATE='PASS',QWEN_FINAL_HIDDEN_INDEXING_GATE='PASS',model_GPU_integrity='PASS',fresh_GPU_guard_sha256=A.sha(guard),fresh_CPU_audit_sha256=A.sha(ROOT/'qualification/posttraining_closure_cpu_checkpoint.json'),actual_cuda_runtime=E.torch.version.cuda,teacher_cache_rebuild=False,scientific_runner='unmodified archived amended_a800_posttraining.main()',audit_before_first_Shared16_model_computation=True))

def exports(E):
    rows={}
    for arm in E.ARMS:
        s=A.read(f'training/{arm}/checkpoint_selection.json')['selected']
        theta=E.torch.load(s['theta_path'],map_location='cpu',weights_only=False)
        frozen=E.torch.load(s['frozen_delta_path'],map_location='cpu',weights_only=False)
        A.require(frozen['theta_checkpoint_sha256']==s['theta_sha256'] and frozen['update']==theta['update']==s['update'] and frozen['exposure']==theta['exposure']==s['exposure'],'export provenance')
        bank=E.fresh_bank('cuda:0');bank.load_state_dict(theta['bank'],strict=True);layers=[]
        with E.torch.no_grad():
            for l in E.core.GDN_LAYERS:
                v=frozen['matrices'][str(l)]
                A.require(v.dtype==E.torch.float32 and v.shape==(128,128) and bool(E.torch.isfinite(v).all()),'export shape/dtype/finite')
                materialized=bank.layer(l).matrix().detach().float().cpu().contiguous().clone()
                exact=E.torch.equal(materialized,v);orth=float((v.T@v-E.torch.eye(128)).abs().max())
                A.require(exact,'checkpoint to frozen export NOT EXACT; no tolerance authorized')
                A.require(orth<=E.core.ORTHOGONALITY_THRESHOLD,'frozen orthogonality')
                h=E.runtime.tensor_sha256(v);A.require(h==s['per_layer_sha256'][str(l)],'layer hash')
                layers.append({'layer':l,'shape':[128,128],'dtype':str(v.dtype),'sha256':h,'orthogonality_max_abs':orth,'checkpoint_loaded_vs_frozen_exact':exact})
        del bank;E.M.clean()
        rows[arm]={'status':'PASS','EXPORT_PARITY_GATE':'PASS','rule':'original training bank.load_state_dict + layer.matrix().detach().float().cpu().contiguous().clone(); comparison only, no file rematerialization','existing_export_rewritten':False,'path':s['frozen_delta_path'],'sha256':s['frozen_delta_sha256'],'source_checkpoint_path':s['theta_path'],'source_checkpoint_sha256':s['theta_sha256'],'update':s['update'],'exposure':s['exposure'],'layers':layers,'orthogonality_threshold':E.core.ORTHOGONALITY_THRESHOLD}
    A.save('analysis/frozen_rotation_export_audit.json',{'status':'PASS','conditions':rows,'backward_calls':0,'optimizer_step_calls':0,'new_export_files':0})

def main():
    A.require(A.read('analysis/posttrain_cpu_prerequisites.json')['status']=='PASS','preaudit incomplete')
    import amended_a800_posttraining as P
    E=P.E
    old=ROOT/'scripts/amended_a800_posttraining.py'
    A.require(A.sha(old)==A.OLD_RUNNER,'archived runner bytes changed')
    original_expected=dict(P.EXPECTED)
    amendment=A.read('provenance/posttrain_hash_constant_amendment_v1.json')
    P.EXPECTED=dict(amendment['resolved_EXPECTED'])
    original_once,original_J,original_load,original_guard=E.once,E.J,E.load,E.guard
    original_table=P.table
    destinations={ROOT/'logs/posttraining.completion.json':ROOT/'logs/posttrain_closure_runner.completion.json',ROOT/'logs/posttraining.error.json':ROOT/'logs/posttrain_closure_runner.error.json'}
    def once(path):return original_once(destinations.get(path,path))
    def write(path,value):return original_J(destinations.get(path,path),value)
    def table(path,rows,fields):
        if path==ROOT/'analysis/cross_objective_learning_curves.csv' and path.exists():
            b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
            A.require(path.read_bytes()==b.getvalue().encode(),'preserved old curve differs from same archived runner output')
            return
        return original_table(path,rows,fields)
    def guard(model,sources,stage,point):return original_guard(model,sources,'posttraining_closure' if stage=='posttraining' else stage,point)
    def load(stage):
        result=original_load('posttraining_closure' if stage=='posttraining' else stage)
        effective(E)
        return result
    E.once,E.J,E.load,E.guard,P.table=once,write,load,guard,table
    # Explicit audit: no replacement of any metric, teacher, QDQ, model-loading implementation or main scientific function.
    references={'archived_main':A.sha(old),'engine':A.sha(ROOT/'scripts/a800_matched.py'),'archived_payload_function':hashlib.sha256(inspect.getsource(P.payload).encode()).hexdigest(),'evaluate_function':hashlib.sha256(inspect.getsource(E.evaluate).encode()).hexdigest(),'cached_teacher_function':hashlib.sha256(inspect.getsource(E.cached).encode()).hexdigest(),'original_load_function':hashlib.sha256(inspect.getsource(original_load).encode()).hexdigest(),'formal_frozen_inputs':E.protocol()['frozen_inputs']}
    A.require(references['engine']==A.ENGINE and references['archived_main']==A.OLD_RUNNER,'scientific function identity')
    A.save('analysis/posttrain_runner_scientific_path_identity.json',{'status':'PASS','POSTTRAIN_RUNNER_SCIENTIFIC_PATH_IDENTITY_GATE':'PASS','method':'unchanged archived main/evaluate/payload/cached and scientific modules; wrapper changes only EXPECTED resolution and audit artifact I/O','old_expected':original_expected,'corrected_expected':P.EXPECTED,'source_references':references,'wrapper_sha256':A.sha(Path(__file__)),'changes':['EXPECTED resolved from independently proven append-only amendment','completion/error and immutable-guard output filenames separated from failed historical attempt','existing learning-curves output checked byte-for-byte instead of overwritten','new provenance and export verification artifacts'],'unchanged':['model loading computation','teacher cache loading','Hadamard and all seven rotations','C128 INT8 QDQ','recurrent writeback','Shared16 order/captures','State/Functional/Final Hidden relative-MSE'],'not_used':['KL','NLL','margin'],'forbidden_calls':{'backward':0,'optimizer_step':0,'training':0,'new_AIME':0,'3090_deployment':0}})
    P.main()
    exports(E)
    print('POSTTRAIN_SHARED16_AND_EXPORT=PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        A.save('logs/posttrain_closure_runner.error.json',{'status':'FAIL','error':str(e),'traceback':traceback.format_exc(),'do_not_retry':True});raise
