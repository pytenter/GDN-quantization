"""CPU-only pre-outcome identity and historical reuse audit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import pilot_engineering as P

def main():
    v=P.V3; recovery=v/'recovery/attempt_002'
    receipts=[]
    for name in ['NEW_NUMERICAL_CONTRACT.json','MULTITOKEN_FULL_MODEL_PARITY.json','FRESH_PROCESS_DETERMINISM.json',
                 'OBSERVER_PARITY_REPORT.json','QUALIFICATION_POLICY.json','recovery/attempt_002/COMPLETE.json',
                 'recovery/attempt_002/QUALIFICATION_REUSE.json','recovery/attempt_002/SCIENTIFIC_VERDICT.json']:
        path=v/name
        receipts.append({'path':str(path),'sha256':P.sha(path),'data':json.loads(path.read_text())})
    P.check(json.loads((v/'NEW_NUMERICAL_CONTRACT.json').read_text())['status']=='PASS','reuse numerical PASS')
    identities=json.loads((recovery/'MODEL_CHECKPOINT_IDENTITY_REUSE.json').read_text())
    for row in identities['model_files']:
        st=Path(row['path']).stat()
        P.check(st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns'],'unchanged sealed checkpoint stat '+row['path'])
    audit=json.loads((v/'EXISTING_ASSET_REUSE_AUDIT.json').read_text())
    rotation=next(r for r in audit['rotations']['anchors'] if r['condition']=='C6R')
    P.check(P.sha(rotation['export_path'])==rotation['export_file_sha256'],'C6R full export SHA')
    sources={str(path):P.sha(path) for path in [P.N.ARCH,P.N.FORMAL,v/'scripts/numerical_recovery.py',
              v/'scripts/scale_science.py',v/'scripts/matched_history.py',P.S.CORE]}
    direction=Path('/data/zypan/worktrees/qwen-gdn-fd-scale-feedback-direction-v1/experiments/QWEN_GDN_FD_SCALE_FEEDBACK_DIRECTION_V1/scripts/direction_replay.py')
    sources[str(direction)]=P.sha(direction)
    P.save('V3_NUMERICAL_CLOSURE_REUSE.json',{'status':'PASS_FOR_PINNED_REUSE_ONLY','receipts':receipts,
        'historical_native_bitwise':'FAIL_PRESERVED','historical_fixed_fullmodel':'FAIL_PRESERVED',
        'initial_V3_document_pairing_STOP':'PRESERVED','fullmodel_closed_loop_fork_and_oracle':'NEW_QUALIFICATION_REQUIRED'})
    P.save('SOURCE_AND_RUNTIME_IDENTITIES.json',{'task':P.TASK,'utc':P.N.now(),'sources':sources,
        'V3_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=v.parent.parent,text=True).strip(),
        'pilot_initial_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.REPO,text=True).strip(),
        'model_identity':identities,'model_aggregate_SHA':P.digest([{k:r[k] for k in ['path','bytes','sha256']} for r in identities['model_files']]),
        'C6R':rotation,'quantizer_source_SHA':P.N.FORMAL_SHA,'quantizer_function_SHA':'6fcbb0aab0aa9d5a324fc1b5fd36a177ba1f7ffa086c6b13f8bd8f9f9e9de82b',
        'runtime_contract':{'torch':'2.5.1+cu121','cuda':'12.1','deterministic':True,'warn_only':False,
            'TF32':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8','scan':'V3 FP32 compensated','device_map':{'entire_model':'GPU0'},'GPU_UUID':P.S.UUIDS[0]},
        'working_tree_isolation':'independent Git checkout; Git1.8.3.1 lacks linked worktree; historical alternates not modified'})
    P.save('INITIAL_RESOURCE_OBSERVATION.json',{'utc':P.N.now(),'resources':P.N.resources(),
        'no_fixed_GPUh_cap':True,'no_VRAM_percent_cap':True,'not_32K_feasibility_proof':True})
    print('PROVENANCE_REUSE_AUDIT_PASS',flush=True)

if __name__=='__main__': main()
