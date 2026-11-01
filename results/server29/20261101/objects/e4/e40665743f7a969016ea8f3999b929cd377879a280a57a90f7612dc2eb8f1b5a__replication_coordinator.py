"""Fixed 8-document serial schedule; fail closed; classify only after all units."""
import json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from archive_assets import sha, save, load, preflight, remote_py, REMOTE_BASE
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent
OLD=Path('/data/zypan/worktrees/qwen-gdn-scale-feedback-fullmodel-functional-causal-pilot-v1/experiments/QWEN_GDN_SCALE_FEEDBACK_FULLMODEL_FUNCTIONAL_CAUSAL_PILOT_V1')
PYTHON='/data/ydai/miniconda3/envs/bitdecode/bin/python'
ARMS=['PARENT','NO_SCALE_INFLATION'];UUID='GPU-a78176ed-f97f-c54f-6201-5a01b570f422'
RESOLUTION=0.000690448106979602

def event(phase,**x):
    p=ROOT/'execution/COORDINATOR_EVENTS.jsonl';p.parent.mkdir(exist_ok=True)
    with p.open('a') as f:f.write(json.dumps({'phase':phase,'time_unix':time.time(),**x},sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def gpu():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    row=out.splitlines()[0].split(',');require(row[2].strip()==UUID,'GPU0_UUID_IDENTITY_FAILURE')
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    require(UUID not in proc,'GPU0_CURRENT_OCCUPANCY_FAILURE')
    return {'gpus_csv':out,'processes_csv':proc,'observed_unix':time.time()}

def unit(args,logname,cpu=False):
    require(not (ROOT/'FIRST_FAILURE_GATE.json').exists(),'FIRST_FAILURE_FORBIDS_NEW_UNIT')
    if not cpu:gpu()
    log=ROOT/'logs'/logname;log.parent.mkdir(exist_ok=True);require(not log.exists(),'UNIT_ALREADY_ATTEMPTED_NO_RESTART')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else '0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    command=[PYTHON]+args;start=time.time()
    with log.open('x') as f:
        p=subprocess.Popen(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        event('UNIT_STARTED',pid=p.pid,command=command,log=str(log),CPU_only=cpu)
        while p.poll() is None:
            if (ROOT/'FIRST_FAILURE_GATE.json').exists() or time.time()-max(start,log.stat().st_mtime)>1800:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                raise RuntimeError('UNIT_STOP_OR_CONFIRMED_NO_PROGRESS_30MIN')
            time.sleep(10)
        event('UNIT_EXIT',command=command,returncode=p.returncode,wall_seconds=time.time()-start,CPU_only=cpu)
        require(p.returncode==0,'UNIT_FAILED_NO_RETRY '+logname)

def prepare():
    require(load(ROOT/'FROZEN_8DOC_SELECTION.json')['status']=='PASS','DOCUMENT_SELECTION_FAILED')
    require(len(load(ROOT/'FROZEN_8DOC_SELECTION.json')['selected_documents'])==8,'EXACTLY8_REQUIRED')
    observed=gpu();historical=load(OLD/'FROZEN_SCIENCE_SPEC.json');ident=load(OLD/'SOURCE_AND_RUNTIME_IDENTITIES.json')
    require(sha(OLD/'FROZEN_SCIENCE_SPEC.json')=='1f24740272d941433afb1e4f8f616e72c3bc2dc29993b89130bdef41b8c9c200','HISTORICAL_SPEC_SHA')
    require(sha(OLD/'RESULT_AND_DELIVERY_HASHES.json')=='bdfe7f26f2ffdd083cfe1dab85cc915362a3e6d7f744bdc0e81f6782d0344a7c','HISTORICAL_SEAL_SHA')
    weight_checks=[]
    for x in historical['model_files']:
        actual=sha(x['path']);require(actual==x['sha256'],'CURRENT_MODEL_FILE_SHA_FAILURE '+x['path'])
        weight_checks.append({'path':x['path'],'SHA':actual,'status':'PASS'})
    require(sha(ident['C6R']['export_path'])==historical['C6R_rotation_SHA'],'ROTATION_SHA_FAILURE')
    for path,s in historical['source_SHA'].items():require(sha(path)==s,'SOURCE_SHA_FAILURE '+path)
    # Persistent archive plus independent local filesystems for current-document I/O.
    temp=Path('/tmp/qwen_functional_replication_20261013');temp.mkdir(mode=0o700,exist_ok=False)
    for n in ['anchors','students']:
        (temp/n).mkdir();(ROOT/n).symlink_to(temp/n,target_is_directory=True)
    preflight()
    remote=remote_py('import os,json\nv=os.statvfs('+repr(REMOTE_BASE)+")\nprint(json.dumps({'free_bytes':v.f_bavail*v.f_frsize}))")
    ledger=load(OLD/'GPU_COST_LEDGER.json');cost=[];storage=[]
    for doc in [1,2]:
        ref=load(OLD/f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json');anc=load(OLD/f'anchors/doc_{doc:02d}/ANCHOR_MANIFEST.json')
        branches=[load(OLD/f'students/doc_{doc:02d}/{a}/BRANCH_MANIFEST.json') for a in ARMS]
        scale_bytes=sum(x['bytes'] for x in ref['chunks']);refbytes=scale_bytes+ref['logits']['bytes']+ref['endpoint_states']['bytes']
        studentbytes=sum(x['sampled_logits']['bytes'] for x in branches)
        cost.append({'historical_doc':doc,'FP_GPUh':ref['GPUh'],'anchor_GPUh':anc['GPUh'],'Parent_GPUh':branches[0]['GPUh'],'NoInflation_GPUh':branches[1]['GPUh'],'two_arm_total_GPUh':ref['GPUh']+anc['GPUh']+sum(x['GPUh'] for x in branches)})
        storage.append({'historical_doc':doc,'scale_chunk_bytes':scale_bytes,'FP_reference_tensor_bytes':refbytes,'anchor_bytes':anc['artifact']['bytes'],'two_student_logit_bytes':studentbytes,'two_arm_tensor_bytes':refbytes+anc['artifact']['bytes']+studentbytes})
    required_data=max(x['FP_reference_tensor_bytes'] for x in storage)+256*24*32*128*4+128*1024**2
    required_tmp=max(x['anchor_bytes']+x['two_student_logit_bytes'] for x in storage)+128*1024**2
    required_archive=8*max(x['two_arm_tensor_bytes'] for x in storage)+1024**3
    free_data=shutil.disk_usage(ROOT).free;free_tmp=shutil.disk_usage(temp).free
    require(free_data>=required_data and free_tmp>=required_tmp and remote['free_bytes']>=required_archive,'CURRENT_STORAGE_PHYSICAL_FAILURE')
    save('RESOURCE_AND_STORAGE_READINESS.json',{'status':'PASS','current_GPU_observation':observed,'historical_full32768_capacity':'PASS_VERIFIED_TWO_DOC_SIX_STUDENT_ENDPOINTS',
         'historical_phase_costs':cost,'historical_tensor_storage':storage,'planned8doc_GPUh_from_two_arm_historical_mean':8*math.fsum(x['two_arm_total_GPUh'] for x in cost)/2,
         'planned8doc_GPUh_is_not_actual':True,'available_compute_data_bytes':free_data,'available_compute_tmp_bytes':free_tmp,'available_archive29_bytes':remote['free_bytes'],
         'required_current_doc_data_bytes_with_atomic_and_metadata_allowance':required_data,'required_current_doc_tmp_bytes_with_allowance':required_tmp,'required_archive8doc_bytes_with_1GiB_allowance':required_archive,
         'deficit_if_all8_retained_in_compute_data_bytes':max(0,required_archive-free_data),'archive_scheme':'one complete document copied, every SHA verified, atomic remote seal, then relocate only its new local tensors',
         'staging_paths':{'FP_reference':str(ROOT/'references'),'new_anchor_and_student_tensors':str(temp),'permanent_archive_server':'202.38.247.29','permanent_archive_directory':REMOTE_BASE},
         'old_scientific_assets_touched':False,'no_GPUh_cap':True,'no_VRAM_percentage_cap':True,'watchdog_seconds':1800,'final_actual_resources':'recorded from unit receipts/progress, never substituted by predictions'})
    (ROOT/'qualification').mkdir(exist_ok=True)
    # Restore the already qualified old 8192 engineering anchor; no prefix rerun.
    for n in ['qualification/ANCHOR_PRODUCTION_RECEIPT.json','qualification/small_1.json','qualification/small_2.json']:
        src=OLD/n;dst=ROOT/n;require(not dst.exists(),'QUALIFICATION_WRITE_ONCE');shutil.copyfile(src,dst)
    save('engineering/QUALIFICATION_REUSE_PROVENANCE.json',{'status':'PINNED_OLD_EVIDENCE_FOR_INCREMENTAL_CHECK','files':[{'source':str(OLD/n),'SHA':sha(OLD/n),'new_copy':str(ROOT/n)} for n in ['qualification/ANCHOR_PRODUCTION_RECEIPT.json','qualification/small_1.json','qualification/small_2.json']],
        'copied_GPUh_is_historical_not_current_cost':True,'no_prefix_recomputed':True,'weight_file_SHA_checks':weight_checks,'rotation_file_SHA':historical['C6R_rotation_SHA']})
    # Preserve the prequalification audit; replace its premature NOT_RUN_SELECTION_STOP metadata label.
    original=ROOT/'SOURCE_AND_RUNTIME_IDENTITY.json'
    original.rename(ROOT/'engineering/SOURCE_AND_RUNTIME_IDENTITY_PREQUALIFICATION_AUDIT.json')
    save('engineering/PREQUALIFICATION_METADATA_CLARIFICATION.json',{'status':'PRE_GPU_METADATA_ONLY','original_preserved_SHA':sha(ROOT/'engineering/SOURCE_AND_RUNTIME_IDENTITY_PREQUALIFICATION_AUDIT.json'),
        'reason':'selector recorded a NOT_RUN_SELECTION_STOP placeholder even when selection passed; authoritative qualified identity produced after P2; no numerical/source change'})
    # Materialize frozen IDs without changing the panel or ranking.
    sys.path.insert(0,'/data/zypan/transformers-qwen35')
    from transformers import AutoTokenizer
    from datasets import Dataset
    tok=AutoTokenizer.from_pretrained('/data/zypan/modelscope_models/Qwen3.5-9B',local_files_only=True,trust_remote_code=True)
    selected=load(ROOT/'FROZEN_8DOC_SELECTION.json')['selected_documents'];sources={};documents=[]
    (ROOT/'inputs').mkdir()
    for i,d in enumerate(selected,1):
        path=d['source_file']
        if path not in sources:sources[path]=Dataset.from_file(path) if path.endswith('.arrow') else [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        row=sources[path][d['row']];text=row['context'];raw=text.encode();require(hashlib_sha(raw)==d['raw_text_SHA'] and row['_id']==d['dataset_id'],'FROZEN_DOCUMENT_SOURCE_IDENTITY')
        require(b'\r' not in raw,'RAW_CR_IDENTITY_READER_MISMATCH_STOP')
        ids=tok.encode(text,add_special_tokens=False)[:32769];require(len(ids)==32769,'DOCUMENT_INPUT_LENGTH')
        from selection_audit import digest
        require(digest(ids)==d['token_sequence_canonical_SHA'],'FROZEN_TOKEN_SEQUENCE_SHA')
        tp=ROOT/f'inputs/doc_{i:02d}.txt';jp=ROOT/f'inputs/doc_{i:02d}_tokens.json'
        with tp.open('xb') as f:f.write(raw)
        with jp.open('x') as f:json.dump(ids,f,sort_keys=True,indent=2);f.write('\n')
        documents.append(dict(d,document_index=i,text_path=str(tp),token_path=str(jp),token_SHA=sha(jp)))
    save('DOCUMENT_INPUT_ARTIFACTS.json',{'status':'PASS_FROZEN_PANEL_MATERIALIZED','documents':documents,'selection_SHA':sha(ROOT/'FROZEN_8DOC_SELECTION.json'),'no_CR_CRLF':True})
    event('PREPARATION_AND_RESOURCE_GATES_PASS',remaining_phases='fresh resume, observer1, observer2, freeze, fixed 32 units')

def hashlib_sha(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()

def freeze():
    from replication_worker import P,W
    historical=load(OLD/'FROZEN_SCIENCE_SPEC.json');old_qual=load(OLD/'qualification/observer_reducer_1.json')
    quals=[load(ROOT/f'qualification/observer_reducer_{i}.json') for i in [1,2]]
    keys=['scale_stream_hashes','Parent_observer_trace','legacy_reducer_parity','FP32_vs_FP64_max_abs_error','stable_time_reducer_max_abs_error','pre_outcome_DELTA_resolution','KL_test_values','all_FP32_FP64_errors']
    for key in keys:require(quals[0][key]==quals[1][key]==old_qual[key],'INHERITED_FRESH_PROCESS_PARITY_FAILURE '+key)
    require(load(ROOT/'ANCHOR_CHECKPOINT_AND_FORK_PARITY.json')['fresh_process_Parent_next8']=='BYTE_EXACT','8192_RESUME_FAILURE')
    require(historical['DELTA_resolution']==RESOLUTION,'FROZEN_RESOLUTION_IDENTITY')
    proof={'status':'PASS','scope':'incremental same-source same-GPU runtime; old full32768 closure reused, new two fresh observer/reducer processes and old8192 fresh Parent next8 tested',
           'fresh_receipts':[{'path':str(ROOT/f'qualification/observer_reducer_{i}.json'),'SHA':sha(ROOT/f'qualification/observer_reducer_{i}.json')} for i in [1,2]],'new8192_prefix_forward_count':0,
           'same_original_scan_quantizer_reducer_sources':True,'original_history_FAIL_STOP_retained':True,'formal_outcomes_used_for_threshold':False}
    save('NUMERICAL_QUALIFICATION.json',dict(proof,DELTA_resolution=RESOLUTION,fresh_parent8192_next8='BYTE_EXACT',production_observer_OFF_ON='BYTE_EXACT'))
    for n in ['KL_REDUCER_QUALIFICATION.json','FP_SCALE_COORDINATE_PARITY.json','QUANTIZER_INTERVENTION_VALIDITY.json','V3_NUMERICAL_CLOSURE_REUSE.json']:
        shutil.copyfile(OLD/n,ROOT/n)
    save('QUANTIZER_INTERVENTION_PARITY.json',dict(proof,old_fullmodel_validity_SHA=sha(OLD/'QUANTIZER_INTERVENTION_VALIDITY.json'),original_AST_round_clamp_writeback=True,clean_scale_coordinates='same C6R/token/layer/group',new_formal_arms=ARMS,observer_no_state_mutation=True))
    identity=load(OLD/'SOURCE_AND_RUNTIME_IDENTITIES.json')
    save('SOURCE_AND_RUNTIME_IDENTITY.json',{'status':'PASS','historical_spec_SHA':sha(OLD/'FROZEN_SCIENCE_SPEC.json'),'historical_seal_SHA':sha(OLD/'RESULT_AND_DELIVERY_HASHES.json'),
         'base_commit':historical['code_commit'],'inherited_source_SHA':historical['source_SHA'],'model_SHA':historical['model_SHA'],'model_file_SHA_current_checks':load(ROOT/'engineering/QUALIFICATION_REUSE_PROVENANCE.json')['weight_file_SHA_checks'],
         'C6R_rotation_SHA':historical['C6R_rotation_SHA'],'quantizer_SHA':historical['quantizer_SHA'],'runtime':historical['runtime'],'runtime_identity':historical['runtime_identity'],'GPU_UUID':UUID,
         'adapter_scope':'only output root/task name/repo path, 2-arm roster and dispatch; original numerical functions and sources imported unchanged','fresh_process_qualification_SHA':sha(ROOT/'NUMERICAL_QUALIFICATION.json')})
    save('PREFORMAL_GATES.json',{'status':'PASS','G0':'PASS_CURRENT_MODEL_ROTATION_TOKENIZER_SOURCE_SHA','G1':'PASS_EIGHT_NEW_TEXT_TOKEN_DEDUP_DOCS_PROJECT_METADATA_UNKNOWN',
         'G2':'PASS_CLEAN_FP_STREAM_COORDINATE','G3':'PASS_ORIGINAL_PARENT_QDQ_WRITEBACK','G4':'PASS_PINNED_MIN_INTERVENTION_CLIPPING_SEMANTICS','G5':'PASS_FRESH_8192_RESTORE_NEXT8_BYTE_EXACT',
         'G6':'PASS_FRESH_INDEPENDENT_RESTORE_NO_SHARED_CACHE','G7':'PASS_FRESH_OBSERVER_OFF_ON_PARITY','G8':'PASS_FRESH_FULL_VOCAB_REDUCER','G9':'PASS_TWO_FRESH_PROCESSES_MATCH_OLD_BYTE_SCALAR_IDENTITIES',
         'G10':'PASS_PINNED_SAME_RUNTIME_ACTUAL_32768_CAPACITY','G11':'PASS_CURRENT_GPU_LOCAL_FILESYSTEM_AND_REMOTE_ATOMIC_SHA_IO','G12':'PASS_PRE_OUTCOME_PANEL_PRIMARY_STAT_RULES',
         'project_independence_limitation':'authoritative repository/project metadata unavailable; formal inference conditional on document independence; 7/8 significance does not prove repository independence'})
    # Commit all new code, inputs and qualification metadata before formal outcomes.
    staged=[str(p.relative_to(REPO)) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.txt'] and not any(x in p.parts for x in ['.cache','__pycache__','logs'])]
    subprocess.run(['git','add','--']+staged,cwd=REPO,check=True)
    subprocess.run(['git','commit','-m','Freeze qualified independent 8-document replication before formal outcomes'],cwd=REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    sources=dict(historical['source_SHA']);sources.update({str(p):sha(p) for p in ROOT.glob('*.py')})
    spec=dict(historical,status='FROZEN_ALL_GATES_PASS',task=P.TASK,code_commit=commit,source_SHA=sources,documents=load(ROOT/'DOCUMENT_INPUT_ARTIFACTS.json')['documents'],
         document_selection_SHA=sha(ROOT/'FROZEN_8DOC_SELECTION.json'),DELTA_resolution=RESOLUTION,
         unit_schedule=[{'document':i,'units':['reference','warmup']+ARMS,'same_GPU':0,'archive_before_next_document':True} for i in range(1,9)],
         output_schema={'per_position':'16 branches *384 full-vocab FP32 KL; all identities bound','doc_effects':'8 document-unit Parent/NoInf K, Delta, RelativeGain, resolution status','archive':'each complete doc full CPU FP64 arithmetic audit then remote every-file SHA and retained relocation catalog'},
         dataset_catalog=load(ROOT/'DATASET_SELECTION_AUDIT.json')['existing_cache_inventory'],exclusion_registry_SHA=sha(ROOT/'EXCLUSION_REGISTRY.json'),
         project_independence_limitation='no authoritative repository/project metadata in compatible snapshots; new text/token dedup docs, unknown project groups; statistics conditional on independence',
         sample_size=8,statistical_unit='DOCUMENT',no_old_pilot_merge=True,
         statistics={'positive':'Delta > frozen resolution','negative':'Delta < -frozen resolution','otherwise':'UNRESOLVED','directional_support':'all8complete, noUNRESOLVED, >=7positive','exact_one_sided_sign_test':'sum(comb(n,k),k>=positive)/2**n; n=8 only noUNRESOLVED, otherwise resolved-only descriptive with full8denominator retained',
          'verdict_rule':'supported if directional_support; otherwise >=2 distinguishable negatives => FUNCTIONAL_EFFECT_NOT_GENERAL_IN_THIS_PANEL; otherwise MIXED_OR_HETEROGENEOUS_FUNCTIONAL_EFFECT; incomplete/failed => INCOMPLETE_OR_ENGINEERING_STOP',
          'CI':'95% percentile CI for mean Delta by 100000 whole-document bootstrap resamples; seed20261013; numpy percentile linear interpolation; conditional smallN independent sampling interpretation',
          'counterexamples_retained':True,'weak_activation_not_excluded':True},
         required_gates={f'G{i}':'PASS' for i in range(13)},utc=P.N.now(),reducer_qualification_SHA=sha(ROOT/'KL_REDUCER_QUALIFICATION.json'))
    spec['intervention']={k:v for k,v in historical['intervention'].items() if k!='NO_SCALE_DEFLATION'}
    save('FROZEN_SCIENCE_SPEC.json',spec)
    event('ALL_GATES_PASS_SCIENCE_FROZEN',code_commit=commit,science_spec_SHA=sha(ROOT/'FROZEN_SCIENCE_SPEC.json'),N=8)

def main():
    with (ROOT/'ACTIVE_COORDINATOR_LOCK').open('x') as f:f.write(str(os.getpid())+'\n')
    event('COORDINATOR_STARTED',pid=os.getpid(),scope='authorized8doc only; old pilot untouched')
    prepare()
    worker=str(ROOT/'replication_worker.py')
    for args,name in [(['resume'],'incremental_8192_resume.log'),(['observer','1'],'incremental_observer1.log'),(['observer','2'],'incremental_observer2.log')]:
        unit([worker]+args,name)
    freeze()
    for doc in range(1,9):
        require(shutil.disk_usage(ROOT).free>=load(ROOT/'RESOURCE_AND_STORAGE_READINESS.json')['required_current_doc_data_bytes_with_atomic_and_metadata_allowance'],'PHYSICAL_DISK_CHANGED_BEFORE_DOC')
        unit([worker,'reference',str(doc)],f'formal_doc{doc}_FP.log')
        require(load(ROOT/f'references/doc_{doc:02d}/FP_REFERENCE_MANIFEST.json')['status']=='COMPLETE','FP_NO_COMPLETION_RECEIPT')
        unit([worker,'warmup',str(doc)],f'formal_doc{doc}_WARMUP.log')
        require(load(ROOT/f'anchors/doc_{doc:02d}/ANCHOR_MANIFEST.json')['status']=='COMPLETE','ANCHOR_NO_COMPLETION_RECEIPT')
        for arm in ARMS:
            unit([worker,'branch',str(doc),arm],f'formal_doc{doc}_{arm}.log')
            require(load(ROOT/f'students/doc_{doc:02d}/{arm}/BRANCH_MANIFEST.json')['status']=='COMPLETE','BRANCH_NO_COMPLETION_RECEIPT')
        unit([str(ROOT/'archive_assets.py'),str(doc)],f'archive_doc{doc}.log',cpu=True)
        require(load(ROOT/f'archive/DOC_{doc:02d}_ARCHIVE_VERIFICATION.json')['status']=='PASS_ALL_FILE_SHA_BYTES','ARCHIVE_NO_COMPLETE_RECEIPT')
        event('DOCUMENT_COMPLETE_ARCHIVED_NO_DIRECTION_CLASSIFIED',document_index=doc)
    event('ALL_8_DOCUMENTS_32_FORMAL_UNITS_COMPLETE')
    unit([str(ROOT/'finalize_replication.py')],'finalization.log',cpu=True)
    event('COMPLETE_SEALED_STOP_NO_NEW_RESEARCH')

if __name__=='__main__':
    try:main()
    except Exception as e:
        if not (ROOT/'FIRST_FAILURE_GATE.json').exists():save('FIRST_FAILURE_GATE.json',{'status':'FAIL_AND_STOP','stage':'COORDINATOR_GATE','message':str(e),'exception':type(e).__name__,'traceback':traceback.format_exc(),'time_unix':time.time(),'no_retry':True})
        event('FAIL_AND_STOP',message=str(e))
        subprocess.run([PYTHON,str(ROOT/'finalize_replication.py')],cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        raise
