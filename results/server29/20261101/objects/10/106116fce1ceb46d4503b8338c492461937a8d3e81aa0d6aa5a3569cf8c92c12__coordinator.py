"""Bounded qualification gates, then exactly2 FP/2 warmup/6 continuations."""
import json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import pilot_engineering as P
import formal_worker as W

def read(name):return json.loads((P.ROOT/name).read_text())

def event(phase,**obj):W.event('execution/COORDINATOR_EVENTS.jsonl',{'utc':P.N.now(),'phase':phase,**obj})

def wait_existing_anchor():
    log=P.ROOT/'logs/anchor_qualification.log'
    while not (P.ROOT/'ANCHOR_CHECKPOINT_AND_FORK_PARITY.json').exists():
        P.check(not (P.ROOT/'FIRST_FAILURE_GATE.json').exists(),'existing engineering STOP; no subsequent qualification or science')
        P.check(time.time()-log.stat().st_mtime<1800,'confirmed no engineering progress for30min')
        # Only wait on the already authorized exact engineering process.
        found=False
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except (OSError,UnicodeError):continue
            if 'python' in command and '/scripts/anchor_qualification.py' in command and any(s in command for s in [' produce',' resume']):found=True
        P.check(found,'anchor worker ended without complete parity receipt')
        time.sleep(10)
    P.check(read('ANCHOR_CHECKPOINT_AND_FORK_PARITY.json')['status']=='PASS','complete8192 fork PASS')
    # Let the successful existing producer/resume exit and release its model.
    for _ in range(60):
        rows=P.N.resources()['processes_csv'].splitlines()
        if not any(P.S.UUIDS[0] in row for row in rows):return
        time.sleep(2)
    raise RuntimeError('GPU0 still occupied after existing anchor qualification completion')

def unit(script,args,log_name):
    P.check(not (P.ROOT/'FIRST_FAILURE_GATE.json').exists(),'STOP receipt forbids new unit')
    P.check(not any(P.S.UUIDS[0] in row for row in P.N.resources()['processes_csv'].splitlines()),'GPU0 one-model physical resource availability')
    log=P.ROOT/'logs'/log_name;P.check(not log.exists(),'unit exactly once; no implicit retry')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',
             HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    command=[sys.executable,str(P.ROOT/'scripts'/script)]+list(args)
    started=time.time()
    with log.open('x') as f:
        proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=P.REPO)
        event('UNIT_STARTED',command=command,pid=proc.pid,log=str(log))
        while proc.poll() is None:
            if (P.ROOT/'FIRST_FAILURE_GATE.json').exists() or time.time()-max(started,log.stat().st_mtime)>1800:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise RuntimeError('confirmed unit STOP/no-progress; current child terminated; no continuation')
            time.sleep(10)
        event('UNIT_EXIT',command=command,returncode=proc.returncode,wall_seconds=time.time()-started)
        P.check(proc.returncode==0,'unit failed; scientific workflow fail-closed')

def freeze():
    P.check(read('DOCUMENT_SELECTION_AND_EXCLUSIONS.json')['status']=='PASS','prospective doc gate')
    qual=[read(f'qualification/observer_reducer_{i}.json') for i in [1,2]]
    keys=['scale_stream_hashes','Parent_observer_trace','legacy_reducer_parity','FP32_vs_FP64_max_abs_error',
          'stable_time_reducer_max_abs_error','pre_outcome_DELTA_resolution','KL_test_values','all_FP32_FP64_errors']
    for key in keys:P.check(qual[0][key]==qual[1][key],'fresh-process production-dataflow/KL repeatability '+key)
    reused=read('SOURCE_AND_RUNTIME_IDENTITIES.json');anchor=read('qualification/ANCHOR_PRODUCTION_RECEIPT.json')
    P.check(anchor['status']=='PASS' and read('QUANTIZER_INTERVENTION_VALIDITY.json')['status']=='PASS','physical/intervention qualification')
    proof={'status':'PASS','raw_receipts':[{'path':f'qualification/observer_reducer_{i}.json','sha256':P.sha(P.ROOT/f'qualification/observer_reducer_{i}.json')} for i in [1,2]],
           'frozen_before_formal_outcomes':True,'strict_same_GPU0_runtime':True}
    P.save('FP_SCALE_COORDINATE_PARITY.json',dict(proof,coordinate='direct same-C6R clean pre-QDQ/cache, current token/layer/group',
        lossless_chunk_streaming=True,FP_QDQ_writebacks=0))
    P.save('FULLMODEL_NUMERICAL_QUALIFICATION.json',dict(proof,V3_reuse_SHA=P.sha(P.ROOT/'V3_NUMERICAL_CLOSURE_REUSE.json'),
        new_closed_loop_teacher_student_and_fork='PASS',history_native_bitwise_FAIL='PRESERVED',
        original_Parent_vs_production_observer='BYTE_EXACT',fresh_process_resume_8192='BYTE_EXACT',
        scope='small/8192 engineering qualification; long formal trajectories retain finite and consumption audits'))
    error=max(read('qualification/small_1.json')['KL_max_abs_FP32_vs_FP64_error'],qual[0]['FP32_vs_FP64_max_abs_error'])
    reduction=max(read('qualification/small_1.json')['KL_time_reducer_error'],qual[0]['stable_time_reducer_max_abs_error'])
    resolution=2*error+2*reduction
    P.save('KL_REDUCER_QUALIFICATION.json',dict(proof,FP32_max_abs_error=error,time_reduction_error=reduction,
        DELTA_resolution=resolution,formula='2*max independent per-position KL32/KL64 error +2*time reduction error',
        time_mean_helper='unchanged V3 scale_science.mean/math.fsum',full_vocabulary=248320,
        sample_count=384,legacy_32K_signed_reducer_source=str(W.LEGACY_REDUCER_SOURCE),
        legacy_32K_signed_reducer_SHA=P.sha(W.LEGACY_REDUCER_SOURCE)))
    # Real serialize sizes plus exact mandated tensor shapes, no percent/GPUh gates.
    scale_bytes=2*24576*24*32*128*4
    score_bytes=2*4*384*248320*4  #2 FP plus6 sampled student logits, CPU/disk only
    endpoint_bytes=2*4*24*32*128*128*4
    anchor_bytes=2*anchor['anchor']['bytes']
    atomic_buffer=256*24*32*128*4
    serialization_allowance=192*65536+8*65536+2*1024**2
    required=scale_bytes+score_bytes+endpoint_bytes+anchor_bytes+atomic_buffer+serialization_allowance
    available=shutil.disk_usage(P.ROOT).free
    P.check(available>=required,'actual disk insufficient to retain both full scientific evidence sets')
    P.save('RESOURCE_AND_STORAGE_READINESS.json',{'status':'PASS','GPU_UUID':P.S.UUIDS[0],
        'physical_GPU_total_bytes':anchor['memory_measurements'][-1]['total'],
        'closed_loop_measured_cache_bytes_per_token':anchor['cache_bytes_per_token'],
        'additional_cache_bytes_to32768':anchor['additional_cache_bytes_to32768'],
        'measured_transient_bytes':anchor['measured_transient_bytes'],
        'available_GPU_bytes_including_reusable_pool':anchor['available_device_bytes_including_reusable_pool'],
        'disk_available_bytes_before_formal':available,'required_formal_evidence_bytes':required,
        'scale_bytes':scale_bytes,'all_sampled_FP_student_logit_bytes':score_bytes,'endpoint_state_bytes':endpoint_bytes,
        'two_anchor_bytes':anchor_bytes,'atomic_buffer_bytes':atomic_buffer,'serialization_allowance':serialization_allowance,
        'no_fixed_GPUh_cap':True,'no_VRAM_percentage_cap':True,'full32768_endpoint_execution':'PENDING_FORMAL_CONFIRMATION'})
    source_paths=list((P.ROOT/'scripts').glob('*.py'))+[P.N.ARCH,P.N.FORMAL,P.V3/'scripts/numerical_recovery.py',
        P.V3/'scripts/scale_science.py',P.V3/'scripts/matched_history.py',P.S.CORE,W.LEGACY_REDUCER_SOURCE,W.LEGACY_REDUCER_SOURCE.parent/'common.py']
    P.save('SECONDARY_PREMEASUREMENT_CLARIFICATION.json',{'utc':P.N.now(),'status':'FROZEN_BEFORE_ANY_FORMAL_OUTCOME',
        'state_relative_error':'unchanged V3/core.relative_state_mse in same C6R coordinates at consumed16384/24576/32768; sparse endpoints',
        'clean_state_cache':'FP32 clean states at8192/16384/24576/32768, CPU/disk; never reset student with them',
        'legacy_signed_scale_histogram':'unchanged original6 time bins supplemental; primary secondary windows stay original3',
        'all_sampled_student_logits':'384 full-vocabulary FP32 logits/branch stored on disk for audit; no full32768 GPU logit retention'})
    # Commit code + engineering protocol and receipts, never model/tensor archives.
    staged=[str(p.relative_to(P.REPO)) for p in source_paths if P.REPO in p.parents]
    staged += [str(p.relative_to(P.REPO)) for p in P.ROOT.glob('*.json')]
    staged += [str(p.relative_to(P.REPO)) for p in (P.ROOT/'qualification').glob('*.json')]
    subprocess.run(['git','add','--']+staged,cwd=P.REPO,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Freeze all qualified full-model pilot code and engineering gates before outcomes'],cwd=P.REPO,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.REPO,text=True).strip()
    spec={'status':'FROZEN_ALL_GATES_PASS','utc':P.N.now(),'task':P.TASK,'documents':read('DOCUMENT_SELECTION_AND_EXCLUSIONS.json')['documents'],
        'document_selection_SHA':P.sha(P.ROOT/'DOCUMENT_SELECTION_AND_EXCLUSIONS.json'),'model_SHA':reused['model_aggregate_SHA'],
        'model_files':reused['model_identity']['model_files'],'C6R_rotation_SHA':reused['C6R']['export_file_sha256'],
        'quantizer_SHA':reused['quantizer_function_SHA'],'code_commit':commit,'source_SHA':{str(p):P.sha(p) for p in source_paths},
        'runtime':reused['runtime_contract'],'runtime_identity':P.digest(reused['runtime_contract']),'GPU_UUID':P.S.UUIDS[0],
        'anchor_consumed':8192,'horizon_consumed':32768,'input_count':32769,'positions':W.POSITIONS,
        'logit_alignment':'t zero-based consumed input index; logits after ids[t] predict ids[t+1]; consumed t+1',
        'primary':'mean384 fixed-position full-vocabulary FP32 KL(FP||student); unchanged V3 stable mean',
        'DELTA':'K_PARENT-K_NO_SCALE_INFLATION','DELTA_resolution':resolution,'reducer_qualification_SHA':P.sha(P.ROOT/'KL_REDUCER_QUALIFICATION.json'),
        'intervention':{'PARENT':'student adaptive original C128 scale','NO_SCALE_INFLATION':'torch.minimum(student,clean)',
            'NO_SCALE_DEFLATION':'torch.maximum(student,clean)','scale':'amax(-2,keepdim=True).clamp_min(1e-12)/127',
            'codes':'torch.round(source/scale).clamp(-127,127)','zero_point':0,'every_token_QDQ':True,'persistent_writeback':True,
            'clean_oracle':'same C6R full-model FP pre-QDQ-equivalent state; observer only; no writeback'},
        'secondary_windows':[[8192,16384],[16384,24576],[24576,32768]],'weak_activation_fraction':.001,
        'state_damage':'sparse frozen window endpoints; original relative_state_mse; separate from KL',
        'unit_schedule':[{'document':doc,'units':['reference','warmup']+W.ARMS,'same_GPU':0} for doc in [1,2]],
        'no_first_document_adaptation':True,'failure_policy':'first numerical/scientific/physical gate failure STOP, preserve evidence; no silent retry',
        'progress_failure_predicate':'worker active but no log/progress for30min, or exited without completion receipt',
        'required_gates':{k:'PASS' for k in ['NUMERICAL_QUALIFICATION','FP_REFERENCE_PAIRING','INTERVENTION_VALIDITY',
            'CHECKPOINT_FORK_PARITY','DETERMINISM','MEMORY_AND_STORAGE_FEASIBILITY','FROZEN_SCIENCE_SPEC']},
        'output_schema':{'FP_REFERENCE_MANIFEST':'identity/chunks/384 logits/endpoint states/complete receipt/cost',
            'STUDENT_BRANCH_MANIFEST':'identity/shared anchor/complete32768/384 KL/activation/clipping/cost',
            'SAMPLED_FULLVOCAB_FUTURE_KL.csv':'bound identities/token_index/consumed/predicted_token_index/full_vocabulary_size/KL_FP32',
            'DOCUMENT_LEVEL_CAUSAL_EFFECT.csv':'bound identities/K_PARENT/K_NO_SCALE_INFLATION/K_NO_SCALE_DEFLATION/DELTA_PRIMARY/DELTA_NO_DEFLATION/direction',
            'SCIENTIFIC_VERDICT.json':'all2doc completeness/preregistered resolution/weak activation/allowed directional small-pilot verdict',
            'other_outputs':'scale activation, clipping/code saturation, sparse state damage, resources, first failure if applicable, report, hashes'}}
    P.save('FROZEN_SCIENCE_SPEC.json',spec)
    event('ALL_ENGINEERING_GATES_PASS_SCIENCE_FROZEN',code_commit=commit,spec_SHA=P.sha(P.ROOT/'FROZEN_SCIENCE_SPEC.json'))

def main():
    with (P.ROOT/'ACTIVE_COORDINATOR_LOCK').open('x') as f:f.write(str(os.getpid())+'\n')
    event('COORDINATOR_STARTED',pid=os.getpid(),scope='authorized pilot only; no training/AIME/free generation')
    wait_existing_anchor()
    for index in [1,2]:unit('observer_reducer_qualification.py',[str(index)],f'observer_reducer_{index}.log')
    freeze()
    for doc in [1,2]:
        unit('formal_worker.py',['reference',str(doc)],f'formal_doc{doc}_FP.log')
        unit('formal_worker.py',['warmup',str(doc)],f'formal_doc{doc}_COMMON_WARMUP.log')
        for arm in W.ARMS:unit('formal_worker.py',['branch',str(doc),arm],f'formal_doc{doc}_{arm}.log')
    event('ALL_FORMAL_UNITS_COMPLETE_NO_DIRECTION_YET')
    subprocess.run([sys.executable,str(P.ROOT/'scripts/finalize_pilot.py')],cwd=P.REPO,check=True)
    event('COMPLETE_AND_STOPPED_NO_FOLLOWUP_EXPERIMENT')
    print('PILOT_COMPLETE_AND_SEALED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if not (P.ROOT/'FIRST_FAILURE_GATE.json').exists():P.save('FIRST_FAILURE_GATE.json',{'status':'FAIL_AND_STOP',
            'stage':'COORDINATOR_GATE','utc':P.N.now(),'exception':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'no_retry':True})
        event('FAIL_AND_STOP',message=str(exc))
        subprocess.run([sys.executable,str(P.ROOT/'scripts/finalize_pilot.py')],cwd=P.REPO)
        raise
