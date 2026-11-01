"""CPU-only premeasurement preparation. No AIME forward and no scoring."""
import argparse, ast, json, os, socket
import canonical60_common as A

def prepare():
    A.require(A.git('rev-parse','HEAD')==A.PARENT,'wrong actual V2 final source')
    A.require(A.git('rev-parse','--abbrev-ref','HEAD')==A.BRANCH,'wrong branch')
    for d in ('configs','manifests','raw/c6r','raw/c7','scores','analysis','reports','logs','provenance'):
        (A.ROOT/d).mkdir(parents=True,exist_ok=True)
    A.require(not (A.ROOT/'configs/canonical60_protocol.json').exists(),'preparation already exists')
    for arm,h in A.SHAS.items(): A.require(A.sha(A.rotation_paths(arm)[0])==h,'rotation SHA mismatch')
    A.require(A.sha(A.DATASET)==A.DATASET_SHA and A.sha(A.SCORER)==A.SCORER_SHA,'dataset/scorer identity')
    checks=A.protected_revalidate()
    A.save('provenance/premeasurement_protected_revalidation.json',checks)
    g=A.generator()
    import torch
    from transformers import AutoTokenizer
    A.require(not torch.cuda.is_initialized(),'CPU preparation initialized CUDA')
    parent=A.read(A.V2/'configs/v2_protocol.json')
    future=A.read(A.V2/'configs/v2_future_canonical_protocol.json')
    assignment=A.read(A.V2/'configs/v2_future_canonical_assignment.json')
    # Select runtime-only keys. Never inspect or carry historical endpoint results.
    inherited=future['canonical_config']
    protocol=inherited['protocol']
    flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    A.require(flags==parent['deterministic_flags'],'deterministic runtime defaults changed')
    A.require(g.runtime_version()==parent['runtime'],'runtime version changed')
    constants=dict(temperature=g.TEMPERATURE,top_p=g.TOP_P,top_k=g.TOP_K,min_p=g.MIN_P,presence_penalty=g.PRESENCE_PENALTY,repetition_penalty=g.REPETITION_PENALTY,max_new_tokens=g.MAX_NEW_TOKENS)
    A.require(all(protocol[k]==v for k,v in constants.items()),'source sampler config mismatch')
    A.require(assignment['condition_order']=='C6R_THEN_C7','parent condition order')
    jobs=assignment['jobs']
    data=[json.loads(x) for x in A.DATASET.read_text().splitlines() if x.strip()]
    problems={x['problem_id']:x for x in data}
    A.require(len(problems)==len(data)==30 and len(jobs)==60,'canonical dataset cardinality')
    A.require(len({(j['problem_id'],j['seed']) for j in jobs})==60,'canonical sample duplicate')
    A.require(all({j['seed'] for j in jobs if j['problem_id']==p}=={1,2} for p in problems),'seed pairs')
    tok=AutoTokenizer.from_pretrained(str(A.MODEL),local_files_only=True,trust_remote_code=True)
    prompts=[];gold=[]
    for rank,j in enumerate(jobs):
        A.require(j['rank']==rank and j['gpu']==rank%4 and j['GPU_uuid']==A.UUIDS[rank%4],'parent assignment drift')
        d=problems[j['problem_id']];messages,ids=g.render_input(tok,d['problem'])
        A.require(ids.device.type=='cpu','CPU prompt identity created on GPU')
        idslist=ids[0].tolist()
        prompts.append(dict(j,problem_idx=d['problem_idx'],problem=d['problem'],raw_messages=messages,input_ids_hash=g.ids_sha256(idslist),input_token_count=len(idslist),prompt_sha256=A.digest(messages)))
        gold.append(dict(problem_id=j['problem_id'],seed=j['seed'],answer=d['answer']))
    # Gold is isolated; generation reads only the prompt/sample manifest.
    A.save('manifests/canonical60_gold_manifest.json',dict(source=str(A.DATASET),source_sha256=A.DATASET_SHA,records=gold),True)
    A.save('manifests/canonical60_sample_manifest.json',dict(source_assignment=str(A.V2/'configs/v2_future_canonical_assignment.json'),source_assignment_sha256=A.sha(A.V2/'configs/v2_future_canonical_assignment.json'),dataset_source=str(A.DATASET),dataset_sha256=A.DATASET_SHA,questions=30,seeds=[1,2],samples=prompts,gold_included=False),True)
    A.save('configs/canonical60_worker_assignment.json',dict(rule='RANK_MOD_4_SAME_GPU_ACROSS_CONDITIONS',GPU_UUIDs=A.UUIDS,jobs=jobs,samples_per_gpu_per_condition=15))
    A.save('configs/canonical60_generation_config.json',dict(protocol=protocol,manual_HF=True,dtype='torch.bfloat16',cache='unchanged canonical manual past_key_values',prefill_quantized=False,QDQ='unchanged quantize_c128 post-decode',writeback='canonical state.copy_',rotation='q/k separate H128 then direct frozen Delta; value untouched',stop_rule='unchanged canonical stop_ids(tokenizer)',generation_source=str(A.FORMAL),generation_source_sha256=A.FORMAL_SHA,source_formal_config_sha256=future['formal_config_sha256'],deterministic_inference=True,deterministic_runtime_flags=flags,determinism_evidence='V2 same-GPU fresh-process exact repeatability; do not enable new torch flags'))
    A.save('configs/canonical60_protocol.json',dict(task=A.TASK,parent_final_commit=A.PARENT,condition_order='C6R_THEN_C7',conditions=['C6R','C7'],planned_generations=120,C5R_generations=0,new_training='NO',runtime=parent['runtime'],GDN_layers=list(g.GDN_LAYERS),rotation_SHA=A.SHAS,architecture='four independent single-GPU model-resident workers per condition; fresh C7 processes',gold_separation=True,score_only_after_raw120_frozen=True,disk_stop_threshold_bytes=inherited['disk_stop_threshold_bytes'],normal_exit_reclaim='exact NVML_BYTES baseline',forbidden=['training','backward','optimizer.step','reselection','dynamic_assignment','SGLang','TP','FSDP','ZeRO','new_mechanism_execution','push']))
    A.save('configs/canonical60_condition_order.json',dict(order=['C6R','C7'],barrier='all C6R workers normal exit, reclaim and raw60 freeze PASS before any C7 forward',inherited_from=str(A.V2/'configs/v2_future_canonical_assignment.json')))
    A.save('configs/canonical60_resume_policy.json',dict(valid_final='validate SHA/schema; never overwrite or regenerate; reconstruct missing completion metadata only',preforward_failure='preserve incident; manual wrapper-only repair permitted with durable proof MODEL_FORWARD_STARTED=NO',postforward_failure='STOP affected condition; no automatic retry; no C7/scoring; wait explicit user authorization',peer_workers='finish already-started requests naturally; shared stop sentinel prevents next sample',started_without_final='STOP; never silently regenerate',ceiling='81920 normal length is valid, never retry',eos='canonical EOS/stop valid, never retry',disk='below 15GiB stop new samples; preserve artifacts',dynamic_balancing=False,automatic_retries=0))
    A.save('configs/canonical60_statistical_plan.json',dict(primary='C7_vs_C6R',test='two-sided exact paired discordant binomial; n=0 p=1',alpha=0.05,multiplicity='one confirmatory comparison; no adjustment',categories=['RESCUE','REGRESSION','BOTH_CORRECT','BOTH_WRONG'],effect='(C7_correct-C6R_correct)/60',CI=dict(method='question-cluster bootstrap',questions=30,seeds_per_question=2,replicates=10000,seed=20261003,rng='numpy.random.default_rng',percentile_method='linear',percentiles=[2.5,97.5]),labels=['SUPPORTED','DIRECTIONALLY_SUPPORTIVE_BUT_UNDERPOWERED','NO_OBSERVED_NET_GAIN','NEGATIVE_DIRECTION_INCONCLUSIVE','CONTRADICTED'],secondary='load frozen historical results only after primary complete; contextual only; no secondary p-values',evidence_scope='AIME26 previously used in method development/diagnosis; not untouched external benchmark'))
    histreader=A.SOURCE.parent/'qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/analyze_existing.py'
    A.save('configs/canonical60_failure_analysis_plan.json',dict(stage='after scoring+primary frozen; raw-output-only',canonical_utility=str(A.POST),canonical_utility_sha256=A.sha(A.POST),historical_reader=str(histreader),historical_reader_sha256=A.sha(histreader),metrics=['total/mean/median generated tokens','EOS/length counts','abstain','first committed claim char position','answer changes','retractions','4/8/16-gram repeated fraction and unique ratio','existing oscillation_proxy'],transitions=['normal->normal','ceiling->normal','normal->ceiling','abstain->committed','committed->abstain'],ceiling_association='rescue vs C6R/H ceiling; regression vs C7 ceiling; descriptive not causal',question_split='NOT_RUN: no preexisting canonical Q1-15/Q16-30 split rule established',new_model_execution=False))
    model_files=[r for r in checks['files'] if r['path'].startswith(str(A.MODEL)+'/')]
    A.require(any(r['path'].endswith('.safetensors') for r in model_files),'missing frozen model weights identity')
    ident=dict(CANONICAL60_RUNTIME_IDENTITY_GATE='PASS',hostname=socket.gethostname(),source_branch='exp/qwen-gdn-c6r-c7-persistent-worker-readiness-v2',source_HEAD=A.PARENT,new_branch=A.BRANCH,new_checkout=str(A.REPO),checkout_type='isolated shared local Git clone; Git1.8.3.1 has no worktree command',model_identity=dict(path=str(A.MODEL),files=model_files,revision='frozen local files and full SHA list inherited from V2'),generation_source=dict(path=str(A.FORMAL),sha256=A.FORMAL_SHA),rotation_files={a:dict(path=str(A.rotation_paths(a)[0]),sha256=A.SHAS[a],manifest_path=str(A.rotation_paths(a)[1]),manifest_sha256=A.sha(A.rotation_paths(a)[1])) for a in A.SHAS},runtime=parent['runtime'],GPU_UUIDs=A.UUIDS,dataset_SHA=A.DATASET_SHA,source_protected_revalidation_SHA=A.sha(A.ROOT/'provenance/premeasurement_protected_revalidation.json'),V2_final_artifact_manifest_SHA=A.sha(A.V2/'manifests/v2_final_artifact_sha256.txt'),preparation_model_forwards=0)
    A.save('provenance/canonical60_runtime_identity.json',ident)
    A.save('provenance/preparation_audit.json',dict(cpu_only=True,MODEL_FORWARD_STARTED='NO',gold_split_from_frozen_dataset=True,new_answers_created=False,historical_result_exposure='During source-only review, parent future-runtime JSON containing already-frozen historical aggregate baseline fields was printed once. No new endpoint existed, no new scores were computed/read, no per-sample historical results loaded. Preparation copies only runtime protocol keys; subsequent historical result loading remains gated after primary. This disclosure is retained prospectively.',diff_review_required_before_commit=True))
    src=ast.parse((A.ROOT/'scripts/canonical60_worker.py').read_text())
    functions=[n.func.attr for n in ast.walk(src) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    A.require('generate_one' in functions and not {'backward','step','empty_cache','gc'}&set(functions),'worker scientific path audit')
    A.save('provenance/scientific_path_audit.json',dict(status='PASS',generator_called_directly=True,canonical_generator_unchanged_SHA=A.FORMAL_SHA,no_sampler_or_QDQ_copy=True,only_methods_metadata_alias_added=True,no_generation_limit_override=True,model_or_rotation_unload_in_process=False,postprocessing_canonical_utilities_sha256=A.sha(A.POST)))
    # Include every prospective script, config and immutable prompt/gold metadata.
    paths=sorted([A.ROOT/'.gitignore']+[p for folder in ('scripts','configs','manifests','provenance') for p in (A.ROOT/folder).glob('*') if p.is_file()])
    A.write_text('manifests/canonical60_premeasurement_sha256.txt',''.join(A.sha(p)+'  '+str(p)+'\n' for p in paths))
    A.verify_manifest(A.ROOT/'manifests/canonical60_premeasurement_sha256.txt')
    A.save('analysis/premeasurement_validation.json',dict(CANONICAL60_PREMEASUREMENT_FREEZE_GATE='PASS',JSON_parse='PASS',SHA_manifest='PASS',SHA_reverify='PASS',sample_keys=60,questions=30,seeds=[1,2],arm_samples=60,generation_started=False))
    A.require(not torch.cuda.is_initialized(),'preparation used CUDA')
    print('CPU_PREMEASUREMENT_PREPARED_NO_FORWARD',flush=True)

if __name__=='__main__': prepare()
