"""CPU-only prospective audit and freeze. No model loading or generation."""
import ast,json,subprocess,time
import persistent_v2_common as A
def main():
 A.require(A.cmd(['git','rev-parse','HEAD'])==A.PARENT,'parent drift')
 A.require(A.cmd(['git','symbolic-ref','--short','HEAD'])==A.BRANCH,'branch drift')
 status=A.cmd(['git','status','--porcelain'])
 A.require(all(A.TASK in l for l in status.splitlines()),'unrelated dirty files')
 A.save('provenance/parent_identity.json',dict(parent_branch='exp/qwen-gdn-c6r-c7-4x3090-deployment-v1',parent_HEAD=A.PARENT,git_status_before_branch='',current_status=status,git_log=A.cmd(['git','log','-8','--oneline']),time=time.time()))
 rows={r['path']:r for r in A.read(A.V1/'provenance/observer_repair_prechange_inventory.json')['files']}
 # Preserve all historical evidence plus all immutable and diagnostic V1 artifacts.
 for p in A.V1.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:rows.setdefault(str(p),dict(path=str(p),sha256=A.sha(p)))
 checks=[]
 for i,(p,r) in enumerate(sorted(rows.items())):
  A.require(A.sha(p)==r['sha256'],'protected artifact changed '+p)
  checks.append(dict(path=p,sha256=r['sha256']))
  if i%400==0:print(json.dumps(dict(event='read_only_identity',files=i+1)),flush=True)
 for arm in A.SHAS:A.require(A.sha(A.rotation_paths(arm)[0])==A.SHAS[arm],'rotation identity')
 source=A.read(A.ROOT/'provenance/a800_source_revalidation.json');A.require(source['status']=='PASS' and source['count']==51,'A800 revalidation missing')
 A.save('provenance/source_artifact_revalidation.json',dict(status='PASS',files=checks,count=len(checks),A800_revalidation_sha256=A.sha(A.ROOT/'provenance/a800_source_revalidation.json'),old_V1_read_only=True))
 g=A.gmodule();runtime=g.runtime_version();prior=A.read(A.PRIOR/'protocol/canonical_endpoint_amended.json')
 A.require(runtime==prior['runtime'],'runtime mismatch')
 import torch
 flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
 A.require(flags==A.read(A.V1/'provenance/deployment_runtime_identity.json')['deterministic_flags'],'flag mismatch')
 tree=ast.parse(A.FORMAL.read_text());funcs={n.name:ast.get_source_segment(A.FORMAL.read_text(),n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 names=['load_model_and_tokenizer','load_frozen_rotation','QwenFrozenRotationPatch','QwenKeyHadamardPatch','get_state','quantize_c128','generate_one','render_input','build_warpers','sample_next','stop_ids','setup_seed']
 identifiers={n:A.digest(funcs[n]) for n in names}
 source_text=(A.ROOT/'scripts/persistent_v2_worker.py').read_text();wt=ast.parse(source_text)
 calls=[n for n in ast.walk(wt) if isinstance(n,ast.Call)]
 A.require(any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='g' and n.func.attr=='generate_one' for n in calls),'not original generation loop')
 A.require(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('register_forward_pre_hook','register_forward_hook','backward','step','empty_cache','collect') for n in calls),'extra instrumentation/cleanup')
 A.require('MAX_NEW_TOKENS=128' in source_text,'short generation limit missing')
 A.require('output_hidden_states=True' in source_text,'public Final Hidden capture missing')
 A.require('gc.collect' not in funcs['generate_one'] and 'empty_cache' not in funcs['generate_one'],'canonical cleanup assumption')
 A.save('provenance/production_runner_source_identity.json',dict(status='PASS',formal_path=str(A.FORMAL),formal_sha256=A.sha(A.FORMAL),core_sha256=A.sha(A.CORE),rotation_implementation_sha256=A.sha(A.ROTATION),function_sources=funcs,function_hashes=identifiers,runtime=runtime,deterministic_flags=flags))
 A.save('analysis/production_runner_scientific_path_identity.json',dict(status='PASS',generation='direct call to unchanged formal.generate_one; no copied generation/quantizer formulas',model_loader='direct unchanged formal.load_model_and_tokenizer',rotation='direct unchanged formal.load_frozen_rotation and QwenFrozenRotationPatch',qdq_writeback='direct unchanged formal.quantize_c128; reference final-token hashes use unchanged training qwen_c128_qdq',semantic_hidden_capture='public output_hidden_states=True, Transformers capture_outputs ties final entry to post-norm last_hidden_state; no added diagnostic hooks/wrapped operators. Not used in production requests.',canonical_rotation_patch='original mandatory scientific q/k transform is preserved; no additional diagnostic observer',allowed_differences=['IPC/lifecycle and memory telemetry','non-AIME engineering item labels','prospective max_new_tokens=128','semantic hash capture via public outputs'],request_cleanup='generate_one returns CPU record; request-local cache/outputs naturally leave function scope; no GC or empty_cache added',function_hashes=identifiers))
 nv=A.Nvml();hardware=[nv.snapshot(i) for i in range(4)]
 A.require(all(h['uuid']==A.UUIDS[i] and not h['processes'] for i,h in enumerate(hardware)),'GPU busy or identity mismatch')
 A.save('provenance/four_gpu_hardware_identity.json',dict(status='PASS',GPUs=hardware,nvidia_smi=A.cmd(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader']),runtime=runtime))
 # Inspect historical smoke source names, never use AIME text or outputs in smoke.
 candidates=[]
 for root in (A.BASE/'experiments',A.PRIOR.parent):
  for p in root.rglob('*smoke*.py'):
   text=p.read_text(errors='replace')
   if any(k in text for k in ('generate_one(','.generate(','def generate')):
    candidates.append(dict(path=str(p),sha256=A.sha(p),reason='AIME dataset/prompt or noncanonical dev/other-model path; not a frozen canonical non-AIME production fixture'))
 A.save('provenance/non_aime_smoke_search.json',dict(status='NO_HISTORICAL_FROZEN_CANONICAL_NON_AIME_GENERATION_SMOKE_FOUND',search_roots=[str(A.BASE/'experiments'),str(A.PRIOR.parent)],candidates=candidates,other_smokes='V1 fixed-token and mechanism/dev teacher forcing are not production generation; 81920 smoke launcher uses AIME dataset and is excluded',selection='prospective TRAIN:first-row engineering request'))
 core=A.mod('v2_cpu_core',A.CORE);row=core.corpus_rows('TRAIN')[0]
 cfg=A.read(A.V1/'configs/frozen_smoke_protocol.json');ids=cfg['input_token_ids']
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(A.MODEL),trust_remote_code=True,local_files_only=True)
 A.require(tok(row['raw_text'],add_special_tokens=False).input_ids[:32]==ids,'TRAIN input provenance drift')
 messages,prompt_ids=g.render_input(tok,row['raw_text'])
 A.save('configs/v2_smoke_input.json',dict(seed=1,input_token_ids=ids,input_token_sha256=cfg['input_token_sha256'],semantic_source='frozen V1 TRAIN:first-row 32-token prefix; no retokenization in workers',production_problem=row['raw_text'],production_raw_text_sha256=A.digest(row['raw_text']),production_source=row['document_id'],production_prompt_messages=messages,production_input_ids=prompt_ids[0].tolist(),production_input_ids_sha256=g.ids_sha256(prompt_ids[0].tolist()),ENGINEERING_PRODUCTION_REQUEST_SMOKE='YES',max_new_tokens=128,AIME_data_used=False,semantic_hash_scope='32 per-token logits/Final Hidden and final-token per-layer pre-state/scale/qcode/post-state fingerprints'))
 matrix=[dict(gpu=0,arm=a,repeat=r,key=f'gpu0_{a.lower()}'+('' if r=='A' else '_repeat_b')) for r in ('A','B') for a in ('C6R','C7')]
 matrix += [dict(gpu=i,arm=a,repeat='A',key=f'gpu{i}_{a.lower()}') for i in (1,2,3) for a in ('C6R','C7')]
 A.save('configs/v2_worker_matrix.json',dict(runs=matrix,total=10,primary_runs=8,fresh_GPU0_duplicates=2,schedule='sequential fail-closed; no other worker starts after any failure',workers_independent_single_GPU=True))
 A.save('configs/v2_memory_policy.json',dict(warmup=1,measured_cycles=3,allocated_end='C1==C2==C3 exact bytes',reserved_end='C1==C2==C3 exact bytes',peaks='descriptive only',cleanup='no added GC/empty_cache; canonical function scope release only',semantic_repeat_sequence='initial semantic smoke; production warmup; each measured cycle semantic smoke then original production request then endpoint memory',process_exit='normal, atomic result; PID and all observed descendants absent; NVML PID absent; exact baseline bytes',exit_timeout_seconds=10,exit_poll_interval_seconds=1,external_GPU_activity='INCONCLUSIVE_EXTERNAL_GPU_ACTIVITY fail-closed',extra_tolerance_bytes=0))
 A.save('configs/v2_protocol.json',dict(task=A.TASK,parent_HEAD=A.PARENT,runtime=runtime,deterministic_flags=flags,rotation_SHA=A.SHAS,manual_HF=True,one_model_one_rotation_per_fresh_worker=True,seed=1,model_resident=True,warmup=1,measured_cycles=3,same_GPU_exact_repeatability=True,cross_GPU_numerical_equality='NOT_REQUIRED',A800_3090_policy='NO_PREEXISTING_CROSS_HARDWARE_TOLERANCE',V1_result='FAIL',V1_final_allocated_bytes=8519680,V1_root_cause='UNRESOLVED_AFTER_OBSERVER_HYPOTHESIS_REJECTED',V2_does_not_override_V1=True,new_training='NO',new_AIME_generation=0,scoring='NO',stop_after_readiness=True))
 # Existing assignment includes only metadata here: never fetch AIME problem text.
 assignment=A.read(A.OLD/'configs/hardware_assignment.json')
 jobs=assignment['jobs'] if 'jobs' in assignment else assignment.get('assignments',[])
 if isinstance(jobs,dict):jobs=list(jobs.values())
 primary=[j for j in jobs if j.get('condition')=='C5' or j.get('method') in ('c5','QWEN_INT8_C128_KEY_C5')]
 if not primary:primary=jobs[:60]
 A.require(len(primary)==60,'historical assignment structure needs explicit audit')
 future=[]
 for rank,j in enumerate(primary):
  seed=j.get('seed',j.get('sampling_seed'));gpu=j.get('gpu',j.get('physical_gpu',rank%4))
  A.require(int(gpu)==rank%4 and seed in (1,2),'historical assignment differs')
  future.append(dict(rank=rank,problem_id=j['problem_id'],seed=seed,gpu=int(gpu),GPU_uuid=A.UUIDS[int(gpu)],conditions=['C6R','C7']))
 A.save('configs/v2_future_canonical_assignment.json',dict(status='PROSPECTIVE_FROZEN_PLAN_NOT_AUTHORIZED_TO_RUN',activation_requires_all_V2_PASS=True,source=str(A.OLD/'configs/hardware_assignment.json'),source_sha256=A.sha(A.OLD/'configs/hardware_assignment.json'),jobs=future,same_pair_same_GPU=True,condition_order='C6R_THEN_C7',earlier_order_source=str(A.PRIOR/'scripts/amended_canonical3090_pipeline.py')))
 A.save('configs/v2_future_canonical_protocol.json',dict(status='PROSPECTIVE_FROZEN_NOT_STARTED',conditions={'C6R':60,'C7':60,'C5R':0},questions=30,seeds=[1,2],max_new_tokens=81920,manual_HF=True,canonical_config=A.read(A.OLD/'configs/formal_config.json'),formal_config_sha256=A.sha(A.OLD/'configs/formal_config.json'),condition_order='PREEXISTING_FROZEN_ORDER_C6R_THEN_C7',raw_outputs_freeze_before_Frozen_V4=True,new_AIME_generation=0))
 A.require(g.MAX_NEW_TOKENS==81920,'historical max changed')
 A.save('provenance/historical_81920_runtime_compatibility.json',dict(status='PASS',evidence=[str(A.OLD/'configs/formal_config.json'),str(A.PRIOR/'protocol/canonical_endpoint_amended.json'),str(A.V1/'provenance/protected_artifact_final_revalidation.json')],base_model_tokenizer_runtime_sources_unchanged=True,canonical_generation_function_hash=identifiers['generate_one'],cache_implementation='same frozen Transformers DynamicCache and GDN recurrent/conv state shapes; bank never enters cache',dtype='bfloat16 model / fp32 frozen bank',flags=flags,max_new_tokens_historical=81920,V2_request_limit=128,limitation='128-token smoke is NOT an 81920-token memory proof; long runtime capability is inherited only from unchanged historical deployment architecture'))
 paths=[p for p in A.ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'logs' not in p.relative_to(A.ROOT).parts]
 m=A.ROOT/'manifests/v2_premeasurement_sha256.txt';m.parent.mkdir(parents=True,exist_ok=True)
 with m.open('x') as f:
  for p in sorted(paths):f.write(A.sha(p)+'  '+str(p)+'\n')
 A.verify_manifest()
 print('V2_PREMEASUREMENT_CPU_GATES_PASS_READY_FOR_COMMIT_A',flush=True)
if __name__=='__main__':main()
