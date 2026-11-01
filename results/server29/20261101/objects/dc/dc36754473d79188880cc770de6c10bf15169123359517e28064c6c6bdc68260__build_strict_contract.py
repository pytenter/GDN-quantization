"""Stdlib-only binding of sealed prior evidence to prospective Strict qualification."""
import ast,hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent.parent
P=R.parent/'QWEN_GDN_RUNTIME_THROUGHPUT_ROOT_CAUSE_V1'
H=R.parent/'qwen_gdn_historical_throughput_reconciliation_v1/delivery'
S=R.parent/'qwen_gdn_dual_m_equivalent_speedup_v1/delivery'
E=R/'evidence/previous';E.mkdir(exist_ok=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
manifest=read(P/'SHA256_MANIFEST.json');entries=[]
for x in manifest['files']:
 p=P/x['path'];raw=p.read_bytes();entries.append(dict(path=str(p),bytes=len(raw),sha256=sha(raw),expected_bytes=x['bytes'],expected_sha256=x['sha256'],status='PASS' if len(raw)==x['bytes'] and sha(raw)==x['sha256'] else 'FAIL'))
sums=[]
for line in (P/'SHA256SUMS.txt').read_text().splitlines():
 expected,name=line.split('  ',1);raw=(P/name).read_bytes();sums.append(dict(path=name,expected_sha256=expected,sha256=sha(raw),status='PASS' if sha(raw)==expected else 'FAIL'))
assert len(entries)==manifest['files_count']==80 and all(x['status']=='PASS' for x in entries+sums)
assert {(x['path'],x['sha256']) for x in manifest['files']}=={(x['path'],x['sha256']) for x in sums if x['path']!='SHA256_MANIFEST.json'}
assert sum(x['path']=='SHA256_MANIFEST.json' for x in sums)==1
pr=P/'qualification/GPU_DIAGNOSTIC_PREREGISTRATION.json';assert sha(pr.read_bytes())=='404e2b2e07519434e589acfe71a2a1355b224a038d3d8d707f2eb94885ea1884'
copyfiles={
 'ROOTCAUSE_SHA256_MANIFEST.json':P/'SHA256_MANIFEST.json','ROOTCAUSE_SHA256SUMS.txt':P/'SHA256SUMS.txt','ROOTCAUSE_GPU_DIAGNOSTIC_PREREGISTRATION.json':pr,
 'ROOTCAUSE_FINAL_REPORT.md':P/'FINAL_REPORT.md','ROOTCAUSE_CALLPATH_DIFF.md':P/'analysis/CALLPATH_DIFF.md','ROOTCAUSE_TRACE_KERNEL_DELTA.json':P/'analysis/trace_kernel_delta.json',
 'ROOTCAUSE_ENVIRONMENT_SOURCE_VERIFICATION.json':P/'analysis/environment_remote_source_verification.json','ROOTCAUSE_ENVIRONMENT_METADATA_VERIFICATION.json':P/'analysis/environment_remote_metadata_verification.json',
 'ROOTCAUSE_ENVIRONMENT_MEASUREMENTS.json':P/'analysis/environment_historical_measurements.json','ROOTCAUSE_SOURCE_AND_ENVIRONMENT_AUDIT.md':P/'SOURCE_AND_ENVIRONMENT_AUDIT.md',
 'HIST_SOURCE_SHA256_MANIFEST.json':H/'SOURCE_SHA256_MANIFEST.json','SPEED_SOURCE_MANIFEST.json':S/'SOURCE_MANIFEST.json','SPEED_REFERENCE_BINDING.json':S/'REFERENCE_BINDING.json',
 'HIST_SHARED_CHECKPOINT_IDENTITY.json':H/'SHARED_CHECKPOINT_IDENTITY.json','HIST_NEW_FP_ENVIRONMENT.json':H/'execution/NEW_FP_R0/ENVIRONMENT.json','HIST_OLD_FP_ENVIRONMENT.json':H/'execution/OLD_FP_R0/ENVIRONMENT.json',
 'SPEED_BASELINE_WORKER.py':S/'source/baseline_worker.py','HIST_MATCHED_WORKER.py':H/'source/matched_worker.py'
}
copies=[]
for name,p in copyfiles.items():
 raw=p.read_bytes();target=E/name
 if target.exists():assert target.read_bytes()==raw
 else:target.write_bytes(raw)
 copies.append(dict(source=str(p),copy=str(target),bytes=len(raw),sha256=sha(raw),exact_copy=True))
verification=dict(status='PASS_PRIOR_SEAL_AND_PREREG_RAW_IDENTITY',prior_root=str(P),prior_status=manifest['scientific_status'],prior_manifest_sha256=sha((P/'SHA256_MANIFEST.json').read_bytes()),manifest_entries=entries,sums_entries=sums,copies=copies,prior_FINAL_SEAL_RECEIPT='NOT_PRESENT_NOT_REQUIRED_BY_PRIOR_TASK',no_remote_trace_reparse=True,GPU_calls=0)
save(R/'analysis/contract_prior_evidence_verification.json',verification)
hm=read(H/'SOURCE_SHA256_MANIFEST.json');sm=read(S/'SOURCE_MANIFEST.json');binding=read(S/'REFERENCE_BINDING.json');env=read(H/'execution/NEW_FP_R0/ENVIRONMENT.json')
remoteH='/data/zypan/analyses/qwen_gdn_historical_throughput_reconciliation_v1';remoteS='/data/zypan/analyses/qwen_gdn_dual_m_equivalent_speedup_v1';future=str(Path(binding['core']['path']).parent.parent)
required={}
for root,m in [(remoteH,hm),(remoteS,sm)]:
 for category in ('execution_sources','external_dependencies'):
  for x in m[category]:
   path=str(Path(root)/x['path']);y=dict(x,path=path)
   if path in required:assert required[path]['sha256']==y['sha256'] and required[path]['bytes']==y['bytes']
   else:required[path]=y
sources={k:binding[k] for k in ['core','worker','codec']}
source_find={
 'baseline_worker':remoteS+'/source/baseline_worker.py',
 'architecture':'/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py',
 'formal_runner':'/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/formal_c128_runner.py',
 'numerical_recovery':'/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/runtime_sources/numerical_recovery.py',
 'dual_axis_codec':'/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1/src/dual_axis_codec.py',
 'matched_worker':remoteH+'/source/matched_worker.py',
 'old_runner':'/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/aime26/81920/run_qwen_aime26_formal.py'
}
for k,path in source_find.items():sources[k]=required[path]
fixture=dict(binding['fixture']);fixture['tokens_path']=str(Path(remoteS)/fixture['tokens_path']);fixture.update(warmup_total_forwards=128,measured_forwards=128,measured_positions=[128,255],batch=1,seed=20261015,sampling='NOT_RUN_FIXED_INPUT_REPLAY')
weights=read(H/'SHARED_CHECKPOINT_IDENTITY.json')['files'];newtokenizer=read(R/'analysis/contract_current_model_tokenizer_hashes.json')
for x in newtokenizer:
 if x['path'] in required:assert required[x['path']]['sha256']==x['sha256'] and required[x['path']]['bytes']==x['bytes']
 else:required[x['path']]={k:x[k] for k in ('path','bytes','sha256')}
libs=read(R/'analysis/contract_prospective_library_hashes.json')
for x in libs:
 if x['status']=='STABLE_CURRENT_CPU_HASH':
  required[x['path']]={k:x[k] for k in ('path','bytes','sha256')}
contract={
 'task_id':'QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1','status':'CPU_RESTORED_STRICT_CONTRACT_GPU_QUALIFICATION_NOT_RUN','ALLOW_GPU_DIAG':'NO','created_unix':time.time(),
 'remote_roots':{'HIST':remoteH,'SPEED':remoteS,'FUTURE':future,'MODEL':'/data/zypan/modelscope_models/Qwen3.5-9B','TRANSFORMERS':'/data/zypan/transformers-qwen35','STATE_QUANT':'/data/zypan/experiments/qwen35_gdn_quant'},
 'source_bindings':sources,'required_files':list(required.values()),
 'prospective_library_bindings':libs,
 'prospective_library_scope':f"CPU-stream-hashed{sum(x['status']=='STABLE_CURRENT_CPU_HASH' for x in libs)}physicalpathbindings/{len(set(x['resolved_path'] for x in libs if x['status']=='STABLE_CURRENT_CPU_HASH'))}uniquephysicalfiles; noimports/downloads; condaandpipcuBLAScoexist; actualselectedloadedmapsMUSTbecheckedinfutureauthorizedB0/B1, historicalselectionUNKNOWN. MissingcondacuDNNcandidateisUNKNOWNnotCPUfailure.",
 'reference_binding':{'path':remoteS+'/REFERENCE_BINDING.json','local_path':str(E/'SPEED_REFERENCE_BINDING.json'),'bytes':(S/'REFERENCE_BINDING.json').stat().st_size,'sha256':sha((S/'REFERENCE_BINDING.json').read_bytes())},
 'reference_binding_content':binding,'closure_manifests':binding['manifests'],'fixture':fixture,
 'model':{'path':'/data/zypan/modelscope_models/Qwen3.5-9B','dtype':'torch.bfloat16','device_map':{'':0},'eval':True,'requires_grad':False,'local_files_only':True,'trust_remote_code':True,'tokenizer_class':'Qwen2Tokenizer','current_nonweight_file_hashes':newtokenizer,'weight_shards':weights,'weight_proof_scope':'REUSED_PREVIOUS_CURRENT_SHARED_CHECKPOINT_STREAM_SHA; NOT_FRESH_WEIGHT_HASH_THIS_TASK; revalidate_before_GPU_dispatch','historical_original_AIME_full_weights_tokenizer_identity':'UNKNOWN'},
 'environment':{'python':'3.10.18','torch':'2.5.1+cu121','cuda_build':'12.1','transformers':'5.16.0.dev0','transformers_source_commit':'a353632607c59463e6ced86a44c2de3c2cd62d5e','CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','intraop_threads':1,'interop_threads':1,'deterministic_algorithms':True,'deterministic_warn_only':False,'matmul_TF32':False,'cudnn_TF32':False,'cudnn_deterministic':False,'deterministic_uninitialized_memory_fill':True,'fill_historical_dynamic_flag':'NOT_RETAINED; original source no setter; prospective qualification must explicitly requireTrue for both B0/B1','driver_runtime_full_library_binary_identity':'UNKNOWN_NOT_RETAINED; must observe before futureGPUexecution without rewriting historical claims','preferred_GPU_UUID':'GPU-a78176ed-f97f-c54f-6201-5a01b570f422','historical_new_ENVIRONMENT':env},
 'seed':{'value':20261015,'python_random':20261015,'torch_CPU':20261015,'torch_CUDA_all':20261015,'sampler':'NOT_RUN','numpy':'NOT_USED_BY_FIXED_INPUT_CONTRACT'},
 'math_and_state':{'architecture_sha256':sources['architecture']['sha256'],'GDN_layers':[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30],'attention_layers':[3,7,11,15,19,23,27,31],'state_dtype':'torch.float32','state_shape_per_layer':[1,32,128,128],'chunk':'original numerical_recovery.compensated_scan/chunk_binding AST; only firstinput prefill; unchanged','recurrent':'actual 90d architecture torch_recurrent_gated_delta_rule unchanged; wrapper selection source+trace must not be inferred from filename alone','state_read_write_order':'forward consumes incoming cache; recurrent updates native cache; readout/logits use native currentforward; postforward QDQ writes state in-place before nexttoken forquantizedconditions','persistent_carrier_disclosure':'NativeFP32 recurrent-cache QDQ carrier (24x1x32x128x128 =48MiB). Encoded INT8 codes/scales exist in codec path; not physical packedINT8 persistentkernel. Never advertise as packedpersistentINT8.','model_forward_calls':256,'GDN_chunk_calls':24,'GDN_recurrent_calls':6120,'measured_GDN_recurrent_calls':3072,'all_decoder_layer_calls':8192,'position_rule':'native rotary0..255; all8attentionKV length=t+1'},
 'interfaces':{
  'load':'original core.configure(); core.load(); formal(True) cachedH128; no loader/math rewrite','FP':'Observe(model,qmod,formal,on=False); full-model nativeFP with state/logitsfinite gate; noQDQ','qualify':'same original Observe.onTrue or observation-only equivalent; retain allstate/cache/logits byte hashes +q/k/readout/initial-state; hashing allowedoutsideTPS only','H_C128':'formal.QwenKeyHadamardPatch beforeGDNq/k; normalizewhenoriginalflagrequires, FP32H128/sqrt128; core.formal(True) memoizesmatrix; formal.quantize_c128 aftereveryforwardincludingt0','C128_math':'sourceFP32; abs.amax(dim=-2,keepdim=True).clamp_min(1e-12)/127; round(source/scale).clamp(-127,127); dequant=codes*scale; finite; state.copy_(dequant.to(state.dtype)); no timingoftheQDQmoved','C128_validation':'Original worker.original_c128 trace helper captures codes/scales/dequant/pre/post at original lines;24layerwrites and next-consumptionpointer/hash; GDN/logits/cache parity separately required','H_DUAL_M':'Original worker.dual AST; C.trace_encode(DUAL_M_INT8), FP16row/columnscales/columnLS/finalreencode, encoded.decode, scale-risk gates, originalcopyback; do not replacecodec','Dual_M_validation':'codes/rows/columns/dequant/pre/post and nextconsumptionbyte/pointer; officialtorch.equal+signedzero caveat mustpreserve; noautomaticFPtoCodecqualification','cache_bytes':'core.cache_tensors/cache_hash for keys/values/conv_states/recurrent_states; fullmetadata plus original worker.describe/deep_hash whenneeded','GDN_intermediate':'Observe.onTrue q/k/initial/updated_state/current_readout digests at each originalhelper; finite+shape/dtype+24calls; notjustfinaloutput','fresh_repeat':'B0 twice andB1 twice independent processes,256tokenbyte traces; comparewithinB0,withinB1,betweenconditions; earliesttoken/tensor divergence retained','unprofiled_measure':'128warmup+128suffix, windowboundarysync only, common fixedinput, nohash/profile extra synclogic insideTPS; originalfiniteQDQsyncs retained'},
 'qualification_status':{'FP':'NOT_RUN_GPU_NOT_AUTHORIZED','H_C128':'NOT_QUALIFIED_FOR_FAST_THIS_CODEC','H_DUAL_M':'NOT_QUALIFIED_FOR_FAST_THIS_CODEC','fill_off':'FILL_OPT_SAFETY_UNPROVEN_NO_GPU_AUTHORIZATION','fast_script':'NOT_CREATED'},
 'candidate_limits':{'B0':':4096:8','B1':':16:8','only_allowed_workspace_source_delta':'one module-level os.environ CUBLAS assignment literal via exactAST; environment must be setbeforetorch/CUDAimport; numericmathbody untouched','strict_safety_preserved':True,'fill_off_not_in_cublas_task':True,'GPU_stage_admission':'HumanYES+frozennewplan+sourceSHA+freshdevice/resource+numericqualification; currentCPUcontractnotexecutionPASS'},
 'historical_provenance':{'prior_root':str(P),'prior_scientific_status':'DIRECTIONALLY_SUPPORTED_ONLY','prior_manifest_sha256':verification['prior_manifest_sha256'],'prior_prereg_sha256':sha(pr.read_bytes()),'prior_prereg_copy':str(E/'ROOTCAUSE_GPU_DIAGNOSTIC_PREREGISTRATION.json'),'fresh_source_checks_reused':{'historical':201,'speed':175,'all376PASS':True,'proof':str(E/'ROOTCAUSE_ENVIRONMENT_SOURCE_VERIFICATION.json'),'not_new_fresh_remote_verification':True},'new_model_tokenizer_hashes_current_only':True},
 'unresolved':['Historicaldriver/cuBLAS/cuDNNdynamicbinaryversions notretained','OriginalAIMEtokenizerfullbyteproof UNKNOWN','Originalprofilefillflag runtimevalue notexplicitlyrecorded','CurrentfullmodelweightSHA notrereadthisCPUphase; priorcurrentcheckpointproofbound and futureadmissionrequiresfresh','Whole32K/81Kbyteequivalence notclaimed','No fast candidate hasGPUproof orwallmeasurements inthis task']}
save(R/'analysis/STRICT_BASELINE_CONTRACT.json',contract)
print(json.dumps({'prior_files_verified':len(entries),'sums_verified':len(sums),'copies':len(copies),'required_files':len(required),'contract_sha256':sha((R/'analysis/STRICT_BASELINE_CONTRACT.json').read_bytes()),'source_binding_keys':list(sources)},indent=2))
