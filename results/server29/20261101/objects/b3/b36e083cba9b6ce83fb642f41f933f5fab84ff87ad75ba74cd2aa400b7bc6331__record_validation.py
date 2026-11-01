"""Complete payload validation; no scoring, CUDA or answer-dependent dispatch."""
import execution_control as C

def validate(wrapper,job,root):
 C.require(wrapper['experiment_id']==C.TASK and wrapper['codec_source_sha256']==C.sha(root/'src/int4_codec.py') and wrapper['runtime_source_sha256']==C.sha(root/'src/runtime_engine.py'),'INT4 experiment/runtime/codec binding changed')
 row=wrapper['result'];C.require(wrapper['status']=='PASS' and wrapper['unit']==job['id'],'Worker result/unit not PASS')
 copy=dict(wrapper);saved=copy.pop('payload_sha256');C.require(C.objsha(copy)==saved,'Worker payload SHA differs')
 C.require(wrapper['source_freeze_sha256']==C.sha(root/'SOURCE_FREEZE.json') and wrapper['permit_sha256']==C.sha(root/'GPU_PERMIT.json'),'Worker execution source/permit differs')
 C.require(row['status']=='COMPLETE' and not row['nonfinite'] and row['runtime_error'] is None,'Technical/incomplete generation')
 C.require(row['condition']==job['condition'] and row['problem_id']==job['problem_id'] and row['seed']==1 and row['gpu']==job['gpu'] and row['gpu_uuid']==C.UUIDS[job['gpu']],'Configuration/question/GPU binding differs')
 C.require(row['input_ids_sha256']==job['input_ids_sha256'] and row['input_tokens']==job['input_tokens'],'Frozen actual prompt differs')
 ts=row['generated_token_ids'];n=len(ts);C.require(0<n<=81920 and all(type(t)==int and 0<=t<248320 for t in ts) and n==row['generated_token_count'] and C.objsha(ts)==row['generated_token_ids_sha256'],'Token payload/range/hash differs')
 eos=ts[-1] in row['stop_token_ids'];C.require(not any(t in row['stop_token_ids'] for t in ts[:-1]) and row['finish_reason']==('eos' if eos else 'length') and (eos or n==81920),'Illegal EOS/length termination')
 C.require(row['max_new_tokens']==81920,'Actual formal cap changed');total=row['input_tokens']+n-1;q=row['quantizer_audit'];C.require(q['logical_bits']==4 and q['integer_range']==[0,15] and q['coordinate']==('NATIVE' if job['condition']=='I4-N' else 'H128'),'INT4 coordinate/bitwidth audit changed');C.require(row['actual_model_forward_calls']==total and q['tokens']==total and q['qdq_layer_updates']==24*total and row['prompt_layer_QDQ']==24*row['input_tokens'] and row['decode_layer_QDQ']==24*(n-1),'Prompt/token0/decode QDQ count differs')
 C.require(len(row['writebacks_by_layer'])==24 and all(v==total for v in row['writebacks_by_layer'].values()) and row['sampler_calls']==n,'Actual layer/sampling coverage incomplete')
 expected=24*sum(t>0 and (t in range(1,128) or t%2048==0) for t in range(total));C.require(q['next_consumption_checks']==expected,'Sparse actual next-consumption roster differs')
 C.require(row['sampler_source_sha256']==C.sha(root/'protocol/sampler.py') and row['scorer_source_sha256']==C.sha(root/'protocol/aime26_scorer_v4.py') and row['dataset_freeze_sha256']==C.sha(root/'AIME26_DATASET_FREEZE.json'),'Sampler/scorer/actual input identity differs')
 return row
