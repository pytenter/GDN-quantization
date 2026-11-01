"""Original production capture, parameterized only by frozen input length/path.

The read-only instrumentation is the same as the passed256-token capture.
No copied recurrence/quantization formula and no patched model computation.
"""
import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,ROOT,FORMAL,read,save,sha,tensor_sha,pt_save,load_module

def execute(mode,document):
    import torch
    policy_path=AMEND/'protocol/extended_capture_policy.json'
    p=read(policy_path);receipt=read(AMEND/'extended_capture_freeze.json')
    assert sha(policy_path)==receipt['policy_sha256']
    assert sha(Path(__file__))==receipt['script_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(p['physical_gpu'])
    base=p['production_policy']
    if mode=='engineering2048':
        assert document is None
        item=p['engineering2048'];required=[32,2048]
        for phase in ['factorization','schema','fp','invariance','decomposition','isolation']:
            assert read(AMEND/'engineering/gates'/(phase+'.json'))['gate']=='PASS'
    else:
        assert document in [1,2,3,4]
        item=p['documents'][document-1];required=[2048,8192,32768]
        authorization=read(AMEND/'scientific_execution_authorization.json')
        assert authorization['DISCOVERY_EXECUTION_AUTHORIZED']=='YES'
        assert authorization['PREMEASUREMENT_COMMIT_SHA'] is not None
        for phase in ['factorization','schema','fp','invariance','decomposition','isolation','determinism','streaming']:
            assert read(AMEND/'engineering/gates'/(phase+'.json'))['gate']=='PASS'
        if document>1:
            previous=read(Path(p['documents'][document-2]['output_directory'])/'drive_release_record.json')
            assert previous['release_complete'] and previous['all_result_hashes_verified']
    n=item['token_count'];out=Path(item['output_directory'])
    assert n==(2048 if mode=='engineering2048' else 32768)
    st=os.statvfs(ROOT);available=st.f_bavail*st.f_frsize
    assert available>=item['required_free_bytes'],'Storage capacity check failed before capture'
    out.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()),'new empty capture directory required; no retry'
    for path,expected in base['source_sha256'].items():assert sha(path)==expected
    assert sha(item['token_file'])==item['token_file_sha256']
    all_ids=read(item['token_file']);ids=all_ids[:n];assert len(ids)==n
    print('MODEL_IDENTITY_VERIFY_START',flush=True)
    for row in base['model_identity']['files']:assert sha(row['path'])==row['sha256']
    print('MODEL_IDENTITY_VERIFY_PASS',flush=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(False)
    F=load_module(FORMAL,'extended_original_formal');F.setup_seed(base['capture_seed'])
    model,tokenizer=F.load_model_and_tokenizer()
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    assert Path(qmod.__file__).resolve()==Path(next(k for k in base['source_sha256'] if k.endswith('modeling_qwen3_5.py'))).resolve()
    layers=list(F.GDN_LAYERS);assert layers==base['expected_layers']
    model_layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
    active=[None];current=[1];handles=[];saved=[]
    buffer={l:[] for l in layers};state_hashes=[];shards=[];checkpoints=[];step_records={}
    counts={'chunk':0,'recurrent':0};dtypes={};fields=base['tensors'];start=time.monotonic()
    initial_available=available;minimum_available=available
    for layer in layers:
        block=model_layers[layer];module=getattr(block,'linear_attn',None) or getattr(block,'self_attn',None)
        assert module is not None
        handles.append(module.register_forward_pre_hook(lambda _m,_a,l=layer:active.__setitem__(0,l)))

    def wrap(name,function):
        def captured(query,key,value,*args,**kwargs):
            assert not args
            layer=active[0];step=current[0]
            assert layer in layers and layer not in step_records
            assert kwargs.get('use_qk_l2norm_in_kernel') is True and kwargs.get('output_final_state') is True
            assert kwargs.get('cu_seqlens') is None
            record={k:t.detach().cpu().clone() for k,t in [('query',query),('key',key),('value',value),('g',kwargs['g']),('beta',kwargs['beta'])]}
            for key,t in record.items():
                assert str(t.dtype)==base['expected_dtype'][key]
                assert list(t.shape)==(base['required_qkv_shape'] if key in ['query','key','value'] else base['required_g_beta_shape'])
                assert bool(torch.isfinite(t).all());dtypes[key]=str(t.dtype)
            result,state=function(query,key,value,**kwargs)
            assert list(state.shape)==[1,32,128,128] and str(state.dtype)==base['expected_dtype']['state']
            state_cpu=state.detach().cpu().clone();assert bool(torch.isfinite(state_cpu).all())
            record['_state_sha256']=tensor_sha(state_cpu)
            if step in required:record['_state_checkpoint']=state_cpu
            step_records[layer]=record;counts[name]+=1
            return result,state
        return captured

    def flush(first,last):
        payload={'first_token':first,'last_token':last,'layers':{}};metadata=[]
        for layer in layers:
            rows=buffer[layer];assert len(rows)==last-first+1
            payload['layers'][str(layer)]={}
            for name in fields:
                t=torch.cat([r[name] for r in rows],dim=1).contiguous()
                payload['layers'][str(layer)][name]=t
                metadata.append(dict(layer=layer,name=name,shape=list(t.shape),dtype=str(t.dtype),sha256=tensor_sha(t)))
        path=out/'drive_shards'/('tokens_%05d_%05d.pt'%(first,last));pt_save(path,payload)
        shards.append(dict(relative_path=str(path.relative_to(out)),file_sha256=sha(path),file_size_bytes=path.stat().st_size,
            first_token=first,last_token=last,tensors=metadata))
        for layer in layers:buffer[layer].clear()

    for helper,role in [('torch_chunk_gated_delta_rule','chunk'),('torch_recurrent_gated_delta_rule','recurrent')]:
        fn=getattr(qmod,helper);saved.append((helper,fn));setattr(qmod,helper,wrap(role,fn))
    try:
        with torch.inference_mode():
            cache=None
            for step,token in enumerate(ids,1):
                current[0]=step;step_records.clear()
                kw=dict(input_ids=torch.tensor([[token]],dtype=torch.long,device='cuda:0'),use_cache=True)
                if cache is not None:kw['past_key_values']=cache
                result=model(**kw);cache=result.past_key_values
                assert sorted(step_records)==layers
                hashes={str(l):step_records[l]['_state_sha256'] for l in layers}
                # Every-token identity is checked in acquisition. Engineering
                # stores all hashes; science retains only sparse reference hashes.
                if mode=='engineering2048' or step in required:state_hashes.append(dict(token=step,per_layer_sha256=hashes))
                for layer in layers:
                    assert tensor_sha(F.get_state(cache,layer))==hashes[str(layer)]
                    buffer[layer].append({name:step_records[layer][name] for name in fields})
                if step in required:
                    path=out/'fp_reference'/('token_%05d.pt'%step)
                    values={str(l):step_records[l]['_state_checkpoint'] for l in layers};pt_save(path,values)
                    checkpoints.append(dict(token=step,relative_path=str(path.relative_to(out)),file_sha256=sha(path),
                        per_layer_sha256={l:tensor_sha(t) for l,t in values.items()},file_size_bytes=path.stat().st_size))
                if step%base['shard_tokens']==0 or step==n:flush(step-len(buffer[layers[0]])+1,step)
                st=os.statvfs(ROOT);minimum_available=min(minimum_available,st.f_bavail*st.f_frsize)
                assert minimum_available>=p['minimum_runtime_free_bytes'],'Runtime disk safety floor failed; preserve incomplete drive'
                if step%256==0:print('CAPTURE_TOKEN='+str(step),flush=True)
    finally:
        for helper,fn in saved:setattr(qmod,helper,fn)
        for h in handles:h.remove()
    assert counts=={'chunk':24,'recurrent':24*(n-1)}
    assert len(shards)==math.ceil(n/base['shard_tokens']) and len(checkpoints)==len(required)
    hashes_path=out/'fp_state_hashes.json';save(hashes_path,state_hashes)
    manifest=dict(status='SEALED_VALID_CAPTURE',mode=mode,scientific_measurement=mode=='discovery',
        doc_id=item['doc_id'],doc_text_sha256=item['text_sha256'],fixture_token_ids_file_sha256=item['token_file_sha256'],
        token_count=n,layer_ids=layers,source_sha256=base['source_sha256'],model_identity=base['model_identity'],
        capture_policy_sha256=sha(policy_path),capture_script_sha256=sha(Path(__file__)),freeze_commit=receipt['freeze_commit'],
        physical_gpu=p['physical_gpu'],runtime={'torch':torch.__version__,'cuda':torch.version.cuda,
            'deterministic_enforcement':False,'tf32':False,'model_dtype':'torch.bfloat16'},
        tensors=fields,dtypes=dtypes,normalization='Original helper, inputs before normalization',
        initial_state='None at first token; original chunk zero initialization; no quantization/intervention',
        original_production_call_counts=counts,shards=shards,fp_reference_checkpoints=checkpoints,
        fp_state_hashes_file='fp_state_hashes.json',fp_state_hashes_file_sha256=sha(hashes_path),
        all_complete_finite=True,generated_tokens=0,model_forward_count=n,full_hidden_trajectory_saved=False,
        elapsed_seconds=time.monotonic()-start,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),initial_available_bytes=initial_available,
        minimum_observed_available_bytes=minimum_available,observed_disk_consumption_bytes=initial_available-minimum_available)
    save(out/'drive_manifest.json',manifest)
    verified=[]
    for row in shards+checkpoints:
        path=out/row['relative_path'];assert sha(path)==row['file_sha256']
        verified.append(dict(path=str(path),sha256=row['file_sha256']))
    save(out/'capture_seal.json',dict(DOCUMENT_DRIVE_SEAL_GATE='PASS',drive_manifest_sha256=sha(out/'drive_manifest.json'),
        verified_files=verified,read_only=True,engineering_only=mode=='engineering2048'))
    print('EXTENDED_CAPTURE_SEALED_PASS',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['engineering2048','discovery'],required=True)
    parser.add_argument('--document',type=int);a=parser.parse_args()
    try:execute(a.mode,a.document)
    except Exception as e:
        p=AMEND/'engineering/extended_capture.error.json' if a.mode=='engineering2048' else AMEND/'scientific'/('doc_%02d_capture.error.json'%a.document)
        if not p.exists():save(p,dict(status='FAIL_AND_STOP',message=str(e),exception=type(e).__name__,
            traceback=traceback.format_exc(),retry_authorized=False,drive_release_allowed=False))
        raise

if __name__=='__main__':main()
