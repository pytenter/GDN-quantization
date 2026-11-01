"""Instrumentation-only original production teacher-forced drive acquisition.

The hooks copy unchanged helper arguments/results. No modified cumsum, no
quantizer, no rotation, no autoregressive generation, no state intervention.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT=REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
AMEND=ROOT/'amendments/STORAGE_ACCOUNTING_CORRECTION_V1'
POLICY=AMEND/'protocol/capture_engineering_preregistration.json'
FORMAL=Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('write-once artifact already exists: '+str(path))
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.rename(temp,path)
    path.chmod(0o444)

def tensor_sha(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def pt_save(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('write-once tensor artifact exists')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.rename(temp,path);path.chmod(0o444)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def execute(run):
    import torch
    p=read(POLICY)
    frozen=read(AMEND/'capture_freeze_receipt.json')
    assert sha(POLICY)==frozen['capture_policy_sha256']
    assert sha(Path(__file__))==frozen['capture_script_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(p['physical_gpu'])
    out=AMEND/'engineering'/('capture_'+run)
    out.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()),'capture directory must be new and empty; no retry'
    for path,expected in p['source_sha256'].items():assert sha(path)==expected,'canonical source changed'
    assert sha(p['fixture_file'])==p['fixture_file_sha256']
    ids=read(p['fixture_file']);assert len(ids)==256
    print('MODEL_IDENTITY_VERIFY_START',flush=True)
    for row in p['model_identity']['files']:
        assert sha(row['path'])==row['sha256'],'model file identity changed: '+row['path']
    print('MODEL_IDENTITY_VERIFY_PASS',flush=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    F=load_module(FORMAL,'frozen_drive_original_formal')
    F.setup_seed(p['capture_seed'])
    # Capture only: original cumsum is allowed, unchanged; replay separately
    # requires strict deterministic enforcement. No F.setup generation calls.
    model,tokenizer=F.load_model_and_tokenizer()
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    assert Path(qmod.__file__).resolve()==Path(next(path for path in p['source_sha256'] if path.endswith('modeling_qwen3_5.py'))).resolve()
    layer_ids=list(F.GDN_LAYERS);assert layer_ids==p['expected_layers']
    model_layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
    active=[None];current=[1];handles=[];saved=[]
    buffer={l:[] for l in layer_ids};state_hashes=[];shards=[];checkpoints=[]
    step_records={};call_counts={'chunk':0,'recurrent':0};original_tensor_dtypes={}
    fields=p['tensors'];start=time.monotonic()
    for layer in layer_ids:
        block=model_layers[layer]
        module=getattr(block,'linear_attn',None) or getattr(block,'self_attn',None)
        assert module is not None
        handles.append(module.register_forward_pre_hook(lambda _m,_a,l=layer:active.__setitem__(0,l)))

    def wrap(name,function):
        def captured(query,key,value,*args,**kwargs):
            assert not args,'unexpected positional helper controls'
            layer=active[0];step=current[0]
            assert layer in layer_ids and layer not in step_records
            assert kwargs.get('use_qk_l2norm_in_kernel') is True
            assert kwargs.get('output_final_state') is True
            assert kwargs.get('cu_seqlens') is None
            record={n:t.detach().cpu().clone() for n,t in [('query',query),('key',key),('value',value),('g',kwargs['g']),('beta',kwargs['beta'])]}
            for n,t in record.items():
                assert str(t.dtype)==p['expected_dtype'][n], 'drive dtype mismatch '+n
                assert list(t.shape)==(p['required_qkv_shape'] if n in ['query','key','value'] else p['required_g_beta_shape'])
                assert bool(torch.isfinite(t).all()),'nonfinite drive'
                original_tensor_dtypes[n]=str(t.dtype)
            # Invoke the unchanged decorated production helper with unchanged
            # controls. Read-only copies do not replace its state or inputs.
            result,state=function(query,key,value,**kwargs)
            assert list(state.shape)==[1,32,128,128] and str(state.dtype)==p['expected_dtype']['state']
            state_cpu=state.detach().cpu().clone()
            assert bool(torch.isfinite(state_cpu).all()),'nonfinite captured reference state'
            record['_state_sha256']=tensor_sha(state_cpu)
            if step in p['FP_reference_tensor_checkpoints']:record['_state_checkpoint']=state_cpu
            step_records[layer]=record
            call_counts[name]+=1
            return result,state
        return captured

    def flush_shard(first,last):
        payload={'first_token':first,'last_token':last,'layers':{}}
        tensor_metadata=[]
        for layer in layer_ids:
            rows=buffer[layer];assert len(rows)==last-first+1
            payload['layers'][str(layer)]={}
            for n in fields:
                t=torch.cat([r[n] for r in rows],dim=1).contiguous()
                payload['layers'][str(layer)][n]=t
                tensor_metadata.append(dict(layer=layer,name=n,shape=list(t.shape),dtype=str(t.dtype),sha256=tensor_sha(t)))
        path=out/'drive_shards'/('tokens_%05d_%05d.pt'%(first,last))
        pt_save(path,payload)
        shards.append(dict(relative_path=str(path.relative_to(out)),file_sha256=sha(path),file_size_bytes=path.stat().st_size,
            first_token=first,last_token=last,tensors=tensor_metadata))
        for layer in layer_ids:buffer[layer].clear()

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
                assert sorted(step_records)==layer_ids,'capture layer/token coverage mismatch'
                hashes={str(l):step_records[l]['_state_sha256'] for l in layer_ids}
                state_hashes.append(dict(token=step,per_layer_sha256=hashes))
                for layer in layer_ids:
                    actual=F.get_state(cache,layer)
                    assert tensor_sha(actual)==hashes[str(layer)],'cache writeback differs from captured state'
                    buffer[layer].append({n:step_records[layer][n] for n in fields})
                if step in p['FP_reference_tensor_checkpoints']:
                    path=out/'fp_reference'/('token_%05d.pt'%step)
                    values={str(l):step_records[l]['_state_checkpoint'] for l in layer_ids}
                    pt_save(path,values)
                    checkpoints.append(dict(token=step,relative_path=str(path.relative_to(out)),file_sha256=sha(path),
                        per_layer_sha256={l:tensor_sha(t) for l,t in values.items()},file_size_bytes=path.stat().st_size))
                if step%p['shard_tokens']==0 or step==len(ids):flush_shard(step-len(buffer[layer_ids[0]])+1,step)
                if step%32==0:print('CAPTURE_TOKEN='+str(step),flush=True)
    finally:
        for helper,fn in saved:setattr(qmod,helper,fn)
        for h in handles:h.remove()
    assert call_counts=={'chunk':24,'recurrent':24*255}
    assert len(state_hashes)==256 and len(shards)==4 and len(checkpoints)==2
    hashes_path=out/'fp_state_hashes.json';save(hashes_path,state_hashes)
    manifest=dict(status='SEALED_VALID_ENGINEERING_CAPTURE',run=run,role=p['acquisition_role'],scientific_measurement=False,
        doc_id=p['fixture_doc_id'],fixture_token_ids_file_sha256=p['fixture_file_sha256'],token_count=256,layer_ids=layer_ids,
        source_sha256=p['source_sha256'],model_identity=p['model_identity'],capture_policy_sha256=sha(POLICY),
        capture_script_sha256=sha(Path(__file__)),freeze_commit=frozen['engineering_freeze_commit'],
        physical_gpu=p['physical_gpu'],runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'deterministic_enforcement':False,
            'tf32':False,'model_dtype':'torch.bfloat16'},tensors=fields,dtypes=original_tensor_dtypes,
        normalization='Original production helper use_qk_l2norm_in_kernel=True; captured inputs are BEFORE normalization',
        initial_state='None at first token; original chunk helper zero initialization; no quantization or intervention',
        original_production_call_counts=call_counts,shards=shards,fp_reference_checkpoints=checkpoints,
        fp_state_hashes_file='fp_state_hashes.json',fp_state_hashes_file_sha256=sha(hashes_path),
        all_complete_finite=True,generated_tokens=0,model_forward_count=256,
        full_hidden_trajectory_saved=False,elapsed_seconds=time.monotonic()-start,
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    save(out/'drive_manifest.json',manifest)
    verified=[]
    for row in shards+checkpoints:
        path=out/row['relative_path'];assert sha(path)==row['file_sha256']
        verified.append(dict(path=str(path),sha256=row['file_sha256']))
    save(out/'capture_seal.json',dict(DOCUMENT_DRIVE_SEAL_GATE='PASS',drive_manifest_sha256=sha(out/'drive_manifest.json'),
        verified_files=verified,read_only=True,engineering_only=True))
    print('ENGINEERING_CAPTURE_'+run+'_SEALED_PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',choices=['01','02'],required=True);args=parser.parse_args()
    try:execute(args.run)
    except Exception as e:
        path=AMEND/'engineering'/('capture_'+args.run)/'capture.error.json'
        if not path.exists():save(path,dict(status='FAIL_AND_STOP',exception=type(e).__name__,message=str(e),traceback=traceback.format_exc(),
            retry_authorized=False,scientific_measurement_count=0,drive_release_allowed=False))
        raise
