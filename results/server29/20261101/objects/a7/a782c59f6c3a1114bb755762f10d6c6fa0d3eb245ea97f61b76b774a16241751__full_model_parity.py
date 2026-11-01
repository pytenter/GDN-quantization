"""Non-AIME fixed-input two-cache comparison through real downstream modules.

Both references are validation-only. No answers, generation or score selection.
The old raw-V replay is extended to period16 only for numerical reference.
"""
import argparse,json,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import pace_runtime as R
from validate_unroll import ReplayReference,metrics
from lq_unroll_buffer import LQUnrollBuffer
from model_engineering import runtime_identity

def run(output):
    model,tokenizer,native=R._load_frozen_model(ROOT)
    ident=runtime_identity();masks,masksha,_=R.load_head_mask(ROOT/'calibration/head_mask.json')
    fixture=json.loads((ROOT/'protocol/non_aime_fixture.json').read_text())['tokens']
    prompt=fixture[:16];decode=(fixture[16:]+fixture)*2
    rows=[];contexts={};handles=[]
    layers=R._layers(model)
    def hook(i):
        def capture(module,args,value):
            tensor=value[0] if isinstance(value,(tuple,list)) else value
            assert isinstance(tensor,torch.Tensor)
            contexts[i]=tensor.detach().clone()
        return capture
    for i in masks:
        handles.append(layers[i].linear_attn.register_forward_hook(hook(i)))
    for period in (8,16):
        ref=R.PACEAdapter(model,native,masks,period,len(prompt),buffer_factory=ReplayReference)
        direct=R.PACEAdapter(model,native,masks,period,len(prompt),buffer_factory=lambda p,m,**kw:LQUnrollBuffer(p,m,verify_batched_keys=True,**kw))
        ca=cb=None
        for index,token in enumerate(prompt+decode[:33]):
            contexts.clear()
            with torch.inference_mode(),ref:
                a=model(input_ids=torch.tensor([[token]],device='cuda:0'),past_key_values=ca,use_cache=True)
                ca=a.past_key_values;ref.assert_cache(ca,index+1)
            old=dict(contexts);contexts.clear()
            with torch.inference_mode(),direct:
                b=model(input_ids=torch.tensor([[token]],device='cuda:0'),past_key_values=cb,use_cache=True)
                cb=b.past_key_values;direct.assert_cache(cb,index+1)
            marker_equal=all(torch.equal(ca.layers[i].recurrent_states[0],cb.layers[i].recurrent_states[0]) for i in masks)
            assert marker_equal
            row=dict(period=period,input_forward=index+1,prefill=index<len(prompt),
                logits=metrics(b.logits,a.logits),native_markers_equal=marker_equal,
                layer_outputs={str(i):metrics(contexts[i],old[i]) for i in masks})
            assert row['logits']['nonfinite_count']==0 and all(x['nonfinite_count']==0 for x in row['layer_outputs'].values())
            if row['prefill']:
                assert row['logits']['mismatch_count']==0 and all(x['mismatch_count']==0 for x in row['layer_outputs'].values()),'Original prefill was changed'
            rows.append(row)
            if index%8==0:print(json.dumps(dict(phase='REAL_MODEL_PARITY',period=period,forward=index+1,logit_nrmse=row['logits']['normalized_RMSE'])),flush=True)
        for obj in (ref,direct):
            for buf in obj.buffers.values():buf.finalize('preserve')
        del a,b,ca,cb,ref,direct,old;contexts.clear();gc.collect();torch.cuda.empty_cache()
    for h in handles:h.remove()
    result=dict(status='COMPLETE_NON_AIME_REAL_DOWNSTREAM_COMPARISON',runtime=ident,
        head_mask_sha256=masksha,rows=rows,model_loads=1,new_AIME_samples=0,
        comparisons=len(rows),max_logits_normalized_RMSE=max(x['logits']['normalized_RMSE'] for x in rows),
        max_layer_output_normalized_RMSE=max(v['normalized_RMSE'] for x in rows for v in x['layer_outputs'].values()),
        first_logits_divergence={str(p):next((x['input_forward'] for x in rows if x['period']==p and x['logits']['mismatch_count']),None) for p in (8,16)},
        unchanged_prefill_bitwise=True,actual_markers_equal=True,
        scope='Teacher-forced non-AIME inputs through actual32-layer BF16 model; logits and24GDN module outputs. Divergent quantized trajectories are not claimed bitwise equivalent.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
