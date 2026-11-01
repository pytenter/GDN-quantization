"""Gold-free short full-model FP identity, H parity and four-config smoke."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import pace_runtime as R
from hmmt_adapters import adapter_for,NativeFPAdapter,hadamard128
from validate_unroll import metrics
from model_engineering import runtime_identity
from execution_common import regular_maps,model_pins,atomic

def emit(phase,**kw):print(json.dumps(dict(phase=phase,monotonic=time.monotonic(),**kw)),flush=True)
def main(a):
    emit('VERIFY_MODEL_17_FULL_SHA')
    pins=model_pins(ROOT,full=True)
    emit('LOAD_PINNED_BF16_9B')
    model,tokenizer,native=R._load_frozen_model(ROOT)
    masks,masksha,_=R.load_head_mask(ROOT/'calibration/head_mask.json')
    fixture=json.loads((ROOT/'protocol/non_aime_fixture.json').read_text())['tokens'][:16]
    ident=runtime_identity();loaded=regular_maps();results={}
    def run(config,rotation=False,raw=False):
        cache=None;outputs=[];ad=None if raw else (NativeFPAdapter(model,native,rotation=True) if rotation else adapter_for(config,model,native,masks,len(fixture)))
        from contextlib import nullcontext
        with torch.inference_mode(),(nullcontext() if raw else ad):
            for i in range(49):
                out=model(input_ids=torch.tensor([[fixture[i%len(fixture)]]],dtype=torch.long,device='cuda'),past_key_values=cache,use_cache=True)
                cache=out.past_key_values;assert bool(torch.isfinite(out.logits).all())
                if ad is not None:ad.assert_cache(cache,i+1)
                outputs.append(out.logits[:,-1,:].float().clone())
                if i%16==0:emit('NON_HMMT_TEACHER_FORWARD',config=config,rotation=rotation,raw=raw,forwards=i+1)
            states={}
            if config=='C0':
                states={i:cache.layers[i].recurrent_states[0].clone() for i in R.GDN_LAYERS}
            if ad is not None and hasattr(ad,'buffers'):
                for b in ad.buffers.values():
                    before=(b.tokens,b.pending_tokens,b.writeback_count);b.finalize('preserve')
                    assert before==(b.tokens,b.pending_tokens,b.writeback_count)
                    assert b.decode_tokens==33 and b.pending_tokens==33%b.decode_period
                    assert b.decode_direct_calls==33 and b.decode_dense_materializations==33//b.decode_period
                    assert b.validation_dense_materializations==0 and b._fp_state is None
            audit=None if ad is None else ad.audit()
        del cache,out,ad;gc.collect();torch.cuda.empty_cache()
        return outputs,states,audit
    if a.deep:
        native_outputs,native_states,_=run('C0',raw=True)
        c0_outputs,c0_states,c0audit=run('C0')
        assert all(torch.equal(x,y) for x,y in zip(native_outputs,c0_outputs))
        assert all(torch.equal(native_states[i],c0_states[i]) for i in R.GDN_LAYERS)
        results['C0_native_parity']=dict(bitwise_logits=True,bitwise_FP32_cache=True,audit=c0audit)
        del native_outputs,native_states
        hy,hs,ha=run('C0',rotation=True)
        hm=hadamard128('cuda')
        logit_metrics=[metrics(x,y) for x,y in zip(hy,c0_outputs)]
        state_metrics={str(i):metrics(hs[i],hm@c0_states[i]) for i in R.GDN_LAYERS}
        spec=json.loads((ROOT/'validation/NUMERICAL_PREREGISTRATION.json').read_text())
        maximum=max(x['normalized_RMSE'] for x in logit_metrics)
        results['FP_H']=dict(logits=logit_metrics,state=state_metrics,max_logits_normalized_RMSE=maximum,
                            threshold=spec['full_model_FP_rotation']['logits_normalized_RMSE'],passed=maximum<=spec['full_model_FP_rotation']['logits_normalized_RMSE'],audit=ha)
        del hy,hs,c0_outputs,c0_states
    for config in ['C0','C1','C2','C3']:
        output,state,audit=run(config)
        results[config]=dict(status='PASS_NON_HMMT_49_FORWARDS_16_PROMPT_33_DECODE',audit=audit)
        del output,state;gc.collect();torch.cuda.empty_cache()
        emit('FOUR_CONFIG_SMOKE_COMPLETE',config=config)
    post=regular_maps()
    passed=not a.deep or results['FP_H']['passed']
    output=dict(status='PASS_NON_HMMT_FULL_MODEL' if passed else 'STOP_ROTATION_PARITY_FAILED',passed=passed,runtime=ident,
                source_native_sha=R.file_sha(native.__file__),model17_fresh_SHA=pins,mask_sha=masksha,results=results,
                actual_regular_maps_LOADED=loaded,actual_regular_maps_POST=post,new_HMMT_generations=0,thresholds_changed=False)
    atomic(a.output,output)
    emit('FULL_MODEL_VALIDATION_FINISHED',status=output['status'],max_FP_H_logits=None if not a.deep else results['FP_H']['max_logits_normalized_RMSE'])
    return 0 if passed else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deep',action='store_true');p.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
