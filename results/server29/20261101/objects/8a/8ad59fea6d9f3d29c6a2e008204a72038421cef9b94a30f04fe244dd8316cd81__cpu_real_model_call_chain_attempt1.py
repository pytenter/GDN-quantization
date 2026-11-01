"""CPU integration fixture for a proposed kwargs-only repair, never a 9B test.

Imports the unchanged frozen Qwen module and instantiates its actual model,
TextModel, decoder and GDN classes with small random dimensions. No saved model
weights, tokenizer, task questions, CUDA, calibration Plan or generation runs.
"""
import argparse
import ast
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

NATIVE_SHA = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
OLD_ADAPTER_SHA = '09175da5111f7e9527987209ee9171804e4c77fcf050dc52ec41489f93b304c2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_adapter(path, source=None):
    module = types.ModuleType('CPU_SYNTHETIC_ADAPTER_' + hashlib.sha256(str(path).encode()).hexdigest()[:8])
    module.__file__ = str(path)
    exec(compile(source if source is not None else Path(path).read_text(), str(path), 'exec'), module.__dict__)
    return module.NativeShadowStatistics


def candidate_source(old):
    # Only the keyword-admission contract changes. Native call and every
    # normalization, recurrence, observation and return statement stay exact.
    needle = "if set(extra) - {'cu_seqlens'}:"
    assert old.count(needle) == 1
    new = old.replace(needle, "if set(extra) - {'cu_seqlens', 'use_cache'}:")
    point = "            initial = a['initial_state']"
    assert new.count(point) == 1
    return new.replace(point, "            if 'use_cache' in extra and extra['use_cache'] is not True:\n"
                       "                raise ValueError('Calibration use_cache must be exactly True')\n"
                       "            if extra.get('cu_seqlens') is not None:\n"
                       "                raise ValueError('Calibration requires unpadded single-request cu_seqlens=None')\n" + point)


def tensor_sha(torch, value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensors(torch, value, prefix=''):
    if torch.is_tensor(value):
        return {prefix: value.detach().clone()}
    out = {}
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(tensors(torch, v, prefix + '/' + str(i)))
    elif isinstance(value, dict):
        for k, v in value.items():
            out.update(tensors(torch, v, prefix + '/' + str(k)))
    return out


def cache_snapshot(torch, cache):
    result = {}
    for i, layer in enumerate(cache.layers):
        for name, value in vars(layer).items():
            result.update(tensors(torch, value, str(i) + '/' + name))
    assert result
    return result


def recurrent(cache, layer):
    x = cache.layers[layer].recurrent_states
    return x[0] if isinstance(x, (dict, list, tuple)) else x


@contextlib.contextmanager
def record_native(qmod, torch):
    original = {n:getattr(qmod,n) for n in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule')}
    calls = []
    try:
        for name, function in original.items():
            @functools.wraps(function)
            def wrapped(*args, _name=name, _function=function, **kwargs):
                answer = _function(*args, **kwargs)
                calls.append({'kernel':_name,'kwargs':{k:(v if v is None or type(v) in (int,bool,str,float) else str(type(v)))
                                                       for k,v in kwargs.items() if k in ('use_cache','cu_seqlens')},
                              'state':answer[1], 'return':answer})
                return answer
            setattr(qmod,name,wrapped)
        yield calls
    finally:
        for name,function in original.items():
            setattr(qmod,name,function)


def run(args):
    start=time.monotonic()
    sys.path.insert(0,str(args.transformers_source))
    sys.path.insert(0,str(args.official_source))
    import torch
    assert not torch.cuda.is_initialized()
    def forbid_cuda(*a,**k):
        raise AssertionError('CUDA initialization forbidden in CPU integration fixture')
    torch.cuda._lazy_init=forbid_cuda
    torch.set_num_threads(1)
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    assert sha(qmod.__file__)==NATIVE_SHA
    assert sha(args.old_adapter)==OLD_ADAPTER_SHA
    old=args.old_adapter.read_text()
    proposal=candidate_source(old) if args.candidate_adapter is None else args.candidate_adapter.read_text()
    Original=load_adapter(args.old_adapter)
    Candidate=load_adapter(args.candidate_adapter or args.old_adapter,proposal)
    # Preserve every original AST statement apart from this tiny predicate and
    # its explicit two value guards. Full source digest records the proposal.
    ast.parse(proposal)
    config_values=dict(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=256,
        linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,
        linear_num_value_heads=2,linear_conv_kernel_dim=4,
        layer_types=['linear_attention']*3+['full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.25,'mrope_section':[2,0,0]})
    records=[]
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(41017)
        config=Qwen3_5TextConfig(**config_values)
        config._attn_implementation='sdpa'
        model=qmod.Qwen3_5ForCausalLM(config).to(dtype=dtype).eval()
        assert all(p.device.type=='cpu' for p in model.parameters())
        model.requires_grad_(False)
        ids=torch.tensor([[1,2,3,4,5,6,7,8,9]],dtype=torch.long)
        next_ids=[torch.tensor([[x]],dtype=torch.long) for x in (10,11,12)]
        old_error=None
        with torch.inference_mode():
            try:
                with Original(model,qmod,8,64):
                    model(input_ids=ids,use_cache=True,logits_to_keep=1)
            except ValueError as error:
                old_error=str(error)
            assert old_error=="Unaudited calibration kernel kwargs: ['cu_seqlens', 'use_cache']",old_error
            baseline=[]; baseline_cache=None
            with record_native(qmod,torch) as native_calls:
                for x in [ids]+next_ids:
                    out=model(input_ids=x,past_key_values=baseline_cache,use_cache=True,logits_to_keep=1)
                    baseline_cache=out.past_key_values
                    baseline.append((out.logits.detach().clone(),cache_snapshot(torch,baseline_cache)))
            assert len(native_calls)==12
            actual=[]; actual_cache=None
            with record_native(qmod,torch) as wrapped_calls:
                with Candidate(model,qmod,8,64) as observer:
                    for x in [ids]+next_ids:
                        out=model(input_ids=x,past_key_values=actual_cache,use_cache=True,logits_to_keep=1)
                        actual_cache=out.past_key_values
                        for layer,call in zip((0,1,2),wrapped_calls[-3:]):
                            state=recurrent(actual_cache,layer)
                            assert state is call['state']
                            assert state.dtype==torch.float32 and tuple(state.shape)==(1,2,8,8)
                        actual.append((out.logits.detach().clone(),cache_snapshot(torch,actual_cache)))
                    stats=observer.export_statistics()
                    assert set(observer.layer_name_to_index.values())=={0,1,2}
                    assert observer.calls=={name:4 for name in observer.statistics}
            assert len(wrapped_calls)==12
            for expected,obtained in zip(baseline,actual):
                assert torch.equal(expected[0],obtained[0]),'Actual model logits changed'
                assert expected[1].keys()==obtained[1].keys()
                assert all(torch.equal(expected[1][k],obtained[1][k]) for k in expected[1]),'Actual model cache changed'
            for call in wrapped_calls:
                assert call['kwargs']=={'cu_seqlens':None,'use_cache':True},call['kwargs']
            assert all(v['tokens']==12 and torch.isfinite(v['omega']).all() and torch.isfinite(v['log_decay']).all()
                       and torch.isfinite(v['snapshots']).all() for v in stats.values())
            refused={}
            for label,extra in [('unknown',{'unqualified_interface_kwarg':1}),
                                ('packed',{'cu_seq_lens_q':torch.tensor([0,9],dtype=torch.int32)}),
                                ('cache_not_bool_true',{'use_cache':1})]:
                kwargs=dict(use_cache=True,logits_to_keep=1);kwargs.update(extra)
                try:
                    with Candidate(model,qmod,8,64):
                        model(input_ids=ids,**kwargs)
                except ValueError as error:
                    refused[label]=str(error)
                else:
                    raise AssertionError('Unqualified kwargs accepted: '+label)
            records.append({'dtype':str(dtype),'old_failure_reproduced':old_error,
                'actual_model_classes':[type(model).__name__,type(model.model).__name__,
                    type(model.model.layers[0]).__name__,type(model.model.layers[0].linear_attn).__name__],
                'native_prefill_calls':3,'native_decode_calls':9,'observed_kernel_extra_kwargs':wrapped_calls[0]['kwargs'],
                'prefill_and_each_decode_logits_byte_equal':True,'all_cache_tensors_byte_equal':True,
                'original_native_state_identity_preserved':True,'actual_state_dtype':'torch.float32',
                'actual_state_shape':[1,2,8,8],'official_observer_layers':len(stats),'official_observer_tokens_per_layer':12,
                'prefill_logits_sha256':tensor_sha(torch,actual[0][0]),'refused':refused})
        del model,baseline,actual
    assert not torch.cuda.is_initialized()
    return {'status':'PASS_CPU_SYNTHETIC_REAL_MODEL_CALL_CHAIN_ONLY',
        'scope':'Actual unchanged frozen Qwen classes/kernel call chain, tiny random weights/config. Not pretrained9B, official32x2048 calibration, GPU, AIME generation or repair scientific admission.',
        'native_source_sha256':NATIVE_SHA,'old_adapter_sha256':OLD_ADAPTER_SHA,
        'candidate_adapter_path':str(args.candidate_adapter) if args.candidate_adapter else 'IN_MEMORY_PRECISE_KWARGS_ONLY_PROPOSAL',
        'candidate_adapter_sha256':hashlib.sha256(proposal.encode()).hexdigest(),
        'test_source_sha256':sha(__file__),'config':config_values,'records':records,
        'CUDA_initialized':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
        'elapsed_seconds':time.monotonic()-start,'full9B_repair_qualification':'NOT_RUN'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--transformers-source',type=Path,required=True)
    p.add_argument('--official-source',type=Path,required=True)
    p.add_argument('--old-adapter',type=Path,required=True)
    p.add_argument('--candidate-adapter',type=Path)
    args=p.parse_args()
    print(json.dumps(run(args),ensure_ascii=False,indent=2,allow_nan=False))
