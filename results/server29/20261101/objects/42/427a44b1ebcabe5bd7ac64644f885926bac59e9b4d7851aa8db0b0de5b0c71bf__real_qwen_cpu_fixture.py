#!/usr/bin/env python3
"""CPU-only tiny real-class exercise; never a 9B/GPU/long-horizon qualification."""
import hashlib, json, os, sys, traceback
from pathlib import Path
sys.dont_write_bytecode = True
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.path.insert(0, '/data/zypan/transformers-qwen35/src')
import torch
def denied(*a, **k):
    raise RuntimeError('GPU initialization forbidden in CPU fixture')
torch.cuda._lazy_init = denied
ambient = {'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(),
           'deterministic': torch.are_deterministic_algorithms_enabled(),
           'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
           'cudnn_TF32': torch.backends.cudnn.allow_tf32,
           'cudnn_deterministic': torch.backends.cudnn.deterministic}
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
arch = Path(Q.__file__)
assert hashlib.sha256(arch.read_bytes()).hexdigest() == '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
checks = []
def check(name, condition, **details):
    checks.append(dict(name=name, passed=bool(condition), **details))
    assert condition, name
def sha_tensor(t):
    assert t.device.type == 'cpu'
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
try:
    # Independent float64 scalar recurrence, separated from the installed native helper.
    torch.random.default_generator.manual_seed(20261015)
    q,k,v=[torch.randn(1,4,2,8)*.1 for _ in range(3)]
    g=-torch.rand(1,4,2)*.2; beta=torch.rand(1,4,2)
    initial=torch.randn(1,2,8,8)*.1
    y,s=Q.torch_recurrent_gated_delta_rule(q,k,v,g,beta,initial,True,False)
    ref=initial.double().clone(); outs=[]
    for i in range(4):
        ref=ref*g[:,i].double().exp()[...,None,None]
        mem=(ref*k[:,i].double()[...,None]).sum(-2)
        ref=ref+k[:,i].double()[...,None]*((v[:,i].double()-mem)*beta[:,i].double()[...,None])[...,None,:]
        outs.append((ref*(q[:,i].double()/(8**.5))[...,None]).sum(-2))
    refy=torch.stack(outs,1)
    err_y=float((y.double()-refy).abs().max());err_s=float((s.double()-ref).abs().max())
    check('native_recurrent_vs_independent_FP64_small_fixture',err_y<1e-6 and err_s<1e-6,
          output_max_abs=err_y,state_max_abs=err_s,bound=1e-6)
    # Use the real model classes and full native attention/MLP/GDN/cache calls, tiny random weights.
    runs={}
    for dtype in [torch.float32,torch.bfloat16]:
        results=[]
        for repeat in range(2):
            torch.random.default_generator.manual_seed(20261015)
            c=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                linear_num_key_heads=4,linear_num_value_heads=4,linear_key_head_dim=8,
                linear_value_head_dim=8,linear_conv_kernel_dim=4,
                layer_types=['linear_attention','full_attention'],
                rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,1,1]})
            c._attn_implementation='sdpa'
            model=Q.Qwen3_5ForCausalLM(c).to(dtype=dtype,device='cpu').eval()
            cache=None; trace=[]; calls={'chunk':0,'recurrent':0};previous=None
            saved={n:getattr(Q,n) for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']}
            for name,kind in [('torch_chunk_gated_delta_rule','chunk'),('torch_recurrent_gated_delta_rule','recurrent')]:
                def wrap(fn=saved[name],kind=kind):
                    def invoke(*args,**kw):
                        nonlocal_dummy=None
                        calls[kind]+=1
                        if kind=='recurrent':
                            initial_state=kw['initial_state']
                            check('tiny_next_token_consumes_previous_state_'+str(dtype)+'_'+str(repeat)+'_'+str(calls[kind]),
                                  previous is not None and sha_tensor(initial_state)==previous)
                        return fn(*args,**kw)
                    return invoke
                setattr(Q,name,wrap())
            try:
                with torch.inference_mode():
                    for t,token in enumerate([3,7,11,13]):
                        out=model(input_ids=torch.tensor([[token]],device='cpu'),past_key_values=cache,use_cache=True)
                        cache=out.past_key_values
                        state=cache.layers[0].recurrent_states[0]
                        check('tiny_finite_'+str(dtype)+'_'+str(repeat)+'_'+str(t),
                              bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(state).all()))
                        check('tiny_state_dtype_shape_'+str(dtype)+'_'+str(repeat)+'_'+str(t),
                              state.dtype==torch.float32 and list(state.shape)==[1,4,8,8])
                        check('tiny_cache_cursor_'+str(dtype)+'_'+str(repeat)+'_'+str(t),
                              cache.get_seq_length()==t+1 and cache.layers[1].keys.shape[-2]==t+1)
                        previous=sha_tensor(state)
                        trace.append(dict(t=t,input=token,logits=sha_tensor(out.logits),state=previous,
                                          conv=sha_tensor(cache.layers[0].conv_states[0]),
                                          keys=sha_tensor(cache.layers[1].keys),values=sha_tensor(cache.layers[1].values)))
            finally:
                for name,fn in saved.items():setattr(Q,name,fn)
            check('tiny_native_chunk_recurrent_counts_'+str(dtype)+'_'+str(repeat),calls=={'chunk':1,'recurrent':3},calls=calls)
            results.append(trace)
        check('tiny_same_process_reconstruction_repeat_'+str(dtype),results[0]==results[1])
        runs[str(dtype)]=results
    mapped=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        f=line.split(maxsplit=5)
        if len(f)==6 and f[5].startswith('/') and '.so' in f[5]:mapped.append(f[5])
    result=dict(status='PASS_REAL_CLASS_TINY_CPU_ONLY',scope='TINY_RANDOM_CPU_NOT_9B_NOT_GPU_NOT_FRESH_PROCESSES_NOT_LONG_HORIZON',
        architecture_path=str(arch),architecture_sha256=hashlib.sha256(arch.read_bytes()).hexdigest(),
        torch=torch.__version__,ambient_CPU_effective_settings=ambient,fixture_threads=[1,1],
        checks=checks,traces=runs,mapped_library_paths=sorted(set(mapped)),GPU_dispatches=0,GPU_queries=0,
        GPU_initialization_guard='FORBIDDEN',native_chunk_cumsum_unchanged=True)
except BaseException:
    result=dict(status='FAIL_CPU_FIXTURE_NO_GPU_ATTEMPT',checks=checks,traceback=traceback.format_exc(),GPU_dispatches=0,GPU_queries=0)
print('CPU_FIXTURE_RESULT='+json.dumps(result,sort_keys=True,allow_nan=False))
if result['status'].startswith('FAIL'):sys.exit(1)
