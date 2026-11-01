#!/usr/bin/env python3
"""Independent CPU-only empty-cache native chunk-T1 boundary diagnostic.

No production source is modified. Native functions are original AST bodies,
including native chunk_size64 padding, with decorators removed only to avoid
external kernel dispatch. This is not a model/9B/CUDA qualification.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NATIVE_SHA='90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_functions(path):
    import torch
    import torch.nn.functional as F
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=NATIVE_SHA:raise ValueError('Native source SHA differs')
    names={'l2norm','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'}
    nodes=[x for x in ast.parse(data).body if isinstance(x,ast.FunctionDef) and x.name in names]
    if {x.name for x in nodes}!=names:raise ValueError('Original native helper closure incomplete')
    ast_hash={x.name:hashlib.sha256(ast.dump(x,include_attributes=False).encode()).hexdigest() for x in nodes}
    for node in nodes:node.decorator_list=[]
    namespace={'torch':torch,'F':F}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['torch_chunk_gated_delta_rule'],namespace['torch_recurrent_gated_delta_rule'],ast_hash


def inputs(seed,mode,count=5):
    import torch
    rng=torch.Generator(device='cpu').manual_seed(seed)
    result=[]
    for t in range(count):
        q=torch.randn(1,16,128,generator=rng).to(torch.bfloat16)
        k=torch.randn(1,16,128,generator=rng).to(torch.bfloat16)
        v=torch.randn(1,32,128,generator=rng).to(torch.bfloat16)
        beta=torch.rand(1,32,generator=rng).to(torch.bfloat16)
        g=-torch.rand(1,32,generator=rng)*.7
        if mode=='all_zero':q.zero_();k.zero_();v.zero_();beta.zero_();g.zero_()
        elif mode=='zero_q':q.zero_()
        elif mode=='zero_k':k.zero_()
        elif mode=='beta_zero':beta.zero_()
        elif mode=='beta_one_decay_zero':beta.fill_(1);g.zero_()
        elif mode=='strong_decay':g.fill_(-80)
        elif mode=='tiny_qk':q.mul_(2**-12);k.mul_(2**-12)
        elif mode=='signed_constant':
            q.fill_(-.5 if t%2 else .5);k.fill_(.25 if t%2 else -.25)
            v.fill_(2 if t%2 else -2);beta.fill_(.5);g.fill_(-.25)
        # The installed Qwen GDN repeats unique16 q/k to value32 before helpers.
        result.append((q.repeat_interleave(2,dim=1),k.repeat_interleave(2,dim=1),v,beta,g))
    return result


def comparison(actual,expected,*,exact,atol=0.,rtol=0.):
    import torch
    difference=(actual.float()-expected.float()).abs()
    exact_equal=torch.equal(actual,expected)
    bad=actual!=expected if exact else difference>(atol+rtol*expected.float().abs())
    coords=bad.nonzero()[:8].tolist()
    return dict(pass_=not bool(bad.any()),exact=exact_equal,mismatched=int(bad.sum().item()),
                max_abs_error=float(difference.max().item()),atol=atol,rtol=rtol,
                first_mismatches=[dict(index=c,actual=float(actual[tuple(c)]),expected=float(expected[tuple(c)])) for c in coords])


def run(native_path,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit empty CUDA_VISIBLE_DEVICES required')
    import torch
    from src.pace_buffer import PACEBuffer
    torch.set_num_threads(1)
    def forbid(*args,**kwargs):raise AssertionError('CUDA initialization forbidden in independent CPU diagnostic')
    torch.cuda._lazy_init=forbid
    if torch.cuda.is_initialized():raise AssertionError('CUDA was already initialized')
    chunk,recurrent,AST=native_functions(native_path)
    fixtures=[(i,'random') for i in range(36)]+[(1000+i,mode) for i,mode in enumerate(
        ('all_zero','zero_q','zero_k','beta_zero','beta_one_decay_zero','strong_decay','tiny_qk','signed_constant'))]
    cases=[];first_failure=None
    start=time.monotonic()
    for seed,mode in fixtures:
        data=inputs(seed,mode)
        # Exactly25% INT8 mask remains structurally admitted, but both state and
        # key quantization are OFF only for this explicit FP parity fixture.
        buffers={p:PACEBuffer(p,[False]*24+[True]*8,state_quantization=False,
                              key_quantization=False,key_heads=16) for p in (8,1)}
        for b in buffers.values():b.set_prompt_length(3)
        state=None;steps=[]
        with torch.no_grad():
            for t,(q,k,v,beta,g) in enumerate(data):
                args=(q[:,None],k[:,None],v[:,None],g[:,None],beta[:,None])
                if t==0:
                    out,state=chunk(*args,chunk_size=64,initial_state=None,output_final_state=True,
                                    use_qk_l2norm_in_kernel=True)
                else:
                    out,state=recurrent(*args,initial_state=state,output_final_state=True,
                                        use_qk_l2norm_in_kernel=True)
                for period,buffer in buffers.items():
                    actual=buffer.step(q,k,v,beta,g)
                    oc=comparison(actual,out[:,0],exact=True)
                    sc=comparison(buffer.state_view_for_test(),state,exact=False,atol=1e-5,rtol=1e-5)
                    row=dict(token=t+1,native_helper='chunk_T1' if t==0 else 'recurrent_T1',
                             decode_period=period,BF16_output=oc,FP32_state=sc)
                    steps.append(row)
                    if (not oc['pass_'] or not sc['pass_']) and first_failure is None:
                        first_failure=dict(seed=seed,fixture=mode,**row)
                        print(json.dumps(dict(event='FIRST_REAL_CPU_PARITY_FAILURE',detail=first_failure)),flush=True)
        good=all(x['BF16_output']['pass_'] and x['FP32_state']['pass_'] for x in steps)
        cases.append(dict(seed=seed,fixture=mode,status='PASS' if good else 'FAIL',steps=steps))
        print(json.dumps(dict(event='FIRST_CHUNK_CASE_COMPLETE',seed=seed,fixture=mode,status=cases[-1]['status'])),flush=True)
    value=dict(status='PASS' if all(x['status']=='PASS' for x in cases) else 'FAIL_CPU_FIRST_CHUNK_PARITY',
        scope='Independent CPU synthetic actual-shape native empty-cache chunk T1 then recurrent T1 versus unquantized PACEBuffer; not a model or GPU admission',
        native_source_path=str(native_path),native_source_sha256=sha(native_path),native_AST_sha256=AST,
        diagnostic_source_sha256=sha(__file__),hold_source_sha256={n:sha(ROOT/n) for n in
            ('src/pace_buffer.py','src/pace_quantizer.py','src/pace_runtime.py','tests/test_pace_math.py')},
        shapes=dict(batch=1,unique_qk_heads=16,value_heads=32,key_dim=128,value_dim=128,tokens=5,prompt_length=3),
        first_native_call=dict(helper='torch_chunk_gated_delta_rule',T=1,chunk_size=64,initial_state=None),
        subsequent_native_calls=dict(helper='torch_recurrent_gated_delta_rule',T=1,count=4),
        thresholds=dict(BF16_output_exact=True,FP32_state_atol=1e-5,FP32_state_rtol=1e-5),
        fixture_count=len(cases),comparison_count=sum(len(x['steps']) for x in cases),
        case_failures=sum(x['status']!='PASS' for x in cases),first_failure=first_failure,cases=cases,
        environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
                         CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized=torch.cuda.is_initialized()),
        GPU_calls=0,model_loads=0,production_source_changes=0,GPU_qualification_credit=False,
        elapsed_seconds=time.monotonic()-start)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],fixture_count=value['fixture_count'],case_failures=value['case_failures'],report=str(output),report_sha256=sha(output))),flush=True)
    return 0 if value['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native-source',default=str(ROOT/'vendor/modeling_qwen3_5.py'))
    p.add_argument('--out',required=True);args=p.parse_args()
    raise SystemExit(run(Path(args.native_source),Path(args.out)))
