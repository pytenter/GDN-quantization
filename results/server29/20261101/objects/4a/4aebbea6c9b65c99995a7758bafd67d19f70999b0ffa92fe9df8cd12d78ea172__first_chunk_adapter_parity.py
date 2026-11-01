#!/usr/bin/env python3
"""Bounded native-first-chunk compatibility qualification; CPU default only.

The same callable is a CUDA Gate B subcheck only when the authorized worker
passes the actual installed, pinned native helpers. No weights are loaded.
Original failed Buffer-only diagnostic and thresholds remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
from first_chunk_parity import NATIVE_SHA, comparison, inputs, native_functions, sha


def run_adapter_parity(root, *, device='cpu', native_module=None, native_identity=None, progress=None):
    import torch
    from pace_buffer import PACEBuffer
    from pace_runtime import MARKER_MAGIC, PACEAdapter
    root=Path(root)
    actual_cuda=str(device).startswith('cuda')
    if native_module is None:
        if actual_cuda:raise ValueError('CUDA subgate requires actual installed native module')
        path=root/'vendor/modeling_qwen3_5.py'
        chunk,recurrent,ast_shas=native_functions(path)
        l2norm=chunk.__globals__['l2norm']
        native_identity=dict(path=str(path),sha256=sha(path),kind='CPU_PINNED_NATIVE_AST_ORIGINAL_BODIES',AST_sha256=ast_shas)
    else:
        chunk=native_module.torch_chunk_gated_delta_rule
        recurrent=native_module.torch_recurrent_gated_delta_rule
        l2norm=native_module.l2norm
        if native_identity is None or native_identity.get('sha256')!=NATIVE_SHA:
            raise ValueError('Actual installed native SHA identity is required')
    # The namespace is private; PACE patches only its pointers, never installed
    # module global state or the separately saved original oracle functions.
    def namespace():
        return SimpleNamespace(l2norm=l2norm,torch_chunk_gated_delta_rule=chunk,
                               torch_recurrent_gated_delta_rule=recurrent)
    class MockCache:
        def __init__(self):self.state=None
    class MockLinear(torch.nn.Module):
        def __init__(self,native):super().__init__();self.native=native;self.force_kind=None
        def forward(self,values,cache):
            q,k,v,beta,g=values
            kind=self.force_kind or ('chunk' if cache.state is None else 'recurrent')
            fn=self.native.torch_chunk_gated_delta_rule if kind=='chunk' else self.native.torch_recurrent_gated_delta_rule
            out,state=fn(q[:,None],k[:,None],v[:,None],g=g[:,None],beta=beta[:,None],
                         initial_state=cache.state,output_final_state=True,use_qk_l2norm_in_kernel=True)
            # Faithful marker cache update behavior: one original state slot.
            if cache.state is None:cache.state=torch.zeros_like(state)
            cache.state.copy_(state)
            return out
    class MockModel(torch.nn.Module):
        def __init__(self,native):
            super().__init__();self.linear=MockLinear(native)
            self.model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=self.linear)])
            self.actual_forward_calls=0
        def forward(self,values,cache):
            self.actual_forward_calls+=1
            return self.linear(values,cache)
    def fp_factory(period,mask,prefill_period=256):
        return PACEBuffer(period,mask,prefill_period=prefill_period,state_quantization=False,
                          key_quantization=False,key_heads=16,_validation_only=actual_cuda)
    fixtures=[(i,'random') for i in range(36)]+[(1000+i,mode) for i,mode in enumerate(
        ('all_zero','zero_q','zero_k','beta_zero','beta_one_decay_zero','strong_decay','tiny_qk','signed_constant'))]
    cases=[];first_failure=None;start=time.monotonic()
    for seed,mode in fixtures:
        data=[tuple(x.to(device) for x in values) for values in inputs(seed,mode)]
        state=None;oracle=[]
        with torch.no_grad():
            for t,(q,k,v,beta,g) in enumerate(data):
                if t==0:
                    out,state=chunk(q[:,None],k[:,None],v[:,None],g[:,None],beta[:,None],
                                    chunk_size=64,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
                else:
                    out,state=recurrent(q[:,None],k[:,None],v[:,None],g[:,None],beta[:,None],
                                        initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
                oracle.append((out[:,0].clone(),state.clone()))
            for period in (8,1):
                native=namespace();model=MockModel(native);cache=MockCache();steps=[]
                adapter=PACEAdapter(model,native,{0:[False]*24+[True]*8},period,3,buffer_factory=fp_factory)
                with adapter:
                    for t,values in enumerate(data):
                        actual=model(values,cache)[:,0]
                        oc=comparison(actual,oracle[t][0],exact=True)
                        sc=comparison(adapter.buffers[0].state_view_for_test(),oracle[t][1],exact=False,atol=1e-5,rtol=1e-5)
                        expected_marker=torch.tensor([MARKER_MAGIC,float(t+1)],dtype=torch.float32,device=device)
                        if cache.state.shape!=(2,) or not torch.equal(cache.state,expected_marker):
                            raise AssertionError('MockModel actual cache retained dense state or wrong marker')
                        row=dict(token=t+1,native_helper='chunk_T1' if t==0 else 'recurrent_T1',BF16_output=oc,FP32_state=sc)
                        steps.append(row)
                        if (not oc['pass_'] or not sc['pass_']) and first_failure is None:
                            first_failure=dict(seed=seed,fixture=mode,decode_period=period,**row)
                    audit=adapter.audit()
                    if model.actual_forward_calls!=5 or audit['helper_calls']!={'0':5}:
                        raise AssertionError('Compatibility path added a model forward')
                    if audit['native_first_chunk_compat_calls']!={'0':1} or audit['native_first_chunk_state_checked']!={'0':1}:
                        raise AssertionError('Initial chunk not checked exactly once')
                    if audit['native_first_chunk_state_retained'] or not audit['native_first_chunk_state_discarded']:
                        raise AssertionError('Native temporary state retention reported')
                    # Only the explicitly FP fixture buffer may own dense state;
                    # the adapter itself and native cache never receive it.
                    if any(isinstance(x,torch.Tensor) and x.shape==(1,32,128,128) for x in vars(adapter).values()):
                        raise AssertionError('Native dense state persisted on adapter')
                if native.torch_chunk_gated_delta_rule is not chunk or native.torch_recurrent_gated_delta_rule is not recurrent:
                    raise AssertionError('Scoped original helpers not restored')
                good=all(x['BF16_output']['pass_'] and x['FP32_state']['pass_'] for x in steps)
                cases.append(dict(seed=seed,fixture=mode,decode_period=period,status='PASS' if good else 'FAIL',
                                  steps=steps,native_first_chunk_compat_calls=audit['native_first_chunk_compat_calls'],
                                  native_first_chunk_state_checked=audit['native_first_chunk_state_checked'],
                                  native_first_chunk_max_state_abs_error=audit['native_first_chunk_max_state_abs_error'],
                                  actual_model_forward_calls=model.actual_forward_calls,native_dense_state_retained=False))
        if progress:progress('NATIVE_FIRST_CHUNK_PARITY_CASE',seed=seed,fixture=mode,cases=len(cases))
    # A prompt_length1 commits immediately. Check the production packed path:
    # first output stays native, QDQ writeback remains once and no dense state
    # survives. This is a fixture, never an extra formal configuration.
    packed=[]
    for period in (8,1):
        native=namespace();model=MockModel(native);cache=MockCache()
        data=[tuple(x.to(device) for x in values) for values in inputs(20261025,'random',count=1)]
        q,k,v,beta,g=data[0]
        with torch.no_grad():
            expected,_=chunk(q[:,None],k[:,None],v[:,None],g[:,None],beta[:,None],chunk_size=64,
                             initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
            adapter=PACEAdapter(model,native,{0:[False]*24+[True]*8},period,1)
            with adapter:
                actual=model(data[0],cache)
                if not torch.equal(actual,expected):raise AssertionError('Production first-chunk BF16 readout differs')
                b=adapter.buffers[0]
                if b._fp_state is not None or b.packed_state is None or b.quantization_count!=1 or b.writeback_count!=1 or b.pending_tokens!=0:
                    raise AssertionError('Prompt-length1 compatibility changed packed writeback or retained dense state')
                audit=adapter.audit()
                packed.append(dict(status='PASS',decode_period=period,prompt_length=1,
                                   base_nbytes=b.packed_state.storage_nbytes(),quantization_count=1,
                                   native_first_chunk_compat_calls=audit['native_first_chunk_compat_calls'],
                                   native_dense_state_retained=False))
    rejected=[]
    for mode in ('first_recurrent','repeated_chunk'):
        native=namespace();model=MockModel(native);cache=MockCache()
        values=tuple(x.to(device) for x in inputs(20261025,'random',count=1)[0])
        adapter=PACEAdapter(model,native,{0:[False]*24+[True]*8},8,3,buffer_factory=fp_factory)
        with torch.no_grad(),adapter:
            if mode=='first_recurrent':model.linear.force_kind='recurrent'
            else:
                model(values,cache)
                model.linear.force_kind='chunk'
            try:model(values,cache)
            except RuntimeError as exc:
                expected='first empty-cache helper' if mode=='first_recurrent' else 'only for the initial'
                if expected not in str(exc):raise
                rejected.append(dict(case=mode,status='PASS_REJECTED',error=str(exc)))
            else:raise AssertionError('Illegal initial/repeated native chunk was accepted')
    good=first_failure is None
    return dict(status='PASS' if good else 'FAIL_NATIVE_FIRST_CHUNK_ADAPTER_PARITY',actual_cuda=actual_cuda,
        scope='GateB only, actual-shape MockModel empty native chunk T1 then4 recurrent calls, original thresholds unchanged',
        native_identity=native_identity,source_sha256={n:sha(root/n) for n in
          ('src/pace_runtime.py','src/pace_buffer.py','src/pace_quantizer.py','tests/first_chunk_adapter_parity.py','tests/first_chunk_parity.py')},
        fixture_count=44,paired_period_cases=len(cases),token_comparisons=sum(len(c['steps']) for c in cases),
        cases=cases,packed_prompt_length1_checks=packed,illegal_helper_path_checks=rejected,first_failure=first_failure,
        thresholds=dict(BF16_output_exact=True,FP32_state_atol=1e-5,FP32_state_rtol=1e-5),
        shape=dict(batch=1,unique_qk_heads=16,value_heads=32,key_dim=128,value_dim=128),
        native_first_chunk_state_persistent=False,model_weights_loaded=False,formal_AIME_samples=0,
        GPU_qualification_credit=actual_cuda and good,elapsed_seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--out',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU diagnostic requires CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(1)
    def forbid(*a,**k):raise AssertionError('CUDA forbidden for local CPU qualification')
    torch.cuda._lazy_init=forbid
    result=run_adapter_parity(args.root)
    result['environment']=dict(executable=sys.executable,torch=torch.__version__,CUDA_initialized=torch.cuda.is_initialized())
    out=Path(args.out)
    with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=result['status'],fixture_count=result['fixture_count'],cases=result['paired_period_cases'],token_comparisons=result['token_comparisons'],report_sha256=sha(out))))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
