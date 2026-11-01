"""Operator gate, small calibration and prospective tolerance freeze before full panel."""
import argparse, sys
import closure_common as C

def determinism():
    runs=[C.read(C.ROOT/f'operator/strict_{i}.json') for i in range(1,6)]
    names=[r['name'] for r in runs[0]['rows']]; rows=[]
    for i,n in enumerate(names):
        hashes=[r['rows'][i]['output']['sha256'] for r in runs]
        ok=all(r['strict'] and r['rows'][i]['same_process_exact'] for r in runs) and len(set(hashes))==1
        rows.append(dict(name=n,fresh_process_hashes=hashes,PASS=ok,output=runs[0]['rows'][i]['output']))
    gate='PASS' if all(r['PASS'] for r in rows) else 'FAIL'
    C.save('provenance/operator_determinism.json',dict(FIXED_ORDER_CUMSUM_DETERMINISM_GATE=gate,same_process_repetitions=5,fresh_processes=5,rows=rows))
    C.check(gate=='PASS','operator determinism failed; STOP')

def freeze():
    import torch
    C.check(C.read(C.ROOT/'provenance/operator_determinism.json')['FIXED_ORDER_CUMSUM_DETERMINISM_GATE']=='PASS','ordering')
    # Inspect only the first head, first eight real scan prefixes from one dev fixture.
    ref=C.load_pt('operator/reference_1.pt')['historical32'][0,0,0,:8]
    out=C.load_pt('operator/strict_1.pt')['historical32'][0,0,0,:8]
    pilot=C.metric(ref,out)
    gamma64=64*2**-24/(1-64*2**-24)
    search=dict(parent_policy='8*gamma128 operand-scaled absolute mapping bound; no max_rel/relative_L2/full-forward policy',
        formal_replay_policy='exact paired identical association; not applicable to changed association',
        readiness_policy='bitwise same-host runtime identity, not applicable to changed association',
        directly_applicable_historical_tolerance=False,
        source_hashes={str(p):C.sha(p) for p in [C.PARENT/'protocol/preregistration_repaired_001.json',
            C.SRC/'experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/configs/deployment_checkpoint_format.json',
            C.READINESS]})
    # No full chunk/model comparison has been run/read. These are fixed, dtype-based
    # engineering caps, not scientific metrics or tuning from endpoint outcomes.
    policy=dict(frozen_before_full_panel=True,search=search,calibration=dict(fixture='historical32',subset='head0 first8 prefixes only',metrics=pilot),
       formula='PASS requires all three limits and finiteness. abs is scale-adjusted; max_rel uses explicitly frozen reference floor.',
       operator=dict(max_abs_scale=4*gamma64,max_rel=4*gamma64,relative_L2=4*gamma64,relative_floor=1e-12),
       chunk_fp32=dict(max_abs_scale=1e-3,max_rel=0.1,relative_L2=1e-4,relative_floor_scale=0.01),
       chunk_bf16=dict(max_abs_scale=0.008,max_rel=0.1,relative_L2=0.002,relative_floor_scale=0.1),
       model_bf16=dict(max_abs_scale=0.008,max_rel=0.1,relative_L2=0.002,relative_floor_scale=0.1),
       model_fp32=dict(max_abs_scale=0.002,max_rel=0.1,relative_L2=0.001,relative_floor_scale=0.02),
       rationale='FP32 scan bound 4*gamma64; downstream FP32 capped at 0.1-0.2% scale and BF16 at ~one output ULP (0.8%), with tighter global L2. Near-zero relative errors are reported raw and floor-stabilized separately. Every cap is frozen before the full panel; failure stops without relaxation.',
       calibration_cannot_define_downstream_noise='Only a tiny real operator subset was calibrated; downstream dtype/reduction caps are conservative prospective engineering policy, not observed model noise or a claim of mathematical error bounds.',
       full_panel_results_read=False)
    C.save('protocol/numerical_parity_policy.json',policy)
    print('NUMERICAL_POLICY_FROZEN',flush=True)

def compare(a,b,kind):
    import torch
    p=C.read(C.ROOT/'protocol/numerical_parity_policy.json')[kind]; m=C.metric(a,b)
    scale=max(1.,m['reference_max_abs']); floor=p.get('relative_floor',p.get('relative_floor_scale',0)*scale)
    d=(a.double()-b.double()).abs(); rel=float((d/a.double().abs().clamp_min(floor)).max())
    m.update(max_rel_floor_stabilized=rel,relative_floor=floor,max_abs_tolerance=p['max_abs_scale']*scale,
             max_rel_tolerance=p['max_rel'],relative_L2_tolerance=p['relative_L2'])
    m['PASS']=m['finite'] and m['max_abs']<=m['max_abs_tolerance'] and rel<=p['max_rel'] and m['relative_L2']<=p['relative_L2']
    return m

def parity():
    C.check((C.ROOT/'protocol/numerical_parity_policy.json').is_file(),'policy not frozen')
    ref=C.load_pt('operator/reference_1.pt'); out=C.load_pt('operator/strict_1.pt')
    rows=[dict(name=n,**compare(ref[n],out[n],'operator')) for n in ref]
    gate='PASS' if all(r['PASS'] for r in rows) else 'FAIL'
    C.save('provenance/operator_parity.json',dict(CUMSUM_NUMERICAL_PARITY_GATE=gate,rows=rows,policy_sha256=C.sha(C.ROOT/'protocol/numerical_parity_policy.json')))
    C.check(gate=='PASS','operator numerical parity failed; STOP')
    print('OPERATOR_PARITY=PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['determinism','freeze','parity'],required=True)
    globals()[p.parse_args().phase]()
