"""CPU observer augmentation; retains the earlier eight-test report unchanged."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import codec_adapter as codec

ROOT=Path(__file__).resolve().parent
D=Path('/data/zypan/experiment_runs/LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1)
prior=D/'CPU_CODEC_PARITY.json'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='c5b40198ccce5cc8067d4dab65a5c90048f120e9287fda315f56ff127dbd52df'
gen=torch.Generator().manual_seed(610162026)
panels={'random':torch.randn((2,3,7,11),generator=gen),'zero':torch.zeros(1,2,7,11),
        'tiny':torch.full((1,2,7,11),1e-30),'large':torch.full((1,2,7,11),1e20)}
observations=[]
def bytes_equal(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
for name,x in panels.items():
    plain=codec.encode_dualm(x);traced,trace=codec.trace_encode_dualm(x)
    for a,b in zip((plain.codes,plain.rows,plain.columns,plain.decode()),(traced.codes,traced.rows,traced.columns,traced.decode())):
        assert bytes_equal(a,b),(name,'observer bytes changed')
    again=codec.encode_dualm(x)
    assert bytes_equal(again.decode(),plain.decode()),'trace namespace not restored'
    assert [z['name'] for z in trace['scale_records']]==['rows','initial_columns','LS_columns']
    for r in trace['scale_records']:
        assert torch.equal(r['raw_floor_hits'],(r['raw_input']<codec.LOW).sum())
        assert torch.equal(r['raw_ceiling_hits'],(r['raw_input']>codec.HIGH).sum())
    y=(x/traced.rows.float())/traced.columns.float()
    assert torch.equal(trace['unrounded_over127_count'],(y.abs()>127).sum())
    assert torch.equal(trace['rounded_clamp_count'],(torch.round(y).abs()>127).sum())
    assert trace['affected_energy_fraction_per_batch_head'].shape==x.shape[:2]
    assert trace['affected_relative_mse_per_batch_head'].shape==x.shape[:2]
    for k in ['affected_energy_fraction_per_batch_head','affected_relative_mse_per_batch_head','local_relative_mse_per_batch_head']:
        assert bool(torch.isfinite(trace[k]).all())
    observations.append(dict(panel=name,output_byte_exact=True,
        unrounded_over127_count=int(trace['unrounded_over127_count']),
        rounded_clamp_count=int(trace['rounded_clamp_count']),
        raw_scale_bounds=[dict(name=r['name'],floor_hits=int(r['raw_floor_hits']),ceiling_hits=int(r['raw_ceiling_hits'])) for r in trace['scale_records']],
        max_affected_energy_fraction=float(trace['affected_energy_fraction_per_batch_head'].max()),
        max_affected_relative_mse=float(trace['affected_relative_mse_per_batch_head'].max())))
assert not torch.cuda.is_initialized()
report={'task_id':'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1','status':'PASS_CPU_TRACE_OBSERVER_AUGMENTATION',
        'CPU_ONLY':True,'CUDA_initialized':False,'GPU_forward':False,'prior_cpu_report_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
        'original_eight_test_report_unchanged':True,'trace_output_byte_exact':True,
        'per_head_metrics_not_cross_head_dilution':True,'trace_policy':'low-frequency probes only; transient statistics, no next-token state',
        'no_new_safety_threshold_invented':True,'observations':observations,'created_unix':time.time(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'codec_adapter.py',Path(__file__).resolve()]}}
out=D/'audits/CPU_CODEC_TRACE_AUGMENTATION.json'
with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':report['status'],'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
