"""CPU-only codec tests; no model or CUDA forward is imported or initialized."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import codec_adapter as codec

ROOT = Path(__file__).resolve().parent


def tensor_digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def canonical_panel():
    generator = torch.Generator().manual_seed(17102026)
    x = torch.randn((2, 3, 7, 11), generator=generator)
    x[..., 0, 0] = 12345.0
    x[..., -1, -1] = -1e-15
    return x


def digest_worker():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    encoded = codec.encode_dualm(canonical_panel())
    assert not torch.cuda.is_initialized()
    print(tensor_digest([encoded.codes, encoded.rows, encoded.columns, encoded.decode()]))


def main(output):
    started = time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY_ENV_REQUIRED'
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    results = []
    def record(name, fn):
        tick = time.perf_counter()
        details = fn()
        results.append(dict(test=name, status='PASS', seconds=time.perf_counter()-tick, details=details))

    def shape_axis():
        x = canonical_panel(); e = codec.encode_dualm(x)
        assert e.shape == (2,3,7,11) and e.rows.shape == (2,3,7,1) and e.columns.shape == (2,3,1,11)
        assert e.codes.shape == x.shape and e.decode().shape == x.shape
        bad = [torch.zeros(2,3,7), torch.full((1,1,2,3), float('nan')), torch.zeros((1,0,2,3))]
        for x in bad:
            try: codec.encode_dualm(x)
            except ValueError: pass
            else: raise AssertionError('invalid shape/value accepted')
        return {'native_axis':'B,H,Key,Value', 'non_square_shape':[2,3,7,11], 'transpose_performed':False}
    record('1_shape_axis_and_invalid_input', shape_axis)

    def signed_codes():
        e=codec.encode_dualm(canonical_panel())
        assert e.codes.dtype == torch.int8 and int(e.codes.min()) >= -127 and int(e.codes.max()) <=127
        assert not bool((e.codes==-128).any())
        return {'min':int(e.codes.min()),'max':int(e.codes.max()),'bitwidth':8,'pivots':0}
    record('2_signed_INT8_range', signed_codes)

    def scale_metadata():
        e=codec.encode_dualm(canonical_panel())
        assert e.rows.dtype==e.columns.dtype==torch.float16
        for scale in (e.rows,e.columns):
            assert bool((scale>=codec.LOW).all()) and bool((scale<=codec.HIGH).all())
        expected=e.codes.numel()+2*e.rows.numel()+2*e.columns.numel()
        assert e.tensor_payload_bytes()==expected
        return {'row_dtype':'float16','column_dtype':'float16','tensor_payload_bytes':expected,
                'metadata_layout':'rows Key, columns Value; shared shape/header not included',
                'functional_FP32_carrier_is_physical_packed_INT8':False}
    record('3_FP16_row_column_scales_and_payload', scale_metadata)

    panels={'random':canonical_panel(), 'zeros':torch.zeros(1,2,7,11),
            'tiny':torch.full((1,2,7,11),1e-30),
            'large':torch.full((1,2,7,11),1e20),
            'outlier':torch.cat((torch.zeros(1,2,7,10),torch.ones(1,2,7,1)*1e6),-1)}
    def stress():
        assert torch.equal(codec.levels(torch.tensor([-2.5,-1.5,-.5,.5,1.5,2.5])),torch.tensor([-2.,-2.,-0.,0.,2.,2.]))
        assert torch.equal(codec.safe_half(torch.tensor([0.,codec.LOW/2,codec.HIGH*2])),torch.tensor([codec.LOW,codec.LOW,codec.HIGH]).half())
        assert torch.equal(codec.encode_dualm(panels['zeros']).decode(),panels['zeros'])
        for x in panels.values(): assert bool(torch.isfinite(codec.encode_dualm(x).decode()).all())
        return {'panels':list(panels),'rounding':'torch.round ties-to-even','codes_clamp':[-127,127],
                'scale_floor':codec.LOW,'scale_ceiling':codec.HIGH,'row_mean_floor':1e-20,'LS_den_floor':1e-30,
                'large_state_accuracy_claim':False}
    record('4_zero_outlier_ties_extreme_scale', stress)

    def reference_parity():
        observed=[]
        for name,x in panels.items():
            e=codec.encode_dualm(x); direct=codec.direct_equation_reference(x)
            for actual,expected in zip((e.codes,e.rows,e.columns,e.decode()),direct): assert torch.equal(actual,expected),name
            official=codec.official_reference(x,'kda'); gdn=codec.official_reference(x,'gdn')
            z,r,c,r2,dequant=official
            for actual,expected in zip((e.codes,e.rows.squeeze(-1),e.columns.squeeze(-2),e.decode()),(z.to(torch.int8),r,c,dequant)):
                assert torch.equal(actual,expected),(name,'official reference mismatch')
            assert r2 is None and all(torch.equal(a,b) for a,b in zip((official[0],official[1],official[2],official[4]),(gdn[0],gdn[1],gdn[2],gdn[4])))
            zero_diff=int(((e.decode()==0)&(dequant==0)&(torch.signbit(e.decode())!=torch.signbit(dequant))).sum())
            observed.append({'panel':name,'torch_equal_codes_scales_dequant':True,'signed_zero_dequant_differences':zero_diff})
        return {'official_commit':'f2deac58131219e6aecff274f8eabe88e08d2e3c','architecture':'kda',
                'bits':'all8','impact':'ones','pivots':0,'equal_semantics':'torch.equal, signed-zero differences separately reported',
                'observations':observed,'direct_equation_reference_exact':True}
    record('5_qualified_and_official_all8_ones_parity',reference_parity)

    def qdq_sim():
        state=torch.zeros(1,2,7,11);previous=None;writes=consumed=0
        for i in range(16):
            if previous is not None:
                assert torch.equal(state,previous);consumed+=1
            update=state*.9+canonical_panel()[:1,:2]*(i+1)*.001
            encoded=codec.encode_dualm(update); state=encoded.decode(); writes+=1
            assert state.dtype==torch.float32 and state.data_ptr()!=update.data_ptr()
            previous=state.clone()
            del update,encoded
        return {'writes':writes,'next_step_consumed':consumed,'hidden_high_precision_state_for_next_step':False,
                'scope':'CPU synthetic persistent recurrence only; not real model next-kernel proof'}
    record('6_every_token_persistent_QDQ_simulation',qdq_sim)

    def synthetic_recurrence():
        gen=torch.Generator().manual_seed(16102026)
        a=b=torch.zeros(1,2,7,11); digest=hashlib.sha256()
        for i in range(128):
            k=torch.randn((1,2,7),generator=gen)*.15;v=torch.randn((1,2,11),generator=gen)
            gate=torch.sigmoid(torch.randn((1,2,7),generator=gen))
            for which in range(2):
                state=a if which==0 else b
                retained=state*gate.unsqueeze(-1)
                residual=v-torch.einsum('bhk,bhkv->bhv',k,retained)
                updated=retained+k.unsqueeze(-1)*(residual*.3).unsqueeze(-2)
                if which==0:
                    e=codec.encode_dualm(updated);a=e.decode()
                else:
                    b=codec.official_reference(updated)[4]
            assert torch.equal(a,b),(i,'synthetic recurrence parity')
            digest.update(a.contiguous().view(torch.uint8).numpy().tobytes())
        return {'steps':128,'nonuniform_Key_gate':True,'adapter_vs_official_every_step_exact':True,
                'digest':digest.hexdigest(),'scope':'codec recurrence reference, no KDA model execution'}
    record('7_128_step_nonuniform_gate_synthetic_recurrence',synthetic_recurrence)

    def fresh_process():
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
        env['PYTHONHASHSEED']='0'
        values=[subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--digest'],env=env,
                    capture_output=True,text=True,check=True).stdout.strip() for _ in range(2)]
        assert values[0]==values[1] and len(values[0])==64
        return {'fresh_processes':2,'digest':values[0],'exact':True,'CUDA_initialized':False}
    record('8_fresh_process_deterministic_digest',fresh_process)
    assert not torch.cuda.is_initialized()
    report={'task_id':'LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1','status':'PASS_CPU_CODEC_ONLY',
            'CPU_ONLY':True,'CUDA_initialized':False,'GPU_model_forward':False,'test_count':len(results),
            'all8_ones_pivots0':True,'official_reference_access':'ACTUAL_PINNED_SOURCE_AST_VERIFIED',
            'physical_packed_INT8_runtime':'NOT_QUALIFIED','tests':results,'started_unix':started,
            'completed_unix':time.time(),'python':sys.version,'torch':torch.__version__,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'codec_adapter.py',Path(__file__).resolve(),ROOT/'reference_sources/qualified_dual_axis_codec.py',ROOT/'reference_sources/stepquant_quantization.py']}}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'tests':len(results),'report':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--digest',action='store_true');parser.add_argument('--output')
    args=parser.parse_args()
    if args.digest:digest_worker()
    else:main(args.output)
