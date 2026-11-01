"""G0/G1/G2 CPU requalification using byte-identical sealed V1 code only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,ast,hashlib,json,pathlib,platform,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
V1=ROOT/'inherited_cpu_v1'
sys.path[:0]=[str(V1/'src'),str(V1/'tests')]
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from dual_axis_codec import NAMES,encode
from panel import make_panel,thash,PARAMS
import test_reference_parity,test_codec,test_recurrent_semantics

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(path,obj):
    with pathlib.Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def digest(panel):return {n:{c:hashlib.sha256(encode(x,c).to_bytes()).hexdigest() for c in NAMES} for n,x in panel.items()}
def static_checks():
    def text(name):return (ROOT/'runtime_sources'/name).read_text()
    def segment(name,node):
        s=text(name);n=next(n for n in ast.parse(s).body if getattr(n,'name',None)==node);return ast.get_source_segment(s,n)
    arch=text('current_modeling_qwen3_5.py');formal=text('formal_c128_runner.py');science=text('scale_science.py')
    recurrent=segment('current_modeling_qwen3_5.py','torch_recurrent_gated_delta_rule')
    rules=[('Key_H128_after_optional_normalization','return value.float() @ hadamard(128, dtype=torch.float32, device=value.device)' in formal),
       ('Query_and_Key_transform_only','query = self._transform(query, use_norm)' in formal and 'key = self._transform(key, use_norm)' in formal),
       ('FP32_recurrent_operands','contiguous().to(torch.float32)' in recurrent),
       ('decay_then_residual_then_updated_then_readout',recurrent.index('last_recurrent_state = last_recurrent_state * g_t')<recurrent.index('kv_mem =')<recurrent.index('last_recurrent_state = last_recurrent_state +')<recurrent.index('core_attn_out[:, :, i] =')),
       ('architecture_cache_updated_after_helper','cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)' in arch),
       ('formal_scale_axis_epsilon_round_clamp','amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0' in formal and 'torch.round(source / scale).clamp(-127, 127)' in formal),
       ('formal_inplace_dtype_writeback_and_nonfinite_guard','state.copy_(dequant.to(state.dtype))' in formal and 'nonfinite C128 dequant' in formal),
       ('per_token_fullmodel_then_QDQ',science.index('out=self.model(input_ids=self.t.tensor([[token]]')<science.index('self.F.quantize_c128(cache,a)')),
       ('24_layers_each_token_count_check',"a['calls']==1 and a['layers']==set(LAYERS)" in science),
       ('next_consumption_explicit_assertion',"self.t.equal(initial,self.previous[c][l])" in science),
       ('compensated_route_explicit',"qmod.torch_chunk_gated_delta_rule=N.chunk_binding(qmod,'compensated')" in science),
       ('one_original_scan_callsite',segment('numerical_recovery.py','chunk_binding').count("original.count('g = g.cumsum(dim=-1)') == 1")==1)]
    assert all(v for _,v in rules),rules
    contract=json.loads(text('NEW_NUMERICAL_CONTRACT.json'));assert contract['status']=='PASS' and contract['selected_route']=='compensated'
    assert contract['old_historical_parity']=='FAIL_UNCHANGED'
    identity=json.loads((ROOT/'SOURCE_IDENTITY_AUDIT.json').read_text())
    for a in identity['files']:assert sha(ROOT/a['local_copy'])==a['sha256']
    return dict(scope='Static source identity and operation ordering only; no model runtime executed',rules=[dict(name=n,status='PASS') for n,v in rules],historical_old_parity='FAIL_UNCHANGED',selected_route='compensated',GPU_numerical_qualification='NOT_RUN')

def main(args):
    if args.worker:
        result=digest(make_panel()) if args.worker=='digest' else test_recurrent_semantics.run()
        assert not torch.cuda.is_initialized();save(args.output,result);return
    out=ROOT/'analysis/CPU_QUALIFICATION_attempt_001';out.mkdir(exist_ok=False)
    gates={f'G{i}':dict(status='NOT_RUN') for i in range(4)};active='G0';first=None
    try:
        start=time.time();verified=[]
        for line in (V1/'hashes/artifact_sha256.txt').read_text().splitlines():
            expected,rel=line.split('  ',1);assert sha(V1/rel)==expected,rel;verified.append(rel)
        assert sha(V1/'hashes/artifact_sha256.txt')=='d659b34a8d8f7dc6196192bcad9d2f95a0aad3d9b1237f58f2e501777c58b015'
        assert sha(V1/'hashes/DELIVERY_VERIFICATION.json')=='562c63ace8179fb9f519e09ec4953b138621f63a1148892d00b1716eb6689754'
        old=json.loads((V1/'analysis/attempt_001/FIRST_FAILURE.json').read_text());assert old['gate']=='G3'
        current=json.loads((V1/'GATE_STATUS_FINAL.json').read_text());assert all(v['status']=='PASS' for v in current['gates'].values())
        gates['G0']=dict(status='PASS',verified_sealed_files=len(verified),failure_history_recoverable=True,elapsed_s=time.time()-start)
        save(out/'G0.json',gates['G0'])
        active='G1';start=time.time();static=static_checks();save(out/'runtime_static_checks.json',static)
        gates['G1']=dict(status='PASS',scope='STATIC_SEMANTICS_ONLY',checks=len(static['rules']),elapsed_s=time.time()-start);save(out/'G1.json',gates['G1'])
        active='G2';start=time.time();panel=make_panel()
        save(out/'CPU_INPUTS_BEFORE_TESTS.json',dict(parameters=PARAMS,inputs={n:dict(shape=list(x.shape),dtype=str(x.dtype),sha256=thash(x)) for n,x in panel.items()},source_sha256={str(p.relative_to(V1)):sha(p) for d in ('src','tests') for p in sorted((V1/d).glob('*.py'))},entry_sha256=sha(__file__)))
        a=test_reference_parity.canonical(panel);save(out/'canonical_exact_parity.json',a)
        d=test_reference_parity.stepquant(panel);save(out/'stepquant_exact_parity.json',d)
        roundtrip=test_codec.run(panel);save(out/'roundtrip_scale_safety.json',roundtrip)
        expected=digest(panel);original=json.loads((V1/'analysis/attempt_002/fresh_process_codes_1.json').read_text());assert expected==original
        for n in (1,2):
            p=out/f'fresh_process_codes_{n}.json'
            subprocess.run([sys.executable,__file__,'--worker','digest','--output',str(p)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert json.loads(p.read_text())==expected
        recurrent=test_recurrent_semantics.run()
        inherited=json.loads((V1/'analysis/attempt_002/recurrent_audit.json').read_text())['snapshots'];assert recurrent==inherited
        p=out/'fresh_process_recurrent.json'
        subprocess.run([sys.executable,__file__,'--worker','recurrent','--output',str(p)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        assert json.loads(p.read_text())==recurrent
        save(out/'recurrent_semantics.json',dict(snapshots=recurrent,fresh_process_exact=True,old_V1_exact=True,steps=[1,8,32,128],next_step_carry='int8 encoded bytes only',formal_GPU_runtime='NOT_RUN'))
        assert not torch.cuda.is_initialized()
        gates['G2']=dict(status='PASS',canonical_cases=len(a),stepquant_cases=len(d),roundtrip_checks=len(roundtrip),code_digest_fresh_processes=2,recurrent_fresh_processes=1,inherited_V1_hashes_exact=True,elapsed_s=time.time()-start)
        save(out/'G2.json',gates['G2'])
    except Exception as exc:
        first=dict(gate=active,error=repr(exc),traceback=traceback.format_exc());gates[active]=dict(status='FAIL',first_failure=first)
        save(out/'FIRST_FAILURE.json',first)
    save(out/'CPU_ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,torch_cuda_build=torch.version.cuda,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=torch.cuda.is_initialized(),MPS_USED=False,model_loaded=False,model_forward=False,threads=torch.get_num_threads(),platform=platform.platform(),GPU_HOURS_USED=0))
    gates['G3']=dict(status='NOT_RUN',reason='GPU_CANARY_AUTHORIZATION=NO; GPU admission intentionally not attempted')
    save(out/'GATE_STATUS.json',dict(gates=gates,first_failure=first,GPU_HOURS_USED=0,CPU_ONLY=True))
    print(json.dumps(dict(gates=gates,first_failure=first,GPU_HOURS_USED=0)))
    if first:sys.exit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['digest','recurrent']);p.add_argument('--output');main(p.parse_args())
