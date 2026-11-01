#!/usr/bin/env python3
"""CPU-only entry. New attempt output required; never imports a model runner."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,csv,hashlib,json,pathlib,platform,statistics,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from dual_axis_codec import NAMES,encode,EncodedState
from panel import make_panel,thash,PARAMS
from references import FORMAL
import test_reference_parity,test_codec,test_recurrent_semantics

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def deterministic_digest(panel):
    return {name:{c:hashlib.sha256(encode(x,c).to_bytes()).hexdigest() for c in NAMES} for name,x in panel.items()}

def scale_summary(x):
    if x is None:return None
    q=torch.quantile(x.float().flatten(),torch.tensor([0.,.01,.25,.5,.75,.99,1.]))
    return dict(zip(('min','p01','p25','median','p75','p99','max'),map(float,q)))

def metric(x,e):
    out=e.decode();err=(out.double()-x.double());square=err.square();norm=x.double().square().sum().item()
    unbounded=(x/e.rows.float())/e.columns.float() if e.rows is not None else x/e.columns.float()
    result=dict(relative_mse=square.sum().item()/norm if norm else 0.,absolute_mse=square.mean().item(),
                max_absolute_error=err.abs().max().item(),p99_absolute_error=torch.quantile(err.abs().flatten(),.99).item(),
                zero_code_fraction=(e.codes==0).double().mean().item(),saturation_fraction=(e.codes.abs()==127).double().mean().item(),
                overflow_clipping_proxy=(unbounded.abs()>127).double().mean().item(),
                actual_rounded_clamp_fraction=(torch.round(unbounded).abs()>127).double().mean().item(),
                row_scale_distribution=scale_summary(e.rows),column_scale_distribution=scale_summary(e.columns),
                **e.byte_audit())
    assert torch.isfinite(out).all()
    # Timing excludes serialization; includes public validation in encode and FP32 materialization in decode.
    for _ in range(PARAMS['latency_warmup']):encode(x,e.codec).decode()
    en=[];de=[]
    for _ in range(PARAMS['latency_repeats']):
        t=time.perf_counter_ns();item=encode(x,e.codec);en.append((time.perf_counter_ns()-t)/1e6)
        t=time.perf_counter_ns();item.decode();de.append((time.perf_counter_ns()-t)/1e6)
    result.update(encode_median_ms=statistics.median(en),decode_median_ms=statistics.median(de),encode_samples_ms=en,decode_samples_ms=de)
    return result

def real_state(out,manifest):
    # The optional old FP candidate is a native-coordinate numerical qualification
    # capture, not an existing Hadamard FP trajectory. Do not silently promote it.
    indexes=manifest['real_state_index_audit']
    evidence=dict(REAL_STATE_DATA_AVAILABLE='NO',scope='Eligible existing Hadamard-coordinate FP pre-quantization trajectory snapshots in audited small manifests only',
       inspected_small_indexes=len(indexes),inspected_sources=['historical_rw_runner.py','historical_rw_protocol.json','historical_functional_protocol.json','historical_numerical_recovery.py'],
       findings=[{'candidate':'historical read/write feasibility','decision':'NOT_A_FULL_STATE','reason':'torch.load retrieves G metric diagonal, not recurrent-state tensor'},
                 {'candidate':'historical magnitude-conditioned study','decision':'NOT_A_FULL_STATE','reason':'paired scalar manifests and aggregate metrics do not reconstruct full FP state'},
                 {'candidate':'V3 numerical recovery qualification/full_model','decision':'EXCLUDED_FROM_THIS_HADAMARD_PANEL','reason':'Full FP cache captures exist but are native coordinates and include reference-only FP64-scan or compensated-scan trajectories. No previously qualified Hadamard-coordinate snapshot identity. No tensor loaded or copied.'},
                 {'candidate':'8-doc NoInflation archive','decision':'EXCLUDED','reason':'C6R coordinate; not permitted for this task. No archive tensor scan or download.'}],
       limits='NO means none eligible and verified in this bounded audit; not proof no FP tensor exists on the servers. Native-to-H conversion is a possible future separately identified derived panel, not claimed to equal an existing H runtime trajectory.',
       continuous_q_k_g_beta_after8192='NOT_AVAILABLE',omega='NOT_AVAILABLE',w='NOT_AVAILABLE',paired_errors='NOT_AVAILABLE')
    save(out/'real_state_availability_audit.json',evidence)
    return evidence

def worker(args):
    panel=make_panel()
    if args.worker=='digest':payload=deterministic_digest(panel)
    elif args.worker=='recurrent':payload=test_recurrent_semantics.run()
    else:raise ValueError(args.worker)
    assert not torch.cuda.is_initialized()
    save(args.output,payload)

def main(args):
    out=ROOT/'analysis'/args.attempt
    out.mkdir(exist_ok=False)
    gates={f'G{i}':{'status':'NOT_RUN'} for i in range(7)};first_failure=None;active='G0'
    def gate(n,fn):
        nonlocal active
        active=f'G{n}';start=time.time();value=fn();gates[active]={'status':'PASS','elapsed_s':time.time()-start}
        save(out/f'{active}_PASS.json',dict(gates[active],details=value));return value
    try:
        def provenance():
            assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
            manifest=json.loads((ROOT/'source_manifest.json').read_text())
            assert manifest['stepquant']['actual_head']==manifest['stepquant']['pinned_commit']
            assert manifest['stepquant']['status']==''
            for f in manifest['files']:assert sha(ROOT/f['local_copy'])==f['sha256'],f['local_copy']
            assert ROOT.parent.name=='quantizer' and 'QWEN_GDN_HADAMARD_DUAL_AXIS' in ROOT.name
            environment=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),cpu=platform.processor(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=False,MPS_USED=False,model_loaded=False,model_forward=False,GPU_HOURS_USED=0)
            save(out/'environment.json',environment)
            real_state(out,manifest)
            return dict(source_files_verified=len(manifest['files']),historical_worktrees_dirty=[bool(r['status']) for r in manifest['repositories']],environment=environment)
        gate(0,provenance)
        panel=make_panel()
        prereg=dict(task=ROOT.name,phase='CPU_ONLY',parameters=PARAMS,input_manifest={name:dict(shape=list(x.shape),dtype=str(x.dtype),sha256=thash(x),seed=PARAMS['seed']+i) for i,(name,x) in enumerate(panel.items())},
            exact_parity={'G1':'torch.equal codes/scales/dequant; tolerance zero','G2':'torch.equal codes/FP16 scales/reconstruction; tolerance zero'},
            operation_order='DUAL sequential x/r then /c; final reconstruction (r*c)*z matching official fit_state; never x/(r*c)',
            scale_bounds={'half_min':2.**-24,'half_max':65504.,'canonical_epsilon':1e-12,'row_mean_min':1e-20,'LS_den_min':1e-30,'zero_denominator':'retain previous column scale'},
            aggregation='Every case reported; unweighted arithmetic mean across the nine preregistered synthetic cases descriptive only; no selection, inference or sign-test',
            first_failure='Stop subsequent gates; preserve traceback and deterministic reproducer. New-code repairs only with history; no criteria relaxation.',
            code_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ('src','tests') for p in sorted((ROOT/folder).glob('*.py'))},entry_sha256=sha(__file__))
        save(out/'preregistration.json',prereg)
        gate(1,lambda:test_reference_parity.canonical(panel))
        gate(2,lambda:test_reference_parity.stepquant(panel))
        def roundtrip():
            checks=test_codec.run(panel);expected=deterministic_digest(panel)
            repeat=[]
            for n in (1,2):
                path=out/f'fresh_process_codes_{n}.json'
                subprocess.run([sys.executable,str(ROOT/'run_cpu.py'),'--worker','digest','--output',str(path)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                actual=json.loads(path.read_text());assert actual==expected,('fresh-process',n);repeat.append(sha(path))
            save(out/'roundtrip_audit.json',dict(checks=checks,fresh_process_match=True,repeat_artifact_sha256=repeat))
            storage={c:encode(torch.zeros((1,32,128,128)),c).byte_audit() for c in NAMES}
            save(out/'storage_audit.json',dict(shape=[1,32,128,128],codecs=storage,metadata='32-byte versioned serialized header; no hidden full float state in EncodedState',formal_runtime='FP32 QDQ values in recurrent cache; no physical resident INT8 implementation established by CPU tests'))
            return dict(checks=len(checks),fresh_processes=2,storage=storage)
        gate(3,roundtrip)
        def comparison():
            rows=[]
            for name,x in panel.items():
                assert thash(x)==prereg['input_manifest'][name]['sha256']
                for c in NAMES:
                    e=encode(x,c);rows.append(dict(case=name,codec=c,input_sha256=thash(x),**metric(x,e)))
            save(out/'codec_comparison.json',rows)
            flat=[]
            for r in rows:
                d=dict(r)
                for key in ('row_scale_distribution','column_scale_distribution','encode_samples_ms','decode_samples_ms'):
                    value=d.pop(key)
                    if isinstance(value,dict):d.update({key+'_'+k:v for k,v in value.items()})
                    elif key.endswith('distribution'):d[key]='N/A'
                flat.append(d)
            keys=list(dict.fromkeys(k for d in flat for k in d))
            with (out/'codec_comparison.csv').open('x',newline='') as f:
                writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(flat)
            pairs=[]
            for name in panel:
                d={r['codec']:r for r in rows if r['case']==name}
                for a,b in ((NAMES[0],NAMES[1]),(NAMES[1],NAMES[2]),(NAMES[2],NAMES[3]),(NAMES[0],NAMES[3])):
                    old,new=d[a]['relative_mse'],d[b]['relative_mse']
                    pairs.append(dict(case=name,baseline=a,candidate=b,baseline_relative_mse=old,candidate_relative_mse=new,raw_delta=old-new,relative_gain=(old-new)/old if old else None))
            save(out/'paired_comparisons.json',pairs)
            summary={c:{m:statistics.mean(r[m] for r in rows if r['codec']==c) for m in ('relative_mse','absolute_mse','max_absolute_error','p99_absolute_error','zero_code_fraction','saturation_fraction','overflow_clipping_proxy','encode_median_ms','decode_median_ms')} for c in NAMES}
            save(out/'descriptive_summary.json',dict(panel_cases=len(panel),weighting='equal synthetic case weight; not document or model statistics',codecs=summary))
            return summary
        gate(4,comparison)
        def recurrent():
            expected=test_recurrent_semantics.run()
            path=out/'fresh_process_recurrent.json'
            subprocess.run([sys.executable,str(ROOT/'run_cpu.py'),'--worker','recurrent','--output',str(path)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert json.loads(path.read_text())==expected
            save(out/'recurrent_audit.json',dict(snapshots=expected,fresh_process_exact=True,carry='encoded bytes only',unquantized_state_cross_step=False,full_model_qualification='NOT_RUN'))
            return dict(codecs=4,steps=[1,8,32,128],fresh_process_exact=True)
        gate(5,recurrent)
    except Exception as exc:
        first_failure=dict(gate=active,error=repr(exc),traceback=traceback.format_exc())
        gates[active]={'status':'FAIL','reason':repr(exc)}
        save(out/'FIRST_FAILURE.json',first_failure)
        # Inputs are deterministic, preregistered, hash-addressed; preserve concrete state too.
        if 'panel' in locals():torch.save(panel,out/'failure_inputs.pt')
    assert not torch.cuda.is_initialized()
    save(out/'gate_status.json',dict(gates=gates,first_failure=first_failure,GPU_HOURS_USED=0,full_model_QDQ_runtime='NOT_RUN',attempt=args.attempt))
    print(json.dumps(dict(attempt=args.attempt,gates={k:v['status'] for k,v in gates.items()},first_failure=first_failure),ensure_ascii=False),flush=True)
    return 1 if first_failure else 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='attempt_001');parser.add_argument('--worker',choices=['digest','recurrent']);parser.add_argument('--output')
    args=parser.parse_args()
    if args.worker:worker(args)
    else:sys.exit(main(args))
