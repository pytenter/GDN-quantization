"""Same-GPU3 qualification using unchanged historical scientific functions."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, traceback
sys.dont_write_bytecode=True
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
W=Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A=W/'hardware_expansion_gpu3_v1'
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
HD=Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
UUID='GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def git(*args,cwd=W):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()

def hardware(torch):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    assert torch.cuda.device_count()==1
    host=subprocess.check_output(['hostname'],text=True).strip();assert host=='nlpg-SYS-4029GP-TRT'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True).splitlines()
    g=next([v.strip() for v in s.split(',')] for s in rows if s.split(',')[0].strip()=='3')
    assert g[1]==UUID and '3090' in g[2] and g[3]=='580.173.02'
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    assert all(int(s.split(',')[1])==os.getpid() for s in processes if UUID in s)
    old=read(Q/'original_historical_reference.json')['hardware']
    flags=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    assert flags=={k:old[k] for k in flags}
    assert sys.version==old['python'] and torch.__version__==old['torch'] and torch.version.cuda==old['CUDA']
    expected=read(A/'GPU3_LAUNCH_IDENTITY.json')['environment']
    assert all(os.environ.get(k)==v for k,v in expected.items())
    return dict(host=host,physical_gpu=3,gpu_uuid=UUID,driver=g[3],python=sys.version,
        torch=torch.__version__,CUDA=torch.version.cuda,environment=expected,**flags)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--hooks',choices=['off','on'],required=True);a=ap.parse_args()
    out=A/'qualification'/a.run_id;assert not out.exists();out.mkdir(parents=True)
    assert not (A/'QUALIFICATION_STOP.json').exists()
    assert git('rev-parse','HEAD',cwd=R)=='e428b5deb0579b926b2ad098a8699b3f4f2233f0'
    assert not git('status','--porcelain=v1','--untracked-files=no',cwd=R)
    assert not git('status','--porcelain=v1','--untracked-files=no')
    amendment=read(A/'HARDWARE_ASSIGNMENT_AMENDMENT.json')
    for row in amendment['parent_protocol_identity']:assert sha(row['path'])==row['sha256']
    assert sha(HD/'scripts/bridge.py')=='e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
    assets=read(R/'provenance_capsule_v3/scientific_asset_manifest.json')['files']
    for row in assets:assert sha(row['path'])==row['sha256'],('SCIENTIFIC_ASSET_MISMATCH',row['path'])
    for row in read(Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json')['files']:
        assert sha(row['path'])==row['sha256'],('RUNTIME_IDENTITY_MISMATCH',row['path'])
    b=load(HD/'scripts/bridge.py','unchanged_historical_gpu3_backend');b.D=out
    # Assignment and output plumbing only; numerical functions remain original code objects.
    def guard():
        assert not (A/'QUALIFICATION_STOP.json').exists()
        assert not git('status','--porcelain=v1','--untracked-files=no')
        return dict(commit=git('rev-parse','HEAD'),formal_authorized=False)
    def modules():
        co=b.imp(b.OLD_N/'scripts/coherence.py','gpu3_original_coherence')
        co.GPU=3;co.UUID=UUID
        t,c,cap,core,stats=co.modules()
        c.D=out;c.R=W;c.TASK=b.TASK;cap.D=out;cap.R=W;cap.TASK=b.TASK
        return co,t,c,cap,core,stats
    b.guard=guard;b.hardware=hardware;b.modules=modules
    for name in ['source_identity.json','document_manifest.json','rotation_manifest.json','runtime_manifest.json','preregistration.json']:
        with (out/name).open('xb') as f:f.write((HD/name).read_bytes())
    co,t,c,cap,core,stats=b.modules();original_capture=cap.Capture
    class ReadOnlyCapture(original_capture):
        def observed_kernel(self,**kwargs):
            values=super().observed_kernel(**kwargs)
            b.tensor_sha(values[1])
            return values
    if a.hooks=='on':cap.Capture=ReadOnlyCapture
    b.verify();b.smoke(0)
    result=read(out/'smoke/repeat0/result.json');fields=['FP_step_hashes','scores','target','endpoint']
    historical=read(Q/'original_historical_reference.json')
    # Cross-GPU comparison is diagnostic only, never substituted for same-GPU pairing.
    cross={k:result[k]==historical[k] for k in fields}
    emit(out/'CROSS_GPU6_DIAGNOSTIC.json',dict(comparisons=cross,not_a_same_gpu_reference=True,
        historical_cross_hardware_failure='FAIL_PRESERVED',GPU6_reference_used_for_measurement=False))
    if a.hooks=='off':
        emit(out/'LOCAL_FP_FIXTURE_READY.json',dict(gate='PASS',hardware=result['hardware'],
            result_sha256=sha(out/'smoke/repeat0/result.json'),formal_measurements=0));return
    off=read(A/'qualification/reference_off/smoke/repeat0/result.json')
    assert all(result[k]==off[k] for k in fields),'SAME_GPU3_INSTRUMENTATION_OR_NUMERICAL_PARITY_FAIL'
    b.parity()
    co,t,c,cap,core,stats=b.modules();q,r,f=b.historical();torch=core.torch
    trace=read(out/'smoke/repeat0/trace/trace_freeze.json');bank=read(out/'rotation_manifest.json')['bank']
    fixture=c.V3/'inputs/traces/wikitext2raw-train-000.pt'
    ids=[int(v) for v in torch.load(fixture,map_location='cpu',weights_only=False)['token_ids'][:128]]
    learned=next(row for row in bank['conditions'] if row['condition_id']=='L6')
    states,scores,endpoint=b.replay(trace,learned,t,c,core,q,r,[64],128)
    model=cap.Capture()
    try:learned_target=b.pulse_target(model,f,ids,64,128,states[64]['Q'].cuda())
    finally:model.close()
    assert learned_target['E_anchor_sha256']==scores[64]['E_anchor_sha256']
    assert learned_target['FP_anchor_sha256']==b.tensor_sha(states[64]['FP'])
    query=torch.tensor([[[1.,2.,-1.],[.5,-.5,1.]]]);state=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)/10
    got=q.query_readout(state,query);expected=torch.sum(state*query.unsqueeze(-1),dim=-2).unsqueeze(1)
    assert torch.equal(got,expected),'HISTORICAL_K3_PARITY_FAIL'
    rotated={};model=cap.Capture()
    try:
        model.probe.globals['fused_recurrent_kda']=model.probe._wrap(model.probe.orig_fused)
        for label in ['H','L6']:
            cond=next(row for row in bank['conditions'] if row['condition_id']==label)
            rot=torch.load(t.mapped(cond['artifact']['path']),map_location='cpu',weights_only=False)['rotations']
            numeric=model.P2.forward_stream(model.model,model.probe,model.cap,ids[:5],label,{label:rot},list(range(4)),keep_full=False)
            assert len(numeric)==4
            for row in numeric.values():
                assert len(row['qdq'])==len(b.LAYERS)
                for layer,qdq in row['qdq'].items():
                    assert qdq['state_dtype']=='torch.float32'
                    assert row['state'][str(layer)]==qdq['postquant_sha256'],'PERSISTENT_WRITEBACK_IDENTITY_FAIL'
            rotated[label]=numeric
    finally:model.close()
    parity=read(out/'smoke/basis_and_jprop_parity.json')
    numeric=dict(H={k:result[k] for k in fields},learned=dict(scores=scores,target=learned_target,endpoint=endpoint),
        rotated_model=rotated,k3=dict(actual_hash=b.tensor_sha(got),expected_hash=b.tensor_sha(expected)),
        basis_and_jprop={k:v for k,v in parity.items() if k!='dependencies'})
    emit(out/'NUMERIC_FINGERPRINT.json',numeric)
    emit(out/'QUALIFICATION_COMPLETE.json',dict(gate='PASS',formal_measurements=0,GPU3_FORMAL_AUTHORIZED='NO',
        hardware=hardware(torch),SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE='PASS',RUNTIME_ENVIRONMENT_IDENTITY_GATE='PASS',
        OBSERVABILITY_INSTRUMENTATION_TRANSPARENCY_GATE='PASS',BASIS_SEMANTICS_GATE='PASS',
        HISTORICAL_K3_PARITY_GATE='PASS',HISTORICAL_JPROP_PARITY_GATE='PASS',FUTURE_KL_HARNESS_PARITY_GATE='PASS',
        FP_REFERENCE_PAIRING_GATE='PASS',HARDWARE_ASSIGNMENT_GATE='PASS',STORAGE_CAPACITY_GATE='PASS',
        FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='PENDING_THREE_PROCESS_COMPARISON',
        FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',tolerance_relaxed=False))
    print('GPU3_QUALIFICATION_PROCESS_COMPLETE',a.run_id,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (A/'QUALIFICATION_STOP.json').exists():emit(A/'QUALIFICATION_STOP.json',dict(error=repr(e),traceback=traceback.format_exc(),
            GPU3_FORMAL_AUTHORIZED='NO',formal_measurements=0,FINAL_ACTION='FAIL_AND_STOP'))
        raise
