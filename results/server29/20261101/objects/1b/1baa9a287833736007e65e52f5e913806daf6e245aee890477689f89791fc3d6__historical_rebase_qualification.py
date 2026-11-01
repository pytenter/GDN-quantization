"""Measurement-only qualification on the original historical numerical backend."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, sys, traceback
sys.dont_write_bytecode=True
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
Q=R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
BASE=Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1')
H=BASE/'checkout';D=H/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
RUNS=Path('/data/zypan/experiment_runs/ling_historical_runtime_rebase_v1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def cp(p,d):
    d.parent.mkdir(parents=True,exist_ok=True)
    with d.open('xb') as f:f.write(p.read_bytes())
    assert sha(p)==sha(d)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--hooks',choices=['on','off'],default='on');a=ap.parse_args()
    out=RUNS/a.run_id;out.mkdir(parents=True)
    assert read(Q/'HISTORICAL_RUNNER_SELF_REPLAY_GATE.json')['gate']=='PASS'
    assert sha(D/'scripts/bridge.py')=='e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
    spec=importlib.util.spec_from_file_location('unchanged_historical_observability_bridge',D/'scripts/bridge.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    # Output relocation only. ROOT/SNAP/modules/model/runtime stay historical.
    b.D=out
    for name in ['premeasurement_commit.json','premeasurement_sha256.json','source_identity.json','document_manifest.json','rotation_manifest.json','runtime_manifest.json','preregistration.json']:
        cp(D/name,out/name)
    cp(D/'historical_assets/asset_manifest.json',out/'historical_assets/asset_manifest.json')
    for row in read(Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json')['files']:
        assert sha(row['path'])==row['sha256'],('HISTORICAL_RUNTIME_DRIFT',row['path'])
    assets=read(R/'provenance_capsule_v3/scientific_asset_manifest.json')['files']
    for row in assets:assert sha(row['path'])==row['sha256'],('SCIENTIFIC_ASSET_MISMATCH',row['path'])
    emit(out/'SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY.json',dict(gate='PASS',files=len(assets),manifest_sha256=sha(R/'provenance_capsule_v3/scientific_asset_manifest.json')))
    co,t,c,cap,core,stats=b.modules();original_capture=cap.Capture
    class ReadOnlyMeasurementCapture(original_capture):
        def __init__(self):
            self.measurement_hashes=[]
            super().__init__()
        def observed_kernel(self,**kwargs):
            values=super().observed_kernel(**kwargs)
            # Values are neither cloned back into caches nor replaced. Read only.
            self.measurement_hashes.append(dict(layer=int(self.probe.current_layer),state=b.tensor_sha(values[1])))
            return values
    if a.hooks=='on':cap.Capture=ReadOnlyMeasurementCapture
    b.verify();b.smoke(0)
    result=read(out/'smoke/repeat0/result.json');historical=read(Q/'original_historical_reference.json')
    fields=['FP_step_hashes','target','scores','endpoint']
    match={k:result[k]==historical[k] for k in fields}
    emit(out/'HISTORICAL_NUMERICAL_PARITY.json',dict(gate='PASS' if all(match.values()) else 'FAIL',comparisons=match,hooks=a.hooks))
    assert all(match.values()),('HISTORICAL_REFERENCE_PARITY_FAIL',match)
    if a.hooks=='off':
        emit(out/'QUALIFICATION_COMPLETE.json',dict(gate='PASS',hooks='off',formal_measurements=0));return
    # OFF = successful exact original Phase-A replay; ON adds read-only hooks.
    off=read(Q/'historical_self_replay_result.json')
    transparency={k:off[k]==result[k] for k in fields}
    assert all(transparency.values()),('INSTRUMENTATION_TRANSPARENCY_FAIL',transparency)
    emit(out/'OBSERVABILITY_INSTRUMENTATION_TRANSPARENCY.json',dict(gate='PASS',comparisons=transparency,
         OFF_result_sha256=sha(Q/'historical_self_replay_result.json'),ON_result_sha256=sha(out/'smoke/repeat0/result.json'),
         K3_comparison_field='scores/LOCAL_K3_ACCUMULATED; all original scores exact',no_scientific_intervention_added=True))
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
    got=q.query_readout(state,query);expected_k3=torch.sum(state*query.unsqueeze(-1),dim=-2).unsqueeze(1)
    assert torch.allclose(got,expected_k3)
    delta=got-expected_k3;rotated={};model=cap.Capture()
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
    numeric=dict(H={k:result[k] for k in fields},learned=dict(scores=scores,target=learned_target,endpoint=endpoint),
         rotated_model=rotated,k3=dict(max_abs=float(delta.abs().max()),actual_hash=b.tensor_sha(got),expected_hash=b.tensor_sha(expected_k3)),
         basis_and_jprop={k:v for k,v in read(out/'smoke/basis_and_jprop_parity.json').items() if k!='dependencies'})
    flags=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
         cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    assert flags=={k:historical['hardware'][k] for k in flags}
    emit(out/'NUMERIC_FINGERPRINT.json',numeric)
    emit(out/'QUALIFICATION_COMPLETE.json',dict(gate='PASS',hooks='on',historical_runtime='WHOLE_ORIGINAL_BACKEND',
         SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE='PASS',RUNTIME_ENVIRONMENT_IDENTITY_GATE='PASS',
         OBSERVABILITY_INSTRUMENTATION_TRANSPARENCY_GATE='PASS',BASIS_SEMANTICS_GATE='PASS',
         HISTORICAL_K3_PARITY_GATE='PASS',HISTORICAL_JPROP_PARITY_GATE='PASS',FUTURE_KL_HARNESS_PARITY_GATE='PASS',
         HARDWARE_ASSIGNMENT_GATE='PASS',STORAGE_CAPACITY_GATE='PASS',FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='PENDING_3_PROCESS_COMPARISON',
         runtime=c.runtime(),torch_flags=flags,coverage=['historical FP128 fixture','H and learned L6 full-model rotated 4token stream',
         'persistent every-token INT8 QDQ/writeback','KDA','K3','JPROP','futureKL H/L6','basis all16'],
         FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    print('WHOLE_RUNTIME_REBASE_QUALIFICATION_COMPLETE',a.run_id,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        out=RUNS/sys.argv[1];out.mkdir(parents=True,exist_ok=True)
        if not (out/'STOP.json').exists():emit(out/'STOP.json',dict(error=repr(e),traceback=traceback.format_exc(),FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
        raise
