"""Shared deterministic replay/reducer for engineering and frozen Discovery.

Uses unchanged production helper/rotation/QDQ via the passed Replay class.
FP64 algebra and metric normalization are those frozen before G4 results.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from isolated_replay_engineering import Replay,LAYERS,CONDITIONS,GATES,RULES,require
from capture_original_drive import AMEND,ROOT,read,save,sha,tensor_sha

class LayerReducer:
    def __init__(self,first_token):
        self.first=first_token;self.count=0
        self.normalized=[[0.]*6 for _ in LAYERS];self.raw=[[0.]*6 for _ in LAYERS]
    def update(self,token,normalized,raw):
        assert len(normalized)==len(raw)==24
        if token<self.first:return
        self.count+=1
        for i in range(24):
            for j in range(6):
                assert math.isfinite(normalized[i][j]) and math.isfinite(raw[i][j])
                self.normalized[i][j]+=normalized[i][j];self.raw[i][j]+=raw[i][j]
    def snapshot(self):
        assert self.count>0
        per_layer=[[v/self.count for v in row] for row in self.normalized]
        raw_layer=[[v/self.count for v in row] for row in self.raw]
        return dict(token_count=self.count,first_token=self.first,per_layer=per_layer,
            normalized_mean=[math.fsum(row[j] for row in per_layer)/24 for j in range(6)],
            raw_energy_mean=[math.fsum(row[j] for row in raw_layer)/24 for j in range(6)],raw_per_layer=raw_layer)

def observe(R,c,fp_old,stored,records):
    t=R.torch;fp={l:R.step('Native',l,fp_old[l],records[l]) for l in LAYERS}
    hist={};pre={};reference={}
    for l in LAYERS:
        native_old=t.zeros_like(fp[l]) if stored[l] is None else R.physical(c,l,stored[l])
        hist[l]=R.step('Native',l,native_old,records[l])
        pre[l]=R.step(c,l,stored[l],records[l])
        clean_old=t.zeros_like(fp[l]) if fp_old[l] is None else fp_old[l]
        reference[l]=R.step(c,l,R.inverse(c,l,clean_old),records[l])
    R.qdq(pre,'path');R.qdq(reference,'reference')
    norm=[];raw=[];vector_max=0.;energy_max=0.
    u64=2**-53;n=32*128*128;gamma=n*u64/(1-n*u64)
    for l in LAYERS:
        sq=R.physical(c,l,pre[l]).double();f=fp[l].double();history=hist[l].double();clean=R.physical(c,l,reference[l]).double()
        E=sq-f;H=history-f;IP=sq-history;IR=clean-f;interaction=IP-IR
        vector=float((E-H-IP).abs().max());scale=max(1.,float(sq.abs().max()),float(f.abs().max()),float(history.abs().max()))
        energies=t.stack([(E*E).sum(),(H*H).sum(),(IP*IP).sum(),(IR*IR).sum(),2*(H*IP).sum(),(interaction*interaction).sum()])
        numeric=energies.cpu().tolist();energy=abs(numeric[0]-numeric[1]-numeric[2]-numeric[4])
        bound=16*gamma*max(1.,numeric[0],numeric[1],numeric[2],abs(numeric[4]))
        assert all(math.isfinite(x) for x in numeric),'nonfinite energy'
        assert vector<=8*u64*scale,'vector closure failed'
        assert energy<=bound,'energy closure failed'
        denom=float((f*f).sum())+R.rules['epsilon']
        raw.append(numeric);norm.append([x/denom for x in numeric])
        vector_max=max(vector_max,vector);energy_max=max(energy_max,energy)
    return fp,pre,norm,raw,vector_max,energy_max

def execute(mode,length,run,document,condition):
    frozen=read(AMEND/'long_qualification_freeze.json')
    for relative,expected in frozen['code_sha256'].items():assert sha(AMEND/relative)==expected
    assert sha(RULES)==frozen['rules_sha256']
    require('isolation')
    science=mode=='discovery'
    policy=read(AMEND/'protocol/extended_capture_policy.json')
    if science:
        assert length==32768 and document in [1,2,3,4] and condition in CONDITIONS
        auth=read(AMEND/'scientific_execution_authorization.json')
        assert auth['DISCOVERY_EXECUTION_AUTHORIZED']=='YES' and auth['PREMEASUREMENT_COMMIT_SHA']
        for phase in ['determinism','streaming']:require(phase)
        item=policy['documents'][document-1];path=Path(item['output_directory'])
        out=path/'results'/condition;conditions=[condition];first=1024;horizons=[2048,8192,32768]
    else:
        assert length in [256,2048] and run in ['01','02','03'] and document is None and condition is None
        if length==2048:assert run!='03' and read(GATES/'determinism_256.json')['gate']=='PASS'
        path=AMEND/'engineering/capture_01' if length==256 else Path(policy['engineering2048']['output_directory'])
        out=AMEND/'engineering/repeatability'/('length_%05d_run_%s'%(length,run))
        conditions=CONDITIONS;first=1;horizons=[length]
    assert not out.exists(),'write-once replay; no retry'
    m=read(path/'drive_manifest.json');seal=read(path/'capture_seal.json')
    drive_sha=sha(path/'drive_manifest.json');assert seal['DOCUMENT_DRIVE_SEAL_GATE']=='PASS' and seal['drive_manifest_sha256']==drive_sha
    assert m['token_count']>=length and m['all_complete_finite'] and m['layer_ids']==LAYERS
    refs={r['token']:r['per_layer_sha256'] for r in read(path/'fp_state_hashes.json')}
    R=Replay();start=time.monotonic();identities={};results={};peak_before=R.torch.cuda.memory_allocated()
    with R.guard():
        for c in conditions:
            fp={l:None for l in LAYERS};stored={l:None for l in LAYERS};reducer=LayerReducer(first)
            states_hash=hashlib.sha256();metrics_hash=hashlib.sha256();snapshots={};endpoints={}
            maximum_vector=maximum_energy=0.
            for token,records in R.drive(length,path):
                fp,stored,normalized,raw,vr,er=observe(R,c,fp,stored,records)
                maximum_vector=max(maximum_vector,vr);maximum_energy=max(maximum_energy,er)
                reducer.update(token,normalized,raw)
                # State hashes every token for fresh-process determinism. In
                # science only sparse endpoints are persisted, not full state.
                if not science or token in horizons:
                    row={str(l):dict(FP=tensor_sha(fp[l]),stored_Q=tensor_sha(stored[l])) for l in LAYERS}
                    states_hash.update(json.dumps(dict(token=token,layers=row),sort_keys=True,separators=(',',':')).encode())
                    if token in horizons:endpoints[str(token)]=row
                if token in refs:
                    for l in LAYERS:assert tensor_sha(fp[l])==refs[token][str(l)],'Captured FP/replay hash mismatch'
                for values in normalized+raw:
                    for v in values:metrics_hash.update(struct.pack('<d',v))
                if token in horizons:
                    assert reducer.count==token-first+1
                    snapshots[str(token)]=reducer.snapshot()
                if token%256==0:print('ANCHOR='+c+' TOKEN='+str(token),flush=True)
            assert token==length and set(snapshots)==set(map(str,horizons))
            identities[c]=dict(state_hash=states_hash.hexdigest(),metric_hash=metrics_hash.hexdigest(),endpoints=endpoints,
                horizons=snapshots,max_vector_closure_residual=maximum_vector,max_energy_closure_residual=maximum_energy)
            results[c]=snapshots
            del fp,stored,records
    assert R.calls['blocked_cumsum']==R.calls['full_model_forward']==0
    assert R.calls['QDQ_path']==R.calls['QDQ_reference']==length*len(conditions)
    identity=dict(conditions=conditions,length=length,drive_manifest_sha256=drive_sha,
        policy_sha256=sha(RULES),anchor_manifest_sha256=sha(ROOT/'protocol/frozen_anchor_conditions.json'),
        outputs=identities,calls=R.calls,strict_deterministic_algorithms=True,
        tf32=False,persistent_path_never_inverse_remapped=True,metric_names=R.rules['scientific_metric_names'])
    payload=dict(status='PASS',mode=mode,identity=identity,elapsed_seconds=time.monotonic()-start,
        peak_gpu_allocated_bytes=R.torch.cuda.max_memory_allocated(),initial_gpu_allocated_bytes=peak_before,
        capture_manifest=drive_sha,source_sha256=R.policy['source_sha256'],
        token_document_id=m['doc_id'],freeze=frozen)
    if science:
        payload['PREMEASUREMENT_COMMIT_SHA']=auth['PREMEASUREMENT_COMMIT_SHA']
        payload['document_token_sha256']=m['fixture_token_ids_file_sha256']
    save(out/'replay_result.json',payload)
    save(out/'result_seal.json',dict(RESULT_SEAL_GATE='PASS',replay_result_sha256=sha(out/'replay_result.json'),
        drive_manifest_sha256=drive_sha,all_horizons_complete=True,condition_count=len(conditions),read_only=True))
    print('PERSISTENT_REPLAY_SEALED_PASS',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['engineering','discovery'],required=True)
    p.add_argument('--length',type=int,required=True);p.add_argument('--run');p.add_argument('--document',type=int);p.add_argument('--condition')
    a=p.parse_args()
    try:execute(a.mode,a.length,a.run,a.document,a.condition)
    except Exception as e:
        name=('length_%05d_run_%s'%(a.length,a.run)) if a.mode=='engineering' else ('doc_%02d_%s'%(a.document,a.condition))
        path=AMEND/('engineering/repeatability' if a.mode=='engineering' else 'scientific')/(name+'.error.json')
        if not path.exists():save(path,dict(status='FAIL_AND_STOP',mode=a.mode,message=str(e),exception=type(e).__name__,
            traceback=traceback.format_exc(),retry_authorized=False,drive_release_allowed=False))
        raise

if __name__=='__main__':main()
