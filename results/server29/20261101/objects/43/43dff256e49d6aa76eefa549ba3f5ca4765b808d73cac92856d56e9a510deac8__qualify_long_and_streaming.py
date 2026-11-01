"""CPU-only exact fresh-process comparison and frozen reducer parity."""
import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,read,save,sha
from isolated_replay_engineering import GATES,LAYERS,CONDITIONS,require
from persistent_anchor_replay import LayerReducer

def determinism(length):
    require('isolation')
    count=3 if length==256 else 2;identities=[];files=[]
    for run in range(1,count+1):
        root=AMEND/'engineering/repeatability'/('length_%05d_run_%02d'%(length,run))
        v=read(root/'replay_result.json');seal=read(root/'result_seal.json')
        assert v['status']=='PASS' and seal['RESULT_SEAL_GATE']=='PASS'
        assert sha(root/'replay_result.json')==seal['replay_result_sha256']
        assert v['identity']['length']==length and v['identity']['conditions']==CONDITIONS
        ids=json.dumps(v['identity'],sort_keys=True,separators=(',',':')).encode()
        identities.append(hashlib.sha256(ids).hexdigest());files.append(dict(path=str(root/'replay_result.json'),sha256=sha(root/'replay_result.json')))
    exact=len(set(identities))==1
    result=dict(gate='PASS' if exact else 'FAIL',length=length,fresh_processes=count,
        conditions=CONDITIONS,identity_hashes=identities,files=files,
        criterion='Every token FP/Q state hash, all metric bytes, endpoint states, horizon reducer outputs and runtime call counts exact',
        strict_deterministic_algorithms=True,retry_authorized=False)
    save(GATES/('determinism_%d.json'%length),result)
    assert exact,'Fresh-process output identity mismatch; STOP'
    if length==2048:
        assert read(GATES/'determinism_256.json')['gate']=='PASS'
        save(GATES/'determinism.json',dict(gate='PASS',REPLAY_DETERMINISM_GATE='PASS',
            length256_report_sha256=sha(GATES/'determinism_256.json'),length2048_report_sha256=sha(GATES/'determinism_2048.json'),
            conditions=CONDITIONS,strict_deterministic_algorithms=True,state_and_metric_byte_exact=True))

def streaming():
    require('determinism')
    v=read(GATES/'engineering_energy_observations.json');rows=[]
    u=2**-53;gamma=256*u/(1-256*u)
    shared=read(AMEND/'engineering/repeatability/length_00256_run_01/replay_result.json')
    for c in CONDITIONS:
        values=v['observations'][c];assert len(values)==256
        reducer=LayerReducer(1)
        for token,row in enumerate(values,1):
            assert row['token']==token;reducer.update(token,row['normalized'],row['raw_energies'])
        actual=reducer.snapshot();comparison=shared['identity']['outputs'][c]['horizons']['256']
        for form,key in [('normalized','per_layer'),('raw_energies','raw_per_layer')]:
            for layer_index,layer in enumerate(LAYERS):
                for term in range(6):
                    expected=math.fsum(r[form][layer_index][term] for r in values)/256
                    got=actual[key][layer_index][term];live=comparison[key][layer_index][term]
                    bound=16*gamma*max(1.,abs(expected),abs(got),abs(live))
                    result=dict(condition=c,layer=layer,term=term,form=form,full_materialization=expected,
                        streaming=got,shared_replay_streaming=live,bound=bound,
                        difference=abs(expected-got),shared_difference=abs(expected-live),
                        PASS=abs(expected-got)<=bound and abs(expected-live)<=bound)
                    rows.append(result)
                    assert result['PASS'],'Streaming reducer parity failure'
        for form,key in [('normalized','normalized_mean'),('raw_energies','raw_energy_mean')]:
            for j in range(6):
                expected=math.fsum(math.fsum(r[form][i][j] for r in values)/256 for i in range(24))/24
                bound=16*gamma*max(1.,abs(expected),abs(actual[key][j]))
                assert abs(expected-actual[key][j])<=bound
    save(GATES/'streaming.json',dict(gate='PASS',STREAMING_REDUCER_PARITY_GATE='PASS',rows=rows,
        normalized_terms=6,raw_energy_terms=6,conditions=CONDITIONS,length=256,
        materialization_source_sha256=sha(GATES/'engineering_energy_observations.json'),
        shared_replay_source_sha256=sha(AMEND/'scripts/persistent_anchor_replay.py'),
        criterion='Frozen16*gamma256_FP64 bound, all24 layers and aggregate means; engineering window1..256 only'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['determinism256','determinism2048','streaming'],required=True);a=p.parse_args()
    try:
        if a.phase=='streaming':streaming()
        else:determinism(int(a.phase.removeprefix('determinism')))
    except Exception as e:
        gate='streaming' if a.phase=='streaming' else 'determinism'
        path=GATES/(gate+'.error.json')
        if not path.exists():save(path,dict(status='FAIL_AND_STOP',phase=a.phase,message=str(e),traceback=traceback.format_exc(),retry_authorized=False))
        raise
    print(a.phase+'_PASS',flush=True)

if __name__=='__main__':main()
