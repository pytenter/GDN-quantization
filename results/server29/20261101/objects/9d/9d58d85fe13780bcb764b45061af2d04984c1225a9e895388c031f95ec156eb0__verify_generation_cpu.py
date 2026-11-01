"""CPU-only integrity and accounting after all99 admitted trajectories; no scorer."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    C.source(ROOT,a.source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    jobs=json.loads((ROOT/'FORMAL_JOBS.json').read_text())['jobs'];assert len(jobs)==99
    session=json.loads((ROOT/'runtime/FORMAL_SESSION_RESULT.json').read_text());assert set(session['completed'])=={x['id'] for x in jobs}
    rows=[]
    for j in jobs:
        target=ROOT/'formal'/j['id'];closed=json.loads((target/'OWNING_CLOSED.json').read_text());arc=json.loads((target/'ARCHIVE29.json').read_text());x=json.loads((target/'RESULT.json').read_text())
        assert closed['owning_Popen_exit']==0 and closed['release_observed'] and closed['RESULT_sha256']==C.sha(target/'RESULT.json')
        assert arc['status']=='PASS_29_DURABLE_NOREPLACE'
        if x.get('execution_origin')=='V2_COMPLETE_VALIDATED_PRESERVED':
            from legacy_admission import validate_installed
            validate_installed(ROOT,j['id'],a.source_sha,a.permit_sha)
        else:
            assert closed['source_sha256']==a.source_sha and closed['permit_sha256']==a.permit_sha
            assert x['source_manifest_sha256']==a.source_sha and x['permit_sha256']==a.permit_sha
        assert x['job']==j
        digest=x.pop('payload_sha256');assert C.objsha(x)==digest
        r=x['result'];ids=r['generated_token_ids']
        assert r['status']=='COMPLETE' and r['seed']==1 and r['config_id']==j['config_id']
        assert r['generated_token_count']==len(ids) and 0<len(ids)<=65536 and C.objsha(ids)==r['generated_token_ids_sha256']
        assert r['actual_model_forward_calls']==r['input_tokens']+len(ids)-1 and r['actual_decode_forwards']==len(ids)-1
        assert not r['nonfinite'] and r['runtime_error'] is None and not r['diagnostic_only']
        dp=target/('ORIGINAL_DISPATCH.json' if x.get('execution_origin')=='V2_COMPLETE_VALIDATED_PRESERVED' else 'DISPATCH.json')
        ident=json.loads(dp.read_text())['child']
        try:current=C.identity(ident['pid'])
        except FileNotFoundError:current=None
        assert current is None or current['starttick']!=ident['starttick'],'Exact owning worker still live'
        audit=r['quantizer_audit']
        if j['config_id']=='C0':
            assert audit['State_dtype']=='torch.float32' and audit['recurrent_state_nbytes']==24*32*128*128*4
            assert audit['state_quantization_calls']==audit['key_quantization_calls']==audit['buffer_payload_bytes']==0
        else:
            p=8 if j['config_id']=='C1' else 16
            assert audit['actual_layer_coverage']==24
            for layer in audit['storage_by_layer'].values():
                assert layer['decode_tokens']==len(ids)-1 and layer['decode_period']==p and layer['pending_tokens']==(len(ids)-1)%p
                assert layer['decode_direct_calls']==len(ids)-1 and layer['decode_dense_materializations']==(len(ids)-1)//p
                assert layer['validation_dense_materializations']==0 and not layer['dense_state_persistent']
        rows.append(dict(unit=j['id'],config_id=j['config_id'],problem_idx=j['problem_idx'],tokens=len(ids),finish_reason=r['finish_reason'],RESULT_sha256=closed['RESULT_sha256']))
        print(json.dumps(dict(phase='CPU_INTEGRITY_VERIFIED',count=len(rows),unit=j['id'])),flush=True)
    C.atomic(ROOT/'runtime/CPU_GENERATION_VERIFIED.json',dict(status='PASS99_TRUE_CLOSED_PAYLOAD_AND_STORAGE',source_sha256=a.source_sha,permit_sha256=a.permit_sha,rows=rows,accuracy=None,C2='NOT_RUN_FIXED_FP_H_GATE_FAILURE'))
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);raise SystemExit(run(p.parse_args()))
