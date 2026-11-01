"""Frozen doc-rank/GPU worker. One trace, then sixteen single long rollouts."""
from common import *
from replay_core import *
from capture import Capture
import argparse,gc,traceback

def frozen_guard():
    assert not (D/'run/FORMAL_STOP.json').exists()
    receipt=load(D/'run/premeasurement_commit.json')
    assert command(['git','rev-parse','HEAD'])==receipt['commit']
    inv=load(D/'manifests/premeasurement_sha256.json')
    for f in inv['files']:assert sha(f['path'])==f['sha256'],('SOURCE_OR_INPUT_CHANGED',f['path'])
    assert load(D/'provenance/engineering_parser_recovery_v2_completion.json')['gate']=='PASS'
    assert not any((D/'run'/name).exists() for name in ['POST_METADATA_RECOVERY_V2_STOP.json','FEASIBILITY_STOP_GPU0.json','FEASIBILITY_STOP_GPU1.json','CAPTURE_CANARY_STOP_GPU0.json','CAPTURE_CANARY_STOP_GPU1.json'])
    return receipt

def unit(doc,condition,trace,K,signs,commit):
    device='cuda';artifact=condition['artifact'];assert sha(artifact['path'])==artifact['sha256']
    r=torch.load(artifact['path'],map_location='cpu',weights_only=False)['rotations']
    R=torch.stack([r[l] for l in LAYERS]).to(device)
    fp,quant,z,decay=initial_states(K,device)
    sums=torch.zeros((len(FIELDS),len(LAYERS)),device=device,dtype=torch.float64)
    scores={};started=time.time();n=0
    with torch.inference_mode():
        for entry in trace['chunks']:
            assert not (D/'run/FORMAL_STOP.json').exists()
            f=entry['artifact'];assert sha(f['path'])==f['sha256']
            x=torch.load(f['path'],map_location=device,weights_only=False)
            for j in range(x['q'].shape[1]):
                t=entry['start_zero_based']+j+1
                rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                fp,quant,z,decay,metrics=replay_step(fp,quant,z,decay,rec,R,signs[t-1])
                if t>=WINDOW_START:sums+=metrics.double();n+=1
                if t in HORIZONS:
                    assert n==t-WINDOW_START+1
                    assert torch.isfinite(sums).all() and torch.isfinite(z).all() and torch.isfinite(fp).all() and torch.isfinite(quant).all()
                    avg=(sums/n).cpu()
                    values={name:float(avg[i].mean()) for i,name in enumerate(FIELDS)}
                    values['sqrt_J_ACC']=values['J_ACC']**.5
                    values['CROSS_TERM_RESIDUAL']=values['J_ACC']-values['J_PROP']
                    scores[str(t)]=dict(window=[WINDOW_START,t],tokens=n,**values,per_layer={str(l):{name:float(avg[i,k]) for i,name in enumerate(FIELDS)} for k,l in enumerate(LAYERS)})
            assert torch.equal(fp,x['fp_after']),('FORMAL_CAPTURE_REPLAY_BITWISE_PARITY',doc['document_id'],condition['condition_id'],entry['end_one_based'])
            del x
        assert n==32768-1024+1 and set(scores)=={'2048','8192','32768'}
    result=dict(gate='PASS',task=TASK,phase=doc['phase'],document_id=doc['document_id'],selection_rank=doc['selection_rank'],physical_gpu=doc['physical_gpu'],gpu_uuid=doc['gpu_uuid'],condition_id=condition['condition_id'],rotation_sha256=artifact['sha256'],token_ids_sha256=doc['token_ids_sha256'],trace_sha256=sha(D/'run/traces'/doc['document_id']/'trace_freeze.json'),premeasurement_commit=commit,probe_count=K,sign_schedule_sha256=sha(D/'protocol/probe_signs.pt'),scores=scores,tokens_consumed=32768,single_rollout_all_horizons=True,capture_replay_bitwise_parity='PASS',elapsed_seconds=time.time()-started,source_manifest_sha256=sha(D/'manifests/premeasurement_sha256.json'))
    f=D/'raw'/doc['phase']/(doc['document_id']+'__'+condition['condition_id']+'.json')
    emit(f,result);f.chmod(0o444)
    print('REPLAY_UNIT_COMPLETE',doc['phase'],doc['document_id'],condition['condition_id'],'seconds',round(time.time()-started,2),flush=True)
    del fp,quant,z,decay,sums,R,metrics,rec;gc.collect();torch.cuda.empty_cache()

def main(gpu,phase):
    receipt=frozen_guard();runtime();hw=hardware()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu) and torch.cuda.device_count()==1
    assert hw['gpus'][gpu]['used_mib']<2000,('GPU_EXTERNAL_OCCUPANCY',hw)
    if phase=='CONFIRMATION':
        s=load(D/'analysis/discovery_selection.json');assert s['DISCOVERY_SURROGATE_GATE']=='PASS'
        assert load(D/'manifests/discovery_raw_freeze.json')['complete']==64
        assert len(list((D/'raw/CONFIRMATION').glob('*.json')))==0
    docs=[d for d in load(D/'protocol/frozen_documents.json')['streams'] if d['phase']==phase and d['physical_gpu']==gpu]
    assert len(docs)==2
    bank=load(D/'protocol/frozen_rotation_bank.json');assert bank['gate']=='PASS' and bank['count']==16
    K=load(D/'protocol/frozen_metrics_and_execution.json')['probe_count']
    signs=torch.load(D/'protocol/probe_signs.pt',map_location='cuda',weights_only=True)
    assert signs.shape==(32768,K) and torch.equal(signs.cpu(),sign_schedule(K,32768))
    for d in docs:
        frozen_guard();tf=d['token_file'];assert sha(tf['path'])==tf['sha256']
        ids=load(tf['path']);assert len(ids)==32769
        assert hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids)).hexdigest()==d['token_ids_sha256']
        trace_root=D/'run/traces'/d['document_id']
        assert not trace_root.exists(),('NO_SCIENTIFIC_RETRY',d['document_id'])
        emit(D/'run'/f'{phase}_DOC_STARTED_{d["selection_rank"]}.json',dict(document_id=d['document_id'],physical_gpu=gpu,commit=receipt['commit'],time_unix=time.time()))
        c=Capture()
        try:trace=c.document(ids[:32768],trace_root)
        finally:c.close()
        assert trace['tokens']==32768
        # All records and writes frozen before replay. No outcome aggregation.
        for cond in bank['conditions']:unit(d,cond,trace,K,signs,receipt['commit'])
    emit(f'run/{phase}_GPU{gpu}_COMPLETE.json',dict(gate='PASS',units=32,physical_gpu=gpu,commit=receipt['commit']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--phase',choices=['DISCOVERY','CONFIRMATION'],required=True);a=p.parse_args()
    try:main(a.gpu,a.phase)
    except BaseException as e:
        emit(f'run/{a.phase}_GPU{a.gpu}_FAILURE.json',dict(gate='FAIL',error=repr(e),traceback=traceback.format_exc(),no_auto_retry=True));raise
