"""Historical development trace only: batch identity and memory/finite feasibility."""
from common import *
from replay_core import *
import argparse,gc,traceback

def main(gpu):
    assert not (D/'run/DISCOVERY_STARTED.json').exists()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    hw=hardware();runtime();assert torch.cuda.device_count()==1
    source=V3/'inputs/traces/wikitext2raw-train-000.pt'
    x=torch.load(source,map_location='cpu',weights_only=False);rr=x['layer_records']
    device='cuda';torch.set_num_threads(4)
    parity=[]
    with torch.inference_mode():
        for t in [0,7,31,127,511]:
            rec=batch_record(rr,t,device)
            state=torch.cat([rr[l]['initial_teacher_state'] for l in LAYERS],dim=0).float().to(device)
            batched=update_pre_q(state,rec)
            scalar=torch.cat([update_pre_q(state[i:i+1],{k:v[i:i+1] for k,v in rec.items()}) for i in range(len(LAYERS))])
            assert torch.equal(batched,scalar),('LAYER_BATCH_BITWISE_PARITY',t)
            rep=repeated_record(rec,8);z=state.repeat(8,1,1,1)
            actual=propagate_zero_write(z,rep)
            expected=propagate_zero_write(state,rec).repeat(8,1,1,1)
            assert torch.equal(actual,expected),('PROBE_BATCH_BITWISE_PARITY',t)
            parity.append(dict(token_zero_based=t,layer_batch_bitwise='PASS',probe_batch_bitwise='PASS'))
        del state,batched,scalar,rep,z,actual,expected,rec
        R=torch.stack([torch.eye(128) for l in LAYERS]).to(device)
        canaries=[]
        for K in [8,4]:
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            fp,quant,z,decay=initial_states(K,device);sign=sign_schedule(K,512).to(device)
            torch.cuda.synchronize();started=time.time();finite=True
            for t in range(512):
                rec=batch_record(rr,t,device)
                fp,quant,z,decay,metrics=replay_step(fp,quant,z,decay,rec,R,sign[t])
                # Only finite stability, no quality scores are recorded or used.
                if t%64==63:finite &= bool(torch.isfinite(metrics).all() and torch.isfinite(z).all())
            torch.cuda.synchronize();elapsed=time.time()-started
            free,total=torch.cuda.mem_get_info();peak=torch.cuda.max_memory_reserved()
            actual_hw=hardware()['gpus'][gpu]
            headroom=min(free/total,1-actual_hw['used_mib']/actual_hw['total_mib'])
            canaries.append(dict(K=K,tokens=512,seconds=elapsed,tokens_per_second=512/elapsed,finite=finite,peak_reserved_bytes=peak,headroom_fraction=headroom,stable=finite and elapsed>0 and headroom>=.20))
            del fp,quant,z,decay,metrics,sign,rec
    chosen=max([r['K'] for r in canaries if r['stable']],default=0)
    assert chosen in [4,8],('NO_FEASIBLE_K',canaries)
    emit(f'runtime/feasibility_gpu{gpu}.json',dict(gate='PASS',source=row(source),physical_gpu=gpu,gpu_uuid=UUIDS[gpu],parity=parity,canaries=canaries,selected_max_K=chosen,no_fresh_documents=True,no_quality_selection=True,resource_policy='replay after FP capture model is unloaded; not co-resident with teacher model'))
    print('FEASIBILITY_PASS',gpu,chosen,canaries,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);a=p.parse_args()
    try:main(a.gpu)
    except BaseException as e:
        emit(f'run/FEASIBILITY_STOP_GPU{a.gpu}.json',dict(gate='FAIL',error=repr(e),traceback=traceback.format_exc(),scientific_measurements=0,no_auto_retry=True));raise
