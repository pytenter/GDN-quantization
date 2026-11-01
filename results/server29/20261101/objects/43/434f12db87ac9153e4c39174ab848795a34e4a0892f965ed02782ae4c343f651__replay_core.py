"""Full actual transition replay; no model training or outcome-dependent choices."""
from common import *
from transition import update_pre_q,propagate_zero_write,qdq_r128
import torch
EPS=1e-12
HORIZONS=[2048,8192,32768]
WINDOW_START=1024
SIGN_SEED=20261005
FIELDS=['J_ACC','J_INJ','J_DECAY','J_PROP','historical_relative_injection']

def sign_schedule(K,T):
    # A counter hash depends only on fixed seed, 1-based token, probe rank.
    rows=[[1 if hashlib.sha256(f'{SIGN_SEED}:{t}:{r}'.encode('ascii')).digest()[0]&1 else -1 for r in range(K)] for t in range(1,T+1)]
    return torch.tensor(rows,dtype=torch.int8)

def batch_record(records,t,device):
    return {key:torch.cat([records[l][key][:,t:t+1] for l in LAYERS],dim=0).to(device).contiguous() for key in ['q','k','v','beta','log_decay']}

def repeated_record(rec,K):
    return {key:value.repeat((K,)+(1,)*(value.ndim-1)).contiguous() for key,value in rec.items()}

def replay_step(fp,quant,z,decay,rec,R,signs):
    L=len(LAYERS);K=len(signs)
    fp=update_pre_q(fp,rec)
    v=rec['v'].float().matmul(R[:,None]).to(rec['v'].dtype)
    y=update_pre_q(quant,rec,v)
    quant,codes,scale=qdq_r128(y);eta=quant-y
    zi=propagate_zero_write(z.reshape(K*L,16,128,128),repeated_record(rec,K))
    z=zi.reshape(K,L,16,128,128)+signs[:,None,None,None,None].float()*eta[None]
    norm=lambda a:a.square().sum(dim=(-3,-2,-1))
    den_fp=norm(fp)+EPS
    inj=norm(eta)/(norm(y)+EPS)
    # log_decay layout [layer-as-batch,token=1,true-head,key]. Erase absent.
    d2=rec['log_decay'].float().mul(2).exp().mean(dim=(1,2,3))
    decay=d2*decay+inj
    metrics=torch.stack([norm(quant.matmul(R[:,None].transpose(-1,-2))-fp)/den_fp,
        inj,decay,norm(z).mean(dim=0)/den_fp,
        norm(eta).sqrt()/(norm(y).sqrt()+EPS)],dim=0)
    return fp,quant,z,decay,metrics

def initial_states(K,device):
    shape=(len(LAYERS),16,128,128)
    return (torch.zeros(shape,device=device,dtype=torch.float32),
        torch.zeros(shape,device=device,dtype=torch.float32),
        torch.zeros((K,)+shape,device=device,dtype=torch.float32),
        torch.zeros(len(LAYERS),device=device,dtype=torch.float32))
