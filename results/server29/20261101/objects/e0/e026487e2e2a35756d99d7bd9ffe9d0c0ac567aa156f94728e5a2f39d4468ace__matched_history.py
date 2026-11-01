"""Same-runtime clean drive + persistent four-anchor replay, no generation.

The unchanged full model acquires inputs; unchanged formal recurrent functions
consume them online. No FP drive is spliced from a different runtime. Complete
actual model/replay caches are sealed at8192/8256, with fresh-process resume
parity before long labels. Short scores are globally frozen before labels.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numerical_recovery as N
import scale_science as S

ROOT=N.ROOT
PARENT=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1/experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1/amendments/FOUR_GPU_ANCHOR_PARALLELISM_V1')
REPLAY=PARENT/'scripts/isolated_replay_engineering.py'
REDUCER=PARENT/'scripts/persistent_anchor_replay.py'


def device_tree(x,device,memo=None):
    """Isolated complete cache copy; no in-place migration of the main cache."""
    import torch
    if memo is None:memo={}
    if id(x) in memo:return memo[id(x)]
    if torch.is_tensor(x):
        result=x.detach().to(device=device,copy=True);memo[id(x)]=result;return result
    if isinstance(x,dict):
        result={};memo[id(x)]=result
        result.update((k,device_tree(v,device,memo)) for k,v in x.items());return result
    if isinstance(x,list):
        result=[];memo[id(x)]=result;result.extend(device_tree(v,device,memo) for v in x);return result
    if isinstance(x,tuple):
        result=tuple(device_tree(v,device,memo) for v in x);memo[id(x)]=result;return result
    if hasattr(x,'__dict__') and not isinstance(x,type):
        result=copy.copy(x);memo[id(x)]=result
        result.__dict__={k:device_tree(v,device,memo) for k,v in vars(x).items()};return result
    return x


class Accumulator:
    """The original parent's J_ACC_FD column, same scalar order/FP64 axes."""
    def __init__(self,first=1024):
        self.first=first;self.count=0;self.layer_sums=[0.]*24

    def update(self,token,values):
        N.check(len(values)==24 and all(math.isfinite(x) for x in values),'finite 24-layer physical J')
        if token<self.first:return
        self.count+=1
        for i,value in enumerate(values):self.layer_sums[i]+=value

    def snapshot(self):
        N.check(self.count>0,'nonempty original J interval')
        return {'first_token':self.first,'token_count':self.count,'per_layer_mean':[x/self.count for x in self.layer_sums],
                'J_ACC_FD':math.fsum(x/self.count for x in self.layer_sums)/24}


class DriveReplay:
    def __init__(self,harness):
        module=N.module(REPLAY,'v3_reuse_original_isolated_replay')
        self.R=module.Replay()
        self.harness=harness;self.t=harness.t
        self.fp={l:None for l in S.LAYERS}
        self.students={c:{l:None for l in S.LAYERS} for c in S.CONDITIONS}
        self.acc={c:Accumulator() for c in S.CONDITIONS}
        self.future={c:Accumulator(8257) for c in S.CONDITIONS}
        self.boundaries={c:{} for c in S.CONDITIONS}
        self.observations={c:{} for c in S.CONDITIONS};self.baselines={c:{} for c in S.CONDITIONS}
        self.count=0;self.drive_chain='0'*64;self.qdq_calls={c:0 for c in S.CONDITIONS}

    def step(self,position,observe=False,qual=False,metrics=True):
        token=position+1;records=self.harness.records;R=self.R;t=self.t
        N.check(set(records)==set(S.LAYERS), 'captured same-token drive coverage')
        fp_old=self.fp
        fp={l:R.step('Native',l,fp_old[l],records[l]) for l in S.LAYERS}
        for l in S.LAYERS:
            N.check(t.equal(fp[l],self.harness.clean[l]),'same-runtime actual clean capture / replay FP state byte parity')
        record_hashes={str(l):{name:N.tsha(value) for name,value in records[l].items()} for l in S.LAYERS}
        self.drive_chain=hashlib.sha256(bytes.fromhex(self.drive_chain)+json.dumps({'token':token,'layers':record_hashes},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for c in S.CONDITIONS:
            old=self.students[c]
            pre={l:R.step(c,l,old[l],records[l]) for l in S.LAYERS}
            if observe:
                values=[];b0=[]
                for l in S.LAYERS:
                    values.append(self.harness.quant.metrics(pre[l],R.inverse(c,l,fp[l])))
                    N.check(old[l] is not None,'history-conditioned previous persistent state')
                    b0.append(float(self.harness.core.relative_state_mse(R.physical(c,l,old[l]),fp_old[l])))
                self.observations[c][position]={key:S.mean([v[key] for v in values]) for key in values[0]}
                self.baselines[c][position]=S.mean(b0)
            expected={l:self.harness.quant.original(pre[l].detach().clone())[2] for l in S.LAYERS} if qual else None
            R.qdq(pre,'path');self.qdq_calls[c]+=1
            if qual:
                N.check(all(t.equal(pre[l],expected[l]) for l in S.LAYERS),'actual replay formal QDQ/writeback byte parity')
            N.check(all(pre[l].dtype==t.float32 for l in S.LAYERS),'no FP64 persistent buffer')
            if metrics:
                errors=[]
                for l in S.LAYERS:
                    sq=R.physical(c,l,pre[l]).double();f=fp[l].double()
                    # The original parent's physical error numerator and
                    # denominator are retained exactly; no B0 substitution.
                    errors.append(float(((sq-f)*(sq-f)).sum())/(float((f*f).sum())+R.rules['epsilon']))
                self.acc[c].update(token,errors);self.future[c].update(token,errors)
                if token in [8192,8224,8256,32768]:self.boundaries[c][str(token)]=self.acc[c].snapshot()
            self.students[c]=pre
        self.fp=fp;self.count=token

    def fingerprint(self,cache,logits):
        return {'complete_clean_cache':S.snapshot_hash(cache),'clean_logits':N.tsha(logits),
                'FP':{str(l):N.tsha(self.fp[l]) for l in S.LAYERS},
                'persistent_Q':{c:{str(l):N.tsha(self.students[c][l]) for l in S.LAYERS} for c in S.CONDITIONS}}

    def state(self,cache):
        return device_tree({'clean_model_cache':cache,'FP_replay_states':self.fp,'persistent_students':self.students,
                            'consumed_tokens':self.count,'accumulators':self.acc,'future_accumulators':self.future,
                            'J_boundaries':self.boundaries,'QDQ_calls':self.qdq_calls,'complete_drive_hash_chain':self.drive_chain},'cpu')

    def restore(self,payload):
        p=device_tree(payload,'cuda:0')
        self.fp=p['FP_replay_states'];self.students=p['persistent_students'];self.count=p['consumed_tokens']
        self.acc=p['accumulators'];self.future=p['future_accumulators'];self.boundaries=p['J_boundaries'];self.qdq_calls=p['QDQ_calls']
        self.drive_chain=p['complete_drive_hash_chain']
        return p['clean_model_cache']


def setup(gpu,doc):
    torch,F,core,policy,model,tokenizer,started=S.load(gpu)
    ap=N.read(ROOT/'HISTORY_PREMEASUREMENT_SPEC.json')
    N.check(N.sha(__file__)==ap['history_code_sha256'],'frozen history code')
    N.check(N.sha(REPLAY)==ap['replay_sha256'] and N.sha(REDUCER)==ap['original_J_source_sha256'],'original recurrent/reducer dependencies')
    N.check(N.read(ROOT/'TRACK_B_SHORT_SCORE_FREEZE.json')['status']=='PASS','completed qualified TrackB before history')
    harness=S.Harness(model,F,core,torch);replay=DriveReplay(harness)
    d=policy['documents'][doc-1]
    N.check(N.sha(d['token_path'])==d['token_sha256'],'frozen history token identity')
    return torch,harness,replay,N.read(d['token_path']),started,d,ap


def prepare():
    N.save('HISTORY_PREMEASUREMENT_SPEC.json',{'utc':N.now(),'history_code_sha256':N.sha(__file__),
        'replay_source':str(REPLAY),'replay_sha256':N.sha(REPLAY),'original_J_source':str(REDUCER),'original_J_source_sha256':N.sha(REDUCER),
        'scientific_spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json'),
        'capture':'unchanged full FP teacher forced original token drive, qualified compensated chunk at first token; no original B teacher rewrite',
        'replay':'unchanged formal q/k normalization and separate rotation products; original recurrent AST; formal every-token quantizer writeback',
        'sharing':'all four frozen rotations on identical same-GPU drive; full FP cache evolves independently of replay students',
        'observer_clean_state':'paired common clean FP post-state inverse-rotated in original basis; scale only; no FP QDQ writeback',
        'anchor_tensors':'complete clean model cache plus all24 FP/replay persistent states; actual tensor files at8192 and8256',
        'resource_check':'256 engineering then8192 actual cache measurements; extrapolate original attention-cache linear storage and actual scratch overhead; no percent gate',
        'resume':'write-once checkpoint8256;3 subsequent tokens qualification without metrics; fresh process compares all complete cache/logits/FP/student hashes byte exact, then restores original8256 cache for actual continuation once',
        'short_before_long':'globally seal all TrackA andTrackB CSVs and spec before any token>8256 scientific label calculation',
        'B1':'TrackA frozen-drive local functional baseline NOT_COMPARABLE; do not invent or substitute qkv approximation',
        'forbidden':['new rotation','training','optimizer','generation','independent KL','AIME','scientific retry']})


def prefix(gpu,doc,engineering=False):
    torch,harness,replay,ids,started,d,ap=setup(gpu,doc)
    cache=None;length=256 if engineering else 8256;memory=[];anchors=[];window_drive=[]
    outroot=f"{'history_qualification' if engineering else 'track_A'}/doc_{doc:02d}"
    try:
        with torch.inference_mode():
            for p in range(length):
                output,cache=harness.step('FP',ids[p],cache,p)
                N.check(bool(torch.isfinite(output.logits).all()),'finite clean acquisition logits')
                replay.step(p,observe=(p>=8192 or (engineering and p>=128)),qual=engineering,metrics=not engineering)
                if p>=8192 and not engineering:window_drive.append(device_tree(harness.records,'cpu'))
                if p+1 in [8192,8256] and not engineering:
                    complete=replay.state(cache)
                    size=sum(t.numel()*t.element_size() for _,t in S.tree_tensors(complete))
                    N.check(N.resources()['disk_available_bytes']>=size+256*1024**2,'actual complete checkpoint physical storage')
                    artifact=N.pt_save(outroot+f'/anchor_{p+1:05d}.pt',complete)
                    receipt={'token':p+1,'artifact':artifact,'complete_cache_fingerprint':S.snapshot_hash(cache),
                             'actual_persistent_replay_fingerprints':replay.fingerprint(cache,output.logits),
                             'no_scientific_reset':True,'document_sha256':d['token_sha256']}
                    N.save(outroot+f'/anchor_{p+1:05d}_receipt.json',receipt);anchors.append(artifact)
                    del complete
                if (p+1)%128==0:
                    memory.append({'tokens':p+1,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                        'clean_cache_bytes':sum(t.numel()*t.element_size() for _,t in S.tree_tensors(cache))})
                    print(json.dumps({'event':'MATCHED_HISTORY_PREFIX','doc':doc,'engineering':engineering,'tokens':p+1}),flush=True)
            if engineering:
                N.save(outroot+'/qualification.json',{'status':'PASS','tokens':256,'all_four_conditions':True,
                    'clean_FP_capture_replay_every_token':'BYTE_EXACT','QDQ_calls':replay.qdq_calls,
                    'finite_observer_count':sum(len(x) for x in replay.observations.values()),'memory':memory,
                    'seconds':time.monotonic()-started,'GPU_hours':(time.monotonic()-started)/3600,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                return
            drive_artifact=N.pt_save(outroot+'/drive_08193_08256.pt',window_drive)
            del window_drive
            # Qualification continuation only: does not enter long integrators
            # and does not overwrite/extend the sealed8256 scientific anchor.
            fingerprints=[]
            for p in range(8256,8259):
                output,cache=harness.step('FP',ids[p],cache,p)
                replay.step(p,metrics=False)
                fingerprints.append(replay.fingerprint(cache,output.logits))
            N.save(outroot+'/RESUME_EXPECTED_3_TOKENS.json',{'status':'QUALIFICATION_REFERENCE','fingerprints':fingerprints,'count':3,'scientific_labels_updated':False})
            rows=[]
            for c in S.CONDITIONS:
                for h in [32,64]:
                    values=[replay.observations[c][p] for p in range(8192,8192+h)]
                    row={'document_index':doc,'condition':c,'H':h,'track':'A','first_0based_token':8192,'last_0based_exclusive':8192+h,
                         'document_sha256':d['text_sha256'],'token_sha256':d['token_sha256'],'GPU_UUID':S.UUIDS[gpu],
                         'B0':S.mean([replay.baselines[c][p] for p in range(8192,8192+h)]),
                         'B1':None,'B2':None,'FUNCTIONAL_BASELINE_STATUS':'NOT_COMPARABLE_FROZEN_DRIVE',
                         'anchor_8192_sha256':anchors[0]['sha256'],'anchor_cumulative_J':replay.boundaries[c]['8192']['J_ACC_FD']}
                    row.update({key:S.mean([v[key] for v in values]) for key in values[0]});rows.append(row)
            # Original actual cache grows linearly in attention KV. Extrapolate
            # from measured original cache bytes, plus measured transient slack.
            last=memory[-1];baseline=memory[0]
            cache_slope=(last['clean_cache_bytes']-baseline['clean_cache_bytes'])/(last['tokens']-baseline['tokens'])
            projected_additional=max(0,int(cache_slope*(32768-last['tokens'])))
            transient=max(0,torch.cuda.max_memory_allocated()-last['allocated'])
            free,total=torch.cuda.mem_get_info()
            available=free+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
            feasible=projected_additional+transient<=available
            resource_gate={'status':'PASS' if feasible else 'FAIL','actual8192_geometry':True,
                           'additional_cache_bytes_to32K':projected_additional,'actual_transient_allowance':transient,
                           'available_device_bytes_including_reusable_pool':available,'cache_bytes_per_token':cache_slope,
                           'no_arbitrary_VRAM_percent_rule':True,'no_long_labels_read':True}
            N.save(outroot+'/LONG_RESOURCE_PREFLIGHT.json',resource_gate)
            N.save(outroot+'/short_result.json',{'status':'PASS','scores':rows,'per_token_observations':replay.observations,
                'anchors':anchors,'short_frozen_drive':drive_artifact,'memory':memory,'QDQ_calls_to8256':{c:8256 for c in S.CONDITIONS},
                'seconds':time.monotonic()-started,'GPU_hours':(time.monotonic()-started)/3600,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'source_commit':N.cmd(['git','rev-parse','HEAD'],N.REPO),'history_spec_sha256':N.sha(ROOT/'HISTORY_PREMEASUREMENT_SPEC.json')})
    finally:harness.close()
    print('TRACK_A_PREFIX_PASS',doc,flush=True)


def continuation(gpu,doc):
    torch,harness,replay,ids,started,d,ap=setup(gpu,doc)
    freeze=N.read(ROOT/'ALL_SHORT_SCORES_FREEZE.json')
    for rel,wanted in freeze['hashes'].items():N.check(N.sha(ROOT/rel)==wanted,'short score global seal before labels')
    outroot=f'track_A/doc_{doc:02d}'
    resource=N.read(ROOT/outroot/'LONG_RESOURCE_PREFLIGHT.json')
    N.check(resource['status']=='PASS','actual long-memory preflight')
    receipt=N.read(ROOT/outroot/'anchor_08256_receipt.json')
    path=ROOT/receipt['artifact']['path']
    N.check(N.sha(path)==receipt['artifact']['sha256'],'real checkpoint8256 identity')
    # This trusted own artifact includes complete dynamic-cache objects. It is
    # not an untrusted checkpoint and never imports downloaded model code.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    N.check(payload['consumed_tokens']==8256,'exact original resumed cursor')
    cache=replay.restore(payload)
    N.check(S.snapshot_hash(cache)==receipt['complete_cache_fingerprint'],'complete actual cache restored')
    wanted=N.read(ROOT/outroot/'RESUME_EXPECTED_3_TOKENS.json')['fingerprints']
    try:
        with torch.inference_mode():
            for i,p in enumerate(range(8256,8259)):
                output,cache=harness.step('FP',ids[p],cache,p)
                replay.step(p,metrics=False)
                N.check(replay.fingerprint(cache,output.logits)==wanted[i],'fresh-process complete-cache continuation byte parity')
            N.save(f'long/doc_{doc:02d}/RESUME_PARITY.json',{'status':'PASS','tokens':3,'main_codes_states_logits_complete_cache':'BYTE_EXACT',
                'anchor_sha256':receipt['artifact']['sha256'],'short_seal_sha256':N.sha(ROOT/'ALL_SHORT_SCORES_FREEZE.json')})
            del cache;gc.collect();torch.cuda.empty_cache()
            cache=replay.restore(payload)
            for p in range(8256,32768):
                output,cache=harness.step('FP',ids[p],cache,p)
                N.check(bool(torch.isfinite(output.logits).all()),'finite clean long acquisition logits')
                replay.step(p,metrics=True)
                if (p+1)%256==0:
                    print(json.dumps({'event':'MATCHED_LONG','doc':doc,'tokens':p+1,'allocated':torch.cuda.memory_allocated(),
                        'reserved':torch.cuda.memory_reserved()}),flush=True)
            rows=[]
            for c in S.CONDITIONS:
                j8=replay.boundaries[c]['8192']['J_ACC_FD'];j32=replay.boundaries[c]['32768']['J_ACC_FD']
                N.check(math.isfinite(j8) and j8>0 and math.isfinite(j32),'legal continuous finite long G denominator')
                future=replay.future[c].snapshot()
                N.check(future['token_count']==32768-8256,'strict nonoverlapping future interval')
                rows.append({'document_index':doc,'condition':c,'J8192':j8,'J32768':j32,'G':j32/j8,
                    'E_future':future['J_ACC_FD'],'future_tokens':future['token_count'],
                    'document_sha256':d['text_sha256'],'token_sha256':d['token_sha256'],'GPU_UUID':S.UUIDS[gpu]})
            N.save(f'long/doc_{doc:02d}/result.json',{'status':'PASS','scores':rows,'boundaries':replay.boundaries,
                'QDQ_calls':replay.qdq_calls,'future_boundaries':{c:replay.future[c].snapshot() for c in S.CONDITIONS},
                'seconds':time.monotonic()-started,'GPU_hours':(time.monotonic()-started)/3600,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'source_commit':N.cmd(['git','rev-parse','HEAD'],N.REPO),'short_seal_sha256':N.sha(ROOT/'ALL_SHORT_SCORES_FREEZE.json'),
                'actual_total_scientific_tokens':32768,'qualification_tokens_excluded':3,
                'complete_120_tensor_per_token_drive_hash_chain':replay.drive_chain})
    finally:harness.close()
    print('MATCHED_LONG_PASS',doc,flush=True)


def child(phase,gpu,**args):
    command=[sys.executable,str(Path(__file__)),phase,'--gpu',str(gpu)]
    for k,v in args.items():
        if isinstance(v,bool):
            if v:command.append('--'+k)
        else:command+=['--'+k,str(v)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    return subprocess.Popen(command,env=env)


def controller():
    N.check(not N.resources()['processes_csv'],'no GPU competition before matched history')
    N.check(child('prefix',0,doc=1,engineering=True).wait()==0,'matched clean-drive engineering qualification')
    workers=[child('prefix',i,doc=i+1) for i in range(4)]
    codes=[w.wait() for w in workers];N.check(codes==[0]*4,'TrackA prefix failure; no retry')
    rows=[]
    for doc in range(1,5):rows.extend(N.read(ROOT/f'track_A/doc_{doc:02d}/short_result.json')['scores'])
    S.csv_write('TRACK_A_8192_SHORT_SCORES.csv',rows)
    files=['TRACK_B_SHORT_SCORES.csv','ORIGINAL_C5_C6_BASELINES.csv','TRACK_A_8192_SHORT_SCORES.csv',
           'SCIENCE_PREMEASUREMENT_SPEC.json','HISTORY_PREMEASUREMENT_SPEC.json','NEW_NUMERICAL_CONTRACT.json']
    N.save('ALL_SHORT_SCORES_FREEZE.json',{'utc':N.now(),'status':'PASS','hashes':{rel:N.sha(ROOT/rel) for rel in files},
        'matched_long_G_read':False,'old_G_pairing':'NOT_AUTHORIZED','independent_documents':4})
    for doc in range(1,5):
        N.check(N.read(ROOT/f'track_A/doc_{doc:02d}/LONG_RESOURCE_PREFLIGHT.json')['status']=='PASS','long physical-memory gate; preserve all short scores')
    workers=[child('continuation',i,doc=i+1) for i in range(4)]
    codes=[w.wait() for w in workers];N.check(codes==[0]*4,'matched long failure; no retry')
    rows=[]
    for doc in range(1,5):rows.extend(N.read(ROOT/f'long/doc_{doc:02d}/result.json')['scores'])
    S.csv_write('MATCHED_LONG_G.csv',rows)
    S.csv_write('FUTURE_ONLY_ERROR.csv',[{k:r[k] for k in ['document_index','condition','E_future','future_tokens','document_sha256','token_sha256']} for r in rows])
    N.save('MATCHED_PANEL_COMPLETE.json',{'status':'PASS','documents':4,'conditions':4,'Track_A_rows':32,'long_rows':16,
        'old_G_used_for_pairing':False,'new_full_model_KL_executed':False,'generation':False,'training':False})
    print('MATCHED_PANEL_COMPLETE_NEXT_FINAL_ANALYSIS',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','prefix','continuation','controller'])
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--doc',type=int,default=1);p.add_argument('--engineering',action='store_true');a=p.parse_args()
    try:
        if a.phase=='prepare':prepare()
        elif a.phase=='prefix':prefix(a.gpu,a.doc,a.engineering)
        elif a.phase=='continuation':continuation(a.gpu,a.doc)
        else:controller()
    except Exception as exc:
        rel=f'execution/HISTORY_{a.phase}_gpu{a.gpu}_doc{a.doc}_STOP.json'
        if not (ROOT/rel).exists():N.save(rel,{'utc':N.now(),'status':'FAIL_AND_STOP','exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'retry':False})
        raise


if __name__=='__main__':main()
