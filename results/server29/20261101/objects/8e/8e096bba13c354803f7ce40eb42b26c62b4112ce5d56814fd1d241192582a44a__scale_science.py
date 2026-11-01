"""V3 inference-only matched science, gated by the sealed numerical contract.

Formal rotation and QDQ are imported unchanged. Original training teacher
collector and loss helpers are called, never replaced by approximate losses.
All scientific design choices are frozen by prepare before any score exists.
"""
import argparse
import ast
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
from types import SimpleNamespace

import numerical_recovery as N

ROOT = N.ROOT
REPO = N.REPO
CORE = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
CORE_SHA = '48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e'
LAYERS = (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)
CONDITIONS = ('Native','H','C6R','C7')
UUIDS = ['GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75']


def mean(xs):
    N.check(bool(xs) and all(math.isfinite(x) for x in xs), 'finite nonempty metric reduction')
    return math.fsum(xs)/len(xs)


def setup(gpu):
    import torch
    N.check(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu), 'frozen physical GPU mapping')
    N.check(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS workspace')
    N.check(torch.__version__ == '2.5.1+cu121' and torch.version.cuda == '12.1', 'qualified runtime identity')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    contract=N.read(ROOT/'NEW_NUMERICAL_CONTRACT.json')
    N.check(contract['status']=='PASS' and contract['selected_route']=='compensated', 'numerical closure prerequisite')
    N.check(N.sha(ROOT/'scripts/numerical_recovery.py') == contract['source_code_sha256'], 'qualified numerical source')
    policy=N.read(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json')
    N.check(N.sha(__file__)==policy['science_code_sha256'], 'science code frozen before measurement')
    for path,wanted in policy['source_sha256'].items():
        N.check(N.sha(path)==wanted, 'scientific source identity '+path)
    N.check(N.resources()['GPU_csv'].splitlines()[gpu].split(',')[1].strip()==UUIDS[gpu], 'GPU UUID identity')
    F=N.module(N.FORMAL, 'v3_science_formal')
    core=N.module(CORE, 'v3_science_original_collector')
    F.setup_seed(N.SEED)
    return torch,F,core,policy


class QuantizerObserver:
    def __init__(self, torch):
        self.t=torch
        text=N.FORMAL.read_text()
        fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='quantize_c128')
        body=next(n for n in fn.body if isinstance(n,ast.For)).body
        assignments=[n for n in body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['source','scale','codes','dequant']]
        N.check([n.targets[0].id for n in assignments]==['source','scale','codes','dequant'], 'exact formal quantizer AST')
        self.all_code=compile(ast.Module(body=assignments,type_ignores=[]),str(N.FORMAL),'exec')
        self.grid_code=compile(ast.Module(body=assignments[2:],type_ignores=[]),str(N.FORMAL),'exec')

    def original(self,u):
        ns={'torch':self.t,'EPS':1e-12,'state':u}
        exec(self.all_code,ns)
        return ns['scale'],ns['codes'],ns['dequant']

    def metrics(self,u,clean):
        t=self.t
        before=N.tsha(u)
        ss,qs,ds=self.original(u.detach().clone())
        sc,_,_=self.original(clean.detach().clone())
        N.check(list(ss.shape)==[1,32,1,128] and ss.dtype==t.float32, 'canonical scale shape/dtype')
        cap=t.minimum(ss,sc)
        ns={'torch':t,'source':u.detach().clone().float(),'scale':cap}
        exec(self.grid_code,ns)
        dc=ns['dequant'];qc=ns['codes']
        r=t.log(ss.double()/sc.double())
        numerator=(ds.double()-dc.double()).square().sum()
        denom=u.double().square().sum()+1e-12
        vals={'S1':float(t.relu(r).square().mean()),'S2':float(numerator/denom),
              'D1':float(t.relu(-r).square().mean()),'D2':float(r.square().mean()),
              'student_clip_fraction':float((u.abs()>127*ss).double().mean()),
              'cap_clip_fraction':float((u.abs()>127*cap).double().mean()),
              'student_saturation_fraction':float((qs.abs()==127).double().mean()),
              'cap_saturation_fraction':float((qc.abs()==127).double().mean())}
        N.check(all(math.isfinite(x) for x in vals.values()), 'finite observer metrics')
        N.check(before==N.tsha(u), 'observer must not mutate main pre-QDQ state')
        return vals


class Harness:
    """Unchanged full-model paths with formal rotation, external every-token QDQ."""
    def __init__(self,model,F,core,torch):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        self.model=model;self.F=F;self.core=core;self.t=torch;self.qmod=qmod
        self.quant=QuantizerObserver(torch)
        self.h=F.hadamard(128,device='cuda:0')
        self.banks={};self.patches={'H':F.QwenKeyHadamardPatch(None)}
        audit=N.read(ROOT/'EXISTING_ASSET_REUSE_AUDIT.json')
        self.rotation_manifest=audit['rotations']
        for c in ['C6R','C7']:
            row=next(a for a in self.rotation_manifest['anchors'] if a['condition']==c)
            N.check(N.sha(row['export_path'])==row['export_file_sha256'], 'frozen direct rotation export')
            p=torch.load(row['export_path'],map_location='cpu',weights_only=True)
            b=F.FrozenRotationBank(p['layer_ids'],p['matrices']).to('cuda:0')
            for parameter in b.parameters(): parameter.requires_grad_(False)
            self.banks[c]=b;self.patches[c]=F.QwenFrozenRotationPatch(None,b,{'sha256':row['export_file_sha256']})
        self.original_chunk=qmod.torch_chunk_gated_delta_rule
        self.original_recurrent=qmod.torch_recurrent_gated_delta_rule
        self.chunk=N.chunk_binding(qmod,'compensated')
        self.condition='FP';self.active=None;self.position=0
        self.observe=False;self.target=None;self.clean={};self.updated={};self.records={}
        self.values={};self.baselines={};self.controls={};self.previous={}
        self.handles=[];self.qual=False;self.codes={}
        layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
        for l in LAYERS:
            block=getattr(layers[l],'linear_attn',None) or getattr(layers[l],'self_attn',None)
            self.handles.append(block.register_forward_pre_hook(lambda m,a,l=l:setattr(self,'active',l)))
            self.handles.append(block.out_proj.register_forward_hook(self.out_hook(l)))
        qmod.torch_chunk_gated_delta_rule=self.wrap(self.chunk)
        qmod.torch_recurrent_gated_delta_rule=self.wrap(self.original_recurrent)

    def inverse(self,c,l,x):
        if c=='Native':return x.float()
        y=self.t.einsum('ij,...jv->...iv',self.h,x.float())
        if c!='H':y=self.t.einsum('ij,...jv->...iv',self.banks[c].layer(l).matrix().T,y)
        return y

    def physical(self,c,l,x):
        if c=='Native':return x.float()
        delta=self.banks[c].layer(l).matrix() if c!='H' else self.t.eye(128,device=x.device)
        return self.core.recover_state(x,delta,self.h)

    def wrap(self,fn):
        def invoke(query,key,value,*args,**kw):
            l=self.active;c=self.condition
            N.check(l in LAYERS, 'mapped GDN layer')
            initial=kw.get('initial_state')
            if c!='FP' and self.qual and self.position>0:
                N.check(self.t.equal(initial,self.previous[c][l]), 'next-token consumption of previous quantized writeback')
            if c!='FP' and self.target is not None:
                N.check(initial is not None, 'baseline actual previous state')
                self.baselines[l]={'B0':float(self.core.relative_state_mse(self.physical(c,l,initial),self.target['prev'][l].to('cuda:0')))}
            if c=='FP':
                result,state=fn(query,key,value,*args,**kw)
                self.records[l]={'query':query.detach(),'key':key.detach(),'value':value.detach(),'g':kw['g'].detach(),'beta':kw['beta'].detach()}
                self.clean[l]=state.detach().clone()
            else:
                norm=kw.pop('use_qk_l2norm_in_kernel',False)
                if c=='Native':
                    result,state=fn(query,key,value,*args,use_qk_l2norm_in_kernel=norm,**kw)
                else:
                    patch=self.patches[c];patch.current_layer=l
                    result,state=fn(patch._transform(query,norm),patch._transform(key,norm),value,*args,use_qk_l2norm_in_kernel=False,**kw)
                    result=result.to(value.dtype)
                N.check(state.dtype==self.t.float32 and list(state.shape)==[1,32,128,128], 'FP32 formal persistent state shape')
                if self.observe:
                    self.values[l]=self.quant.metrics(state,self.inverse(c,l,self.clean[l]))
                if self.qual:
                    scale,codes,dq=self.quant.original(state.detach().clone())
                    self.codes[l]=N.tsha(codes);self.updated[l]=dq
            return result,state
        return invoke

    def out_hook(self,l):
        def hook(m,args,out):
            if self.condition!='FP' and self.target is not None:
                self.baselines[l]['B1']=float(self.core.CAYLEY.relative_mse(out.float(),self.target['out'][l].to('cuda:0')))
        return hook

    def step(self,c,token,cache,position,observe=False,target=None,qual=False):
        self.condition=c;self.position=position;self.observe=observe;self.target=target;self.qual=qual
        self.values={};self.baselines={};self.updated={};self.codes={}
        if c=='FP':self.clean={};self.records={}
        out=self.model(input_ids=self.t.tensor([[token]],device='cuda:0'),past_key_values=cache,use_cache=True)
        cache=out.past_key_values
        if c!='FP':
            a={'calls':0,'layers':set(),'scale_shapes':{},'code_min':0,'code_max':0}
            self.F.quantize_c128(cache,a)
            N.check(a['calls']==1 and a['layers']==set(LAYERS), '24 layers every-token formal QDQ')
            if qual:
                for l in LAYERS:
                    N.check(self.t.equal(self.F.get_state(cache,l),self.updated[l]), 'exact canonical formal cache writeback')
                self.previous[c]={l:self.F.get_state(cache,l).detach().clone() for l in LAYERS}
        return out,cache

    def close(self):
        self.qmod.torch_chunk_gated_delta_rule=self.original_chunk
        self.qmod.torch_recurrent_gated_delta_rule=self.original_recurrent
        for h in self.handles:h.remove()


def load(gpu):
    torch,F,core,policy=setup(gpu)
    started=time.monotonic()
    model,tokenizer=F.load_model_and_tokenizer()
    core.freeze_model(model)
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    N.check(Path(qmod.__file__).resolve()==N.ARCH.resolve(), 'qualified architecture actual import')
    qmod.torch_chunk_gated_delta_rule=N.chunk_binding(qmod,'compensated')
    return torch,F,core,policy,model,tokenizer,started


def tree_tensors(value,prefix='',seen=None):
    import torch
    if seen is None:seen=set()
    if id(value) in seen:return
    if torch.is_tensor(value):
        yield prefix,value;return
    seen.add(id(value))
    if isinstance(value,dict):
        for k in sorted(value,key=str):yield from tree_tensors(value[k],prefix+'/'+str(k),seen)
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):yield from tree_tensors(v,prefix+'/'+str(i),seen)
    elif hasattr(value,'__dict__'):
        for k,v in sorted(vars(value).items()):yield from tree_tensors(v,prefix+'/'+str(k),seen)


def snapshot_hash(cache):
    return {name:N.tsha(t) for name,t in tree_tensors(cache)}


def smoke(gpu,index):
    torch,F,core,policy,model,tokenizer,started=load(gpu)
    ids=N.read(policy['documents'][0]['token_path'])[:64]
    harness=Harness(model,F,core,torch)
    outcomes={};counts={}
    try:
        with torch.inference_mode():
            for enabled in [False,True]:
                caches={c:None for c in ['FP','H','C6R']};digests={c:hashlib.sha256() for c in ['H','C6R']}
                harness.previous={};n=0
                for p,token in enumerate(ids):
                    out,caches['FP']=harness.step('FP',token,caches['FP'],p)
                    for c in ['H','C6R']:
                        out,caches[c]=harness.step(c,token,caches[c],p,observe=enabled,qual=True)
                        receipt={'token':p,'logits':N.tsha(out.logits),'codes':harness.codes,'complete_cache':snapshot_hash(caches[c])}
                        digests[c].update(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
                        if enabled:N.check(len(harness.values)==24, 'observer every-layer coverage');n+=24
                outcomes[str(enabled)]={c:h.hexdigest() for c,h in digests.items()}
                counts[str(enabled)]=n
                del caches,out;gc.collect();torch.cuda.empty_cache()
    finally:harness.close()
    N.check(outcomes['False']==outcomes['True'], 'observer OFF/ON exact code/cache/logit/next consumption parity')
    N.save(f'science_qualification/observer_smoke_{index}.json',{'status':'PASS','index':index,'hashes':outcomes,
        'observer_layer_token_count':counts,'length':64,'conditions':['H','C6R'],'all_complete_cache_tensors_hashed':True,
        'main_codes_cache_writeback_logits_next_consumption':'BYTE_EXACT',
        'seconds':time.monotonic()-started,'GPU_hours':(time.monotonic()-started)/3600,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'GPU_UUID':UUIDS[gpu],'source_commit':N.cmd(['git','rev-parse','HEAD'],REPO),
        'spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json')})
    print('OBSERVER_SMOKE_PASS',index,flush=True)


def teacher_targets(model,F,core,ids,positions):
    # Invoke the exact original training collector, including original
    # multi-token chunk-to-position calls. This is the B0/B1 golden teacher.
    dummy=SimpleNamespace()
    with core.RecurrentExperimentPatch(model,dummy,F.hadamard(128,device='cuda:0')) as patch:
        targets=core.collect_teacher_targets(model,patch,ids,positions)
    N.check(set(targets)==set(positions), 'original eight captures preserved')
    return targets


def track_b(gpu,doc,engineering=False):
    torch,F,core,policy,model,tokenizer,started=load(gpu)
    d=policy['documents'][doc-1];ids=N.read(d['token_path'])[:1024]
    N.check(N.sha(d['token_path'])==d['token_sha256'], 'frozen document token context')
    N.check(tokenizer(Path(d['text_path']).read_text(),add_special_tokens=False).input_ids[:1024]==ids, 'original text/tokenizer first1024 context pairing')
    positions=policy['track_B']['capture_positions'][str(doc)]
    N.check(core.sampled_positions({'raw_text_sha256':d['text_sha256'],'document_id':d['canonical_doc_id']},1024)==positions, 'original sample-position helper')
    targets=teacher_targets(model,F,core,ids,positions)
    target_artifact=N.pt_save(f"{'science_qualification' if engineering else 'track_B'}/doc_{doc:02d}/original_teacher_targets.pt",targets)
    harness=Harness(model,F,core,torch)
    conditions=['H','C6R'] if engineering else list(CONDITIONS)
    caches={c:None for c in ['FP']+conditions}
    obs={c:{} for c in conditions};base={c:[] for c in conditions};memory=[];qdq={c:0 for c in conditions}
    try:
        with torch.inference_mode():
            for p,token in enumerate(ids):
                out,caches['FP']=harness.step('FP',token,caches['FP'],p)
                observe=any(max(0,k-63)<=p<=k for k in positions)
                for c in conditions:
                    out,caches[c]=harness.step(c,token,caches[c],p,observe,targets[p] if p in targets else None)
                    qdq[c]+=1
                    if observe:
                        N.check(set(harness.values)==set(LAYERS), 'paired clean/student scale coverage')
                        obs[c][p]={key:mean([harness.values[l][key] for l in LAYERS]) for key in harness.values[LAYERS[0]]}
                    if p in targets:
                        N.check(set(harness.baselines)==set(LAYERS), 'original actual out_proj baseline coverage')
                        b0=mean([harness.baselines[l]['B0'] for l in LAYERS]);b1=mean([harness.baselines[l]['B1'] for l in LAYERS])
                        base[c].append({'position':p,'B0':b0,'B1':b1,'B2':b1+.1*b0})
                if (p+1)%64==0:
                    # Original H64 graph boundary; never reset numerical state
                    # or change the separate clean acquisition cache.
                    for c in conditions:core.detach_cache(caches[c])
                if (p+1)%128==0:
                    memory.append({'tokens':p+1,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                        'complete_cache_bytes':sum(t.numel()*t.element_size() for cache in caches.values() for _,t in tree_tensors(cache))})
                    print(json.dumps({'event':'TRACK_B','doc':doc,'engineering':engineering,'tokens':p+1}),flush=True)
    finally:harness.close()
    rows=[]
    for c in conditions:
        N.check(qdq[c]==1024 and len(base[c])==8, 'exact QDQ/capture counts')
        for horizon in [32,64]:
            values=[obs[c][p] for k in positions for p in range(k-horizon+1,k+1)]
            row={'document_index':doc,'condition':c,'H':horizon,'track':'B','token_context':'first1024',
                 'window_token_exposures':len(values),'window_geometry':'last_H_tokens_including_each_original_capture_position',
                 'document_sha256':d['text_sha256'],'token_sha256':d['token_sha256'],'GPU_UUID':UUIDS[gpu],
                 'teacher_target_sha256':target_artifact['sha256']}
            row.update({key:mean([v[key] for v in values]) for key in values[0]})
            row.update({key:mean([v[key] for v in base[c]]) for key in ['B0','B1','B2']})
            rows.append(row)
    outroot=f"{'science_qualification' if engineering else 'track_B'}/doc_{doc:02d}"
    N.save(outroot+'/result.json',{'status':'PASS','engineering_only':engineering,'scores':rows,'per_capture_baselines':base,
        'per_token_short_observations':obs,'formal_QDQ_calls':qdq,'formal_QDQ_layers':list(LAYERS),
        'original_teacher_target_artifact':target_artifact,'baseline_teacher':'UNCHANGED_CHUNK_TO_POSITION_COLLECTOR',
        'scale_teacher':'SEPARATE_MATCHED_FULL_FP_TOKEN_DRIVE; not substituted for baseline teacher',
        'seconds':time.monotonic()-started,'GPU_hours':(time.monotonic()-started)/3600,'cache_memory':memory,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'source_commit':N.cmd(['git','rev-parse','HEAD'],REPO),'spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json'),
        'strict':True,'TF32':False,'backward_calls':0,'generation_calls':0,'training':False})
    print('TRACK_B_PASS',doc,'engineering',engineering,flush=True)


def prepare():
    N.check(N.read(ROOT/'NEW_NUMERICAL_CONTRACT.json')['status']=='PASS','numerical contract')
    audit=N.read(ROOT/'EXISTING_ASSET_REUSE_AUDIT.json')
    core=N.module(CORE,'v3_prepare_original_core')
    docs=audit['documents']
    positions={str(i):core.sampled_positions({'raw_text_sha256':d['text_sha256'],'document_id':d['canonical_doc_id']},1024) for i,d in enumerate(docs,1)}
    N.save('SCIENCE_PREMEASUREMENT_SPEC.json',{
        'utc':N.now(),'science_code_sha256':N.sha(__file__),'numerical_contract_sha256':N.sha(ROOT/'NEW_NUMERICAL_CONTRACT.json'),
        'documents':docs,'conditions':list(CONDITIONS),'main_ranking_conditions':['H','C6R','C7'],
        'source_sha256':{str(CORE):CORE_SHA,str(N.FORMAL):N.FORMAL_SHA,str(N.ARCH):N.ARCH_SHA,
                          str(core.CAYLEY_SOURCE):N.sha(core.CAYLEY_SOURCE),str(core.BASE_RUNNER):N.sha(core.BASE_RUNNER)},
        'GPU_document_mapping':{str(i+1):{'physical_GPU':i,'UUID':UUIDS[i]} for i in range(4)},
        'observer':{'S1':'mean(relu(log(ss/sc))^2)','D1':'mean(relu(-log(ss/sc))^2)','D2':'mean(log(ss/sc)^2)',
                    'S2':'same student pre-QDQ u, canonical AST grids at ss and min(ss,sc); relative Frobenius squared error with denominator +1e-12; no writeback',
                    'scales':'formal FP32 [1,32,1,128]; matched clean FP state inverse-rotated by separate products',
                    'reduction_dtype':'FP64 observation only; no FP64 persistent state',
                    'weights':'equal scale elements then equal24layers then equal tokens; repeated overlapping capture-window tokens weighted per capture exposure',
                    'finite_rule':'all finite; fail closed otherwise','parity':'OFF/ON codes/all cache tensors/logits/next consumption exact; 2 fresh processes'},
        'track_B':{'tokens':1024,'capture_positions':positions,'horizons':[32,64],
                   'S_windows':'[p-H+1,p+1) 0-based,8 unchanged original captured positions; no student state reset',
                   'detach':'original every64 student detach; no numerical reset and no backward; H32/H64 observational windows on one continuous inference trajectory',
                   'B0_B1_B2':'exact original helpers and original8 chunk-to-position targets; same baseline for both observational horizons',
                   'FP_scale_pairing':'separate same-runtime first1024 per-token clean FP trajectory; explicit, not a replacement of original collector',
                   'student_quantizer':'unchanged formal inference quantize_c128; no training STE/backward qualification claimed'},
        'track_A':{'anchor':8192,'H32':[8192,8224],'H64':[8192,8256],
                   'semantics':'new matched common-clean frozen-drive continuous persistent QDQ; native negative control',
                   'baseline_B0':'original per-head previous physical state helper vs same-drive clean previous state',
                   'baseline_B1_B2':'NOT_COMPARABLE unless actual matched out_proj forward is retained; never approximate'},
        'long':{'panel':'NEW_SELF_CONSISTENT_MATCHED_PANEL','old_G':'DESCRIPTIVE_ONLY',
                'J':'unchanged parent FP64 per-token/per-layer physical squared Frobenius error/(FP norm squared+1e-12); tokens1-based1024..T inclusive, equal24layer',
                'G':'J32768/J8192','E_future':'same J error 0-based[8256,32768); no overlap with short window',
                'read_labels_only_after':'all TrackB andTrackA short scores globally SHA frozen',
                'checkpoint':'complete actual clean model cache and persistent24-layer replay states; cursor and accumulators; exact fresh-process resume parity'},
        'analysis':{'independent_unit':'document; n4 Discovery only','per_doc':['Spearman','pairwise concordance','discordant pairs'],
                    'no_confirmatory_pvalue':True,'no_gradient_training':True,'no_AIME_free_generation':True},
        'STOP':'any identity/numerical/observer/pairing/QDQ/determinism/OOM/growth/storage/resume failure; preserve earliest evidence; no automatic retry',
        'GPUh_cap':None,'VRAM_percentage_cap':None})
    print('SCIENCE_SPEC_FROZEN',flush=True)


def csv_write(rel,rows):
    import csv
    N.check(bool(rows),'no empty unexecuted CSV')
    path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_child(phase,gpu,**options):
    args=[sys.executable,str(Path(__file__)),phase,'--gpu',str(gpu)]
    for key,value in options.items():
        if isinstance(value,bool):
            if value:args.append('--'+key)
        else:args += ['--'+key,str(value)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    return subprocess.Popen(args,env=env)


def controller():
    N.check(not N.resources()['processes_csv'], 'no GPU competition')
    for index in [1,2]:N.check(run_child('smoke',0,index=index).wait()==0,'observer prerequisite failed')
    one=N.read(ROOT/'science_qualification/observer_smoke_1.json');two=N.read(ROOT/'science_qualification/observer_smoke_2.json')
    N.check(one['hashes']==two['hashes'],'observer fresh-process repeatability')
    N.save('OBSERVER_PARITY_REPORT.json',{'status':'PASS','fresh_processes':2,'exact_parity':True,
        'receipts':[N.sha(ROOT/f'science_qualification/observer_smoke_{i}.json') for i in [1,2]],
        'main_codes_recurrent_state_complete_cache_logits_next_consumption':'PASS','S2_writeback_calls':0})
    N.check(run_child('track_b',0,doc=1,engineering=True).wait()==0,'TrackB Doc1 H/C6R engineering qualification')
    eng=N.read(ROOT/'science_qualification/doc_01/result.json')
    N.save('TRACK_B_ENGINEERING_GATE.json',{'status':'PASS','full_geometry':1024,'conditions':['H','C6R'],
        'baseline':'actual original matched out_proj and per-head state helper','scores_excluded_from_scientific_ranking':True,
        'peak_allocated_bytes':eng['peak_allocated_bytes'],'seconds':eng['seconds'],'receipt_sha256':N.sha(ROOT/'science_qualification/doc_01/result.json')})
    N.check(not N.resources()['processes_csv'],'GPU release before four-document expansion')
    workers=[run_child('track_b',i,doc=i+1) for i in range(4)]
    codes=[w.wait() for w in workers]
    N.check(codes==[0]*4,'formal TrackB worker failed; no retry')
    rows=[]
    for doc in range(1,5):rows.extend(N.read(ROOT/f'track_B/doc_{doc:02d}/result.json')['scores'])
    N.check(len(rows)==32,'TrackB exact four-document four-condition two-horizon coverage')
    csv_write('TRACK_B_SHORT_SCORES.csv',rows)
    csv_write('ORIGINAL_C5_C6_BASELINES.csv',[{key:r[key] for key in ['document_index','condition','H','B0','B1','B2','teacher_target_sha256','document_sha256','token_sha256']} for r in rows])
    N.save('TRACK_B_SHORT_SCORE_FREEZE.json',{'status':'PASS','utc':N.now(),'rows':32,
        'sha256':N.sha(ROOT/'TRACK_B_SHORT_SCORES.csv'),'baseline_sha256':N.sha(ROOT/'ORIGINAL_C5_C6_BASELINES.csv'),
        'spec_sha256':N.sha(ROOT/'SCIENCE_PREMEASUREMENT_SPEC.json'),'long_G_read':False})
    print('TRACK_B_COMPLETE_NEXT_TRACK_A',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','smoke','track_b','controller'])
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--index',type=int,default=1);p.add_argument('--doc',type=int,default=1);p.add_argument('--engineering',action='store_true')
    a=p.parse_args()
    try:
        if a.phase=='prepare':prepare()
        elif a.phase=='smoke':smoke(a.gpu,a.index)
        elif a.phase=='track_b':track_b(a.gpu,a.doc,a.engineering)
        else:controller()
    except Exception as exc:
        rel=f'execution/SCIENCE_{a.phase}_gpu{a.gpu}_doc{a.doc}_index{a.index}_STOP.json'
        if not (ROOT/rel).exists():N.save(rel,{'utc':N.now(),'status':'FAIL_AND_STOP','exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'retry':False})
        raise


if __name__=='__main__':main()
