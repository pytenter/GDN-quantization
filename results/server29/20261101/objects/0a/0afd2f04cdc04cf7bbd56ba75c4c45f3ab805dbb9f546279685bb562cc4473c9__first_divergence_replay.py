"""Frozen 25-panel shared-prefix replay. No generation, scoring or training."""
import argparse, ast, collections, contextlib, csv, gc, hashlib, importlib.util, json
import math, os, shutil, statistics, subprocess, sys, time, traceback
from pathlib import Path

TASK='QWEN_C6R_C7_FIRST_DIVERGENCE_PREFIX_REPLAY_V1'
REPO=Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1')
ROOT=REPO/'experiments'/TASK
PARENT=REPO/'experiments/QWEN_C6R_C7_CANONICAL60_POSTENDPOINT_DIAGNOSIS_V1'
ENDPOINT=REPO/'experiments/QWEN_GDN_C6R_C7_CANONICAL60_V1'
METRIC_SOURCE=Path('/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2/experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/scripts/sparse_probe.py')
CAYLEY=Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py')
SHAS={'C6R':'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6','C7':'57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8'}

def require(v,m):
    if not v: raise RuntimeError(m)
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ids_hash(ids): return hashlib.sha256(b''.join(int(i).to_bytes(8,'little') for i in ids)).hexdigest()
def write(rel,text,freeze=False):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.exists(),'refuse overwrite '+str(p))
    temp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with temp.open('x',encoding='utf-8',newline='') as f: f.write(text);f.flush();os.fsync(f.fileno())
    os.rename(temp,p)
    if freeze:p.chmod(0o444)
def save(rel,obj,freeze=False): write(rel,json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n',freeze)
def table(rel,rows,fields=None,freeze=False):
    import io
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows);write(rel,s.getvalue(),freeze)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def source_function(path,name):
    text=Path(path).read_text();tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.get_source_segment(text,node)
def frozen_metrics(torch):
    # Execute only the original numerical functions, not old experiment runners.
    ns={'torch':torch,'EPS':1e-12}
    for path,name in [(METRIC_SOURCE,'state_metric'),(METRIC_SOURCE,'model_metrics'),(CAYLEY,'relative_mse')]:
        exec(compile(source_function(path,name),str(path), 'exec'),ns)
    return ns
def manifest_rows(root,path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        h,n=line.split(None,1);p=Path(n.strip());p=p if p.is_absolute() else root/p
        rows.append(dict(path=str(p),sha256=h))
    return rows
def verify(entries):
    for r in entries: require(sha(r['path'])==r['sha256'],'SHA drift '+r['path'])
def git_at(path): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
def prepare():
    require(not ROOT.exists(),'experiment already exists: do not duplicate/retry')
    inv=read(PARENT/'provenance/input_artifact_inventory.json')
    require(inv['endpoint_root']==str(ENDPOINT),'actual source endpoint differs')
    protected=manifest_rows(PARENT,PARENT/'SHA256SUMS')+manifest_rows(ENDPOINT,ENDPOINT/'manifests/final_artifact_sha256.txt')
    protected += [dict(path=str(p),sha256=sha(p)) for p in [PARENT/'SHA256SUMS',ENDPOINT/'manifests/final_artifact_sha256.txt',ENDPOINT/'scripts/canonical60_common.py']]
    protected += [dict(path=p,sha256=h) for p,h in read(PARENT/'provenance/read_only_revalidation.json')['consumed_source_hashes'].items()]
    verify(protected)
    panel_path=PARENT/'proposals/discordant_replay_panel.json'; panel=read(panel_path)
    require(panel['samples']==25 and len(panel['rows'])==25,'panel count')
    require(collections.Counter(r['pair_category'] for r in panel['rows'])==dict(A=3,B=10,C=6,D=6),'panel taxonomy')
    taxonomy=read(PARENT/'analysis/pair_taxonomy.json')
    for c in 'CD':require(sorted(r['sample_id'] for r in panel['rows'] if r['pair_category']==c)==sorted(taxonomy['sample_ids'][c])[:6],'control reselection')
    for c in 'AB':require(sorted(r['sample_id'] for r in panel['rows'] if r['pair_category']==c)==sorted(taxonomy['sample_ids'][c]),'discordant omission')
    runtime=read(ENDPOINT/'provenance/canonical60_runtime_identity.json')
    verify(runtime['model_identity']['files'])
    protected+=runtime['model_identity']['files']
    for arm,r in runtime['rotation_files'].items():
        require(sha(r['path'])==SHAS[arm]==r['sha256'] and sha(r['manifest_path'])==r['manifest_sha256'],'rotation identity')
        protected += [dict(path=r['path'],sha256=r['sha256']),dict(path=r['manifest_path'],sha256=r['manifest_sha256'])]
    formal=runtime['generation_source'];require(sha(formal['path'])==formal['sha256'],'formal source')
    jobs={(r['problem_id'],r['seed']):r for r in read(ENDPOINT/'manifests/canonical60_sample_manifest.json')['samples']}
    raws={(r['condition'],r['sample_id']):r for r in inv['raw_outputs']}
    records=[];sequences=[]
    for rank,item in enumerate(panel['rows']):
        sid=item['sample_id'];problem,seed=sid.split(':seed');job=jobs[problem,int(seed)]
        rr={arm:raws[arm,sid] for arm in SHAS};x=read(rr['C6R']['path']);y=read(rr['C7']['path'])
        require(x['input_ids_hash']==y['input_ids_hash']==job['input_ids_hash'] and x['prompt_sha256']==y['prompt_sha256']==job['prompt_sha256'],'shared prompt identity')
        a=x['generated_token_ids'];b=y['generated_token_ids'];t=next((j for j,(u,v) in enumerate(zip(a,b)) if u!=v),None)
        status='TOKEN_DIVERGENCE' if t is not None else 'NONE' if a==b else 'NO_COMPARABLE_TOKEN_DIVERGENCE_SEQUENCE_LENGTH_DIFFERS'
        require(t is None or a[:t]==b[:t],'not first divergence')
        prefix=[] if t is None else a[:t]
        records.append(dict(panel_rank=rank,gpu=rank%4,GPU_uuid=runtime['GPU_UUIDs'][rank%4],sample_id=sid,problem_id=problem,seed=int(seed),pair_category=item['pair_category'],first_divergence=status,t_star=t,C6R_token_at_t_star=None if t is None else a[t],C7_token_at_t_star=None if t is None else b[t],shared_prefix_length=None if t is None else t,C6R_total_length=len(a),C7_total_length=len(b),shared_prefix_token_ids=prefix,shared_prefix_sha256=None if t is None else ids_hash(prefix),window_positions=[] if t is None else list(range(max(0,t-64),t)),C6R_raw_path=rr['C6R']['path'],C7_raw_path=rr['C7']['path'],C6R_raw_sha256=rr['C6R']['sha256'],C7_raw_sha256=rr['C7']['sha256'],problem=job['problem'],input_ids_hash=job['input_ids_hash'],prompt_sha256=job['prompt_sha256']))
        sequences.append(dict(sample_id=sid,**{arm:dict(path=rr[arm]['path'],sha256=rr[arm]['sha256'],token_count=len(z['generated_token_ids']),token_ids_sha256=ids_hash(z['generated_token_ids'])) for arm,z in [('C6R',x),('C7',y)]}))
    layers=read(ENDPOINT/'configs/canonical60_protocol.json')['GDN_layers'];require(len(layers)==24,'layer count')
    regions={name:layers[i*8:(i+1)*8] for i,name in enumerate(['early','middle','late'])}
    metrics=[]
    for path,name in [(METRIC_SOURCE,'state_metric'),(METRIC_SOURCE,'model_metrics'),(METRIC_SOURCE,'physical_state'),(METRIC_SOURCE,'OutputCapture'),(CAYLEY,'relative_mse')]:
        metrics.append(dict(path=str(path),function=name,file_sha256=sha(path),function_sha256=hashlib.sha256(source_function(path,name).encode()).hexdigest(),source_commit=git_at(path.parent)))
        protected.append(dict(path=str(path),sha256=sha(path)))
    protected=list({r['path']:r for r in protected}.values())
    ROOT.mkdir(parents=True)
    for name in ['first_divergence_replay.py','analyze_first_divergence.py','run_first_divergence.sh']:
        write('scripts/'+name,Path(__file__).with_name(name).read_text(encoding='utf-8'),True)
    save('protocol/panel.json',dict(parent_panel_path=str(panel_path),parent_panel_sha256=sha(panel_path),panel_count=25,records=records),True)
    save('protocol/original_sequence_manifest.json',dict(status='PASS',records=sequences),True)
    cols=['sample_id','problem_id','seed','pair_category','panel_rank','gpu','GPU_uuid','first_divergence','t_star','C6R_token_at_t_star','C7_token_at_t_star','shared_prefix_length','C6R_total_length','C7_total_length']
    table('analysis/first_divergence_manifest.csv',[{k:r[k] for k in cols} for r in records],freeze=True)
    protocol=dict(EXPERIMENT=TASK,EVIDENCE_TYPE='PROSPECTIVE_POSTENDPOINT_MECHANISM_DIAGNOSIS',conditions=['FP','H','C6R','C7'],condition_order='FP_THEN_H_THEN_C6R_THEN_C7_WITH_FRESH_CACHE',panel_assignment='parent panel row ordinal (0-based) % 4; every condition on that physical UUID',W=64,indexing='distribution j predicts generated token[j]; prefill predicts j=0; consuming shared token[j-1] predicts j; t_star requires exactly t_star decode tokens; window [max(0,t_star-64),t_star)',prefix='original common generated tokens [0:t_star]; identical rendered canonical prompt all arms',prefill_quantized=False,qdq='original formal.quantize_c128 after each decode; output logits/functional precede current QDQ, captured state is post-QDQ for next decode',distributions='raw model next-token logits softmax; NO presence penalty/temperature/topk/topp because these are model mechanism metrics, not resampling',EOS='primary tokenizer.eos_token_id probability/logit; rank=1+number strictly greater; secondary canonical stop_ids aggregate probability',t_star_topk=32,topk_ties='stable descending logit, ascending token ID',state_metric='existing sparse_probe.state_metric relative_l2; post-QDQ persistent physical-state drift vs FP at j=t_star',state_basis='q/k transform H then Delta unchanged; inverse physical state H @ Delta @ native state, orientation reused from sparse_probe.physical_state for C5/C6',functional_metric='existing sparse_probe.state_metric relative_l2 on GDN linear_attn.out_proj last-position output; same hook point as sparse_probe.OutputCapture',final_hidden='model-level final normalized lm_head input, existing CAYLEY.relative_mse; no fake layerwise hidden',metric_sources=metrics,regions=regions,layer_region_reduction='within sample median paired C7-C6R relative_l2 across 8 layers; then category sample distribution',prefix_flags=dict(KL='10 B; >=7 positive deltaKL(t*) AND median>0',margin='10 B; >=7 negative deltaMargin(t*) AND median<0',accumulation='10 B; >=7 positive per-sample median window deltaKL OR >=7 negative per-sample median window deltaMargin'),specificity_operationalization='Prospectively conservative: no B primary/window flag => NOT_PRESENT; else PRESENT only if every active directional signal has <=3/6 both-correct controls in same direction; otherwise MIXED. Always report C/D exact counts; no significance meaning.',layer_flag_operationalization='EXPLORATORY: for each State/Functional region candidate requires B positive deltas >=7/10, uniquely largest among 3 region deltas >=7/10, group median>0. PRESENT only same unique candidate in both families and corresponding positive/dominance conjunction <=3/6 in each C/D control group. Any family candidate without this closure=>MIXED; no family candidate=>NOT_PRESENT.',next_category='A only at least one primary/window flag and specificity PRESENT and not all active directional rates >=4/6 in both-wrong; else B only layer PRESENT; otherwise C (no stable specific prefix explanation, including control-common patterns)',missing_divergence='NONE kept in accounting, primary/window metrics excluded; no replacement point; sequence exhaustion without unequal token is separately labelled',new_free_generation=0,new_AIME_generation=0,training=False,checkpoint_selection=False,rotation_export=False,scorer_calls=0,full_trajectory_replay='NOT_AUTHORIZED',C8_authorized=False,semantic_retry=False,raw_freeze_before_aggregate=True,storage_policy='no full vocabulary disk tensors; transient CPU FP window logits <=65 positions; only t_star state/output/hidden reference tensors transient; metrics only saved',disk_stop_bytes=15*(1<<30),runtime=runtime['runtime'],flags=read(ENDPOINT/'configs/canonical60_generation_config.json')['deterministic_runtime_flags'],created_utc=utc(),code_sha256=sha(__file__))
    save('protocol/preregistration.json',protocol,True)
    save('provenance/input_identity.json',dict(status='PASS',PANEL_IDENTITY_GATE='PASS',PARENT_ARTIFACT_IDENTITY_GATE='PASS',ROTATION_IDENTITY_GATE='PASS',MODEL_IDENTITY_GATE='PASS',MECHANISM_METRIC_IDENTITY_GATE='PASS',parent_root=str(PARENT),endpoint_root=str(ENDPOINT),endpoint_commit=inv['endpoint_commit'],model_identity=runtime['model_identity'],rotations=runtime['rotation_files'],runtime=runtime,metric_sources=metrics,protected_files=protected),True)
    save('provenance/read_only_parent_revalidation_before.json',dict(status='PASS',files=len(protected),parent_manifest_sha256=sha(PARENT/'SHA256SUMS'),parent_write_operations=0,checked_utc=utc()),True)
    freeze=[p for p in ROOT.rglob('*') if p.is_file()]
    write('protocol/PREMEASUREMENT_SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(freeze)),True)
    print(json.dumps(dict(status='PASS',root=str(ROOT),panel=25,t_stars=[r['t_star'] for r in records],sum_prefix_tokens=sum(r['t_star'] or 0 for r in records),new_forward=0,premeasurement_manifest_sha256=sha(ROOT/'protocol/PREMEASUREMENT_SHA256SUMS'))),flush=True)

def worker(gpu):
    import fcntl
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu),'GPU mapping')
    require(not (ROOT/f'logs/worker{gpu}.error.json').exists(),'prior failure: no retry')
    lock=(ROOT/'logs');lock.mkdir(exist_ok=True);handle=(lock/f'worker{gpu}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(manifest_rows(ROOT,ROOT/'protocol/PREMEASUREMENT_SHA256SUMS'))
    cfg=read(ROOT/'protocol/preregistration.json');require(sha(__file__)==cfg['code_sha256'],'adapter changed after freeze')
    inv=read(ROOT/'provenance/input_identity.json');formal_path=inv['runtime']['generation_source']['path']
    require(sha(formal_path)==inv['runtime']['generation_source']['sha256'],'formal drift')
    sys.path.insert(0,str(ENDPOINT/'scripts'));A=module('canonical60_identity_helpers',ENDPOINT/'scripts/canonical60_common.py')
    nv=A.nvml();before=nv.snapshot(gpu);require(before['uuid']==inv['runtime']['GPU_UUIDs'][gpu] and not before['processes'],'foreign GPU process or wrong UUID')
    g=A.generator()
    import torch
    require(torch.cuda.device_count()==1 and g.runtime_version()==cfg['runtime'],'runtime mismatch')
    flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    require(flags==cfg['flags'],'deterministic flags drift')
    def forbidden(*a,**kw):raise RuntimeError('free generation/scoring/training forbidden')
    g.generate_one=g.sample_next=forbidden
    g.setup_seed(1);model,tok=g.load_model_and_tokenizer()
    for p in model.parameters():p.requires_grad_(False)
    require(not model.training,'model not eval')
    metrics=frozen_metrics(torch)
    banks={};ident={}
    for arm,r in inv['rotations'].items():banks[arm],ident[arm]=g.load_frozen_rotation(r['path'],r['manifest_path'],torch.device('cuda:0'),arm)
    # Exact lm_head checkpoint readback guards historical materialization risk.
    from safetensors import safe_open
    modelpath=Path(inv['model_identity']['path']);index=read(modelpath/'model.safetensors.index.json')
    with safe_open(str(modelpath/index['weight_map']['lm_head.weight']),framework='pt',device='cpu') as f:head_ref=f.get_tensor('lm_head.weight')
    require(torch.equal(model.lm_head.weight.detach().cpu(),head_ref),'lm_head GPU materialization mismatch')
    head_sha=g.tensor_sha256(model.lm_head.weight);del head_ref
    eos=tok.eos_token_id;require(eos is not None,'EOS missing')
    stops=sorted(g.stop_ids(tok));require(int(eos) in stops,'EOS not canonical stop')
    units=[r for r in read(ROOT/'protocol/panel.json')['records'] if r['gpu']==gpu]
    progress=[];started=time.time();baseline_alloc=int(torch.cuda.memory_allocated())
    def state_physical(arm,l,state):
        x=state.detach().float()
        if arm=='FP':return x
        # Exact operations/orientation from existing sparse_probe.physical_state.
        if arm in banks:x=torch.einsum('ij,...jv->...iv',banks[arm].layer(l).matrix().float(),x)
        return torch.einsum('ij,...jv->...iv',g.hadamard(128,dtype=torch.float32,device=x.device),x)
    for r in units:
        require(not (ROOT/'logs/STOP.json').exists(),'peer failure, no next sample')
        live=nv.snapshot(gpu);require(live['uuid']==before['uuid'] and all(int(p['pid'])==os.getpid() for p in live['processes']),'foreign GPU process before sample')
        require(shutil.disk_usage('/data').free>=cfg['disk_stop_bytes'],'disk below15GiB')
        require(not (ROOT/f'raw/{r["panel_rank"]:02d}.json').exists() and not (ROOT/f'logs/started_{r["panel_rank"]:02d}.json').exists(),'existing sample no retry')
        for arm in SHAS:require(sha(r[arm+'_raw_path'])==r[arm+'_raw_sha256'],'source raw drift')
        t=r['t_star'];print('SAMPLE_STARTED',r['panel_rank'],r['sample_id'],'t_star',t,'GPU',gpu,flush=True)
        save(f'logs/started_{r["panel_rank"]:02d}.json',dict(sample_id=r['sample_id'],gpu=gpu,uuid=before['uuid'],pid=os.getpid(),utc=utc(),MODEL_FORWARD_STARTED=t is not None),True)
        if t is None:
            save(f'raw/{r["panel_rank"]:02d}.json',dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],status='NO_COMPARABLE_DIVERGENCE_ACCOUNTED',gpu=gpu,uuid=before['uuid'],conditions=[],new_free_generation=0),True);continue
        positions=set(r['window_positions']+[t]);fp={};outputs=[];layerrows=[];audits=[]
        for arm in cfg['conditions']:
            g.setup_seed(r['seed']);_,prompt=g.render_input(tok,r['problem']);require(g.ids_sha256(prompt[0].tolist())==r['input_ids_hash'],'prompt IDs drift')
            prompt=prompt.cuda();capture={'active':False,'values':{},'hidden':None};hooks=[]
            layers=model.model.layers if hasattr(model.model,'layers') else model.model.language_model.layers
            def output_hook(layer):
                def fn(m,a,o):
                    if capture['active']:
                        v=o[0] if isinstance(o,(tuple,list)) else o
                        capture['values'][layer]=v[:,-1:,:].detach().float().cpu().clone()
                return fn
            for l in g.GDN_LAYERS:hooks.append(layers[l].linear_attn.out_proj.register_forward_hook(output_hook(l)))
            def head_hook(m,a):
                if capture['active']:capture['hidden']=a[0][:,-1:,:].detach().cpu().clone()
            hooks.append(model.lm_head.register_forward_pre_hook(head_hook))
            patch=contextlib.nullcontext() if arm=='FP' else g.QwenKeyHadamardPatch(model) if arm=='H' else g.QwenFrozenRotationPatch(model,banks[arm],ident[arm])
            audit=dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0)
            torch.cuda.reset_peak_memory_stats();unitstart=time.time();prealloc=int(torch.cuda.memory_allocated())
            with patch as pa,torch.inference_mode():
                for j in range(t+1):
                    capture['active']=j==t;capture['values']={};capture['hidden']=None
                    if j==0:out=model(input_ids=prompt,attention_mask=torch.ones_like(prompt),use_cache=True)
                    else:out=model(input_ids=torch.tensor([[r['shared_prefix_token_ids'][j-1]]],device='cuda:0',dtype=torch.long),past_key_values=past,use_cache=True)
                    past=out.past_key_values
                    if j>0 and arm!='FP':g.quantize_c128(past,audit)
                    if j not in positions:continue
                    logit=out.logits[:,-1,:].detach().float();require(bool(torch.isfinite(logit).all()),'nonfinite logits')
                    target=r['shared_prefix_token_ids'][j] if j<t else r['C6R_token_at_t_star']
                    if arm=='FP':fp[j]={'logits':logit.cpu().clone()}
                    mm=metrics['model_metrics'](fp[j]['logits'],logit,target)
                    row=dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],gpu=gpu,uuid=before['uuid'],condition=arm,predicted_generated_token_index=j,role='T_STAR' if j==t else 'LAST64',**mm)
                    probs=torch.softmax(logit,dim=-1);ev=float(logit[0,int(eos)])
                    row.update(EOS_token_id=int(eos),EOS_probability=float(probs[0,int(eos)]),EOS_logit=ev,EOS_rank=1+int((logit[0]>ev).sum()),canonical_stop_probability=float(probs[0,stops].sum()),condition_top1=int(logit.argmax()),next_target_kind='SHARED_SAVED_TOKEN' if j<t else 'C6R_DIVERGENCE_TOKEN_NOT_GROUND_TRUTH')
                    if j==t:
                        require(set(capture['values'])==set(g.GDN_LAYERS) and capture['hidden'] is not None,'capture coverage')
                        require(tuple(capture['hidden'].shape)==(1,1,4096),'final-hidden lm_head input shape')
                        if arm=='FP':
                            fp[j]['states']={l:state_physical('FP',l,g.get_state(past,l)).cpu().clone() for l in g.GDN_LAYERS}
                            fp[j]['outputs']={l:v.clone() for l,v in capture['values'].items()};fp[j]['hidden']=capture['hidden'].clone()
                        row['Final_Hidden_relative_mse']=float(metrics['relative_mse'](capture['hidden'],fp[j]['hidden']))
                        winner=r['C7_token_at_t_star'] if r['pair_category']=='A' else r['C6R_token_at_t_star'] if r['pair_category']=='B' else None
                        alternative=r['C6R_token_at_t_star'] if r['pair_category']=='A' else r['C7_token_at_t_star'] if r['pair_category']=='B' else None
                        row.update(winner_token=winner,alternative_token=alternative,PairMargin=None if winner is None else float(logit[0,winner]-logit[0,alternative]),winner_label='ENDPOINT_SUCCESS_ASSOCIATED_NOT_GROUND_TRUTH' if winner is not None else 'NOT_DEFINED_FOR_CONTROLS',Top32_token_ids=torch.argsort(logit[0],descending=True,stable=True)[:32].cpu().tolist())
                        row['Top32_logits']=[float(logit[0,k]) for k in row['Top32_token_ids']]
                        for l in g.GDN_LAYERS:
                            state=metrics['state_metric'](state_physical(arm,l,g.get_state(past,l)),fp[j]['states'][l])
                            function=metrics['state_metric'](capture['values'][l],fp[j]['outputs'][l])
                            region=next(name for name,ids in cfg['regions'].items() if l in ids)
                            layerrows.append(dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],condition=arm,layer=l,region=region,predicted_generated_token_index=j,State_relative_l2=state['relative_l2'],Functional_relative_l2=function['relative_l2'],state_metrics=state,functional_metrics=function))
                    outputs.append(row)
                require(audit['calls']==(0 if arm=='FP' else t),'QDQ call count')
                require(arm=='FP' or sorted(audit['layers'])==list(g.GDN_LAYERS),'QDQ layers')
                if arm!='FP':require(pa.calls==24*(1+t) and sorted(pa.layers)==list(g.GDN_LAYERS),'rotation patch coverage')
            for h in hooks:h.remove()
            peak=int(torch.cuda.max_memory_allocated());patch_info=None if arm=='FP' else dict(calls=pa.calls,layers=sorted(pa.layers),phase_counts=pa.phase_counts)
            del out,past,prompt,patch,pa,logit,probs;capture.clear();gc.collect();torch.cuda.empty_cache()
            afteralloc=int(torch.cuda.memory_allocated())
            require(afteralloc<=baseline_alloc+(64<<20),'unexpected persistent GPU memory creep')
            audits.append(dict(condition=arm,elapsed_seconds=time.time()-unitstart,prefix_tokens_consumed=t,quantizer_calls=audit['calls'],quantizer_layers=sorted(audit['layers']),patch=patch_info,allocated_before=prealloc,allocated_after=afteralloc,peak_bytes=peak))
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()),'model gradient appeared')
        save(f'raw/{r["panel_rank"]:02d}.json',dict(status='PASS',sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],gpu=gpu,uuid=before['uuid'],t_star=t,shared_prefix_sha256=r['shared_prefix_sha256'],window_positions=r['window_positions'],model_metrics=outputs,layer_metrics=layerrows,condition_audits=audits,runtime=g.runtime_version(),lm_head_sha256=head_sha,new_free_generation=0,new_training=0,scorer_calls=0,completed_utc=utc()),True)
        progress.append(r['sample_id']);fp.clear();del fp,outputs,layerrows;gc.collect()
        print('SAMPLE_COMPLETE',r['panel_rank'],r['sample_id'],flush=True)
    require(g.tensor_sha256(model.lm_head.weight)==head_sha,'lm_head changed')
    save(f'logs/worker{gpu}.complete.json',dict(status='PASS',gpu=gpu,uuid=before['uuid'],pid=os.getpid(),assigned=len(units),completed=len(units),measurement_sample_ids=progress,elapsed_seconds=time.time()-started,normal_exit=True,lm_head_unchanged=True,new_free_generation=0,retries=0),True)
    print('WORKER_COMPLETE',gpu,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','worker'],required=True);ap.add_argument('--gpu',type=int,choices=range(4));a=ap.parse_args()
    if a.phase=='prepare':prepare();return
    require(a.gpu is not None,'gpu required')
    try:worker(a.gpu)
    except BaseException as e:
        if ROOT.exists():
            save(f'logs/worker{a.gpu}.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),gpu=a.gpu,pid=os.getpid(),utc=utc(),retry=False),True)
            if not (ROOT/'logs/STOP.json').exists():save('logs/STOP.json',dict(status='STOP',reason=str(e),gpu=a.gpu,utc=utc(),no_automatic_retry=True),True)
        raise
if __name__=='__main__':main()
