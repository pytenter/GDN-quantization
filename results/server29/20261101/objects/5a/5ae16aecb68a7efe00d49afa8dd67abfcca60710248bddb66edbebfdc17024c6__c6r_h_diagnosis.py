"""Frozen C6R/H discordant shared-prefix diagnosis; no sampling/scoring/training."""
import argparse, ast, collections, contextlib, csv, gc, hashlib, importlib.util, json
import math, os, shutil, statistics, subprocess, sys, time, traceback
from pathlib import Path
HELPER=Path('/data/zypan/worktrees/qwen-gdn-c6r-c7-canonical60-v1/experiments/QWEN_C6R_C7_FIRST_DIVERGENCE_PREFIX_REPLAY_V1/scripts/first_divergence_replay.py')
sp=importlib.util.spec_from_file_location('frozen_prefix_utilities',HELPER);S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
TASK='QWEN_C6R_H_DISCORDANT_MECHANISM_DIAGNOSIS_V1'
ROOT=S.REPO/'experiments'/TASK;S.ROOT=ROOT
ENDPOINT=S.ENDPOINT;PARENT=S.PARENT
SHAS={'C6R':S.SHAS['C6R']}
for name in ['require','utc','read','sha','digest','ids_hash','write','save','table','module','source_function','frozen_metrics','manifest_rows','verify','git_at']:globals()[name]=getattr(S,name)
def key(x):return x['problem_id'],int(x['seed'])
def sid(k):return f'{k[0]}:seed{k[1]}'
def boolval(v):return v if isinstance(v,bool) else str(v).lower() in ('1','true')

def qdq_observer(torch,g,past,audit,metrics):
    """Read-only operator observation around unmodified canonical quantize_c128."""
    from torch.utils._python_dispatch import TorchDispatchMode
    pre={l:g.get_state(past,l).detach().float().cpu().clone() for l in g.GDN_LAYERS}
    class Observe(TorchDispatchMode):
        def __init__(self):super().__init__();self.amax=[];self.scale=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            if isinstance(result,torch.Tensor) and tuple(result.shape)==(1,32,1,128):
                if func==torch.ops.aten.amax.default:self.amax.append(result.detach().cpu().clone())
                if func==torch.ops.aten.div.Tensor and len(args)>1 and isinstance(args[1],(float,int)) and args[1]==127.0:self.scale.append(result.detach().cpu().clone())
            return result
    observer=Observe()
    with observer:g.quantize_c128(past,audit)
    require(len(observer.amax)==len(observer.scale)==24,'QDQ true amax/scale observation coverage')
    def summary(x):
        v=x.reshape(-1);return dict(count=v.numel(),minimum=float(v.min()),mean=float(v.mean()),median=float(v.median()),maximum=float(v.max()))
    rows=[]
    for i,l in enumerate(g.GDN_LAYERS):
        x=pre[l];m=metrics['state_metric'](g.get_state(past,l),x)
        rows.append(dict(layer=l,post_QDQ_relative_l2=m['relative_l2'],QDQ_error_metrics=m,pre_QDQ_L2=m['reference_l2'],pre_QDQ_mean_abs=float(x.abs().mean()),pre_QDQ_max_abs=float(x.abs().max()),group_amax=summary(observer.amax[i]),group_scale=summary(observer.scale[i]),operator_amax='aten.amax.default',operator_scale='aten.div.Tensor with actual divisor 127.0',group_shape=[1,32,1,128],evidence='DESCRIPTIVE_ONLY',RAW_M5='NOT_CAPTURED',outlier_concentration='NOT_AVAILABLE_FROM_REUSED_CAPTURE'))
    return rows

def cpu_qdq_gate():
    A=module('cpu_formal_identity_helper',ENDPOINT/'scripts/canonical60_common.py');g=A.generator()
    import torch
    from types import SimpleNamespace
    metrics=frozen_metrics(torch);torch.manual_seed(1977)
    base={l:torch.randn(1,32,128,128).to(torch.bfloat16) for l in g.GDN_LAYERS}
    # Includes zero groups; original EPS/round/clamp/writeback, never a copied quantizer.
    for l in g.GDN_LAYERS:base[l][:,:,0:128,0]=0
    a=SimpleNamespace(recurrent_states={l:v.clone() for l,v in base.items()});b=SimpleNamespace(recurrent_states={l:v.clone() for l,v in base.items()})
    def audit():return dict(calls=0,layers=set(),scale_shapes={},code_min=0,code_max=0)
    aa=audit();bb=audit();g.quantize_c128(a,aa);diag=qdq_observer(torch,g,b,bb,metrics)
    require(aa==bb and all(torch.equal(a.recurrent_states[l],b.recurrent_states[l]) for l in g.GDN_LAYERS),'observational QDQ CPU parity')
    require(all(m['group_amax']['minimum']==0 for m in diag),'zero-group test')
    return dict(status='PASS',exact_writeback_match=True,exact_audit_match=True,layers=24,CPU_synthetic_only=True,new_GPU_forward=0,source_function='unmodified formal.quantize_c128',observation_changes_returned_tensors=False)

def prepare():
    require(not ROOT.exists(),'existing experiment; no duplicate')
    inv=read(PARENT/'provenance/input_artifact_inventory.json');context=read(PARENT/'analysis/c6r_vs_h_contextual_taxonomy.json')
    protected=manifest_rows(PARENT,PARENT/'SHA256SUMS')+manifest_rows(ENDPOINT,ENDPOINT/'manifests/final_artifact_sha256.txt')
    protected += [dict(path=p,sha256=h) for p,h in read(PARENT/'provenance/read_only_revalidation.json')['consumed_source_hashes'].items()]
    for p in [PARENT/'SHA256SUMS',ENDPOINT/'manifests/final_artifact_sha256.txt',HELPER,ENDPOINT/'scripts/canonical60_common.py']:protected.append(dict(path=str(p),sha256=sha(p)))
    verify(protected)
    hp=Path(context['source_path']);require(sha(hp)==context['source_sha256'],'H scored source hash')
    with hp.open() as f:hs=[r for r in csv.DictReader(f) if r['condition']=='QWEN_INT8_C128_KEY_HADAMARD']
    hm={key(r):r for r in hs};scored=read(ENDPOINT/'scores/c6r_frozen_v4.json');cm={key(r):r for r in scored['records']}
    require(len(hs)==len(hm)==len(cm)==60 and set(hm)==set(cm),'60 exact sample IDs')
    gold={key(r):int(r['answer']) for r in read(ENDPOINT/'manifests/canonical60_gold_manifest.json')['records']}
    pairs=[]
    for k in sorted(cm):
        require(int(hm[k]['gold_answer'])==gold[k]==int(cm[k]['gold_answer']),'gold ID mismatch')
        c=boolval(cm[k]['v4_correct']);h=boolval(hm[k]['v4_correct']);category='A' if c and not h else 'B' if h and not c else 'C' if c else 'D'
        pairs.append(dict(sample_id=sid(k),problem_id=k[0],seed=k[1],pair_category=category,category_name={'A':'C6R_WIN','B':'H_WIN','C':'BOTH_CORRECT','D':'BOTH_WRONG'}[category],C6R_correct=c,H_correct=h,historical_H_evidence='HISTORICAL_CONTEXTUAL_ONLY'))
    require(collections.Counter(r['pair_category'] for r in pairs)==dict(A=5,B=6,C=35,D=14),'pair counts not 5/6/35/14')
    chosen=[r for r in pairs if r['pair_category'] in 'AB']
    for c in 'CD':chosen += [r for r in pairs if r['pair_category']==c][:6]
    chosen.sort(key=key);require(len(chosen)==23,'panel count')
    frozen_external={r['path']:r['sha256'] for r in read(ENDPOINT/'provenance/protected_artifact_revalidation.json')['files']}
    hraw={};sources=[]
    for name in sorted({r['source_file'] for r in hs}):
        require(name in frozen_external and sha(name)==frozen_external[name],'H raw frozen identity')
        protected.append(dict(path=name,sha256=frozen_external[name]));sources.append(dict(path=name,sha256=frozen_external[name]))
        for line_no,line in enumerate(Path(name).read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line);require(key(row) not in hraw and row['method']=='QWEN_INT8_C128_KEY_HADAMARD','H raw duplicate/condition')
            require(isinstance(row.get('generated_token_ids'),list) and all(isinstance(v,int) and not isinstance(v,bool) for v in row['generated_token_ids']),'H_SEQUENCE_RECOVERY_GATE BLOCKED: no original token IDs')
            require(len(row['generated_token_ids'])==row['generated_token_count'] and row['runtime_error'] is None and not row['nonfinite'],'H raw count/runtime')
            hraw[key(row)]=(row,name,line_no)
    require(set(hraw)==set(cm),'H raw 60 complete')
    crinv={key(r):r for r in inv['raw_outputs'] if r['condition']=='C6R'};jobs={key(r):r for r in read(ENDPOINT/'manifests/canonical60_sample_manifest.json')['samples']}
    runtime=read(ENDPOINT/'provenance/canonical60_runtime_identity.json');verify(runtime['model_identity']['files']);protected+=runtime['model_identity']['files']
    rotation=runtime['rotation_files']['C6R'];require(sha(rotation['path'])==SHAS['C6R']==rotation['sha256'] and sha(rotation['manifest_path'])==rotation['manifest_sha256'],'C6R rotation')
    protected += [dict(path=rotation['path'],sha256=rotation['sha256']),dict(path=rotation['manifest_path'],sha256=rotation['manifest_sha256'])]
    records=[];seq=[]
    for rank,p in enumerate(chosen):
        k=key(p);h,hn,ln=hraw[k];ci=crinv[k];c=read(ci['path']);job=jobs[k]
        require(sha(ci['path'])==ci['sha256'] and h['input_ids_hash']==c['input_ids_hash']==job['input_ids_hash'] and h['problem']==c['problem']==job['problem'],'sequence/prompt identity')
        a=h['generated_token_ids'];b=c['generated_token_ids'];t=next((j for j,(u,v) in enumerate(zip(a,b)) if u!=v),None)
        status='TOKEN_DIVERGENCE' if t is not None else 'NONE' if a==b else 'SEQUENCE_LENGTH_DIFFERS_NO_COMPARABLE_TOKEN_DIVERGENCE'
        require(t is None or a[:t]==b[:t],'first divergence')
        records.append(dict(**p,panel_rank=rank,gpu=rank%4,GPU_uuid=runtime['GPU_UUIDs'][rank%4],first_divergence=status,t_star=t,H_token_at_t_star=None if t is None else a[t],C6R_token_at_t_star=None if t is None else b[t],H_total_length=len(a),C6R_total_length=len(b),shared_prefix_length=t,shared_prefix_token_ids=[] if t is None else a[:t],shared_prefix_sha256=None if t is None else ids_hash(a[:t]),window_positions=[] if t is None else list(range(max(0,t-64),t)),H_raw_path=hn,H_raw_sha256=frozen_external[hn],H_raw_line=ln,C6R_raw_path=ci['path'],C6R_raw_sha256=ci['sha256'],problem=job['problem'],input_ids_hash=job['input_ids_hash'],prompt_sha256=job['prompt_sha256']))
        seq.append(dict(sample_id=p['sample_id'],H=dict(path=hn,line=ln,source_sha256=frozen_external[hn],original_token_ids=True,token_count=len(a),token_ids_sha256=ids_hash(a)),C6R=dict(path=ci['path'],sha256=ci['sha256'],original_token_ids=True,token_count=len(b),token_ids_sha256=ids_hash(b)),reconstruction=False))
    metricinfo=[]
    for path,n in [(S.METRIC_SOURCE,'state_metric'),(S.METRIC_SOURCE,'model_metrics'),(S.METRIC_SOURCE,'physical_state'),(S.METRIC_SOURCE,'OutputCapture'),(S.CAYLEY,'relative_mse'),(Path(runtime['generation_source']['path']),'quantize_c128'),(Path(runtime['generation_source']['path']),'hadamard')]:
        metricinfo.append(dict(path=str(path),function=n,file_sha256=sha(path),function_sha256=hashlib.sha256(source_function(path,n).encode()).hexdigest(),commit=git_at(path.parent)));protected.append(dict(path=str(path),sha256=sha(path)))
    # Reuse already-computed behavior; never call Frozen V4 or claim enumeration.
    reuse=read(PARENT/'provenance/diagnostic_definition_reuse.json');util=Path(reuse['source_implementation']);require(sha(util)==reuse['source_implementation_sha256'],'behavior implementation')
    U=module('frozen_behavior_definition',util);hmetric=U.ROOT/'analysis/trajectory_metrics.jsonl';mh=U.ROOT/'hashes/artifact_sha256.txt'
    oldhash={r['path']:r['sha256'] for r in manifest_rows(U.ROOT,mh)};require(str(hmetric) in oldhash and sha(hmetric)==oldhash[str(hmetric)],'H stored behavior frozen hash')
    ht={key(r):r for r in [json.loads(l) for l in hmetric.read_text().splitlines()] if r['condition']=='H'}
    with Path(reuse['source_csv']).open() as f:ct={key(r):r for r in csv.DictReader(f) if r['condition']=='C6R'}
    require(set(ht)==set(ct)==set(cm),'behavior IDs')
    behavior=[]
    for p in chosen:
        if p['pair_category'] not in 'AB':continue
        k=key(p)
        for arm,r,b,s in [('H',hraw[k][0],ht[k],hm[k]),('C6R',read(crinv[k]['path']),ct[k],cm[k])]:
            require(int(b['generated_tokens'])==len(r['generated_token_ids']),'stored behavior length identity')
            behavior.append(dict(sample_id=p['sample_id'],pair_category=p['pair_category'],condition=arm,abstain=s['v4_status'].lower()=='abstain',generated_length=len(r['generated_token_ids']),length_hit=r['finish_reason']=='length',EOS=bool(r['eos_seen']),finish_reason=r['finish_reason'],oscillation_proxy=float(b['oscillation_proxy']),answer_changes=int(b['answer_changes']),retractions=int(b['retractions']),committed_aba_count=int(b['committed_aba_count']),first_committed_claim=b['first_committed_claim'],first_committed_claim_char=b['first_committed_claim_char'],**{f'repeated_{n}gram_fraction':float(b[f'repeated_{n}gram_fraction']) for n in (4,8,16)},evidence='EXISTING_OUTPUT_FROZEN_DEFINITION_NO_SCORER_CALL'))
    for p in [hmetric,mh,util,Path(reuse['source_csv'])]:protected.append(dict(path=str(p),sha256=sha(p)))
    qgate=cpu_qdq_gate();layers=read(ENDPOINT/'configs/canonical60_protocol.json')['GDN_layers'];regions={n:layers[i*8:(i+1)*8] for i,n in enumerate(['early','middle','late'])}
    protected=list({r['path']:r for r in protected}.values());ROOT.mkdir(parents=True)
    for name in ['c6r_h_diagnosis.py','c6r_h_analysis.py','run_c6r_h_diagnosis.sh']:write('scripts/'+name,Path(__file__).with_name(name).read_text(),True)
    gates=dict(PARENT_ARTIFACT_IDENTITY_GATE='PASS',C6R_H_PAIR_ALIGNMENT_GATE='PASS',PANEL_IDENTITY_GATE='PASS',H_SEQUENCE_RECOVERY_GATE='PASS',C6R_SEQUENCE_IDENTITY_GATE='PASS',MODEL_IDENTITY_GATE='PASS',ROTATION_IDENTITY_GATE='PASS',QDQ_IDENTITY_GATE='PASS',MECHANISM_METRIC_IDENTITY_GATE='PASS')
    save('provenance/input_artifact_inventory.json',dict(**gates,endpoint_root=str(ENDPOINT),parent_root=str(PARENT),H_scored_path=str(hp),H_scored_sha256=sha(hp),H_raw_files=sources,HISTORICAL_H='HISTORICAL_CONTEXTUAL_ONLY',C6R_checkpoint_export=rotation,metric_sources=metricinfo,behavior_sources=dict(H=str(hmetric),C6R=reuse['source_csv'],producer=str(util),producer_sha256=sha(util)),protected_files=protected),True)
    save('provenance/input_identity.json',dict(**gates,runtime=runtime,model_identity=runtime['model_identity'],rotations={'C6R':rotation},protected_files=protected),True)
    save('protocol/frozen_panel.json',dict(count=23,selection='all 11 discordants + sorted first six per C/D; full panel sorted by problem_id then seed; rank zero-based %4',pair_counts=dict(A=5,B=6,C=35,D=14),records=records),True)
    save('protocol/panel.json',dict(records=records),True);save('protocol/original_sequence_manifest.json',dict(status='PASS',records=seq),True)
    table('analysis/c6r_h_pair_taxonomy.csv',pairs,freeze=True);table('analysis/existing_output_behavior_triage.csv',behavior,freeze=True)
    cols=['sample_id','problem_id','seed','category_name','panel_rank','gpu','GPU_uuid','first_divergence','t_star','H_token_at_t_star','C6R_token_at_t_star','shared_prefix_length','H_total_length','C6R_total_length']
    table('analysis/first_divergence_manifest.csv',[{k:r[k] for k in cols} for r in records],freeze=True)
    save('provenance/qdq_observer_cpu_parity.json',qgate,True)
    save('protocol/preregistration.json',dict(task=TASK,conditions=['FP','H','C6R'],W=64,indexing='distribution j predicts saved token[j]; prefill predicts0; j>0 consumes shared[j-1]; t* consumes exactly shared[0:t*)',no_sampling=True,prefill_QDQ=False,QDQ_after_decode=True,state_snapshot='post-QDQ vs FP; functional and logits precede current QDQ',metric_sources=metricinfo,final_hidden='existing relative_mse at normalized lm_head input, model-level only',delta='C6R minus H; raw logit KL(FP||X); no sampling warpers',regions=regions,region_order=['early','middle','late'],region_reduction='per sample median paired per-layer C6R-H; category median per-sample region penalties',layer_rule='For each State/Functional family, fixed group-worst region = argmax H_WIN category median regional penalty; ties early/middle/late. Report each sample worst region as well. PRESENT candidate requires >=5/6 H_WIN positive in this same region and <5/6 BOTH_CORRECT positive. Combined PRESENT if any family candidate; MIXED if positive >=5 but common in controls; else NOT_PRESENT. Exploratory only.',qdq_rule='Same regional reduction/group-worst rule; >=5/6 H_WIN positive post-QDQ relative_l2 difference and <5/6 positive in each BOTH_CORRECT and BOTH_WRONG; otherwise MIXED if stable but common, else NOT_PRESENT.',primary_rule='KL positive>=5/6 AND median>0; margin negative>=5/6 AND median<0; FH positive>=5/6 AND median>0',specificity_rule='No active primary/layer/QDQ signal => NOT_PRESENT; PRESENT only an active signal has <5/6 same direction in each C and D; MIXED if any active signal is common in either control category. Raw counts always reported.',next_rule='A only specific layer candidate (C and D each <5/6); B only primary-specific signal without PRESENT layer or QDQ; C if no specific stable signal. QDQ-only specific outcome without an eligible A/B category is protocol category gap: stop and request user direction, never invent verdict.',qdq_capture='read-only TorchDispatchMode records actual original aten.amax / div scale values; original formal.quantize_c128 called exactly once. CPU exact state and audit parity required before GPU.',qdq_summary='per layer actual group count/min/mean/median/max; native post-vs-pre QDQ error uses existing state_metric. No copied quantizer, no M5 ranking.',RAW_M5='NOT_CAPTURED; any raw M5 would be WITHIN_ROTATION_DIAGNOSTIC_ONLY',READOUT_AWARE_METRIC='NOT_AVAILABLE_AS_DISTINCT_VALIDATED_METRIC_IN_REUSED_REPLAY_SOURCE',optional_window_layer_metrics='NOT_CAPTURED; primary LAST64 logit/margin present; all24 state/functional/QDQ captured at t*',top32='t* only; descending stable logits ties ascending token ID',disk_stop_bytes=15*(1<<30),runtime=runtime['runtime'],flags=read(ENDPOINT/'configs/canonical60_generation_config.json')['deterministic_runtime_flags'],code_sha256=sha(__file__),helper_sha256=sha(HELPER),created_utc=utc(),raw_freeze_before_aggregate=True,new_free_generation=0,new_AIME_generation=0,new_training=False,scorer_calls=0,full_trajectory='NOT_AUTHORIZED',C8_authorized=False),True)
    write('protocol/PREMEASUREMENT_SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(ROOT.rglob('*')) if p.is_file()),True)
    print(json.dumps(dict(status='PASS',panel=23,first_divergence=[r['t_star'] for r in records],sum_prefix_tokens=sum(r['t_star'] or 0 for r in records),gates=gates,new_GPU_forward=0,root=str(ROOT))),flush=True)

def worker(gpu):
    import fcntl
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu),'GPU mapping')
    require(not (ROOT/f'logs/worker{gpu}.error.json').exists(),'prior failure: no retry')
    lock=ROOT/'logs';lock.mkdir(exist_ok=True);handle=(lock/f'worker{gpu}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(manifest_rows(ROOT,ROOT/'protocol/PREMEASUREMENT_SHA256SUMS'))
    cfg=read(ROOT/'protocol/preregistration.json');require(sha(__file__)==cfg['code_sha256'] and sha(HELPER)==cfg['helper_sha256'],'adapter/helper changed after freeze')
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
    require(not model.training,'model not eval');metrics=frozen_metrics(torch)
    banks={};ident={}
    for arm,r in inv['rotations'].items():banks[arm],ident[arm]=g.load_frozen_rotation(r['path'],r['manifest_path'],torch.device('cuda:0'),arm)
    from safetensors import safe_open
    modelpath=Path(inv['model_identity']['path']);index=read(modelpath/'model.safetensors.index.json')
    with safe_open(str(modelpath/index['weight_map']['lm_head.weight']),framework='pt',device='cpu') as f:head_ref=f.get_tensor('lm_head.weight')
    require(torch.equal(model.lm_head.weight.detach().cpu(),head_ref),'lm_head GPU materialization mismatch')
    head_sha=g.tensor_sha256(model.lm_head.weight);del head_ref
    eos=tok.eos_token_id;require(eos is not None,'EOS missing');stops=sorted(g.stop_ids(tok));require(int(eos) in stops,'EOS not canonical stop')
    units=[r for r in read(ROOT/'protocol/panel.json')['records'] if r['gpu']==gpu]
    progress=[];started=time.time();baseline_alloc=int(torch.cuda.memory_allocated())
    def state_physical(arm,l,state):
        x=state.detach().float()
        if arm=='FP':return x
        # Two separate inverse transforms, exact existing physical_state orientation.
        if arm in banks:x=torch.einsum('ij,...jv->...iv',banks[arm].layer(l).matrix().float(),x)
        return torch.einsum('ij,...jv->...iv',g.hadamard(128,dtype=torch.float32,device=x.device),x)
    for r in units:
        require(not (ROOT/'logs/STOP.json').exists(),'peer failure, no next sample')
        live=nv.snapshot(gpu);require(live['uuid']==before['uuid'] and all(int(p['pid'])==os.getpid() for p in live['processes']),'foreign GPU process before sample')
        require(shutil.disk_usage('/data').free>=cfg['disk_stop_bytes'],'disk below15GiB')
        require(not (ROOT/f'raw/{r["panel_rank"]:02d}.json').exists() and not (ROOT/f'logs/started_{r["panel_rank"]:02d}.json').exists(),'existing sample no retry')
        for arm in ['H','C6R']:require(sha(r[arm+'_raw_path'])==r[arm+'_raw_sha256'],'source raw drift')
        t=r['t_star'];print('SAMPLE_STARTED',r['panel_rank'],r['sample_id'],'t_star',t,'GPU',gpu,flush=True)
        save(f'logs/started_{r["panel_rank"]:02d}.json',dict(sample_id=r['sample_id'],gpu=gpu,uuid=before['uuid'],pid=os.getpid(),utc=utc(),MODEL_FORWARD_STARTED=t is not None),True)
        if t is None:
            save(f'raw/{r["panel_rank"]:02d}.json',dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],status='NO_COMPARABLE_DIVERGENCE_ACCOUNTED',gpu=gpu,uuid=before['uuid'],conditions=[],new_free_generation=0),True);continue
        positions=set(r['window_positions']+[t]);fp={};outputs=[];layerrows=[];qdqrows=[];audits=[]
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
                    if j>0 and arm!='FP':
                        if j==t:
                            observed=qdq_observer(torch,g,past,audit,metrics)
                            for q in observed:
                                qdqrows.append(dict(sample_id=r['sample_id'],pair_category=r['pair_category'],condition=arm,region=next(n for n,ls in cfg['regions'].items() if q['layer'] in ls),predicted_generated_token_index=j,**q))
                            del observed,q
                        else:g.quantize_c128(past,audit)
                    if j not in positions:continue
                    logit=out.logits[:,-1,:].detach().float();require(bool(torch.isfinite(logit).all()),'nonfinite logits')
                    target=r['shared_prefix_token_ids'][j] if j<t else r['H_token_at_t_star']
                    if arm=='FP':fp[j]={'logits':logit.cpu().clone()}
                    mm=metrics['model_metrics'](fp[j]['logits'],logit,target)
                    row=dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],gpu=gpu,uuid=before['uuid'],condition=arm,predicted_generated_token_index=j,role='T_STAR' if j==t else 'LAST64',**mm)
                    probs=torch.softmax(logit,dim=-1);ev=float(logit[0,int(eos)])
                    row.update(EOS_token_id=int(eos),EOS_probability=float(probs[0,int(eos)]),EOS_logit=ev,EOS_rank=1+int((logit[0]>ev).sum()),canonical_stop_probability=float(probs[0,stops].sum()),condition_top1=int(logit.argmax()),next_target_kind='SHARED_SAVED_TOKEN' if j<t else 'H_DIVERGENCE_TOKEN_NOT_GROUND_TRUTH')
                    if j==t:
                        require(set(capture['values'])==set(g.GDN_LAYERS) and capture['hidden'] is not None,'capture coverage')
                        require(tuple(capture['hidden'].shape)==(1,1,4096),'final-hidden lm_head input shape')
                        if arm=='FP':
                            fp[j]['states']={l:state_physical('FP',l,g.get_state(past,l)).cpu().clone() for l in g.GDN_LAYERS}
                            fp[j]['outputs']={l:v.clone() for l,v in capture['values'].items()};fp[j]['hidden']=capture['hidden'].clone()
                        row['Final_Hidden_relative_mse']=float(metrics['relative_mse'](capture['hidden'],fp[j]['hidden']))
                        winner=r['C6R_token_at_t_star'] if r['pair_category']=='A' else r['H_token_at_t_star'] if r['pair_category']=='B' else None
                        alternative=r['H_token_at_t_star'] if r['pair_category']=='A' else r['C6R_token_at_t_star'] if r['pair_category']=='B' else None
                        row.update(winner_token=winner,alternative_token=alternative,PairMargin=None if winner is None else float(logit[0,winner]-logit[0,alternative]),winner_label='ENDPOINT_SUCCESS_ASSOCIATED_NOT_GROUND_TRUTH' if winner is not None else 'NOT_DEFINED_FOR_CONTROLS',Top32_token_ids=torch.argsort(logit[0],descending=True,stable=True)[:32].cpu().tolist())
                        row['Top32_logits']=[float(logit[0,k]) for k in row['Top32_token_ids']]
                        for l in g.GDN_LAYERS:
                            state=metrics['state_metric'](state_physical(arm,l,g.get_state(past,l)),fp[j]['states'][l]);function=metrics['state_metric'](capture['values'][l],fp[j]['outputs'][l])
                            region=next(name for name,ids in cfg['regions'].items() if l in ids)
                            layerrows.append(dict(sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],condition=arm,layer=l,region=region,predicted_generated_token_index=j,State_relative_l2=state['relative_l2'],Functional_relative_l2=function['relative_l2'],state_metrics=state,functional_metrics=function))
                    outputs.append(row)
                require(audit['calls']==(0 if arm=='FP' else t),'QDQ call count')
                require(arm=='FP' or sorted(audit['layers'])==list(g.GDN_LAYERS),'QDQ layers')
                if arm!='FP':require(pa.calls==24*(1+t) and sorted(pa.layers)==list(g.GDN_LAYERS),'rotation patch coverage')
            for h in hooks:h.remove()
            peak=int(torch.cuda.max_memory_allocated());patch_info=None if arm=='FP' else dict(calls=pa.calls,layers=sorted(pa.layers),phase_counts=pa.phase_counts)
            del out,past,prompt,patch,pa,logit,probs;capture.clear();gc.collect();torch.cuda.empty_cache()
            afteralloc=int(torch.cuda.memory_allocated());require(afteralloc<=baseline_alloc+(64<<20),'unexpected persistent GPU memory creep')
            audits.append(dict(condition=arm,elapsed_seconds=time.time()-unitstart,prefix_tokens_consumed=t,quantizer_calls=audit['calls'],quantizer_layers=sorted(audit['layers']),patch=patch_info,allocated_before=prealloc,allocated_after=afteralloc,peak_bytes=peak))
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()),'model gradient appeared')
        save(f'raw/{r["panel_rank"]:02d}.json',dict(status='PASS',sample_id=r['sample_id'],panel_rank=r['panel_rank'],pair_category=r['pair_category'],gpu=gpu,uuid=before['uuid'],t_star=t,shared_prefix_sha256=r['shared_prefix_sha256'],window_positions=r['window_positions'],model_metrics=outputs,layer_metrics=layerrows,qdq_metrics=qdqrows,condition_audits=audits,runtime=g.runtime_version(),lm_head_sha256=head_sha,new_free_generation=0,new_training=0,scorer_calls=0,completed_utc=utc()),True)
        progress.append(r['sample_id']);fp.clear();del fp,outputs,layerrows,qdqrows;gc.collect();print('SAMPLE_COMPLETE',r['panel_rank'],r['sample_id'],flush=True)
    require(g.tensor_sha256(model.lm_head.weight)==head_sha,'lm_head changed')
    save(f'logs/worker{gpu}.complete.json',dict(status='PASS',gpu=gpu,uuid=before['uuid'],pid=os.getpid(),assigned=len(units),completed=len(units),measurement_sample_ids=progress,elapsed_seconds=time.time()-started,normal_exit=True,lm_head_unchanged=True,new_free_generation=0,retries=0),True)
    print('WORKER_COMPLETE',gpu,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','worker'],required=True);ap.add_argument('--gpu',type=int,choices=range(4));a=ap.parse_args()
    if a.phase=='prepare':prepare();return
    require(a.gpu is not None,'GPU required')
    try:worker(a.gpu)
    except BaseException as e:
        if ROOT.exists():
            save(f'logs/worker{a.gpu}.error.json',dict(status='FAIL',error=str(e),traceback=traceback.format_exc(),gpu=a.gpu,pid=os.getpid(),utc=utc(),no_retry=True),True)
            if not (ROOT/'logs/STOP.json').exists():save('logs/STOP.json',dict(status='STOP',error=str(e),utc=utc(),no_retry=True),True)
        raise
if __name__=='__main__':main()
