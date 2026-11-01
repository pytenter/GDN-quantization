"""Fail-closed observability bridge. No training, AIME, or free generation."""
from pathlib import Path
import argparse, ast, copy, csv, gc, hashlib, importlib.util, json, math, os
import statistics, subprocess, sys, time, traceback
sys.dont_write_bytecode=True
TASK='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
D=Path(__file__).resolve().parents[1]
ROOT=D.parents[1]
SOURCE=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1')
TARGET=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v1')
OLD_REMOTE=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1')
OLD_N=Path('/data/zypan/worktrees/ling-error-coherence-diagnosis-v1/experiments/LING_ERROR_COHERENCE_DIAGNOSIS_V1')
LEGACY=Path('/data01/user2/.codex/worktrees/GDN-quantization/ling-kda-int8-rc-granularity')
SNAP=D/'historical_assets/snapshot'
GPU=6
UUID='GPU-de53f243-b271-521c-d875-d96611ed8574'
LAYERS=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
EXECUTED=set()
PROJECT_DEPS=set()
HISTORICAL_ALLOWED={'run_ling_kda_query_readout_causal_closure_v1.py','pytorch_kda_reference.py',
    'run_ling_kda_source_error_functional_causal_v1.py','run_ling_kda_persistent_error_decomposition_causal_v1.py'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def row(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def tensor_sha(x):
    import torch
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),('NO_OVERWRITE',str(p))
    tmp=p.with_name(p.name+'.partial');assert not tmp.exists()
    with tmp.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def imp(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp)
    sys.modules[name]=m;sp.loader.exec_module(m);return m
def run(a,cwd=None):return subprocess.check_output(a,cwd=cwd,text=True).strip()
def relocate(p):
    p=str(p)
    if p.startswith(str(SOURCE)+'/'):return ROOT/Path(p).relative_to(SOURCE)
    return Path(p)

def profile(frame,event,arg):
    if event!='call':return
    name=frame.f_code.co_filename
    if name.startswith(str(LEGACY)+'/'):
        raise RuntimeError('DIRTY_LEGACY_EXECUTION_FORBIDDEN: '+name)
    if name.startswith(str(SNAP)+'/'):
        rel=str(Path(name).relative_to(SNAP))
        if Path(rel).name not in HISTORICAL_ALLOWED and not rel.startswith('tests/ling/'):
            raise RuntimeError('UNRESOLVED_HISTORICAL_EXECUTED_DEPENDENCY: '+rel)
        if frame.f_code.co_name in ['main','run_unit','run_unit_panel','run_stage','load_model_and_tokenizer']:
            raise RuntimeError('HISTORICAL_MAIN_OR_SCIENTIFIC_RUNNER_FORBIDDEN')
        PROJECT_DEPS.add(rel)
    if name.startswith('/') and name not in EXECUTED:EXECUTED.add(name)

def historical():
    os.environ['LING_REPO_ROOT']=str(SNAP)
    q=imp(SNAP/'experiments/ling/run_ling_kda_query_readout_causal_closure_v1.py','bridge_historical_query')
    r=imp(SNAP/'experiments/ling/pytorch_kda_reference.py','bridge_historical_reference')
    f=imp(SNAP/'experiments/ling/run_ling_kda_source_error_functional_causal_v1.py','bridge_historical_functional')
    assert Path(f.P.__file__).resolve()==(SNAP/'experiments/ling/run_ling_kda_persistent_error_decomposition_causal_v1.py').resolve()
    return q,r,f

def modules():
    co=imp(OLD_N/'scripts/coherence.py','bridge_parent_coherence')
    t,c,cap,core,stats=co.modules()
    c.D=D;c.R=ROOT;c.TASK=TASK;cap.D=D;cap.R=ROOT;cap.TASK=TASK
    return co,t,c,cap,core,stats

def guard():
    assert not (D/'STOP.json').exists(),'PRESERVED_STOP'
    receipt=read(D/'premeasurement_commit.json')
    assert run(['git','rev-parse','HEAD'],ROOT)==receipt['commit'],'HEAD_CHANGED'
    assert run(['git','status','--porcelain=v1','--untracked-files=no'],ROOT)=='','TRACKED_SOURCE_DIRTY'
    for f in read(D/'premeasurement_sha256.json')['files']:
        p=relocate(f['path']);assert sha(p)==f['sha256'],('PREMEASUREMENT_SOURCE_CHANGED',str(p))
    for f in read(D/'historical_assets/asset_manifest.json')['files']:
        p=relocate(f['snapshot_path']);assert sha(p)==f['sha256'],('HISTORICAL_SNAPSHOT_CHANGED',str(p))
    return receipt

def hardware(torch):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
    assert run(['hostname'])=='nlpg-SYS-4029GP-TRT'
    out=run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'])
    g=[l.split(',') for l in out.splitlines() if l.split(',')[0].strip()=='6'][0]
    assert g[1].strip()==UUID and '3090' in g[2]
    assert torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name(0)
    assert os.environ['LING_GATED_RMSNORM_FIXED_CONFIG']=='BT16_W8'
    assert os.environ['PYTHONHASHSEED']=='0'
    # Do not touch unrelated processes. A concurrent occupant fails the assignment.
    processes=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])
    occupied=[l for l in processes.splitlines() if UUID in l]
    assert all(int(l.split(',')[1])==os.getpid() for l in occupied),('GPU6_CONCURRENT_OCCUPANT',occupied)
    return dict(host=run(['hostname']),physical_gpu=6,gpu_uuid=UUID,driver=g[5].strip(),CUDA=torch.version.cuda,
         python=sys.version,torch=torch.__version__,processes=occupied,
         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
         cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
         matmul_tf32=torch.backends.cuda.matmul.allow_tf32,environment={k:os.environ.get(k) for k in
             ['CUDA_VISIBLE_DEVICES','LING_GATED_RMSNORM_FIXED_CONFIG','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','FLA_CONFIG_DIR']})

def execution_inventory():
    paths=set(EXECUTED)
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and Path(p).is_file():paths.add(p)
    files=[]
    for p in sorted(paths):
        if not Path(p).is_file():continue
        if str(p).startswith(str(LEGACY)+'/'):raise RuntimeError('LEGACY_MODULE_EXECUTED')
        files.append(row(p))
    return dict(historical_executed_files=sorted(PROJECT_DEPS),execution_module_files=files,
                bytecode_writes=False,old_dirty_directory_executed=False)

def verify():
    guard();co,t,c,cap,core,stats=modules()
    for f in read(D/'source_identity.json')['inputs']:
        p=t.mapped(f['path']);assert sha(p)==f['sha256'],('INHERITED_RUNTIME_INPUT_CHANGED',str(p))
    docs=read(D/'document_manifest.json')['documents'];bank=read(D/'rotation_manifest.json')['bank']
    for d in docs:assert sha(t.mapped(d['token_file']['path']))==d['token_file']['sha256']
    for r in bank['conditions']:assert sha(t.mapped(r['artifact']['path']))==r['artifact']['sha256']
    assert sha(c.V4/'runtime/v4_deterministic_runtime_manifest.json')=='0fd12303bdb54c6e54ee72963f0fc78f6e1791c5a807869fca9cf7b5d723dbb4'
    rt=c.runtime();torch=core.torch
    free=shutil_disk_usage=__import__('shutil').disk_usage(D).free
    assert free>20*1024**3,('STORAGE_CAPACITY_FAIL',free)
    hw=hardware(torch)
    dump(D/'smoke/preflight.json',dict(gate='PASS',SOURCE_IDENTITY_GATE='PASS',MODEL_CHECKPOINT_GATE='PASS',
         DOCUMENT_IDENTITY_GATE='PASS',ROTATION_BANK_IDENTITY_GATE='PASS',RUNTIME_IDENTITY_GATE='PASS',
         HARDWARE_ASSIGNMENT_GATE='PASS',STORAGE_CAPACITY_GATE='PASS',hardware=hw,runtime=rt,free_bytes=free,
         HISTORICAL_ASSET_BYTE_IDENTITY_GATE='PASS',formal_authorized=False,measurement_count=0))
    print('PREFLIGHT_PASS_NO_DISCOVERY',flush=True)

def sq(x):return x.double().square().sum()

def k3(query,reference,state,rec):
    historical_rec=dict(q=rec['q'],k=rec['k'],v=rec['v'],g=rec['log_decay'],beta=rec['beta'],
        use_qk_l2norm_in_kernel=True,use_gate_in_kernel=False,use_beta_sigmoid_in_kernel=False,state_v_first=False)
    return query.query_readout(state,query.prepared_query(reference,historical_rec))

def target_step(capture,cache,token,t):
    torch=__import__('torch')
    capture.current={};capture.post={};capture.probe.begin('FP','native',None)
    try:
        with torch.inference_mode():
            out=capture.model(input_ids=torch.tensor([[token]],device=capture.device),
                attention_mask=torch.ones((1,t+1),device=capture.device,dtype=torch.long),
                past_key_values=cache,cache_position=torch.tensor([t],device=capture.device),use_cache=True)
    finally:capture.probe.end()
    cache=out.past_key_values
    assert sorted(capture.current)==LAYERS
    for l in LAYERS:assert capture.P2.get_state(cache,l).data_ptr()==capture.post[l].data_ptr()
    return cache,out.logits,capture.current

def pulse_target(capture,functional,ids,anchor,end,actual_q,keep_diagnostics=False):
    torch=__import__('torch')
    ref_cache=None
    for t in range(anchor):ref_cache,_,_=target_step(capture,ref_cache,ids[t],t)
    fp_stack=functional.cache_stack(ref_cache,LAYERS)
    before={str(l):tensor_sha(v) for l,v in fp_stack.items()}
    pulse_cache=functional.clone_cache(ref_cache)
    functional.replace_cache_stack(pulse_cache,{l:actual_q[i:i+1] for i,l in enumerate(LAYERS)})
    # Exact only-KDA-state substitution: unchanged non-KDA keys and every value/
    # short-convolution cache. No FP+E reconstruction is permitted.
    def digest(v):
        if torch.is_tensor(v):return tensor_sha(v)
        if isinstance(v,(tuple,list)):return [digest(x) for x in v]
        return None if v is None else repr(v)
    for l in range(len(ref_cache.layers)):
        a,b=ref_cache.layers[l],pulse_cache.layers[l]
        assert digest(a.values)==digest(b.values),('NON_KDA_OR_CONV_CACHE_CHANGED',l)
        if l not in LAYERS:assert digest(a.keys)==digest(b.keys)
        else:assert torch.equal(functional.P.get_cache_state(pulse_cache,l),actual_q[LAYERS.index(l):LAYERS.index(l)+1])
    fp_anchor=torch.cat([fp_stack[l] for l in LAYERS])
    E=actual_q-fp_anchor
    got=torch.cat([functional.P.get_cache_state(pulse_cache,l)-functional.P.get_cache_state(ref_cache,l) for l in LAYERS])
    assert torch.equal(got,E),'INITIAL_PULSE_IDENTITY_FAIL'
    auc=torch.zeros((),dtype=torch.float64,device='cpu')
    for t in range(anchor,end):
        ref_cache,ref_logits,_=target_step(capture,ref_cache,ids[t],t)
        pulse_cache,pulse_logits,_=target_step(capture,pulse_cache,ids[t],t)
        value=functional.full_logit_metrics(torch,ref_logits,pulse_logits)['KL']
        assert math.isfinite(value),'NONFINITE_TARGET_KL'
        auc+=value
        del ref_logits,pulse_logits
        if (t+1)%512==0:print('TARGET_PROGRESS',anchor,t+1,end,flush=True)
    result=dict(TARGET_FUTURE_KL_AUC=float(auc),E_anchor_sha256=tensor_sha(E),FP_anchor_hashes=before,
         FP_anchor_sha256=tensor_sha(fp_anchor),INITIAL_PULSE_IDENTITY_GATE='PASS',
         FP_REFERENCE_PAIRING_GATE='PASS',NO_FURTHER_QDQ_GATE='PASS',teacher_tokens=end-anchor)
    del ref_cache,pulse_cache,fp_stack;capture.current={};capture.post={};gc.collect();torch.cuda.empty_cache()
    return result

def replay(trace,condition,t,c,core,q,r,anchors,end,out_root=None):
    torch=core.torch
    R=torch.load(t.mapped(condition['artifact']['path']),map_location='cuda',weights_only=False)
    R=torch.stack([R['rotations'][l] for l in LAYERS])
    # The bank stores per-layer matrices, broadcast only on actual head dimension.
    assert R.shape==(18,128,128),('ROTATION_LAYOUT',R.shape)
    RT=R[:,None].transpose(-1,-2)
    K=8;signs=core.sign_schedule(K,end).to('cuda')
    fp,quant,z,decay=core.initial_states(K,'cuda')
    historic=torch.zeros((5,18),device='cuda',dtype=torch.float64);n=0
    states={};scores={};paths={};sums={};largest={}
    with torch.inference_mode():
        for entry in trace['chunks']:
            a=entry['artifact'];assert sha(a['path'])==a['sha256'],'FROZEN_DRIVE_CHANGED'
            x=torch.load(a['path'],map_location='cuda',weights_only=False)
            for j in range(x['q'].shape[1]):
                token=entry['start_zero_based']+j+1
                if token>end:break
                rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                fp,quant,z,decay,metrics=core.replay_step(fp,quant,z,decay,rec,R,signs[token-1])
                if token>=1024:historic+=metrics.double();n+=1
                for anchor in anchors:
                    if token>anchor:
                        A,B=paths[anchor]
                        A=core.update_pre_q(A,rec);B=core.update_pre_q(B,rec)
                        delta=B-A;visible=k3(q,r,B,rec)-k3(q,r,A,rec)
                        pp=sq(delta);por=sq(visible)
                        assert bool(torch.isfinite(pp) and torch.isfinite(por))
                        sums[anchor][0]+=pp;sums[anchor][1]+=por;paths[anchor]=(A,B)
                        event=dict(token=token,state_energy=float(pp),k3_energy=float(por))
                        largest[anchor].append(event)
                        largest[anchor]=sorted(largest[anchor],key=lambda e:(-e['k3_energy'],e['token']))[:5]
                    elif token==anchor:
                        actual=quant.matmul(RT);E=actual-fp
                        A=fp.clone();B=actual.clone();assert torch.equal(B-A,E)
                        states[anchor]=dict(Q=actual.detach().cpu(),FP=fp.detach().cpu(),E_hash=tensor_sha(E))
                        scores[anchor]=dict(RAW_ACC_ERROR=float(sq(E)),LOCAL_K3_ACCUMULATED=float(sq(k3(q,r,E,rec))),
                            HISTORICAL_J_PROP=float((historic[3]/n).mean()) if n else None,
                            J_INJ=float((historic[1]/n).mean()) if n else None,
                            J_ACC=float((historic[0]/n).mean()) if n else None,E_anchor_sha256=tensor_sha(E))
                        paths[anchor]=(A,B);sums[anchor]=torch.zeros(2,dtype=torch.float64,device='cuda');largest[anchor]=[]
                if token==end:
                    E=quant.matmul(RT)-fp
                    endpoint=dict(RAW_ACC_ERROR_32K=float(sq(E)),LOCAL_K3_ACCUMULATED_32K=float(sq(k3(q,r,E,rec))),
                        J_INJ=float((historic[1]/n).mean()) if n else None,HISTORICAL_J_PROP=float((historic[3]/n).mean()) if n else None,
                        J_ACC=float((historic[0]/n).mean()) if n else None,FP=tensor_sha(fp),Q=tensor_sha(quant))
            assert torch.equal(fp,x['fp_after']),'FP_CAPTURE_REPLAY_BITWISE_FAIL'
            if entry['end_one_based']%2048==0:print('EXACT_REPLAY_PROGRESS',condition['condition_id'],entry['end_one_based'],flush=True)
            del x
    for anchor in anchors:
        scores[anchor].update(PULSE_PROP_ENERGY_AUC=float(sums[anchor][0]),POR=float(sums[anchor][1]),
            POR_GAIN=float(sums[anchor][1])/scores[anchor]['RAW_ACC_ERROR'] if scores[anchor]['RAW_ACC_ERROR'] else 0.,
            top_K3_events=largest[anchor],future_tokens=end-anchor)
    del fp,quant,z,decay,paths,sums;gc.collect();torch.cuda.empty_cache()
    return states,scores,endpoint

def smoke(repeat):
    guard();assert read(D/'smoke/preflight.json')['gate']=='PASS'
    sys.setprofile(profile)
    co,t,c,cap,core,stats=modules();q,r,f=historical();torch=core.torch
    hw=hardware(torch);out=D/'smoke'/f'repeat{repeat}';assert not out.exists();out.mkdir()
    # Original historical fixture functions are executed from byte-identical copies.
    test=imp(SNAP/'tests/ling/test_ling_kda_query_readout_causal_closure.py','bridge_old_query_tests')
    test.test_query_readout_matches_actual_kda_tensor_orientation()
    kt=imp(SNAP/'tests/ling/test_ling_kda_kernel_internal_directional_geometry_decomposition.py','bridge_old_kernel_tests')
    kt.test_one_step_reference_exposes_named_kernel_checkpoints()
    kt.test_exact_state_error_transfer_matches_finite_difference()
    fixture_source=c.V3/'inputs/traces/wikitext2raw-train-000.pt'
    ids=[int(v) for v in torch.load(fixture_source,map_location='cpu',weights_only=False)['token_ids'][:128]]
    model=cap.Capture()
    try:trace=model.document(ids,out/'trace',block=32,keep_step_hashes=True)
    finally:model.close()
    bank=read(D/'rotation_manifest.json')['bank'];condition=bank['conditions'][0]
    states,scores,endpoint=replay(trace,condition,t,c,core,q,r,[64],128)
    model=cap.Capture()
    try:target=pulse_target(model,f,ids,64,128,states[64]['Q'].to('cuda'))
    finally:model.close()
    assert target['E_anchor_sha256']==scores[64]['E_anchor_sha256'],'FP_REFERENCE_REPLAY_PAIRING_MISMATCH'
    assert target['FP_anchor_sha256']==tensor_sha(states[64]['FP']),'SAME_GPU_FP_ANCHOR_MISMATCH'
    # Independent import of identical historical implementation, not a new formula.
    f2=imp(SNAP/'experiments/ling/run_ling_kda_source_error_functional_causal_v1.py','bridge_historical_functional_parity')
    logits=torch.arange(41,dtype=torch.float32,device='cuda').reshape(1,1,-1)/41
    branch=logits.flip(-1)
    assert f.full_logit_metrics(torch,logits,branch)==f2.full_logit_metrics(torch,logits,branch)
    deps=execution_inventory();sys.setprofile(None)
    dump(out/'result.json',dict(gate='PASS',scores=scores,target=target,endpoint=endpoint,
         FP_step_hashes=trace['step_hashes'],hardware=hw,dependencies=deps,
         historical_fixture_tests='PASS',HISTORICAL_K3_PARITY_GATE='PASS',FUTURE_KL_HARNESS_PARITY_GATE='PASS',
         INITIAL_PULSE_IDENTITY_GATE='PASS',FP_REFERENCE_PAIRING_GATE='PASS'))
    print('FRESH_PROCESS_SMOKE_COMPLETE',repeat,flush=True)

def parity():
    guard();sys.setprofile(profile)
    co,t,c,cap,core,stats=modules();q,r,f=historical();torch=core.torch
    trace=read(D/'smoke/repeat0/trace/trace_freeze.json')
    bank=read(D/'rotation_manifest.json')['bank']
    x=torch.load(trace['chunks'][0]['artifact']['path'],map_location='cuda',weights_only=False)
    rec={k:x[k][:,0:1].contiguous() for k in ['q','k','v','beta','log_decay']}
    state=torch.arange(18*16*128*128,device='cuda',dtype=torch.float32).reshape(18,16,128,128)/1000000
    readout=k3(q,r,state,rec)
    errors=[]
    for cond in bank['conditions']:
        loaded=torch.load(t.mapped(cond['artifact']['path']),map_location='cpu',weights_only=False)
        R=torch.stack([loaded['rotations'][l] for l in LAYERS]).cuda()
        err=float((R.double().transpose(-1,-2)@R.double()-torch.eye(128,device='cuda',dtype=torch.float64)).abs().max())
        assert err<=bank['orthogonality_max_abs_threshold']
        got=q.query_readout(state.matmul(R[:,None]),q.prepared_query(r,dict(q=rec['q'],k=rec['k'],v=rec['v'],g=rec['log_decay'],beta=rec['beta'],use_qk_l2norm_in_kernel=True))).matmul(R[:,None].transpose(-1,-2))
        relative=float((got-readout).double().square().sum().sqrt()/readout.double().square().sum().sqrt())
        assert relative<=f.IDENTITY_TOL,('COMMON_FUNCTIONAL_BASIS_PARITY_FAIL',cond['condition_id'],relative)
        errors.append(dict(condition=cond['condition_id'],orthogonality_max_abs=err,functional_basis_relative_l2=relative))
    # Historical J_PROP call parity: exact original replay_step is used. Compare
    # all five fields to an independent import of the unchanged source module.
    core2=imp(Path(core.__file__),'bridge_historical_replay_parity')
    R=torch.stack([torch.load(t.mapped(bank['conditions'][0]['artifact']['path']),map_location='cpu',weights_only=False)['rotations'][l] for l in LAYERS]).cuda()
    left=core.initial_states(8,'cuda');right=core2.initial_states(8,'cuda')
    signs=core.sign_schedule(8,32).cuda()
    for j in range(32):
        rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
        a=core.replay_step(*left,rec,R,signs[j]);b=core2.replay_step(*right,rec,R,signs[j])
        assert all(torch.equal(u,v) for u,v in zip(a,b)),('HISTORICAL_JPROP_BITWISE_PARITY_FAIL',j)
        left=a[:4];right=b[:4]
    dump(D/'smoke/basis_and_jprop_parity.json',dict(BASIS_SEMANTICS_GATE='PASS',HISTORICAL_JPROP_PARITY_GATE='PASS',
         KDA_TRANSITION_IDENTITY_GATE='PASS',QDQ_IDENTITY_GATE='PASS',basis_rows=errors,identity_tolerance=f.IDENTITY_TOL,
         JPROP_parity='bitwise; every state plus all five fields; 32 exact historical development transitions',
         dependencies=execution_inventory()))
    sys.setprofile(None);print('BASIS_AND_JPROP_PARITY_PASS',flush=True)

def discovery():
    receipt=guard();assert read(D/'smoke/ALL_BLOCKING_GATES.json')['gate']=='PASS'
    co,t,c,cap,core,stats=modules();q,r,f=historical();torch=core.torch;hardware(torch)
    # Freeze the entire allowed executed source set established by blocking smokes.
    dependencies=read(D/'historical_assets/executed_dependency_closure.json')['all_module_files']
    for i in dependencies:assert sha(i['path'])==i['sha256']
    dump(D/'discovery/STARTED.json',dict(premeasurement_commit=receipt['commit'],time_unix=time.time(),units=128,
         physical_gpu=6,gpu_uuid=UUID,FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO'))
    docs=read(D/'document_manifest.json')['documents'];bank=read(D/'rotation_manifest.json')['bank']
    for doc in docs:
        guard();doc_id=doc['document_id'];path=t.mapped(doc['token_file']['path'])
        assert sha(path)==doc['token_file']['sha256'];ids=read(path)
        assert len(ids)==32769
        assert hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids)).hexdigest()==doc['token_ids_sha256']
        tracepath=OLD_N/'discovery/traces'/doc_id/'trace_freeze.json';trace=read(tracepath)
        assert trace['token_sha256']==hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids[:32768])).hexdigest()
        assert trace['tokens']==32768 and trace['layer_ids']==LAYERS
        dump(D/'discovery/summaries'/(doc_id+'_reused_trace.json'),dict(trace=row(tracepath),files=[e['artifact'] for e in trace['chunks']],
             FP_reference_origin='same GPU6 inherited coherence capture; cross-hardware references not used'))
        for condition in bank['conditions']:
            guard();condition_id=condition['condition_id']
            started=D/'discovery/logs'/(doc_id+'__'+condition_id+'.started.json')
            dump(started,dict(time_unix=time.time(),condition=condition_id,document=doc_id))
            sys.setprofile(profile)
            states,scores,endpoint=replay(trace,condition,t,c,core,q,r,[8192,2048],32768)
            sys.setprofile(None)
            for anchor in [8192,2048]:
                model=cap.Capture()
                try:target=pulse_target(model,f,ids,anchor,32768,states[anchor]['Q'].to('cuda'))
                finally:model.close()
                assert target['E_anchor_sha256']==scores[anchor]['E_anchor_sha256'],'TARGET_FP_REFERENCE_OR_PULSE_MISMATCH'
                assert target['FP_anchor_sha256']==tensor_sha(states[anchor]['FP']),'TARGET_FP_ANCHOR_MISMATCH'
                result=dict(gate='PASS',task=TASK,document=doc_id,condition=condition_id,anchor=anchor,endpoint=32768,
                     evidence='EXPLORATORY_DISCOVERY_ONLY',scores=scores[anchor],target=target,
                     hardware=dict(physical_gpu=6,gpu_uuid=UUID),premeasurement_commit=receipt['commit'],
                     token_sha256=doc['token_ids_sha256'],rotation_sha256=condition['artifact']['sha256'])
                dest=D/'discovery/unit_results'/(doc_id+'__'+condition_id+'__'+str(anchor)+'.json')
                dump(dest,result);dump(dest.with_suffix('.sha256.json'),row(dest));dest.chmod(0o444)
                print('ANCHOR_UNIT_COMPLETE',doc_id,condition_id,anchor,flush=True)
            dump(D/'discovery/summaries'/(doc_id+'__'+condition_id+'__endpoint32K.json'),endpoint)
    paths=[p for p in sorted((D/'discovery/unit_results').glob('*.json')) if not p.name.endswith('.sha256.json')]
    assert len(paths)==128
    dump(D/'discovery/raw_freeze.json',dict(gate='PASS',complete=128,files=[row(p) for p in paths],before_aggregation=True))
    print('DISCOVERY_128_UNITS_COMPLETE',flush=True)

def csvfile(name,rows):
    path=D/'analysis'/name;path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze():
    guard();co,t,c,cap,core,stats=modules();frozen=read(D/'discovery/raw_freeze.json');assert frozen['complete']==128
    rows=[];panels=[]
    for item in frozen['files']:
        assert sha(item['path'])==item['sha256'];a=read(item['path'])
        sc={k:v for k,v in a['scores'].items() if isinstance(v,(int,float)) or v is None}
        rows.append(dict(document=a['document'],condition=a['condition'],anchor=a['anchor'],**sc,TARGET_FUTURE_KL_AUC=a['target']['TARGET_FUTURE_KL_AUC']))
    names=['RAW_ACC_ERROR','LOCAL_K3_ACCUMULATED','HISTORICAL_J_PROP','PULSE_PROP_ENERGY_AUC','POR']
    docs=[d['document_id'] for d in read(D/'document_manifest.json')['documents']]
    order=read(D/'rotation_manifest.json')['bank']['execution_order']
    for anchor in [8192,2048]:
        for doc in docs:
            rr={r['condition']:r for r in rows if r['anchor']==anchor and r['document']==doc};assert set(rr)==set(order)
            y=[rr[k]['TARGET_FUTURE_KL_AUC'] for k in order];p=dict(document=doc,anchor=anchor)
            for name in names:
                x=[rr[k][name] for k in order]
                rho=stats.spearman(x,y);assert rho is not None,('UNDEFINED_PRIMARY_RANKING',doc,anchor,name)
                p[name+'_rho']=rho;p[name+'_concordance']=stats.order(x,y)['concordance']
                for label,fun in [('top',max),('bottom',min)]:
                    A={i for i,v in enumerate(x) if v==fun(x)};B={i for i,v in enumerate(y) if v==fun(y)}
                    p[name+'_'+label+'_agreement']=len(A&B)/len(A|B)
            p['delta_rho_POR_minus_PROP']=p['POR_rho']-p['PULSE_PROP_ENERGY_AUC_rho']
            p['delta_concordance_POR_minus_PROP']=p['POR_concordance']-p['PULSE_PROP_ENERGY_AUC_concordance']
            panels.append(p)
    csvfile('per_rotation_anchor_metrics.csv',rows);csvfile('per_document_ranking.csv',panels)
    summaries={}
    for anchor,name in [(8192,'primary_8k_to_32k.json'),(2048,'secondary_2k_to_32k.json')]:
        a=[p for p in panels if p['anchor']==anchor]
        med={k:statistics.median(p[k] for p in a) for k in a[0] if k not in ['anchor','document']}
        nr=sum(p['delta_rho_POR_minus_PROP']>0 for p in a);nc=sum(p['delta_concordance_POR_minus_PROP']>0 for p in a)
        present=nr>=3 and nc>=3 and med['delta_rho_POR_minus_PROP']>0 and med['POR_rho']>0
        label='DESCRIPTIVE_SIGNAL_PRESENT' if present else 'DESCRIPTIVE_SIGNAL_MIXED' if med['delta_rho_POR_minus_PROP']>0 else 'DESCRIPTIVE_SIGNAL_NOT_PRESENT'
        summary=dict(anchor=anchor,complete=64,document_panels=a,medians=med,POR_beats_PROP_rho_docs=nr,POR_beats_PROP_concordance_docs=nc,label=label,
             primary=anchor==8192,pooled_rotation_level_primary_pvalue='NOT_COMPUTED')
        dump(D/'analysis'/name,summary);summaries[anchor]=summary
    label=summaries[8192]['label']
    nextstep={'DESCRIPTIVE_SIGNAL_PRESENT':'DESIGN_PROSPECTIVE_LONG_HORIZON_OBSERVABILITY_CONFIRMATION',
         'DESCRIPTIVE_SIGNAL_MIXED':'STOP_OBSERVABILITY_ADVANCEMENT_PENDING_REVIEW',
         'DESCRIPTIVE_SIGNAL_NOT_PRESENT':'CONSIDER_REQUANTIZATION_FREQUENCY_CAUSAL_DIAGNOSIS'}[label]
    report=dict(EXPERIMENT=TASK,PREREGISTRATION_GATE='PASS',PROVENANCE_GATE='PASS',**read(D/'smoke/ALL_BLOCKING_GATES.json')['gates'],
         DISCOVERY_PRIMARY_COMPLETE='64/64',DISCOVERY_SECONDARY_COMPLETE='64/64',OBSERVABILITY_DISCOVERY_LABEL=label,
         PRIMARY_NEXT_STEP_CATEGORY=nextstep,CONFIRMATION_EXECUTED=0,NEW_ROTATION_TRAINING_EXECUTED='NO',L9_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',
         FINAL_ACTION='STOP_AFTER_DISCOVERY',primary=summaries[8192],secondary=summaries[2048])
    dump(D/'FINAL_REPORT.json',report)
    with (D/'FINAL_REPORT.md').open('x') as f:
        f.write('# '+TASK+'\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n\n')
        f.write('Discovery only. POR is constructed from future K3 readout energy; downstream full-logit KL, not K3 itself, is the primary target. No reasoning or confirmation claim is authorized.\n\n')
        f.write('No mixed precision, bit allocation, per-state adaptive precision, windowed quantization, or compensator token was used. All historical conclusions remain preserved.\n')
    with (D/'sha256_manifest.txt').open('x') as f:
        for p in sorted(D.rglob('*')):
            if p.is_file() and p.name!='sha256_manifest.txt' and '__pycache__' not in p.parts:f.write(sha(p)+'  '+str(p.relative_to(D))+'\n')
    print('FINAL_ACTION STOP_AFTER_DISCOVERY',label,flush=True)

def gate():
    guard();a=read(D/'smoke/repeat0/result.json');b=read(D/'smoke/repeat1/result.json')
    for k in ['scores','target','endpoint','FP_step_hashes']:assert a[k]==b[k],('FRESH_PROCESS_BITWISE_FAIL',k)
    expected=read(D/'historical_assets/executed_dependency_plan.json')['required_historical_modules']
    got={Path(p).name for p in a['dependencies']['historical_executed_files']}
    assert set(expected).issubset(got),('DEPENDENCY_CLOSURE_INCOMPLETE',set(expected)-got)
    assert a['dependencies']['historical_executed_files']==b['dependencies']['historical_executed_files']
    for item in a['dependencies']['execution_module_files']:assert sha(item['path'])==item['sha256']
    for item in b['dependencies']['execution_module_files']:assert sha(item['path'])==item['sha256']
    gates={g:'PASS' for g in read(D/'preregistration.json')['blocking_gates']}
    # These gates must have dedicated numerical evidence before acceptance. Do
    # not infer JPROP/basis parity from byte identity or merely a repeat PASS.
    assert read(D/'smoke/basis_and_jprop_parity.json')['BASIS_SEMANTICS_GATE']=='PASS'
    assert read(D/'smoke/basis_and_jprop_parity.json')['HISTORICAL_JPROP_PARITY_GATE']=='PASS'
    dump(D/'historical_assets/executed_dependency_closure.json',dict(gate='PASS',
         HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE='PASS',evidence=[row(D/'smoke/repeat0/result.json'),row(D/'smoke/repeat1/result.json')],
         historical_files=a['dependencies']['historical_executed_files'],all_module_files=a['dependencies']['execution_module_files'],
         callable_scope=read(D/'historical_assets/executed_dependency_plan.json')['roots'],old_mains_executed=False))
    dump(D/'smoke/ALL_BLOCKING_GATES.json',dict(gate='PASS',gates=gates,HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',
         proposed_hardware=read(D/'runtime_manifest.json')['proposed_hardware'],GPU6_FORMAL_AUTHORIZED=True,
         FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',cross_hardware_FP_bitwise='FAIL_PRESERVED'))
    print('ALL_BLOCKING_GATES_PASS',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','smoke','parity','gate','discovery','analyze']);p.add_argument('--repeat',type=int);a=p.parse_args()
    try:
        if a.mode=='verify':verify()
        elif a.mode=='smoke':smoke(a.repeat)
        else:globals()[a.mode]()
    except BaseException as e:
        sys.setprofile(None)
        if not (D/'STOP.json').exists():dump(D/'STOP.json',dict(stage=a.mode,error=repr(e),traceback=traceback.format_exc(),
             gate='FAIL',FINAL_ACTION='FAIL_AND_STOP',Discovery_primary_complete=0,Discovery_secondary_complete=0,
             no_auto_retry=True,L9_TRAINING_EXECUTED='NO',NEW_ROTATION_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO'))
        raise

if __name__=='__main__':main()
