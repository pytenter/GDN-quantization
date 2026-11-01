"""Append-only exact finite-transition Discovery; no training or generation."""
from pathlib import Path
import argparse, csv, gc, hashlib, importlib.util, json, math, os, shutil, statistics, subprocess, sys, time, traceback
sys.dont_write_bytecode = True
TASK = 'LING_ERROR_COHERENCE_DIAGNOSIS_V1'
D = Path(__file__).resolve().parents[1]
ROOT = D.parents[1]
SOURCE = Path('/data01/user2/worktrees/ling-error-coherence-diagnosis-v1')
TARGET = Path('/data/zypan/worktrees/ling-error-coherence-diagnosis-v1')
P = Path('/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1')
C = Path('/data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1')
OLD_REMOTE = Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1')
GPU = 6
UUID = 'GPU-de53f243-b271-521c-d875-d96611ed8574'
REMOTE = str(ROOT).startswith('/data/zypan/')
HORIZONS = [2048, 8192, 32768]
LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
KEYS = ['ACC','HISTORY','INJECTION','ALIGNMENT','POS_ALIGNMENT','NEG_ALIGNMENT','ABS_ALIGNMENT',
        'POS_COUNT','NEG_COUNT','ZERO_COUNT','COHERENCE','COSINE','PURE_QDQ','BASIS_DISCREPANCY']

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def row(p): return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(), ('EXCLUSIVE_OUTPUT_EXISTS',str(p))
    tmp=p.with_name(p.name+'.partial');assert not tmp.exists()
    with tmp.open('x') as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def textfile(p,s):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:f.write(s)
def run(a,cwd=None): return subprocess.check_output(a,cwd=cwd,text=True).strip()
def tensor_sha(x): return hashlib.sha256(x.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
def imp(p,n):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def stop(stage,e):
    f=D/'STOP.json'
    if not f.exists():dump(f,dict(stage=stage,error=repr(e),traceback=traceback.format_exc(),time_unix=time.time(),FINAL_ACTION='FAIL_AND_STOP',NO_AUTOMATIC_RETRY=True,CONFIRMATION_EXECUTED=0,L9_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO'))

def prepare():
    assert not REMOTE
    assert run(['git','branch','--show-current'],ROOT)=='exp/ling-error-coherence-diagnosis-v1'
    dirty=run(['git','status','--porcelain','--untracked-files=all'],ROOT)
    for line in dirty.splitlines(): assert line[3:].startswith('experiments/'+TASK+'/'),('UNRELATED_DIRTY_CHANGE',line)
    t=imp(C/'scripts/transport.py','coherence_transport_prepare')
    inputs=read(C/'transport/transport_inventory_v2.json')['files']
    for i in inputs: assert row(i['path'])==i,('INHERITED_IDENTITY_CHANGED',i['path'])
    # Protect the exact historical experiment roots, not unrelated live projects.
    protected=[]
    for root in [P,C]:
        for f in sorted(root.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts: protected.append(row(f))
    dump(D/'source_identity.json',dict(parent_commits={str(P.parents[1]):'dc602d2d2d3b78346b6e89d9b6a71a9ad9ddc53b',str(C.parents[1]):'caf2fff04505d74c1596fb90ec1378fed85b2fc7'},inputs=inputs,protected_historical_files=protected,scientific_source_changed=False))
    docs=[x for x in read(P/'protocol/frozen_documents.json')['streams'] if x['phase']=='DISCOVERY']
    assert [x['document_id'] for x in docs]==['pg19-test-37702','pg19-test-5396','pg19-test-28988','pg19-test-26493']
    for d in docs:
        d['historical_physical_gpu']=d['physical_gpu'];d['historical_gpu_uuid']=d['gpu_uuid']
        d['physical_gpu']=GPU;d['gpu_uuid']=UUID;d['evidence_role']='EXPLORATORY_MECHANISM_DISCOVERY_ONLY'
    dump(D/'discovery_manifest.json',dict(documents=docs,bank=read(P/'protocol/frozen_rotation_bank.json'),source_document_manifest=row(P/'protocol/frozen_documents.json'),source_rotation_manifest=row(P/'protocol/frozen_rotation_bank.json')))
    expected=read(P/'runtime/deterministic_runtime_identity.json')
    dump(D/'runtime_manifest.json',dict(expected=expected,launch_profile=read(P/'runtime/inherited_launch_profile.json')['launch_profile'],old_cross_hardware_FP_bitwise='FAIL_PRESERVED',new_hardware=dict(host='nlpg-SYS-4029GP-TRT',physical_gpu=GPU,gpu_uuid=UUID),new_measurements=0))
    md=row(D/'preregistration.md')
    dump(D/'preregistration.json',dict(EXPERIMENT=TASK,EVIDENCE_CATEGORY='PREREGISTERED_EXPLORATORY_MECHANISM_DIAGNOSIS',text_authority=md,docs=[d['document_id'] for d in docs],rotation_order=read(P/'protocol/frozen_rotation_bank.json')['execution_order'],horizons=HORIZONS,primary_horizon=32768,window_start=1024,state_shape=[18,16,128,128],state_dtype='float32',metric_dtype='float64',equations={'E':'Qnext_phys-FPnext','H':'F(Sq_rot@R.T,u)-FPnext','I':'Qnext_phys-F(Sq_rot@R.T,u)'},injection_scope='actual Q-minus-native-F; pure QDQ and basis discrepancy separately audited',closure=dict(atol=1e-10,relative_l2=1e-10,energy_scaled=1e-10),parent_metric_parity=dict(atol=5e-5,rtol=5e-5),tie_tolerance=0,zero_denominator=0,hardware=dict(gpu=GPU,uuid=UUID,amendment='USER_AUTHORIZED_SINGLE_GPU6_V1',user_authorization='那就只使用GPU6来完成指令吧',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO'),DISCOVERY_UNITS=64,CONFIRMATION_EXECUTED=0,L9_AUTHORIZED='NO',AIME_AUTHORIZED='NO',FINAL_ACTION='STOP_AFTER_DISCOVERY'))
    dump(D/'provenance.json',dict(source_inventory_sha256=sha(D/'source_identity.json'),discovery_manifest_sha256=sha(D/'discovery_manifest.json'),preregistration_sha256=sha(D/'preregistration.json'),parent_global_gate='FAIL_PRESERVED',prior32K_confirmation='SUPPORTED_PRESERVED_NOT_REPEATED',route='B_DETERMINISTIC_REPLAY_WITH_SAME_GPU_FP_CAPTURE',new_training=False,new_AIME=False,hardware_amendment_before_new_measurement=True))
    fixed=[row(f) for f in sorted(D.rglob('*')) if f.is_file()]
    dump(D/'premeasurement_sha256.json',dict(files=fixed,scientific_measurements=0))
    print('PREPARATION_PASS',len(inputs),len(protected),flush=True)

def modules():
    t=imp(OLD_REMOTE/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py','coherence_transport')
    # Only operational relocation/assignment; inherited source files remain unchanged.
    t.GPUS=[GPU];t.UUIDS={GPU:UUID}
    c,cap,core,_,stats=t.scientific_modules()
    c.D=D;c.R=ROOT;c.TASK=TASK;cap.D=D;cap.R=ROOT;cap.TASK=TASK
    return t,c,cap,core,stats

def fixed_guard():
    assert not (D/'STOP.json').exists(), 'PRESERVED_STOP'
    receipt=read(D/'premeasurement_commit.json')
    for f in read(D/'premeasurement_sha256.json')['files']:
        p=D/Path(f['path']).relative_to(SOURCE/'experiments'/TASK)
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],('NEW_SOURCE_CHANGED',str(p))
    assert sha(D/'premeasurement_sha256.json')==receipt['premeasurement_manifest_sha256']
    return receipt

def verify():
    receipt=fixed_guard();t,c,cap,core,stats=modules()
    for f in read(D/'source_identity.json')['inputs']:
        p=t.mapped(f['path']);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],('TRANSPORT_INPUT_CHANGED',str(p))
    c.runtime();hw=c.hardware();selected=[g for g in hw['gpus'] if g['physical_gpu']==GPU]
    assert hw['host']=='nlpg-SYS-4029GP-TRT' and len(selected)==1 and selected[0]['gpu_uuid']==UUID
    runtime=dict(authority=read(D/'runtime_manifest.json'),actual=c.runtime(),hardware=hw,source_commit=receipt['commit'],source_inventory_sha256=sha(D/'source_identity.json'),gate='PASS',new_forward=False)
    dump(D/'smoke/runtime_preflight.json',runtime)
    print('RUNTIME_INPUT_PREFLIGHT_PASS',len(read(D/'source_identity.json')['inputs']),flush=True)

def launch_guard(c,torch):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(GPU) and torch.cuda.device_count()==1
    hw=c.hardware();g=next(x for x in hw['gpus'] if x['physical_gpu']==GPU)
    assert hw['host']=='nlpg-SYS-4029GP-TRT' and g['gpu_uuid']==UUID and '3090' in g['name']
    assert g['used_mib']<2000,('EXTERNAL_GPU_OCCUPANCY',g)
    return hw

def storage(remaining):
    available=shutil.disk_usage(D).free
    required=remaining*13325495104+2*1024**3; reserve=20*1024**3
    assert available>=required+reserve,('INSUFFICIENT_STORAGE',available,required,reserve)
    return dict(required_bytes=required,available_bytes=available,safety_margin_bytes=reserve)

def measure(doc,cond,trace,smoke=False):
    receipt=fixed_guard();t,c,cap,core,stats=modules();torch=core.torch
    art=cond['artifact'];p=t.mapped(art['path']);assert sha(p)==art['sha256']
    R=torch.stack([torch.load(p,map_location='cpu',weights_only=False)['rotations'][l] for l in LAYERS]).cuda()
    RT=R[:,None].transpose(-1,-2);K=8
    signs=torch.load(t.mapped(P/'protocol/probe_signs.pt'),map_location='cuda',weights_only=True)
    assert tensor_sha(signs.cpu())==tensor_sha(core.sign_schedule(K,32768))
    fp,quant,z,decay=core.initial_states(K,'cuda')
    total=torch.zeros((len(KEYS),18),dtype=torch.float64,device='cuda');raw_total=torch.zeros_like(total)
    parent_sum=torch.zeros((5,18),dtype=torch.float64,device='cuda')
    closure=torch.zeros((4,18),dtype=torch.float64,device='cuda')
    segments={str(i):dict(weighted=torch.zeros_like(total),raw=torch.zeros_like(total),n=0,primary=torch.zeros_like(total),primary_n=0) for i in range(3)}
    scores={};windows=[];n=0;start=time.time()
    with torch.inference_mode():
        for entry in trace['chunks']:
            fixed_guard();a=entry['artifact'];assert sha(a['path'])==a['sha256']
            x=torch.load(a['path'],map_location='cuda',weights_only=False)
            window=torch.zeros_like(total);raw_window=torch.zeros_like(total);event_rows=[];historic=torch.zeros((5,18),dtype=torch.float64,device='cuda')
            for j in range(x['q'].shape[1]):
                token=entry['start_zero_based']+j+1
                rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                old_quant=quant.clone()
                old_native=old_quant.matmul(RT)
                native_q_pre=core.update_pre_q(old_native,rec)
                v=rec['v'].float().matmul(R[:,None]).to(rec['v'].dtype)
                matched_rot_pre=core.update_pre_q(old_quant,rec,v)
                # Exact inherited main/shadow transitions; measurement does not replace cache.
                fp,quant,z,decay,parent_metrics=core.replay_step(fp,quant,z,decay,rec,R,signs[token-1])
                assert torch.equal(core.qdq_r128(matched_rot_pre)[0],quant),('MATCHED_NON_QDQ_ENDPOINT_PARITY_FAIL',token)
                qnext=quant.matmul(RT);f=fp.double();qn=qnext.double();fq=native_q_pre.double()
                E=qn-f;H=fq-f;I=qn-fq
                # Separate actual pure-QDQ and basis discrepancy without altering the path.
                recovered_pre=matched_rot_pre.matmul(RT).double()
                pure_qdq=qn-recovered_pre;basis_discrepancy=recovered_pre-fq
                norm=lambda a:a.square().sum(dim=(-3,-2,-1))
                e,h,i=norm(E),norm(H),norm(I);align=2*(H*I).sum(dim=(-3,-2,-1));base=h+i
                res=E-H-I;absmax=res.abs().amax(dim=(-3,-2,-1));resn=norm(res).sqrt()
                rel=torch.where(e>0,resn/e.sqrt(),torch.where(resn==0,torch.zeros_like(resn),torch.full_like(resn,float('inf'))))
                er=(e-h-i-align).abs();er_scaled=er/torch.maximum(torch.ones_like(er),e+h+i+align.abs())
                closure=torch.maximum(closure,torch.stack([absmax,rel,er,er_scaled]))
                assert bool(torch.isfinite(torch.stack([e,h,i,align,absmax,rel,er_scaled])).all())
                assert bool((absmax<=1e-10).all() and (rel<=1e-10).all() and (er_scaled<=1e-10).all()),('CLOSURE_FAIL',doc['document_id'],cond['condition_id'],token)
                coh=torch.where(base>0,align/base,torch.zeros_like(base));cosden=(h*i).sqrt();cos=torch.where(cosden>0,align/2/cosden,torch.zeros_like(cosden))
                assert bool((coh.abs()<=1+1e-10).all() and (cos.abs()<=1+1e-10).all())
                assert bool(((I-pure_qdq-basis_discrepancy).abs().amax()<=1e-10))
                vals=torch.stack([e,h,i,align,align.clamp_min(0),align.clamp_max(0),align.abs(),(align>0).double(),(align<0).double(),(align==0).double(),coh,cos,norm(pure_qdq),norm(basis_discrepancy)])
                den=norm(f)+1e-12;weighted=vals.clone();weighted[:7]/=den;weighted[12:]/=den
                s=0 if token<=2048 else 1 if token<=8192 else 2
                segments[str(s)]['weighted']+=weighted;segments[str(s)]['raw']+=vals;segments[str(s)]['n']+=1
                window+=weighted;raw_window+=vals;historic+=parent_metrics.double()
                event_rows.append(torch.stack([weighted[0],weighted[1],weighted[2],weighted[3]],dim=-1))
                if token>=1024 or smoke:
                    total+=weighted;raw_total+=vals;parent_sum+=parent_metrics.double();n+=1
                    segments[str(s)]['primary']+=weighted;segments[str(s)]['primary_n']+=1
                if token in HORIZONS or smoke and token==trace['tokens']:
                    av=total/n
                    assert torch.allclose(av[0],parent_sum[0]/n,atol=5e-5,rtol=5e-5),('PARENT_JACC_PARITY_FAIL',token)
                    scores[str(token)]=dict(n=n,per_layer=av.cpu().tolist(),weighted_sums=total.cpu().tolist(),raw_sums=raw_total.cpu().tolist(),parent_scores=(parent_sum/n).cpu().tolist())
            assert torch.equal(fp,x['fp_after']),('FP_CAPTURE_REPLAY_BITWISE_FAIL',entry['end_one_based'])
            ev=torch.stack(event_rows).cpu();flat=[dict(token=entry['start_zero_based']+jj+1,layer=LAYERS[ll],ACC=float(ev[jj,ll,0]),HISTORY=float(ev[jj,ll,1]),INJECTION=float(ev[jj,ll,2]),ALIGNMENT=float(ev[jj,ll,3])) for jj in range(len(ev)) for ll in range(18)]
            positive=sorted((r for r in flat if r['ALIGNMENT']>0),key=lambda r:(-r['ALIGNMENT'],r['token'],r['layer']))[:5]
            negative=sorted((r for r in flat if r['ALIGNMENT']<0),key=lambda r:(r['ALIGNMENT'],r['token'],r['layer']))[:5]
            windows.append(dict(start=entry['start_zero_based']+1,end=entry['end_one_based'],weighted=window.cpu().tolist(),raw=raw_window.cpu().tolist(),parent=historic.cpu().tolist(),top_positive=positive,top_negative=negative,closure_running_max=closure.cpu().tolist()))
            del x,ev,flat,event_rows
    result=dict(gate='PASS',task=TASK,schema=1,identity=dict(doc_id=doc['document_id'],token_ids_sha256=doc['token_ids_sha256'],rotation_sha256=art['sha256'],condition=cond['condition_id'],physical_gpu=GPU,gpu_uuid=UUID,premeasurement_commit=receipt['commit'],source_identity_sha256=sha(D/'source_identity.json'),runtime_manifest_sha256=sha(D/'runtime_manifest.json'),trace_sha256=sha(Path(trace['chunks'][0]['artifact']['path']).parent/'trace_freeze.json')),keys=KEYS,scores=scores,segments={k:dict(n=v['n'],primary_n=v['primary_n'],weighted=v['weighted'].cpu().tolist(),raw=v['raw'].cpu().tolist(),primary=v['primary'].cpu().tolist()) for k,v in segments.items()},windows=windows,closure=closure.cpu().tolist(),final_state_hashes={k:tensor_sha(v) for k,v in [('FP',fp),('Q',quant),('SHADOW',z)]},elapsed_seconds=time.time()-start,tokens=trace['tokens'],FP_CAPTURE_REPLAY_BITWISE='PASS',DECOMPOSITION_CLOSURE_GATE='PASS')
    return result

def smoke(repeat):
    fixed_guard();t,c,cap,core,stats=modules();hw=launch_guard(c,core.torch)
    assert read(D/'smoke/runtime_preflight.json')['gate']=='PASS'
    root=D/'smoke'/f'repeat{repeat}';assert not root.exists();root.mkdir()
    dev=c.V3/'inputs/traces/wikitext2raw-train-000.pt'
    ids=[int(v) for v in core.torch.load(dev,map_location='cpu',weights_only=False)['token_ids'][:128]]
    model=cap.Capture()
    try:trace=model.document(ids,root/'trace',block=32,keep_step_hashes=True)
    finally:model.close()
    bank=read(D/'discovery_manifest.json')['bank'];doc=dict(document_id='HISTORICAL_DEVELOPMENT_SMOKE',token_ids_sha256=trace['token_sha256'])
    r=measure(doc,bank['conditions'][0],trace,smoke=True)
    numeric={k:r[k] for k in ['scores','segments','windows','closure','final_state_hashes']}
    tensor_hashes=[]
    for entry in trace['chunks']:
        x=core.torch.load(entry['artifact']['path'],map_location='cpu',weights_only=False)
        tensor_hashes.append({k:tensor_sha(v) for k,v in x.items()})
    dump(root/'result.json',dict(gate='PASS',numeric=numeric,tensor_hashes=tensor_hashes,FP_step_hashes=trace['step_hashes'],hardware=hw,environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','LING_GATED_RMSNORM_FIXED_CONFIG','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','FLA_CONFIG_DIR']},torch_flags=dict(deterministic_algorithms=core.torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=core.torch.backends.cudnn.deterministic,cudnn_benchmark=core.torch.backends.cudnn.benchmark,matmul_tf32=core.torch.backends.cuda.matmul.allow_tf32),CUDA=core.torch.version.cuda,seed=0))
    print('SMOKE_COMPLETE',repeat,flush=True)

def smoke_gate():
    fixed_guard();a=read(D/'smoke/repeat0/result.json');b=read(D/'smoke/repeat1/result.json')
    for k in ['numeric','tensor_hashes','FP_step_hashes','environment','torch_flags','CUDA','seed']:assert a[k]==b[k],('FRESH_PROCESS_REPEAT_BITWISE_FAIL',k)
    dump(D/'smoke/GATE.json',dict(gate='PASS',SAME_GPU_FRESH_PROCESS_REPEAT_BITWISE='PASS',FP_CAPTURE_REPLAY_BITWISE='PASS',DECOMPOSITION_CLOSURE_GATE='PASS',physical_gpu=GPU,gpu_uuid=UUID,cross_hardware_FP_bitwise='FAIL_PRESERVED'))
    print('SMOKE_GATE_PASS',flush=True)

def formal():
    fixed_guard();t,c,cap,core,stats=modules();assert read(D/'smoke/GATE.json')['gate']=='PASS'
    data=read(D/'discovery_manifest.json');docs=data['documents'];bank=data['bank']
    dump(D/'discovery/STARTED.json',dict(commit=read(D/'premeasurement_commit.json')['commit'],time_unix=time.time(),units=64))
    for index,doc in enumerate(docs):
        fixed_guard();gc.collect();core.torch.cuda.empty_cache();launch_guard(c,core.torch)
        capacity=storage(4-index);dump(D/'discovery'/f'storage_doc{index}.json',capacity)
        for f in read(D/'source_identity.json')['inputs']:
            p=t.mapped(f['path']);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],('INPUT_CHANGED',str(p))
        path=t.mapped(doc['token_file']['path']);assert sha(path)==doc['token_file']['sha256']
        ids=read(path);assert len(ids)==32769
        assert hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids)).hexdigest()==doc['token_ids_sha256']
        trroot=D/'discovery/traces'/doc['document_id'];assert not trroot.exists(),('PARTIAL_TRACE_OR_DUPLICATE',str(trroot))
        model=cap.Capture()
        try:trace=model.document(ids[:32768],trroot)
        finally:model.close()
        for condition in bank['conditions']:
            target=D/'discovery/unit_results'/(doc['document_id']+'__'+condition['condition_id']+'.json')
            assert not target.exists(),('NO_DUPLICATE_UNIT',str(target))
            dump(target.with_suffix('.started.json'),dict(doc=doc['document_id'],condition=condition['condition_id'],time_unix=time.time(),commit=read(D/'premeasurement_commit.json')['commit']))
            result=measure(doc,condition,trace);dump(target,result);dump(target.with_suffix('.sha256.json'),row(target));target.chmod(0o444)
            print('UNIT_COMPLETE',doc['document_id'],condition['condition_id'],round(result['elapsed_seconds'],2),flush=True)
    paths=sorted((D/'discovery/unit_results').glob('*.json'));units=[p for p in paths if not p.name.endswith(('.sha256.json','.started.json'))];assert len(units)==64
    dump(D/'discovery/raw_freeze.json',dict(complete=64,files=[row(p) for p in units],before_statistics=True))
    print('DISCOVERY_RAW_FREEZE_PASS',flush=True)

def ratio(a,b):return a/b if b else 0.
def csvfile(name,rows):
    assert rows
    p=D/'analysis'/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(sums,n):
    a={k:float(sum(sums[i])) for i,k in enumerate(KEYS)};base=a['HISTORY']+a['INJECTION']
    return dict(J_ACC=ratio(a['ACC'],n*18),J_NOALIGN=ratio(base,n*18),SUM_HISTORY=a['HISTORY'],SUM_INJECTION=a['INJECTION'],SUM_ALIGNMENT=a['ALIGNMENT'],SUM_BASE=base,NET_COHERENCE_RATIO=ratio(a['ALIGNMENT'],base),POS_ALIGNMENT=a['POS_ALIGNMENT'],NEG_ALIGNMENT=a['NEG_ALIGNMENT'],ABS_ALIGNMENT=a['ABS_ALIGNMENT'],COHERENCE_BIAS=ratio(a['ALIGNMENT'],a['ABS_ALIGNMENT']),positive_alignment_fraction=ratio(a['POS_COUNT'],n*18),negative_alignment_fraction=ratio(a['NEG_COUNT'],n*18),zero_alignment_fraction=ratio(a['ZERO_COUNT'],n*18),mean_token_coherence=ratio(a['COHERENCE'],n*18),mean_cosine=ratio(a['COSINE'],n*18),SUM_PURE_QDQ=a['PURE_QDQ'],SUM_BASIS_DISCREPANCY=a['BASIS_DISCREPANCY'])

def analyze():
    fixed_guard();t,c,cap,core,stats=modules();data=read(D/'discovery_manifest.json');frozen=read(D/'discovery/raw_freeze.json');assert frozen['complete']==64
    assert read(D/'provenance/postmeasurement_parent_integrity.json')['gate']=='PASS'
    records=[]
    for f in frozen['files']:
        assert sha(f['path'])==f['sha256'];r=read(f['path']);assert r['gate']=='PASS' and r['DECOMPOSITION_CLOSURE_GATE']=='PASS';records.append(r)
    per=[];layers=[];segments=[];closures=[];pairs=[];rank=[]
    for r in records:
        identity=r['identity'];doc=identity['doc_id'];cond=identity['condition'];assert identity['physical_gpu']==GPU and identity['gpu_uuid']==UUID
        for h in HORIZONS:
            sc=r['scores'][str(h)];a=summarize(sc['weighted_sums'],sc['n']);a.update(J_INJ=statistics.mean(sc['parent_scores'][1]),J_PROP=statistics.mean(sc['parent_scores'][3]))
            per.append(dict(document_id=doc,condition=cond,horizon=h,**a))
            for j,l in enumerate(LAYERS):
                v=[[x[j]] for x in sc['weighted_sums']];s=summarize(v,sc['n']);s['J_ACC']*=18;s['J_NOALIGN']*=18
                # Per-layer fractions/means use one layer, not the global eighteen-layer count.
                for k in ['positive_alignment_fraction','negative_alignment_fraction','zero_alignment_fraction','mean_token_coherence','mean_cosine']:s[k]*=18
                layers.append(dict(document_id=doc,condition=cond,horizon=h,layer=l,**s))
        for segment,s in r['segments'].items():
            for support,key,nkey in [('FULL_SEGMENT_DIAGNOSTIC','weighted','n'),('PRIMARY_SUPPORT_INTERSECTION','primary','primary_n')]:segments.append(dict(document_id=doc,condition=cond,segment=segment,support=support,tokens=s[nkey],**summarize(s[key],s[nkey])))
        for j,l in enumerate(LAYERS):closures.append(dict(document_id=doc,condition=cond,layer=l,max_abs_residual=r['closure'][0][j],relative_L2_residual=r['closure'][1][j],max_energy_abs_residual=r['closure'][2][j],max_energy_scaled_residual=r['closure'][3][j]))
    order=data['bank']['execution_order']
    for d in data['documents']:
        for h in HORIZONS:
            rows={r['condition']:r for r in per if r['document_id']==d['document_id'] and r['horizon']==h};assert set(rows)==set(order)
            ys=[rows[k]['J_ACC'] for k in order];no=[rows[k]['J_NOALIGN'] for k in order]
            count=0;non=0
            for i in range(16):
                for j in range(i+1,16):
                    x=no[i]-no[j];y=ys[i]-ys[j];tie=(x==0 or y==0);flip=not tie and (x>0)!=(y>0);non+=not tie;count+=flip
                    pairs.append(dict(document_id=d['document_id'],horizon=h,rotation_i=order[i],rotation_j=order[j],noalign_difference=x,acc_difference=y,non_tied=not tie,decisive=flip,amplification_or_cancellation='NOT_CLASSIFIED',alignment_difference=rows[order[i]]['SUM_ALIGNMENT']/((h-1023)*18)-rows[order[j]]['SUM_ALIGNMENT']/((h-1023)*18)))
            a=dict(document_id=d['document_id'],horizon=h,J_INJ_rho=stats.spearman([rows[k]['J_INJ'] for k in order],ys),J_PROP_rho=stats.spearman([rows[k]['J_PROP'] for k in order],ys),J_NOALIGN_rho=stats.spearman(no,ys),J_NOALIGN_concordance=stats.order(no,ys)['concordance'],total_non_tied_pairs=non,alignment_decisive_pair_count=count,alignment_decisive_pair_rate=ratio(count,non),median_NET_COHERENCE_RATIO=statistics.median(rows[k]['NET_COHERENCE_RATIO'] for k in order),median_abs_NET_COHERENCE_RATIO=statistics.median(abs(rows[k]['NET_COHERENCE_RATIO']) for k in order))
            assert all(a[k] is not None for k in ['J_INJ_rho','J_PROP_rho','J_NOALIGN_rho']);rank.append(a)
    long=[r for r in rank if r['horizon']==32768];flips=sum(r['alignment_decisive_pair_count']>0 for r in long);mag=sum(r['median_abs_NET_COHERENCE_RATIO']>=.01 for r in long)
    label='DESCRIPTIVE_SIGNAL_PRESENT' if flips==4 and mag>=3 else 'DESCRIPTIVE_SIGNAL_NOT_PRESENT' if flips==0 and mag==0 else 'DESCRIPTIVE_SIGNAL_WEAK' if flips>0 and mag==0 else 'DESCRIPTIVE_SIGNAL_MIXED'
    summaries={str(h):{k:statistics.median(r[k] for r in rank if r['horizon']==h) for k in ['J_INJ_rho','J_PROP_rho','J_NOALIGN_rho','J_NOALIGN_concordance','alignment_decisive_pair_rate','median_NET_COHERENCE_RATIO']} for h in HORIZONS}
    closure_columns=['max_abs_residual','relative_L2_residual','max_energy_abs_residual','max_energy_scaled_residual']
    closure_summary=dict(global_max={k:max(r[k] for r in closures) for k in closure_columns},per_document={d['document_id']:{k:max(r[k] for r in closures if r['document_id']==d['document_id']) for k in closure_columns} for d in data['documents']})
    signed_by_horizon={str(h):{k:statistics.median(r[k] for r in per if r['horizon']==h) for k in ['positive_alignment_fraction','negative_alignment_fraction','POS_ALIGNMENT','NEG_ALIGNMENT','SUM_PURE_QDQ','SUM_BASIS_DISCREPANCY']} for h in HORIZONS}
    out=dict(EXPERIMENT=TASK,PREREGISTRATION_GATE='PASS',PROVENANCE_GATE='PASS',RUNTIME_IDENTITY_GATE='PASS',DECOMPOSITION_CLOSURE_GATE='PASS',DISCOVERY_COMPLETE='64/64',CONFIRMATION_EXECUTED=0,L9_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',ERROR_COHERENCE_DISCOVERY_LABEL=label,CONFIRMATORY_STATUS='NOT_EVALUATED',ENERGY_DECOMPOSITION_IDENTITY_IS_TAUTOLOGICAL='YES',SCIENTIFIC_QUESTION='IS_ALIGNMENT_STRUCTURED_AND_RANK_DECISIVE',PRIMARY_NEXT_STEP_CATEGORY='DESIGN_PROSPECTIVE_ERROR_COHERENCE_CONFIRMATION' if label=='DESCRIPTIVE_SIGNAL_PRESENT' else 'CONSIDER_READOUT_SENSITIVITY_DIAGNOSIS',RECOMMENDATION_ONLY=True,FINAL_ACTION='STOP_AFTER_DISCOVERY',median_horizon_diagnostics=summaries,per_document_horizon=rank,closure_summary=closure_summary,signed_magnitude_diagnostics=signed_by_horizon,injection_scope='actual Q-minus-native-F, includes numerical basis/write rounding; pure QDQ separately audited',PARENT_GLOBAL_GATE='FAIL_PRESERVED',CROSS_HARDWARE_FP_BITWISE='FAIL_PRESERVED',FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',NO_NEW_MODEL_FORWARD='NO; Route B teacher-forced FP capture',PREMEASUREMENT_COMMIT_SHA=read(D/'premeasurement_commit.json')['commit'])
    csvfile('per_rotation_metrics.csv',per);csvfile('per_layer_metrics.csv',layers);csvfile('horizon_segment_metrics.csv',segments);csvfile('alignment_decisive_pairs.csv',pairs);csvfile('rank_diagnostics.csv',rank);csvfile('per_document_horizon_metrics.csv',rank);csvfile('closure_diagnostics.csv',closures)
    dump(D/'analysis/coherence_analysis.json',out)
    report='# '+TASK+'\n\n```json\n'+json.dumps(out,indent=2,sort_keys=True)+'\n```\n\nDiscovery-only. No reasoning, AIME or training conclusion. Algebraic closure is tautological, not predictive evidence.\n\n## Segment, amplification and cancellation details\n\nSee horizon_segment_metrics.csv for both full-segment and primary-support intersection summaries; per_rotation_metrics.csv reports positive/negative fractions and POS/NEG_ALIGNMENT magnitudes.\n'
    textfile(D/'analysis/coherence_analysis.md',report);textfile(D/'FINAL_REPORT.md',report)
    dump(D/'provenance/COMPLETION.json',dict(FINAL_ACTION='STOP_AFTER_DISCOVERY',AIME_STARTED='NO',TRAINING_CHANGED='NO',CHECKPOINT_CHANGED='NO',DATA_CHANGED='NO',STATISTICS_CHANGED='NO',CONFIRMATION_EXECUTED=0,units=64))
    paths=sorted(f for f in D.rglob('*') if f.is_file() and f.name not in ['sha256_manifest.txt','frozen_checkout.tar'] and not f.name.endswith('.log'))
    textfile(D/'sha256_manifest.txt',''.join(sha(f)+'  '+str(f.relative_to(D))+'\n' for f in paths))
    for f in paths:assert f.exists()
    print('FINAL_ACTION STOP_AFTER_DISCOVERY',label,flush=True)

def controller():
    assert REMOTE;fixed_guard()
    relocation=imp(OLD_REMOTE/'experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py','launch_relocation')
    env=dict(os.environ)
    for k,v in read(D/'runtime_manifest.json')['launch_profile'].items():
        if k=='CUDA_VISIBLE_DEVICES':continue
        if v is None:env.pop(k,None)
        else:env[k]=str(relocation.mapped(v)) if v.startswith('/data01/user2/') else v
    env.update(CUDA_VISIBLE_DEVICES=str(GPU),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8')
    for mode,args in [('verify',[]),('smoke',['--repeat','0']),('smoke',['--repeat','1']),('smoke_gate',[]),('formal',[])]:
        subprocess.run([sys.executable,str(Path(__file__)),mode]+args,env=env,check=True)
    print('READY_FOR_SOURCE_INTEGRITY_AND_FINAL_ANALYSIS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','verify','smoke','smoke_gate','formal','analyze','controller']);p.add_argument('--repeat',type=int);a=p.parse_args()
    try:
        if a.mode=='smoke':smoke(a.repeat)
        else:globals()[a.mode]()
    except BaseException as e:stop(a.mode,e);raise
