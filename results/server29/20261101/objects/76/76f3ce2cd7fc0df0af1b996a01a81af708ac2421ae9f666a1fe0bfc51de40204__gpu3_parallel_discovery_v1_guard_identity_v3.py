"""Prospective two-GPU orchestration. All numerical functions remain historical."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback
sys.dont_write_bytecode = True
R = Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
W = Path('/data/zypan/worktrees/ling-observability-gpu3-expansion-v1')
A = W/'hardware_expansion_gpu3_v1'
P = A/'prospective_parallel'
FD = R/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
Q = R/'provenance_capsule_v3/scientific_reproducibility_v4_whole_runtime_replay'
HD = Path('/data/zypan/experiment_runs/ling_historical_runtime_replay_v1/checkout/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
UUIDS = {3:'GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae', 6:'GPU-de53f243-b271-521c-d875-d96611ed8574'}
ASSIGNMENT = {'pg19-test-37702':6,'pg19-test-5396':3,'pg19-test-28988':6,'pg19-test-26493':3}
PARENT = 'e428b5deb0579b926b2ad098a8699b3f4f2233f0'
BRIDGE_SHA = 'e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
CONFIGS = ['source_identity.json','document_manifest.json','rotation_manifest.json','runtime_manifest.json','preregistration.json']
PARENT_STOP_REL = 'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/STOP.json'
EXPECTED_PARENT_TRACKED_STATUS = ' M '+PARENT_STOP_REL+'\n'

def assert_parent_runtime_tree():
    assert git('rev-parse','HEAD',cwd=R)==PARENT
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=R,text=True)
    assert status==EXPECTED_PARENT_TRACKED_STATUS,('UNEXPECTED_PARENT_TRACKED_STATUS',status)
    handoff=read(P/'HANDOFF_COMPLETION.json');stop=R/PARENT_STOP_REL
    assert sha(stop)==handoff['controlled_handoff_receipt_sha256'],'HANDOFF_CONTROL_RECEIPT_CHANGED'
    value=read(stop)
    assert value.get('kind')=='ENGINEERING_CONTROLLED_HANDOFF_REQUEST' and value.get('scientific_failure') is False

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def row(p): return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def emit(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args,cwd=W):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def check_assignment(gpu,assignment):
    assert gpu in UUIDS and assignment==ASSIGNMENT,'FROZEN_HARDWARE_ASSIGNMENT_CHANGED'
    return [d for d,g in assignment.items() if g==gpu]

def validate_unit(unit,gpu,document,condition,anchor,token_sha,rotation_sha,allowed_commits):
    assert unit['gate']=='PASS' and unit['task']=='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
    assert (unit['document'],unit['condition'],unit['anchor'],unit['endpoint'])==(document,condition,anchor,32768)
    assert unit['hardware']==dict(physical_gpu=gpu,gpu_uuid=UUIDS[gpu]),'WRONG_UNIT_HARDWARE'
    assert unit['premeasurement_commit'] in allowed_commits,'WRONG_UNIT_PREMEASUREMENT'
    assert unit['token_sha256']==token_sha and unit['rotation_sha256']==rotation_sha
    assert unit['scores']['E_anchor_sha256']==unit['target']['E_anchor_sha256']
    for key in ['INITIAL_PULSE_IDENTITY_GATE','FP_REFERENCE_PAIRING_GATE','NO_FURTHER_QDQ_GATE']:
        assert unit['target'][key]=='PASS'
    assert unit['target']['teacher_tokens']==32768-anchor

def existing_pair(paths):
    """No single completed anchor can authorize an automatic pair rerun."""
    present=[Path(p).exists() for p in paths]
    assert not any(present) or all(present),'INCOMPLETE_PAIR_REQUIRES_ENGINEERING_RECOVERY'
    return all(present)

def prepare_context(gpu,mode):
    seal=read(P/'SCIENTIFIC_PREMEASUREMENT_COMMIT.json');commit=seal['commit']
    assert git('rev-parse','HEAD')==commit and not git('status','--porcelain=v1','--untracked-files=no')
    assert_parent_runtime_tree()
    assert sha(HD/'scripts/bridge.py')==BRIDGE_SHA
    plan=read(P/'SCIENTIFIC_PLAN.json');check_assignment(gpu,plan['document_assignment'])
    assert plan['scientific_protocol_changed']=='NO' and plan['FP_REFERENCES_MIXED_ACROSS_HARDWARE']=='NO'
    identity={}
    groups=[seal['input_files'],plan['identity_files'],read(R/'provenance_capsule_v3/scientific_asset_manifest.json')['files'],
        read(Q/'HISTORICAL_EXECUTED_PACKAGE_IDENTITY.json')['files']]
    groups.extend(reference['files'] for doc,reference in plan['references'].items() if ASSIGNMENT[doc]==gpu)
    for group in groups:
        for item in group:
            path=Path(item['path'])
            if path in identity:
                assert identity[path][0]==item['sha256'],'CONFLICTING_FROZEN_SHA'
                continue
            assert sha(path)==item['sha256'],('FROZEN_SOURCE_ASSET_RUNTIME_REFERENCE_CHANGED',str(path))
            st=path.stat();identity[path]=(item['sha256'],st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    expected=plan['launch_profiles'][str(gpu)]['environment']
    assert all(os.environ.get(k)==v for k,v in expected.items()),'ENVIRONMENT_PROFILE_CHANGED'
    out=A/('postcommit_parallel' if mode=='postcommit' else 'discovery_shards')/('gpu'+str(gpu))
    assert not out.exists(),'NO_DUPLICATE_LAUNCH'
    out.mkdir(parents=True)
    for name in CONFIGS:
        with (out/name).open('xb') as f:f.write((FD/name).read_bytes())
    b=load(HD/'scripts/bridge.py','frozen_historical_parallel_backend');b.D=out;b.ROOT=W
    original_hardware=b.hardware
    q3=load(A/'gpu3_expansion_qualification.py','frozen_gpu3_assignment')
    def hardware(torch):
        hw=q3.hardware(torch) if gpu==3 else original_hardware(torch)
        assert hw['physical_gpu']==gpu and hw['gpu_uuid']==UUIDS[gpu]
        baseline=read(Q/'original_historical_reference.json')['hardware']
        for key in ['host','python','torch','CUDA','driver','deterministic_algorithms','cudnn_deterministic','cudnn_benchmark','matmul_tf32']:
            assert hw[key]==baseline[key],('RUNTIME_IDENTITY_CHANGED',key)
        return hw
    def modules():
        co=b.imp(b.OLD_N/'scripts/coherence.py','unchanged_coherence_assignment')
        co.GPU=gpu;co.UUID=UUIDS[gpu]
        t,c,cap,core,stats=co.modules()
        c.D=out;c.R=W;c.TASK=b.TASK;cap.D=out;cap.R=W;cap.TASK=b.TASK
        return co,t,c,cap,core,stats
    def guard():
        assert not (out/'STOP.json').exists() and not (P/'SCIENTIFIC_STOP.json').exists()
        assert not list(P.glob('GPU*_discovery_STOP.json')),'OTHER_SHARD_FORMAL_STOP'
        assert git('rev-parse','HEAD')==commit and not git('status','--porcelain=v1','--untracked-files=no')
        assert_parent_runtime_tree()
        for path,expected_identity in identity.items():
            st=path.stat()
            assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==expected_identity[1:],('FROZEN_FILE_IDENTITY_CHANGED',str(path))
        if mode=='discovery':
            auth=read(P/('GPU'+str(gpu)+'_FORMAL_AUTHORIZATION.json'))
            assert auth['gate']=='PASS' and auth['premeasurement_commit']==commit
            assert auth['physical_gpu']==gpu and auth['gpu_uuid']==UUIDS[gpu]
            handoff=read(P/'HANDOFF_COMPLETION.json')
            assert handoff['gate']=='PASS' and handoff['all_original_task_processes_exited'] is True
            assert handoff['actual_gpu6_original_process_present'] is False
            assert handoff['no_active_anchor_interrupted'] is True
            for item in handoff['files']:assert sha(item['path'])==item['sha256']
            original_stop=FD/'STOP.json'
            assert sha(original_stop)==handoff['controlled_handoff_receipt_sha256']
            assert read(original_stop)['kind']=='ENGINEERING_CONTROLLED_HANDOFF_REQUEST'
        return dict(commit=commit,formal_authorized=mode=='discovery')
    b.hardware=hardware;b.modules=modules;b.guard=guard
    guard();co,t,c,cap,core,stats=modules();hardware(core.torch)
    assert shutil.disk_usage(W).free>20*1024**3,'STORAGE_CAPACITY_FAIL'
    return b,plan,out,commit,(co,t,c,cap,core,stats)

def postcommit(gpu):
    b,plan,out,commit,mods=prepare_context(gpu,'postcommit')
    b.verify();b.smoke(0);b.parity()
    got=read(out/'smoke/repeat0/result.json')
    baseline=read(A/'qualification/reference_off/smoke/repeat0/result.json') if gpu==3 else read(Q/'original_historical_reference.json')
    for key in ['FP_step_hashes','scores','target','endpoint']:assert got[key]==baseline[key],('POSTCOMMIT_PARITY_FAIL',key)
    for doc,reference in plan['references'].items():
        if ASSIGNMENT[doc]!=gpu:continue
        assert reference['physical_gpu']==gpu and reference['gpu_uuid']==UUIDS[gpu]
        for item in reference['files']:assert sha(item['path'])==item['sha256']
        assert reference['FP_CAPTURE_REPLAY_BITWISE_GATE']=='PASS'
    emit(out/'MINIMAL_POSTCOMMIT_IDENTITY_PARITY_GATE.json',dict(gate='PASS',premeasurement_commit=commit,
        physical_gpu=gpu,gpu_uuid=UUIDS[gpu],exact_same_gpu_historical_parity=True,same_gpu_reference_inventory_verified=True,
        FORMAL_MEASUREMENTS=0,FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO'))

def discovery(gpu):
    b,plan,out,commit,(co,t,c,cap,core,stats)=prepare_context(gpu,'discovery')
    torch=core.torch;q,r,f=b.historical();receipt=b.guard()
    docs=read(FD/'document_manifest.json')['documents'];bank=read(FD/'rotation_manifest.json')['bank']
    selected=check_assignment(gpu,plan['document_assignment'])
    for item in read(FD/'historical_assets/executed_dependency_closure.json')['all_module_files']:
        assert sha(item['path'])==item['sha256']
    b.dump(out/'discovery/STARTED.json',dict(premeasurement_commit=commit,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],
        documents=selected,prospective_shard_units=64,time_unix=time.time(),FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO'))
    canonical=[]
    handoff=read(P/'HANDOFF_COMPLETION.json');inherited={Path(item['path']).name:item for item in handoff['unit_files']}
    for doc in docs:
        doc_id=doc['document_id']
        if doc_id not in selected:continue
        b.guard();source=t.mapped(doc['token_file']['path']);assert sha(source)==doc['token_file']['sha256'];ids=read(source)
        assert len(ids)==32769 and hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids)).hexdigest()==doc['token_ids_sha256']
        ref=plan['references'][doc_id]
        assert ref['physical_gpu']==gpu and ref['gpu_uuid']==UUIDS[gpu]
        assert sha(ref['trace_path'])==ref['trace_sha256'];trace=read(ref['trace_path'])
        assert trace['tokens']==32768 and trace['layer_ids']==b.LAYERS
        assert trace['token_sha256']==hashlib.sha256(b''.join(int(v).to_bytes(4,'little') for v in ids[:32768])).hexdigest()
        for cond in bank['conditions']:
            label=cond['condition_id'];b.guard();stem=doc_id+'__'+label+'__'
            old=[inherited.get(stem+str(anchor)+'.json') for anchor in [8192,2048]]
            assert not any(old) or all(old),'INCOMPLETE_INHERITED_PAIR'
            if all(old):
                assert gpu==6,'GPU3_MUST_NOT_REUSE_GPU6_CANONICAL_UNIT'
                for anchor,item in zip([8192,2048],old):
                    assert sha(item['path'])==item['sha256'];unit=read(item['path'])
                    validate_unit(unit,gpu,doc_id,label,anchor,doc['token_ids_sha256'],cond['artifact']['sha256'],[PARENT])
                    canonical.append(item)
                print('INHERITED_COMPLETE_PAIR_PRESERVED_NO_RERUN',doc_id,label,flush=True);continue
            b.dump(out/'discovery/logs'/(doc_id+'__'+label+'.started.json'),dict(time_unix=time.time(),document=doc_id,condition=label))
            sys.setprofile(b.profile)
            states,scores,endpoint=b.replay(trace,cond,t,c,core,q,r,[8192,2048],32768)
            sys.setprofile(None)
            for anchor in [8192,2048]:
                model=cap.Capture()
                try:target=b.pulse_target(model,f,ids,anchor,32768,states[anchor]['Q'].to('cuda'))
                finally:model.close()
                assert target['FP_anchor_sha256']==b.tensor_sha(states[anchor]['FP'])
                result=dict(gate='PASS',task=b.TASK,document=doc_id,condition=label,anchor=anchor,endpoint=32768,
                    evidence='EXPLORATORY_DISCOVERY_ONLY',scores=scores[anchor],target=target,
                    hardware=dict(physical_gpu=gpu,gpu_uuid=UUIDS[gpu]),premeasurement_commit=receipt['commit'],
                    token_sha256=doc['token_ids_sha256'],rotation_sha256=cond['artifact']['sha256'],
                    same_gpu_fp_reference_sha256=ref['trace_sha256'],hardware_assignment_sha256=sha(A/'HARDWARE_ASSIGNMENT_AMENDMENT.json'))
                validate_unit(result,gpu,doc_id,label,anchor,doc['token_ids_sha256'],cond['artifact']['sha256'],[commit])
                dest=out/'discovery/unit_results'/(stem+str(anchor)+'.json')
                b.dump(dest,result);b.dump(dest.with_suffix('.sha256.json'),b.row(dest));dest.chmod(0o444);canonical.append(b.row(dest))
                print('ANCHOR_UNIT_COMPLETE',doc_id,label,anchor,flush=True)
            b.dump(out/'discovery/summaries'/(doc_id+'__'+label+'__endpoint32K.json'),endpoint)
    assert len(canonical)==64 and len({read(item['path'])['document']+'__'+read(item['path'])['condition']+'__'+str(read(item['path'])['anchor']) for item in canonical})==64
    b.dump(out/'discovery/raw_shard_freeze.json',dict(gate='PASS',complete=64,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],
        files=canonical,before_aggregation=True,FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO'))
    print('DISCOVERY_SHARD_64_COMPLETE',gpu,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['postcommit','discovery']);ap.add_argument('--gpu',type=int,choices=[3,6],required=True);args=ap.parse_args()
    try:globals()[args.mode](args.gpu)
    except BaseException as error:
        stop=P/('GPU'+str(args.gpu)+'_'+args.mode+'_STOP.json')
        if not stop.exists():emit(stop,dict(error=repr(error),traceback=traceback.format_exc(),FINAL_ACTION='FAIL_AND_STOP',mode=args.mode,physical_gpu=args.gpu))
        raise
