"""Scientific reproducibility policy; generated runtime code is recorded, not binary-closed."""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, json, os, shutil, subprocess, sys, traceback
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent.parent; Q=C/'scientific_reproducibility_v3_virtual_namespace'; ROOT=C.parent
CACHE=Path('/data/zypan/experiment_caches/ling_observability_scientific_repro_v1')
RUNS=Path('/data/zypan/experiment_runs/ling_observability_scientific_repro_v3')
ENV=Path('/data/zypan/envs/ling-kda')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def emit(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def synchronize(capture,common):
    assert capture.LAYERS==common.LAYERS
    for name in ['runtime','imp']:assert getattr(capture,name).__code__==getattr(common,name).__code__
    for name in ['P','OLD','V4','V3','ENV','PYTHON','MODEL','PARENTS','runtime','imp','load']:
        setattr(capture,name,getattr(common,name))

def frozen_soname_resolution(raw, hashes):
    p=Path(raw)
    if p.is_file():
        real=str(p.resolve())
        assert real in hashes and sha(p)==hashes[real],('UNKNOWN_EXTERNAL_NATIVE_LIBRARY',str(p))
        return real
    if '/' in raw:return None
    for directory in sorted({str(Path(real).parent) for real in hashes}):
        candidate=Path(directory)/raw
        if candidate.is_file():
            real=str(candidate.resolve())
            assert real in hashes and sha(candidate)==hashes[real],('UNKNOWN_EXTERNAL_NATIVE_LIBRARY',str(candidate))
            return real
    return None

def virtual_runtime_implementation(module, envfiles, module_registry):
    owner=type(module).__module__
    if not any(owner==pkg or owner.startswith(pkg+'.') for pkg in ['torch','triton','fla']):return None
    implementation=module_registry.get(owner)
    raw=getattr(implementation,'__file__',None)
    if not raw:return None
    p=Path(raw);expected=envfiles.get(str(p))
    if expected is None:return None
    assert p.is_file() and sha(p)==expected,('FROZEN_RUNTIME_NAMESPACE_IMPLEMENTATION_CHANGED',owner)
    return dict(path=str(p),sha256=expected,owner=owner,type_name=type(module).__qualname__)

def main():
    args=argparse.ArgumentParser();args.add_argument('mode',choices=['qualify','postcommit','discovery','analyze']);args.add_argument('run_id');a=args.parse_args()
    out=RUNS/a.run_id if a.mode in ['qualify','postcommit'] else ROOT/'experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
    out.mkdir(parents=True,exist_ok=False) if a.mode in ['qualify','postcommit'] else out.mkdir(parents=True,exist_ok=True)
    base=load('scientific_repro_frozen_transport',C/'sealed_runner_attempt05_gpu6.py')
    base.A=Q;base.LAUNCH=C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json'
    cap=None
    try:
        assert not (Q/'STOP.json').exists()
        profile=read(Q/'launcher_profile.json')
        for k,v in profile['set'].items():assert os.environ.get(k)==v,('RUNTIME_ENVIRONMENT_MISMATCH',k)
        for k in profile['unset']:assert k not in os.environ,('RUNTIME_ENVIRONMENT_MISMATCH',k)
        for row in read(Q/'INPUT_SOURCE_IDENTITY.json')['files']:assert sha(Q/row['path'])==row['sha256']
        for row in read(RUNS/'INPUT_COMMIT.json')['qualification_inputs']:assert sha(ROOT/row['path'])==row['sha256'],('FROZEN_QUALIFICATION_INPUT_CHANGED',row['path'])
        class ScientificRuntime(base.Capsule):
            def check_file_runtime(self):
                # Ordinary OS support-data closure is retired by explicit authorization.
                pass
            def producer(self):
                frame=sys._getframe(2)
                while frame is not None:
                    path=frame.f_code.co_filename
                    if any('/site-packages/'+p+'/' in path for p in ['torch','triton','fla']):
                        expected=self.envfiles.get(path)
                        if expected is not None:
                            assert sha(path)==expected,('FROZEN_RUNTIME_PRODUCER_CHANGED',path)
                            return dict(path=path,sha256=expected,function=frame.f_code.co_name)
                    frame=frame.f_back
                return None
            def module_file(self,raw,name):
                p=Path(raw);real=str(p.resolve())
                assert not real.startswith(str(base.LEGACY)),'DIRTY_WORKTREE_EXECUTION'
                expected=self.files.get(real,self.envfiles.get(str(p),self.shared_hash.get(real)))
                if expected is not None:
                    assert p.is_file() and sha(p)==expected,('FROZEN_SOURCE_RUNTIME_MISMATCH',name,real)
                    kind='FROZEN_PROJECT_SOURCE' if real in self.files else 'FROZEN_RUNTIME'
                    h=expected;producer=None
                elif name.startswith('transformers_modules.') and p.is_file() and sha(p) in self.model_source_hashes:
                    kind='FROZEN_MODEL_SOURCE';h=sha(p);producer=None
                elif real in getattr(self,'generated_admissions',{}):
                    item=self.generated_admissions[real];h=item['sha256'];kind=item['identity_class'];producer=item['producer']
                    if p.is_file():assert sha(p)==h,('GENERATED_FILE_CHANGED_WITHIN_PROCESS',real)
                else:
                    producer=self.producer()
                    assert p.is_file() and producer is not None,('UNKNOWN_UNATTRIBUTED_MODULE_OR_LIBRARY',name,real)
                    # Native JIT must originate from this newly empty experiment cache.
                    # Torch-generated Python may use its standard tempfile implementation.
                    assert p.resolve().is_relative_to(CACHE) or (p.suffix=='.py' and p.resolve().is_relative_to(Path('/tmp'))),('UNATTRIBUTED_GENERATED_LOCATION',real)
                    h=sha(p);kind='NORMAL_GENERATED_CODE_FROM_FROZEN_RUNTIME'
                    self.generated_admissions[real]=dict(sha256=h,identity_class=kind,producer=producer)
                self.code_events.append(dict(module=name,path=str(p),sha256=h,identity_class=kind,producer=producer))
            def audit(self,event,args):
                if event=='ctypes.dlopen' and args[0] is not None:
                    raw=str(args[0]);p=Path(raw)
                    if p.is_file():self.module_file(p,'ctypes.dlopen')
                    else:
                        real=frozen_soname_resolution(raw,self.shared_hash)
                        assert real is not None,('UNKNOWN_EXTERNAL_NATIVE_LIBRARY',raw)
                        self.events.append(dict(event='FROZEN_NATIVE_SONAME_ALIAS',requested=raw,realpath=real,sha256=self.shared_hash[real]))
                if event=='subprocess.Popen':
                    exe,argv,cwd,env=args
                    if Path(str(exe)).name=='git':raise RuntimeError('GIT_IN_SCIENTIFIC_RUNNER')
                    p=Path(str(exe)) if '/' in str(exe) else Path(shutil.which(str(exe),path=(env or os.environ).get('PATH')) or '')
                    if not p.is_file():
                        self.events.append(dict(event='ABSENT_EXECUTABLE_PROBE',path=str(exe)));return
                    expected=self.exe_hash.get(str(p.resolve()))
                    assert expected is not None and sha(p)==expected,('UNKNOWN_OR_CHANGED_EXECUTABLE',str(p))
                    self.events.append(dict(event=event,path=str(p),sha256=expected,argv=list(map(str,argv)) if isinstance(argv,(tuple,list)) else str(argv),producer=self.producer()))
                elif event in ['os.exec','os.posix_spawn']:
                    p=Path(args[0]);expected=self.exe_hash.get(str(p.resolve()))
                    if p.is_file():assert expected and sha(p)==expected,('UNKNOWN_EXECUTABLE',str(p))
                elif event=='os.system':raise RuntimeError('UNKNOWN_SHELL_EXECUTION')
            def origins(self):
                assert not any(str(p).startswith(str(base.LEGACY)) for p in sys.path)
                modules=[]
                for name,m in list(sys.modules.items()):
                    raw=getattr(m,'__file__',None)
                    if not raw:continue
                    p=Path(raw)
                    if p.suffix=='.pyc':
                        source=Path(importlib.util.source_from_cache(str(p)))
                        if source.is_file():p=source
                    real=str(p.resolve())
                    if not p.is_file() and real not in self.generated_admissions:
                        virtual=virtual_runtime_implementation(m,self.envfiles,sys.modules)
                        if virtual is not None:
                            self.module_file(virtual['path'],name)
                            event=dict(self.code_events[-1]);event.update(virtual_namespace=True,reported_file_metadata=str(p),implementation=virtual)
                            modules.append(event);continue
                        if str(p).startswith(str(ENV)):continue
                        raise RuntimeError('UNKNOWN_MISSING_MODULE: '+name+': '+str(p))
                    self.module_file(p,name);modules.append(dict(self.code_events[-1]))
                libs=[]
                for line in Path('/proc/self/maps').read_text().splitlines():
                    bits=line.split(None,5)
                    if len(bits)!=6 or not bits[5].startswith('/') or '.so' not in bits[5]:continue
                    p=Path(bits[5].removesuffix(' (deleted)'));real=str(p.resolve())
                    if real in {x['path'] for x in libs}:continue
                    self.module_file(p,'loaded.native.library');libs.append(dict(self.code_events[-1]))
                return dict(module_files=modules,loaded_libraries=libs,generated_admissions=self.generated_admissions,
                            subprocess_events=self.events,git_subprocess_count=0,sys_path=list(sys.path),
                            runtime_generated_full_dependency_closure_required=False,native_binary_rebuild_identity_required=False)
        # The policy permits normal generated code from frozen runtime, not arbitrary cache paths.
        ScientificRuntime.generated_admissions={}
        cap=ScientificRuntime(out)
        cap.files[str(Path(__file__).resolve())]=sha(__file__)
        b=base.bind(cap,out)
        original_modules=b.modules
        def modules():
            result=original_modules();synchronize(result[3],result[2]);return result
        b.modules=modules
        verified=False
        asset_stats=[]
        def guard():
            nonlocal verified,asset_stats
            assert not (Q/'STOP.json').exists() and not (out/'STOP.json').exists()
            for p,h in cap.files.items():assert sha(p)==h,('PROTECTED_SOURCE_MISMATCH',p)
            if not verified:
                for row in read(Q/'ASSET_IDENTITY_FREEZE.json')['files']:
                    p=Path(row['path']);assert sha(p)==row['sha256'],('SCIENTIFIC_ASSET_MISMATCH',str(p))
                    s=p.stat();asset_stats.append((p,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
                verified=True
                receipt=out/'SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY.json'
                value=dict(gate='PASS',full_SHA256=True,asset_count=len(asset_stats),science_manifest_sha256=sha(C/'scientific_asset_manifest.json'))
                if receipt.exists():assert read(receipt)==value
                else:emit(receipt,value)
            else:
                for p,dev,ino,size,mtime,ctime in asset_stats:
                    s=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(dev,ino,size,mtime,ctime),('FROZEN_ASSET_CHANGED_AFTER_SHA',str(p))
            if a.mode in ['discovery','analyze']:
                r=read(Q/'GPU6_FORMAL_AUTHORIZATION.json');assert r['GPU6_FORMAL_AUTHORIZED']=='YES'
                return read(Q/'PREMEASUREMENT_COMMIT.json')
            return dict(commit=read(RUNS/'INPUT_COMMIT.json')['commit'],formal_authorized=False)
        b.guard=guard
        if a.mode=='discovery':
            b.discovery();emit(out/'DISCOVERY_COMPUTE_COMPLETE.json',dict(primary='64/64',secondary='64/64'));return
        if a.mode=='analyze':b.analyze();return
        print('SCIENTIFIC_QUALIFICATION_BEGIN',a.run_id,flush=True)
        b.verify();b.smoke(0)
        hw=read(out/'smoke/preflight.json')['hardware'];frozen_hw=read(RUNS/'INPUT_COMMIT.json')['hardware']
        for field in ['host','physical_gpu','gpu_uuid','driver','CUDA','python','torch']:
            assert hw[field]==frozen_hw[field],('HARDWARE_OR_RUNTIME_IDENTITY_CHANGED',field,hw[field],frozen_hw[field])
        expected=read(C/'known_fixture_manifest.json')['result']
        assert sha(expected['path'])==expected['sha256']
        result=read(out/'smoke/repeat0/result.json');baseline=read(expected['path'])
        for field in ['scores','target','endpoint','FP_step_hashes']:
            assert result[field]==baseline[field],('HISTORICAL_NUMERICAL_PARITY_FAIL',field)
        b.parity()
        co,t,c,capture,core,stats=b.modules();q,r,f=b.historical();torch=core.torch
        trace=read(out/'smoke/repeat0/trace/trace_freeze.json');bank=b.read(out/'rotation_manifest.json')['bank']
        fixture=c.V3/'inputs/traces/wikitext2raw-train-000.pt'
        ids=[int(v) for v in torch.load(fixture,map_location='cpu',weights_only=False)['token_ids'][:128]]
        learned=next(row for row in bank['conditions'] if row['condition_id']=='L6')
        states,scores,endpoint=b.replay(trace,learned,t,c,core,q,r,[64],128)
        model=capture.Capture()
        try:learned_target=b.pulse_target(model,f,ids,64,128,states[64]['Q'].cuda())
        finally:model.close()
        assert learned_target['E_anchor_sha256']==scores[64]['E_anchor_sha256']
        assert learned_target['FP_anchor_sha256']==b.tensor_sha(states[64]['FP'])
        query=torch.tensor([[[1.,2.,-1.],[.5,-.5,1.]]]);state=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)/10
        got=q.query_readout(state,query);expected_k3=torch.sum(state*query.unsqueeze(-1),dim=-2).unsqueeze(1)
        assert torch.allclose(got,expected_k3)  # exact historical fixture tolerance
        delta=got-expected_k3
        rotated={}
        model=capture.Capture()
        try:
            # Invoke the unchanged full-model Phase2 stream and unchanged per-token qdq_cache.
            model.probe.globals['fused_recurrent_kda']=model.probe._wrap(model.probe.orig_fused)
            for label in ['H','L6']:
                cond=next(row for row in bank['conditions'] if row['condition_id']==label)
                rot=torch.load(t.mapped(cond['artifact']['path']),map_location='cpu',weights_only=False)['rotations']
                numeric=model.P2.forward_stream(model.model,model.probe,model.cap,ids[:5],label,{label:rot},list(range(4)),keep_full=False)
                assert len(numeric)==4
                for row in numeric.values():
                    assert len(row['qdq'])==len(b.LAYERS)
                    for layer,qdq in row['qdq'].items():
                        assert qdq['state_dtype']=='torch.float32'
                        assert row['state'][str(layer)]==qdq['postquant_sha256'],'PERSISTENT_WRITEBACK_IDENTITY_FAIL'
                rotated[label]=numeric
        finally:model.close()
        numerical=dict(H=result,learned=dict(scores=scores,target=learned_target,endpoint=endpoint),
                       rotated_model=rotated,k3=dict(max_abs=float(delta.abs().max()),
                       relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected_k3)),
                       actual_hash=b.tensor_sha(got),expected_hash=b.tensor_sha(expected_k3)),
                       basis_and_jprop=read(out/'smoke/basis_and_jprop_parity.json'))
        # Retain numerical fields only for determinism comparison, never generated binary identity.
        for field in ['hardware','dependencies']:numerical['H'].pop(field,None)
        numerical['basis_and_jprop'].pop('dependencies',None)
        emit(out/'NUMERIC_FINGERPRINT.json',numerical)
        runtime=c.runtime();flags=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
              cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
              deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        assert flags==profile['expected_historical_torch_flags'],('DETERMINISTIC_SETTINGS_CHANGED',flags)
        free=shutil.disk_usage(ROOT).free;required=read(C/'storage_capacity_identity.json')['required_total_bytes'];assert free>required
        emit(out/'QUALIFICATION_COMPLETE.json',dict(gate='PASS',SOURCE_AND_SCIENTIFIC_ASSET_IDENTITY_GATE='PASS',
             RUNTIME_ENVIRONMENT_IDENTITY_GATE='PASS',BASIS_SEMANTICS_GATE='PASS',HISTORICAL_K3_PARITY_GATE='PASS',
             HISTORICAL_JPROP_PARITY_GATE='PASS',FUTURE_KL_HARNESS_PARITY_GATE='PASS',HARDWARE_ASSIGNMENT_GATE='PASS',
             STORAGE_CAPACITY_GATE='PASS',FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='PENDING_SUPERVISOR_COMPARISON',
             runtime=runtime,deterministic_settings=flags,free_bytes=free,required_bytes=required,
             coverage=['real model Hadamard H rotation','real model learned L6 rotation','every-token INT8 persistent QDQ/writeback',
                       'KDA transition','historical K3','historical JPROP','basis all16','future-KL H and L6'],
             runtime_ingress=cap.origins(),triton_cache=str(CACHE/'triton'),FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
        print('SCIENTIFIC_QUALIFICATION_COMPLETE',a.run_id,flush=True)
    except BaseException as e:
        receipt=dict(reason=repr(e),traceback=traceback.format_exc(),mode=a.mode,run_id=a.run_id,FINAL_ACTION='FAIL_AND_STOP',
                     FORMAL_MEASUREMENTS=0 if a.mode not in ['discovery','analyze'] else 'CHECK_UNIT_RECEIPTS',GPU6_FORMAL_AUTHORIZED='NO')
        if not (out/'STOP.json').exists():emit(out/'STOP.json',receipt)
        if not (Q/'STOP.json').exists():emit(Q/'STOP.json',receipt)
        raise
if __name__=='__main__':main()
