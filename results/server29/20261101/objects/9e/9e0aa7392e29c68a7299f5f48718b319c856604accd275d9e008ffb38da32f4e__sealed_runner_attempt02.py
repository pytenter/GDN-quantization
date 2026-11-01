"""Engineering-only adapter around unchanged historical scientific functions.

This observer verifies explicit source/environment identities and subprocess
origins. It deliberately does not observe or classify ordinary data-file opens.
"""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, importlib.machinery, json, os, shutil, subprocess, sys, traceback
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent
A=C/'attempts/hash_correction_attempt02'
ROOT=C.parent
TASK='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
OLD_SOURCE=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1')
LEGACY=Path('/data01/user2/.codex/worktrees/GDN-quantization/ling-kda-int8-rc-granularity')
PROP=Path('/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1')
ENV=Path('/data/zypan/envs/ling-kda')
SOURCE_ENV='/data01/user2/worktrees/ling-l6-l7-objective-horizon-diagnosis-v1/experiments/LING_L7R_L8_LAYER_AWARE_MATCHED_TRAINING_V2_CLEANROOM/runtime/env'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def emit(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def mapped(p):
    p=str(p)
    src='/data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1'
    if p.startswith(src+'/'):return PROP/Path(p).relative_to(src)
    if p.startswith(SOURCE_ENV+'/'):return ENV/Path(p).relative_to(SOURCE_ENV)
    if p.startswith('/data01/user2/models/'):return Path('/data/zypan/models')/Path(p).relative_to('/data01/user2/models')
    if p.startswith('/data01/user2/'):return PROP/'inherited'/p.lstrip('/')
    return Path(p)

class Capsule:
    def __init__(self,run):
        self.run=run;self.project={};self.files={};self.events=[];self.code_events=[];self.git_count=0
        for r in read(C/'historical_source_snapshot_manifest.json')['files']:
            p=C/r['snapshot_relative_path'];assert sha(p)==r['snapshot_sha256']==r['original_sha256']
            self.project[r['original_absolute_path']]=p
            self.project[str(mapped(r['original_absolute_path']))]=p
            if r['original_absolute_path'].startswith('/data01/user2/worktrees/'):
                self.project[r['original_absolute_path'].replace('/data01/user2/worktrees/','/data/zypan/worktrees/',1)]=p
            self.files[str(p.resolve())]=r['snapshot_sha256']
        self.environment=read(C/'runtime_environment_manifest.json')
        self.envfiles={r['path']:r['sha256'] for r in self.environment['environment_files']}
        self.envfiles.update({r['path']:r['sha256'] for r in read(C/'cached_environment_module_manifest.json')['files']})
        self.generated={r['sha256'] for r in self.environment['generated_environment_modules']}
        self.science=read(C/'scientific_asset_manifest.json')
        self.model_source_hashes={r['sha256'] for r in self.science['files'] if '/models/' in r['path'] and r['path'].endswith('.py')}
        self.allowed_executables=read(C/'executable_runtime_manifest.json')['files']
        self.exe_hash={r['realpath']:r['sha256'] for r in self.allowed_executables}
        self.shared=read(C/'shared_library_manifest.json')
        self.shared_hash={r['realpath']:r['sha256'] for r in self.shared['files']}
        self.launch=read(A/'candidate_launch_identity.json')
        for r in self.launch['files']:
            p=C/r['path'];assert sha(p)==r['sha256'],('ENGINEERING_CANDIDATE_IDENTITY_CHANGED',r['path'])
            if p.suffix=='.py':self.files[str(p.resolve())]=r['sha256']
        self.spec=importlib.util.spec_from_file_location
        importlib.util.spec_from_file_location=self.spec_redirect
        path_find=importlib.machinery.PathFinder.find_spec
        capsule=self
        def redirected_find(cls,fullname,path=None,target=None):
            result=path_find(fullname,path,target)
            if result is not None and result.origin in capsule.project:
                package=result.submodule_search_locations is not None
                p=capsule.project[result.origin]
                return capsule.spec(fullname,p,submodule_search_locations=[str(p.parent)] if package else None)
            return result
        importlib.machinery.PathFinder.find_spec=classmethod(redirected_find)
        source_get_code=importlib.machinery.SourceFileLoader.get_code
        extension_create=importlib.machinery.ExtensionFileLoader.create_module
        def checked_code(loader,fullname):
            capsule.module_file(loader.path,fullname)
            return source_get_code(loader,fullname)
        def checked_extension(loader,spec):
            capsule.module_file(loader.path,loader.name)
            return extension_create(loader,spec)
        importlib.machinery.SourceFileLoader.get_code=checked_code
        importlib.machinery.ExtensionFileLoader.create_module=checked_extension
        sys.addaudithook(self.audit)

    def module_file(self,raw,name):
        p=Path(raw);real=str(p.resolve())
        if not p.is_file():raise RuntimeError('UNKNOWN_IMPORTED_PYTHON_MODULE_MISSING: '+name+': '+str(p))
        h=sha(p)
        if real in self.files:assert h==self.files[real],('SOURCE_CAPSULE_HASH_MISMATCH',name,real)
        elif str(p) in self.envfiles:assert h==self.envfiles[str(p)],('RUNTIME_PACKAGE_IDENTITY_FAILURE',name,str(p))
        elif real in self.shared_hash:assert h==self.shared_hash[real],('RUNTIME_SHARED_LIBRARY_CHANGED',name,real)
        elif str(p).startswith('/tmp/') and h in self.generated:pass
        elif name.startswith('transformers_modules.') and h in self.model_source_hashes:pass
        else:raise RuntimeError('UNKNOWN_IMPORTED_PYTHON_MODULE: '+name+': '+str(p))
        self.code_events.append(dict(module=name,path=str(p),sha256=h,project_source=real in self.files))

    def code(self,p):
        p=Path(p)
        if str(p) in self.project:return self.project[str(p)]
        if p.resolve().is_relative_to(C):return p
        return p
    def spec_redirect(self,name,location=None,*a,**k):
        if location is not None:
            p=self.code(location)
            if p.suffix=='.py' and '/worktrees/' in str(p) and not p.resolve().is_relative_to(C):
                raise RuntimeError('UNFROZEN_PROJECT_LOCAL_EXECUTABLE: '+str(p))
            location=p
        return self.spec(name,location,*a,**k)
    def audit(self,event,args):
        if event=='subprocess.Popen':
            exe,argsv,cwd,env=args
            argv=argsv if isinstance(argsv,(list,tuple)) else [str(argsv)]
            name=Path(str(exe)).name
            if name=='git' or any(Path(str(v)).name=='git' for v in argv[:1]):
                self.git_count+=1;raise RuntimeError('GIT_DURING_SEALED_MEASUREMENT')
            path=Path(str(exe)) if '/' in str(exe) else Path(shutil.which(str(exe),path=(env or os.environ).get('PATH')) or '')
            real=str(path.resolve())
            if not path.is_file():
                self.events.append(dict(event='NONEXISTENT_EXECUTABLE_PROBE',executable=str(exe),successful_ingress=False))
                return
            if real not in self.exe_hash or sha(path)!=self.exe_hash[real]:
                raise RuntimeError('UNKNOWN_EXECUTABLE: '+str(exe))
            self.events.append(dict(event=event,executable=str(path),realpath=real,sha256=self.exe_hash[real],argv=list(map(str,argv))))
        if event in ('os.exec','os.posix_spawn'):
            p=Path(args[0]);real=str(p.resolve())
            if not p.is_file():return
            if p.name=='git':self.git_count+=1;raise RuntimeError('GIT_DURING_SEALED_MEASUREMENT')
            if real not in self.exe_hash or sha(p)!=self.exe_hash[real]:raise RuntimeError('UNKNOWN_EXECUTABLE: '+str(p))
            self.events.append(dict(event=event,executable=str(p),realpath=real,sha256=self.exe_hash[real]))
        if event=='os.system':raise RuntimeError('UNINSTRUMENTED_SHELL_LAUNCH')
    def origins(self):
        origins=[]
        assert not any(str(p).startswith(str(LEGACY)) for p in sys.path),'DIRTY_SYS_PATH'
        assert not (os.environ.get('PYTHONPATH') or '').startswith(str(LEGACY)),'DIRTY_PYTHONPATH'
        for name,m in list(sys.modules.items()):
            raw=getattr(m,'__file__',None)
            if not raw:continue
            p=Path(raw)
            if not p.is_file():
                if str(p).startswith(str(ENV)):continue  # package-owned absent cache file; source checked below
                raise RuntimeError('UNKNOWN_IMPORTED_PYTHON_MODULE_MISSING: '+name+': '+str(p))
            if p.suffix=='.pyc':
                src=Path(importlib.util.source_from_cache(str(p)))
                if src.is_file():p=src
            h=sha(p);real=str(p.resolve())
            if real in self.files:
                assert h==self.files[real],('SOURCE_CAPSULE_HASH_MISMATCH',name,real)
                kind='FROZEN_PROJECT_SOURCE'
            elif str(p) in self.envfiles:
                assert h==self.envfiles[str(p)],('RUNTIME_PACKAGE_IDENTITY_FAILURE',name,str(p))
                kind='FROZEN_ENVIRONMENT_MODULE'
            elif str(p).startswith('/tmp/') and h in self.generated:kind='FROZEN_ENVIRONMENT_GENERATED_MODULE_BYTES'
            elif name.startswith('transformers_modules.') and h in self.model_source_hashes:kind='FROZEN_MODEL_REMOTE_CODE_BYTES'
            elif p.resolve().is_relative_to(C) and p.name=='sealed_runner_attempt02.py':kind='ENGINEERING_ADAPTER'
            else:raise RuntimeError('UNKNOWN_IMPORTED_PYTHON_MODULE: '+name+': '+str(p))
            origins.append(dict(module=name,path=str(p),sha256=h,identity_class=kind))
        libs=[]
        for line in Path('/proc/self/maps').read_text().splitlines():
            bits=line.split(None,5)
            if len(bits)!=6 or not bits[5].startswith('/') or '.so' not in bits[5]:continue
            p=Path(bits[5]);real=str(p.resolve())
            if real in {x['realpath'] for x in libs}:continue
            if not p.is_file():raise RuntimeError('UNKNOWN_LOADED_SHARED_LIBRARY_MISSING: '+str(p))
            h=sha(p)
            if real in self.shared_hash:
                assert h==self.shared_hash[real],('RUNTIME_SHARED_LIBRARY_CHANGED',real)
            elif str(p) in self.envfiles:
                assert h==self.envfiles[str(p)],('RUNTIME_EXTENSION_CHANGED',str(p))
            else:raise RuntimeError('UNKNOWN_LOADED_SHARED_LIBRARY: '+str(p))
            libs.append(dict(path=str(p),realpath=real,sha256=h))
        return dict(module_files=origins,source_and_extension_loader_events=self.code_events,loaded_libraries=libs,sys_path=list(sys.path),PYTHONPATH=os.environ.get('PYTHONPATH'),git_subprocess_count=self.git_count,subprocess_events=self.events,ambient_regular_file_reads_observed=False)

def bind(cap,out):
    original=OLD_SOURCE/'experiments'/TASK/'scripts/bridge.py'
    sp=cap.spec('v3_frozen_bridge',cap.project[str(original)]);b=importlib.util.module_from_spec(sp);sys.modules[sp.name]=b;sp.loader.exec_module(b)
    assert sha(b.__file__)=='e4baf3f7e3ec19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
    b.D=out;b.ROOT=ROOT;b.SNAP=C/'source_snapshot'/LEGACY.relative_to('/')
    b.profile=None  # V1/V2 per-call execution tracing architecture is retired.
    raw_read=b.read
    def reading(p):
        p=Path(p)
        if p.parent==out and (C/'frozen_protocol'/p.name).is_file():p=C/'frozen_protocol'/p.name
        return raw_read(p)
    b.read=reading
    def guard():
        assert sha(C/'STOP.json')=='d2330b426afbea3e443bc44fc0381b9d4e045fcfa3a18739bd6d5570a85e2fe8','FIRST_STOP_IDENTITY_FAILURE'
        assert not (A/'STOP.json').exists(),'PRESERVED_ATTEMPT02_STOP'
        assert not (out/'STOP.json').exists(),'PRESERVED_RUN_STOP'
        for p,h in cap.files.items():assert sha(p)==h,('SOURCE_CAPSULE_HASH_MISMATCH',p)
        for r in cap.launch['files']:assert sha(C/r['path'])==r['sha256'],('ENGINEERING_CANDIDATE_IDENTITY_CHANGED',r['path'])
        for r in cap.science['files']:assert sha(r['path'])==r['sha256'],('SCIENTIFIC_ASSET_HASH_MISMATCH',r['path'])
        receipt=C/'premeasurement_commit.json'
        return read(receipt) if receipt.is_file() else dict(commit=read(A/'candidate_launch_identity.json')['commit'],formal_authorized=False)
    b.guard=guard
    b.execution_inventory=lambda:dict(historical_executed_files=[r['path'] for r in cap.origins()['module_files'] if r['identity_class']=='FROZEN_PROJECT_SOURCE'],execution_module_files=[dict(path=r['path'],sha256=r['sha256']) for r in cap.origins()['module_files']],v3=cap.origins(),bytecode_writes=False,old_dirty_directory_executed=False)
    modules_old=b.modules
    cached=[]
    def modules():
        if cached:return cached[0]
        parent='/data01/user2/worktrees/ling-propagation-surrogate-validation-v1/experiments/LING_PROPAGATION_SURROGATE_VALIDATION_V1/scripts/'
        for name in ['common','transition','replay_core','capture','worker','surrogate_statistics']:
            if name not in sys.modules:b.imp(cap.project[parent+name+'.py'],name)
        mods=modules_old();co,t,c,capture,core,stats=mods
        old_load=c.load
        def remap(x):
            if isinstance(x,dict):return {k:remap(v) for k,v in x.items()}
            if isinstance(x,list):return [remap(v) for v in x]
            if isinstance(x,str) and x.endswith('.py'):return str(cap.code(x))
            return x
        c.load=lambda p:remap(old_load(p))
        def qualified_runtime():
            p=cap.code(c.V4/'scripts/v4_common.py')
            cap.module_file(p,'V4_apply_qualified_runtime_AST')
            helper=Path(c.load(c.V4/'runtime/historical_fixed_forward_manifest.json')['helper'])
            cap.module_file(helper,'historical_fixed_forward_AST_helper')
            fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='apply_qualified_runtime')
            ns=dict(os=os,ast=ast,Path=Path,sys=sys,D=c.V4,c=c,load=c.load)
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
            return ns['apply_qualified_runtime']()
        capture.qualified_runtime=qualified_runtime
        # No historical project directory remains on the import search path.
        sys.path[:]=[p for p in sys.path if '/worktrees/' not in str(p) or Path(p).resolve().is_relative_to(C)]
        sys.path.insert(0,str(cap.project[parent+'common.py'].parent))
        cached.append(mods);return mods
    b.modules=modules
    return b

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','smoke0','smoke1','smoke2','parity']);a=p.parse_args()
    out=C/'runs/hash_correction_attempt02/qualification';out.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
    assert os.environ.get('PYTHONHASHSEED')=='0'
    assert os.environ.get('TOKENIZERS_PARALLELISM')=='false'
    assert os.environ.get('LING_GATED_RMSNORM_FIXED_CONFIG')=='BT16_W8'
    cap=None
    try:
        cap=Capsule(out);b=bind(cap,out)
        if a.mode=='verify':b.verify()
        elif a.mode=='parity':b.parity()
        else:
            repeat=int(a.mode[-1]);b.smoke(repeat)
            # Compare scientific fields against the preserved successful smoke
            # from before the provenance STOP, not a newly chosen baseline.
            baseline=read(C/'known_fixture_manifest.json')
            expected_path=Path(baseline['result']['path'])
            assert sha(expected_path)==baseline['result']['sha256'],'KNOWN_FIXTURE_IDENTITY_FAILURE'
            old=read(expected_path);current=read(out/'smoke'/f'repeat{repeat}'/'result.json')
            fields=['scores','target','endpoint','FP_step_hashes']
            for name in fields:assert current[name]==old[name],('HISTORICAL_FUNCTIONAL_PARITY_FAILURE',name)
            co,t,c,capture,core,stats=b.modules();q,r,f=b.historical();torch=core.torch
            query=torch.tensor([[[1.0,2.0,-1.0],[0.5,-0.5,1.0]]])
            state=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)/10.0
            got=q.query_readout(state,query);expected=torch.sum(state*query.unsqueeze(-1),dim=-2).unsqueeze(1)
            assert torch.allclose(got,expected)  # unchanged historical fixture tolerance
            delta=got-expected
            trace=read(out/'smoke'/f'repeat{repeat}'/'trace/trace_freeze.json')
            x=torch.load(trace['chunks'][0]['artifact']['path'],map_location='cuda',weights_only=False)
            bank=b.read(out/'rotation_manifest.json')['bank']
            rot=torch.load(t.mapped(bank['conditions'][0]['artifact']['path']),map_location='cpu',weights_only=False)['rotations']
            R=torch.stack([rot[l] for l in b.LAYERS]).cuda()
            states=core.initial_states(8,'cuda');signs=core.sign_schedule(8,32).cuda()
            for j in range(32):
                rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
                values=core.replay_step(*states,rec,R,signs[j]);states=values[:4]
            extra=dict(HISTORICAL_K3_PARITY_GATE='PASS',max_abs=float(delta.abs().max()),relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected)),expected_K3_sha256=b.tensor_sha(expected),actual_K3_sha256=b.tensor_sha(got),JPROP_fixed_fixture_state_hashes=[b.tensor_sha(v) for v in values],JPROP_fixed_fixture_metrics=values[4].cpu().tolist(),FUTURE_KL_HARNESS_PARITY_GATE='PASS',historical_expected_result_sha256=baseline['result']['sha256'],historical_scientific_fields_bitwise_identical=fields,future_KL=dict(direction='ref||pulse',dtype='FP32 full_vocabulary per-token reduction',AUC='streaming FP64 sum; no averaging/no clamping',support='anchor+1 through endpoint inclusive',teacher_forcing=True,fixed_fixture_AUC=current['target']['TARGET_FUTURE_KL_AUC']))
            emit(out/(a.mode+'_fixed_fixture_parity.json'),extra)
        emit(out/(a.mode+'_v3_identity.json'),dict(gate='PASS',mode=a.mode,identity=cap.origins(),FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    except BaseException as e:
        sys.setprofile(None)
        emit(A/'STOP.json',dict(reason=repr(e),traceback=traceback.format_exc(),mode=a.mode,FORMAL_MEASUREMENTS=0,DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',GPU6_FORMAL_AUTHORIZED='NO',git_subprocess_count=cap.git_count if cap else None,FINAL_ACTION='FAIL_AND_STOP'))
        raise
if __name__=='__main__':main()
