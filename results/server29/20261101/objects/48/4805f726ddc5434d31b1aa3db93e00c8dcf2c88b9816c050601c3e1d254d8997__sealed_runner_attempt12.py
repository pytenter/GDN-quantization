"""Generated-code engineering preflight/strict qualification. No Discovery mode."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, traceback
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent.parent
A=C/'attempt_12_alias_namespace_relocation'
ROOT=C.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def prepare_output_directory(run):
    # Engineering lifecycle only; historical storage/science validation unchanged.
    out=Path(run)/'qualification'
    out.mkdir(exist_ok=False)
    return out

def synchronize_capture_bindings(capture,canonical,frozen_sources):
    # Frozen transport creates a qualified relocated common instance, but an
    # earlier cached `from common import *` retains the original namespace.
    # Rebind operational aliases, never edit a historical source/config file.
    assert capture.LAYERS==canonical.LAYERS,'SCIENTIFIC_ALIAS_VALUE_MUTATION'
    evidence=[]
    for name in ['runtime','imp']:
        old=getattr(capture,name);new=getattr(canonical,name)
        assert old.__code__==new.__code__,('ALIAS_FUNCTION_BODY_CHANGED',name)
        for function in [old,new]:
            p=str(Path(function.__code__.co_filename).resolve())
            assert p in frozen_sources and sha(p)==frozen_sources[p],('UNQUALIFIED_ALIAS_SOURCE',name,p)
    for function in [capture.Capture.__init__,canonical.load]:
        p=str(Path(function.__code__.co_filename).resolve())
        assert p in frozen_sources and sha(p)==frozen_sources[p],('UNQUALIFIED_ALIAS_SOURCE','constructor_or_load',p)
    for name in ['P','OLD','V4','V3','ENV','PYTHON','MODEL','PARENTS']:
        old=getattr(capture,name);new=getattr(canonical,name)
        if isinstance(new,Path):assert isinstance(old,Path)
        else:assert isinstance(new,list) and isinstance(old,list) and all(isinstance(v,Path) for v in new+old)
        evidence.append(dict(binding=name,previous=str(old),relocated=str(new),role='existing qualified transport operational asset reference'))
    for name in ['P','OLD','V4','V3','ENV','PYTHON','MODEL','PARENTS','runtime','imp','load']:
        setattr(capture,name,getattr(canonical,name))
    return dict(bindings=evidence,source_bytes_modified=False,scientific_values_modified=False,
                policy='reuse existing exact frozen canonical transport bindings; no path whitelist or dependency expansion')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','verify']);parser.add_argument('run_id')
    args=parser.parse_args();run=A/'runs'/args.run_id
    run.mkdir(parents=True,exist_ok=False)
    prepare_output_directory(run)
    guard=None;cap=None
    try:
        assert not (A/'STOP.json').exists(),'ATTEMPT12_ALREADY_STOPPED'
        pre=read(A/'GATE0_REVERIFICATION.json');assert pre['OLD_EVIDENCE_INTEGRITY_GATE']=='PASS'
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
        profile=read(C/'measurement_launcher_environment.json')
        for key,value in profile['set'].items():assert os.environ.get(key)==value,('ENVIRONMENT_CHANGED',key)
        for key in profile['unset']:assert key not in os.environ,('ENVIRONMENT_CHANGED',key)
        clean=read(A/'CLEAN_ENGINEERING_LAUNCH.json')
        for row in clean['files']:assert sha(C/row['path'])==row['sha256'],('CLEAN_LAUNCH_SOURCE_CHANGED',row['path'])
        launch=read(A/'INPUT_SOURCE_IDENTITY.json')
        for row in launch['files']:assert sha(C/row['path'])==row['sha256'],('ATTEMPT12_SOURCE_CHANGED',row['path'])
        old=read(C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json')
        for row in old['files']:assert sha(C/row['path'])==row['sha256'],('OLD_187_CHANGED',row['path'])
        runtime=read(C/'runtime_environment_manifest.json')
        static={row['path']:row['sha256'] for row in runtime['environment_files']}
        static.update({row['path']:row['sha256'] for row in read(C/'cached_environment_module_manifest.json')['files']})
        static.update({str((C/row['path']).resolve()):row['sha256'] for row in old['files'] if row['path'].endswith('.py')})
        instrumentation={str((A/name).resolve()) for name in ['generated_code_guard_attempt12.py','sealed_runner_attempt12.py']}
        static.update({str((C/row['path']).resolve()):row['sha256'] for row in launch['files'] if str((C/row['path']).resolve()) in instrumentation})
        for row in read(C/'historical_source_snapshot_manifest.json')['files']:static[str(C/row['snapshot_relative_path'])]=row['snapshot_sha256']
        for row in clean['files']:
            if str((C/row['path']).resolve()) in instrumentation:static[str((C/row['path']).resolve())]=row['sha256']
        mod=load('attempt12_generated_guard',A/'generated_code_guard_attempt12.py')
        base=load('attempt12_frozen_adapter',C/'sealed_runner_attempt05_gpu6.py')
        base.A=A;base.LAUNCH=C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json'
        manifest=read(A/'generated_executable_manifest.json') if args.mode=='verify' else None
        if args.mode=='verify':
            freeze=read(A/'GENERATED_CAPSULE_FREEZE.json')
            assert sha(A/'generated_executable_manifest.json')==freeze['GENERATED_EXECUTABLE_CAPSULE_ID']
        versions=dict(runtime['actual']['packages'])
        versions.update({'fla':versions['fla-core'],'yaml':versions['PyYAML'],'jinja2':versions['Jinja2'],'markupsafe':versions['MarkupSafe']})
        guard=mod.GeneratedGuard(static,run/'generated_capture',mode='VERIFY' if manifest else 'CAPTURE',manifest=manifest,packages=versions)
        guard.install()  # MUST precede Torch/FLA/KDA-dependent experiment import.
        class Capsule08(base.Capsule):
            def module_file(self,raw,name):
                p=Path(raw);real=str(p.resolve())
                if p.suffix in ('.so','.pyd'):
                    return super().module_file(raw,name)
                if name.startswith('transformers_modules.') and p.is_file() and sha(p) in self.model_source_hashes:
                    # Same existing frozen official model source content identity.
                    guard.static[real]=sha(p)
                kind=guard.module_source(p,name)
                self.code_events.append(dict(module=name,path=str(p),sha256=sha(p),project_source=real in self.files,identity_class=kind))
            def origins(self):
                origins=[]
                assert not any(str(p).startswith(str(base.LEGACY)) for p in sys.path),'DIRTY_SYS_PATH'
                for name,module in list(sys.modules.items()):
                    raw=getattr(module,'__file__',None)
                    if not raw:continue
                    p=Path(raw)
                    if p.suffix=='.pyc':
                        source=Path(importlib.util.source_from_cache(str(p)))
                        if source.is_file():p=source
                    real=str(p.resolve())
                    if real in guard.sources:
                        event=guard.sources[real]
                        if p.is_file():assert sha(p)==event['source_sha256'],'GENERATED_SOURCE_CHANGED'
                        origins.append(dict(module=name,path=str(p),sha256=event['source_sha256'],identity_class='GENERATED_EXECUTABLE_FROM_FROZEN_PRODUCER',captured_copy=event['canonical_captured_copy_path']))
                    elif p.is_file():
                        if real in self.shared_hash or str(p) in self.envfiles or real in guard.static:
                            h=sha(p)
                            expected=self.shared_hash.get(real,self.envfiles.get(str(p),guard.static.get(real)))
                            assert h==expected,('MODULE_HASH_CHANGED',name,str(p))
                            origins.append(dict(module=name,path=str(p),sha256=h,identity_class='FROZEN_PROJECT_SOURCE' if real in self.files else 'STATIC_FROZEN_SOURCE'))
                        else:raise RuntimeError('UNKNOWN_IMPORTED_PYTHON_MODULE: '+name+': '+str(p))
                    elif str(p).startswith(str(base.ENV)):continue  # Existing package-owned absent-cache policy.
                    else:raise RuntimeError('UNKNOWN_IMPORTED_PYTHON_MODULE_MISSING: '+name+': '+str(p))
                libs=[]
                for line in Path('/proc/self/maps').read_text().splitlines():
                    parts=line.split(None,5)
                    if len(parts)!=6 or not parts[5].startswith('/') or '.so' not in parts[5]:continue
                    p=Path(parts[5]);real=str(p.resolve())
                    if real in {row['realpath'] for row in libs}:continue
                    h=sha(p);expected=self.shared_hash.get(real,self.envfiles.get(str(p)))
                    if expected is None:raise RuntimeError('UNKNOWN_LOADED_SHARED_LIBRARY: '+str(p))
                    assert h==expected,('NATIVE_LIBRARY_HASH_CHANGED',real)
                    libs.append(dict(path=str(p),realpath=real,sha256=h))
                return dict(module_files=origins,source_and_extension_loader_events=self.code_events,loaded_libraries=libs,git_subprocess_count=self.git_count,subprocess_events=self.events,sys_path=list(sys.path),ambient_regular_file_reads_observed=False,generated_events=guard.events)
        cap=Capsule08(run/'qualification');b=base.bind(cap,run/'qualification')
        inherited_modules=b.modules
        def qualified_alias_modules():
            result=inherited_modules();_,_,canonical,capture,_,_=result
            evidence=synchronize_capture_bindings(capture,canonical,cap.files)
            receipt=run/'qualification/OPERATIONAL_ALIAS_RELOCATION.json'
            if not receipt.exists():emit(receipt,evidence)
            return result
        b.modules=qualified_alias_modules
        # Reuse the unchanged historical verify/smoke/parity; never bridge.gate().
        b.verify()
        b.smoke(0)
        baseline=read(C/'known_fixture_manifest.json');expected_path=Path(baseline['result']['path'])
        assert sha(expected_path)==baseline['result']['sha256'],'HISTORICAL_FIXTURE_CHANGED'
        expected=read(expected_path);actual=read(run/'qualification/smoke/repeat0/result.json')
        for key in ['scores','target','endpoint','FP_step_hashes']:assert actual[key]==expected[key],('HISTORICAL_SMOKE_PARITY_FAILURE',key)
        b.parity()
        # Required V3 diagnostics reproduce the unchanged Attempt05 fixed fixture.
        # These are engineering qualification evidence, never Discovery data.
        out=run/'qualification'
        fields=['scores','target','endpoint','FP_step_hashes']
        co,t,c,capture,core,stats=b.modules();q,r,f=b.historical();torch=core.torch
        query=torch.tensor([[[1.0,2.0,-1.0],[0.5,-0.5,1.0]]])
        state=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)/10.0
        got=q.query_readout(state,query);expected=torch.sum(state*query.unsqueeze(-1),dim=-2).unsqueeze(1)
        assert torch.allclose(got,expected)  # unchanged historical fixture tolerance
        delta=got-expected
        trace=read(out/'smoke'/'repeat0'/'trace/trace_freeze.json')
        x=torch.load(trace['chunks'][0]['artifact']['path'],map_location='cuda',weights_only=False)
        bank=b.read(out/'rotation_manifest.json')['bank']
        rot=torch.load(t.mapped(bank['conditions'][0]['artifact']['path']),map_location='cpu',weights_only=False)['rotations']
        R=torch.stack([rot[l] for l in b.LAYERS]).cuda()
        states=core.initial_states(8,'cuda');signs=core.sign_schedule(8,32).cuda()
        for j in range(32):
            rec={k:x[k][:,j:j+1].contiguous() for k in ['q','k','v','beta','log_decay']}
            values=core.replay_step(*states,rec,R,signs[j]);states=values[:4]
        extra=dict(HISTORICAL_K3_PARITY_GATE='PASS',max_abs=float(delta.abs().max()),relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected)),expected_K3_sha256=b.tensor_sha(expected),actual_K3_sha256=b.tensor_sha(got),JPROP_fixed_fixture_state_hashes=[b.tensor_sha(v) for v in values],JPROP_fixed_fixture_metrics=values[4].cpu().tolist(),FUTURE_KL_HARNESS_PARITY_GATE='PASS',historical_expected_result_sha256=baseline['result']['sha256'],historical_scientific_fields_bitwise_identical=fields,future_KL=dict(direction='ref||pulse',dtype='FP32 full_vocabulary per-token reduction',AUC='streaming FP64 sum; no averaging/no clamping',support='anchor+1 through endpoint inclusive',teacher_forcing=True,fixed_fixture_AUC=actual['target']['TARGET_FUTURE_KL_AUC']))
        emit(out/'fixed_fixture_numerical_parity.json',extra)
        identity=cap.origins()
        receipt=guard.finish()
        emit(run/'PREFLIGHT_COMPLETE.json',dict(GENERATED_CODE_PREFLIGHT_GATE='PASS',GENERATED_PREEXEC_CAPTURE_GATE='PASS',generated_event_count=receipt['generation_count'],engineering_only=True,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',coverage=['Torch','FLA','torch.compiler.disable','KDA model loading','historical K3','historical JPROP','future KL','QDQ','basis','minimal transition replay','minimal functional KL'],identity=identity,historical_smoke_scientific_fields_bitwise_identical=['scores','target','endpoint','FP_step_hashes']))
        print('ATTEMPT12_ENGINEERING_PREFLIGHT_COMPLETE',args.run_id,receipt['generation_count'],flush=True)
    except BaseException as e:
        if guard:
            guard.active=False
            if not guard.closed:guard.finish(enforce_complete=False)
        emit(A/'STOP.json',dict(reason=repr(e),traceback=traceback.format_exc(),mode=args.mode,run_id=args.run_id,FINAL_ACTION='FAIL_AND_STOP',FORMAL_MEASUREMENTS=0,DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',GPU6_FORMAL_AUTHORIZED='NO',ATTEMPT_05='FAIL_PRESERVED',ATTEMPT_06='FAIL_PRESERVED'))
        raise
if __name__=='__main__':main()
