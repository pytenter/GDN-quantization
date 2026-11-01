"""Generated-code engineering preflight/strict qualification. No Discovery mode."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, traceback
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent.parent
A=C/'attempt_07_generated_identity'
ROOT=C.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','verify']);parser.add_argument('run_id')
    args=parser.parse_args();run=A/'runs'/args.run_id
    run.mkdir(parents=True,exist_ok=False)
    guard=None;cap=None
    try:
        assert not (A/'STOP.json').exists(),'ATTEMPT07_ALREADY_STOPPED'
        pre=read(A/'GATE0_REVERIFICATION.json');assert pre['OLD_EVIDENCE_INTEGRITY_GATE']=='PASS'
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6'
        profile=read(C/'measurement_launcher_environment.json')
        for key,value in profile['set'].items():assert os.environ.get(key)==value,('ENVIRONMENT_CHANGED',key)
        for key in profile['unset']:assert key not in os.environ,('ENVIRONMENT_CHANGED',key)
        clean=read(A/'CLEAN_ENGINEERING_LAUNCH.json')
        for row in clean['files']:assert sha(C/row['path'])==row['sha256'],('CLEAN_LAUNCH_SOURCE_CHANGED',row['path'])
        launch=read(A/'INPUT_SOURCE_IDENTITY.json')
        for row in launch['files']:assert sha(C/row['path'])==row['sha256'],('ATTEMPT07_SOURCE_CHANGED',row['path'])
        old=read(C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json')
        for row in old['files']:assert sha(C/row['path'])==row['sha256'],('OLD_187_CHANGED',row['path'])
        runtime=read(C/'runtime_environment_manifest.json')
        static={row['path']:row['sha256'] for row in runtime['environment_files']}
        static.update({row['path']:row['sha256'] for row in read(C/'cached_environment_module_manifest.json')['files']})
        static.update({str((C/row['path']).resolve()):row['sha256'] for row in old['files'] if row['path'].endswith('.py')})
        static.update({str((C/row['path']).resolve()):row['sha256'] for row in launch['files'] if row['path'].endswith('.py')})
        for row in read(C/'historical_source_snapshot_manifest.json')['files']:static[str(C/row['snapshot_relative_path'])]=row['snapshot_sha256']
        for row in clean['files']:
            if row['path'].endswith('.py'):static[str((C/row['path']).resolve())]=row['sha256']
        mod=load('attempt07_generated_guard',A/'generated_code_guard_attempt07.py')
        base=load('attempt07_frozen_adapter',C/'sealed_runner_attempt05_gpu6.py')
        base.A=A;base.LAUNCH=C/'attempts/gpu6_file_recovery_attempt05/FAILED_LAUNCH_IDENTITY.json'
        manifest=read(A/'generated_executable_manifest.json') if args.mode=='verify' else None
        if args.mode=='verify':
            freeze=read(A/'GENERATED_CAPSULE_FREEZE.json')
            assert sha(A/'generated_executable_manifest.json')==freeze['GENERATED_EXECUTABLE_CAPSULE_ID']
        versions=dict(runtime['actual']['packages'])
        versions.update({'fla':versions['fla-core'],'yaml':versions['PyYAML'],'jinja2':versions['Jinja2'],'markupsafe':versions['MarkupSafe']})
        guard=mod.GeneratedGuard(static,run/'generated_capture',mode='VERIFY' if manifest else 'CAPTURE',manifest=manifest,packages=versions)
        guard.install()  # MUST precede Torch/FLA/KDA-dependent experiment import.
        class Capsule07(base.Capsule):
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
        cap=Capsule07(run/'qualification');b=base.bind(cap,run/'qualification')
        # Reuse the unchanged historical verify/smoke/parity; never bridge.gate().
        b.verify()
        b.smoke(0)
        baseline=read(C/'known_fixture_manifest.json');expected_path=Path(baseline['result']['path'])
        assert sha(expected_path)==baseline['result']['sha256'],'HISTORICAL_FIXTURE_CHANGED'
        expected=read(expected_path);actual=read(run/'qualification/smoke/repeat0/result.json')
        for key in ['scores','target','endpoint','FP_step_hashes']:assert actual[key]==expected[key],('HISTORICAL_SMOKE_PARITY_FAILURE',key)
        b.parity()
        identity=cap.origins()
        receipt=guard.finish()
        emit(run/'PREFLIGHT_COMPLETE.json',dict(GENERATED_CODE_PREFLIGHT_GATE='PASS',GENERATED_PREEXEC_CAPTURE_GATE='PASS',generated_event_count=receipt['generation_count'],engineering_only=True,FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO',coverage=['Torch','FLA','torch.compiler.disable','KDA model loading','historical K3','historical JPROP','future KL','QDQ','basis','minimal transition replay','minimal functional KL'],identity=identity,historical_smoke_scientific_fields_bitwise_identical=['scores','target','endpoint','FP_step_hashes']))
        print('ATTEMPT07_ENGINEERING_PREFLIGHT_COMPLETE',args.run_id,receipt['generation_count'],flush=True)
    except BaseException as e:
        if guard:
            guard.active=False
            if not guard.closed:guard.finish(enforce_complete=False)
        emit(A/'STOP.json',dict(reason=repr(e),traceback=traceback.format_exc(),mode=args.mode,run_id=args.run_id,FINAL_ACTION='FAIL_AND_STOP',FORMAL_MEASUREMENTS=0,DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',GPU6_FORMAL_AUTHORIZED='NO',ATTEMPT_05='FAIL_PRESERVED',ATTEMPT_06='FAIL_PRESERVED'))
        raise
if __name__=='__main__':main()


