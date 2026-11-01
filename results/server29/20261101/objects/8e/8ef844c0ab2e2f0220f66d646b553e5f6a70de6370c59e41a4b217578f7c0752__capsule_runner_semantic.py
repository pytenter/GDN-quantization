"""Append-only V2 operational adapter. Scientific bridge bytes unchanged."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
A1=Path(__file__).resolve().parent
E=A1.parents[1]
ROOT=E.parents[1]
A=E/'provenance/execution_capsule_v2/classifier_correction_v1'
spec=importlib.util.spec_from_file_location('capsule_artifact_semantics_v2',A/'capsule_v2.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
OLD='c9c8e3199e0d017b9712809b77d8f16cbd07f165'


def read(p):return json.loads(Path(p).read_text())


def guard():
    expected=read(A/'new_premeasurement_commit.json')['commit'] if (A/'new_premeasurement_commit.json').exists() else OLD
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==expected
    assert subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True).strip()==''
    assert v.sha(E/'scripts/bridge.py')=='e4baf3f7e3ecc19f014434f8c28f1ecc77e87651487c547f9cd107c4f3eeafedd'
    for x in read(E/'historical_assets/asset_manifest.json')['files']:
        assert v.sha(E/'historical_assets/snapshot'/x['relative_path'])==x['sha256']
    assert read(E/'STOP.json')['FINAL_ACTION']=='FAIL_AND_STOP'
    assert read(A1/'run/FINAL_REPORT.json')['FINAL_ACTION']=='FAIL_AND_STOP'


def baseline():
    import locale
    modules=[]
    for name,m in list(sys.modules.items()):
        p=getattr(m,'__file__',None)
        if p and Path(p).is_file():
            modules.append(dict(module=name,path=str(Path(p).resolve()),sha256=v.sha(p)))
    envkeys=['PYTHONPATH','PYTHONHOME','PYTHONHASHSEED','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE',
        'CUDA_VISIBLE_DEVICES','LING_GATED_RMSNORM_FIXED_CONFIG','FLA_CONFIG_DIR','FLA_CACHE_MODE',
        'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING',
        'PYTORCH_CUDA_ALLOC_CONF','TRITON_CACHE_DIR','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT',
        'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PATH',
        'LANG','LC_ALL','LC_CTYPE','LC_NUMERIC','LC_COLLATE']
    return dict(executable=sys.executable,real_executable=str(Path(sys.executable).resolve()),
        executable_sha256=v.sha(sys.executable),python=sys.version,cwd=str(Path.cwd()),
        sys_path=list(sys.path),environment={k:os.environ.get(k) for k in envkeys},modules=modules,
        active_locale=locale.setlocale(locale.LC_ALL),
        active_locale_categories={k:locale.setlocale(getattr(locale,k)) for k in ('LC_CTYPE','LC_NUMERIC','LC_COLLATE')},
        numerical_locale_configuration=locale.localeconv())


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','smoke','parity'])
    p.add_argument('--repeat',type=int);p.add_argument('--run-root',required=True)
    p.add_argument('--baseline-expectation');a=p.parse_args()
    r=Path(a.run_root).resolve();assert r.is_relative_to(A.resolve())
    r.mkdir(parents=True,exist_ok=True)
    stage=a.mode+(str(a.repeat) if a.repeat is not None else '')
    audit_spec=importlib.util.spec_from_file_location('append_only_python_origin_audit',A/'python_origin_audit.py')
    audit=importlib.util.module_from_spec(audit_spec);audit_spec.loader.exec_module(audit)
    audit_channel=audit.install()
    startup=baseline();v.write(r/('startup_'+stage+'.json'),startup)
    try:
        if a.baseline_expectation:
            old=read(a.baseline_expectation)
            assert startup==old,'STARTUP_BASELINE_ENV_SYSPATH_MODULE_BYTES_MISMATCH'
        guard()
        s=importlib.util.spec_from_file_location('unchanged_frozen_bridge',E/'scripts/bridge.py')
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        oldread=m.read; oldimp=m.imp
        inputs={'source_identity.json','document_manifest.json','rotation_manifest.json','runtime_manifest.json','preregistration.json'}
        def routed_read(path):
            path=Path(path)
            if path.parent==r and path.name in inputs:path=E/path.name
            if path.parent==r/'historical_assets':path=E/'historical_assets'/path.name
            return oldread(path)
        historical=set()
        def imported(path,name):
            result=oldimp(path,name)
            path=Path(path).resolve()
            if path.is_relative_to((E/'historical_assets/snapshot').resolve()):
                historical.add(str(path.relative_to(E/'historical_assets/snapshot')))
            return result
        def inventory():
            return dict(historical_executed_files=sorted(historical),execution_module_files=[],
                artifact_closure='INDEPENDENT_SYSCALL_SUPERVISOR_V2',old_dirty_directory_executed=False,
                execution_source_event_tracing=False)
        m.D=r;m.ROOT=ROOT;m.SNAP=E/'historical_assets/snapshot'
        m.guard=guard;m.read=routed_read;m.imp=imported;m.execution_inventory=inventory
        m.profile=None  # Retire event/call tracing, not a filename exception.
        if a.mode=='verify':m.verify()
        elif a.mode=='smoke':m.smoke(a.repeat)
        elif a.mode=='parity':m.parity()
        final=baseline()
        assert final['sys_path']==startup['sys_path'],'UNEXPECTED_RUNTIME_SYSPATH_ESCAPE'
        for field in ('environment','active_locale','active_locale_categories','numerical_locale_configuration'):
            assert final[field]==startup[field],('LOCALE_OR_ENVIRONMENT_CHANGED',field)
        v.write(r/('loaded_modules_'+stage+'.json'),final)
        guard()
    except BaseException as exc:
        v.write(r/('STOP_'+stage+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),
            stage=stage,FINAL_ACTION='FAIL_AND_STOP',FORMAL_MEASUREMENTS=0,no_auto_retry=True))
        raise


if __name__=='__main__':main()
