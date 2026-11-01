"""Human-requested continuation entry in a separate output root. No retries."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from resume_adapter import (PHASE_A,PHASE_D,CHECKPOINT_SHA,PARENT_SOURCE_SHA,
                            PARENT_PREREG_SHA,GPU_UUID,build_run,build_install,
                            bind_runtime,digest,need)

ROOT=Path(__file__).resolve().parents[1]


def load_qualified(root):
    path=Path(root)/'control/worker_entry.py'
    manifest=json.loads((Path(root)/'SOURCE_MANIFEST.json').read_text())
    expected=next(a for a in manifest['execution_sources'] if a['path']=='control/worker_entry.py')
    need(path.stat().st_size==expected['bytes'] and digest(path)==expected['sha256'],'Frozen original entry changed')
    spec=importlib.util.spec_from_file_location('original_c6r_qualified_entry',path)
    E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    return E


def compile_recovery_install(E):
    old,renamed,_=E.adapted_run();candidate,run_proof=build_run(renamed)
    install_node=next(n for n in ast.parse(Path(E.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='install')
    adapted,install_proof=build_install(install_node)
    def source():return old,candidate,{'H_C128':1,'H_DUAL_M':1}
    E.__dict__['_continuation_adapted_run']=source
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapted],type_ignores=[])),str(ROOT/'control/recovery_worker_entry.py'),'exec'),E.__dict__)
    return dict(status='PASS_CPU_AST_ONLY',run=run_proof,install=install_proof,no_GPU=True,original_scientific_core_unchanged=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0]);p.add_argument('--audit',action='store_true');p.add_argument('--audit-source-root',default=str(ROOT));p.add_argument('phase',nargs='?',choices=[PHASE_A,PHASE_D]);a=p.parse_args()
    if a.audit:
        E=load_qualified(a.audit_source_root);proof=compile_recovery_install(E)
        print(json.dumps(proof,indent=2,allow_nan=False));return
    need(a.gpu==0 and a.phase in (PHASE_A,PHASE_D),'Exact remaining Doc5/GPU0 phase required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Continuation must expose original physicalGPU0 only')
    need(not (ROOT/'FIRST_FAILURE.json').exists(),'Continuation first failure prohibits any later entry')
    admission=json.loads((ROOT/'CONTINUATION_ADMISSION.json').read_text())
    need(admission['status']=='PASS','Continuation gates not passed')
    need(digest(ROOT/'SOURCE_MANIFEST.json')==PARENT_SOURCE_SHA and digest(ROOT/'PREREGISTRATION.json')==PARENT_PREREG_SHA,'Original scientific identity aliases changed')
    need(admission['input_checkpoint']['sha256']==CHECKPOINT_SHA,'Continuation checkpoint binding changed')
    frozen=json.loads((ROOT/'CONTINUATION_SOURCE_MANIFEST.json').read_text())
    for row in frozen['execution_sources']:
        path=ROOT/row['path'];need(path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],'Continuation source changed: '+row['path'])
    E=load_qualified(ROOT);proof=compile_recovery_install(E)
    W,K,output,waits,hooked=E.install(a.gpu,a.phase)
    bind_runtime(W,K,ROOT,admission,a.phase)
    old_science=W.science
    def science(model,qmod,phase):
        result=old_science(model,qmod,phase)
        actual=K.read(ROOT/'analysis'/(phase+'.json'))
        fields=('resume_cursor','new_model_tokens','prefix_score_count','new_score_count','new_writebacks','new_next_token_consumption_checks','restore_audit_status','rotation_calls')
        return dict(result,**{k:actual[k] for k in fields},continuation=actual['continuation'])
    W.science=science
    K.save('analysis/CONTINUATION_ADAPTER_'+a.phase+'.json',dict(proof,phase=a.phase,input_checkpoint_sha256=CHECKPOINT_SHA,physical_GPU_ID=0,GPU_UUID=GPU_UUID,scope='CPU AST source proof; actual checkpoint CPU/GPU byte equality verified inside run before first suffix token',continuation_manifest_sha256=digest(ROOT/'CONTINUATION_SOURCE_MANIFEST.json'),admission_sha256=digest(ROOT/'CONTINUATION_ADMISSION.json')))
    sys.argv=[str(ROOT/'src/worker.py'),a.phase]
    try:
        W.main()
        K.check(hooked==[True] and sys.gettrace() is None,'Original checkpoint hook not removed exactly once')
    finally:
        sys.settrace(None)
        K.save('analysis/CHECKPOINT_LOCK_'+a.phase+'.json',dict(status='MEASURED',phase=a.phase,physical_GPU_ID=0,GPU_UUID=GPU_UUID,checkpoint_function_body_unchanged=True,lock_scope='Complete original checkpoint function',main_call_hook_count=len(hooked),entry_trace_removed_before_main=True,calls=waits,total_lock_wait_seconds=sum(r['lock_wait_seconds'] for r in waits)))


if __name__=='__main__':main()
