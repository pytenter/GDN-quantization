"""True controller/CPU generation-verifier owning chain. Final scoring remains local; no HMMT gold on55."""
import argparse,csv,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import execution_common as C
from archive_hmmt import publish

def owned(command,name):
    target=ROOT/'runtime'/name;target.mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    with (target/'stdout.log').open('wb') as log:
        child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ident=C.identity(child.pid);C.atomic(target/'STARTED.json',dict(owner=C.identity(os.getpid()),child=ident,command=command,start_monotonic=time.monotonic()))
        last_size=-1;last=time.monotonic()
        pending=None
        while child.poll() is None:
            try:current=C.identity(child.pid)
            except FileNotFoundError:current=None
            if current is None or current['cmdline']==['']:
                if current is not None:
                    assert current['starttick']==ident['starttick'] and current['ppid']==os.getpid(),'Empty argv identity epoch/owner changed'
                if pending is None:pending=time.monotonic()
                assert time.monotonic()-pending<120,'Owning child exit unresolved'
                time.sleep(.05);continue
            assert C.same_process(current,ident),'Live owning child identity changed; no signal to unmatched process'
            size=(target/'stdout.log').stat().st_size
            if size!=last_size:last_size=size;last=time.monotonic()
            if time.monotonic()-last>1800:
                if name=='FORMAL_CONTROLLER':
                    request=ROOT/'runtime/STOP_REQUEST.json'
                    if not request.exists():C.atomic(request,dict(reason='True1800s owning-chain phase no-progress',owner=C.identity(os.getpid())))
                elif C.same_process(C.identity(child.pid),ident):child.terminate()
                code=child.wait(timeout=125)
                C.atomic(target/'WATCHDOG_OWNING_EXIT.json',dict(child=ident,owning_Popen_exit=code))
                raise RuntimeError(name+' true phase no-progress watchdog')
            time.sleep(1)
        code=child.wait()
    C.atomic(target/'OWNING_EXIT.json',dict(status='EXITED',child=ident,owning_Popen_exit=code,observed_monotonic=time.monotonic()))
    return code

def finish(a):
    C.source(ROOT,a.source_sha)
    rc=owned([sys.executable,'-B',str(ROOT/'scripts/verify_generation_cpu.py'),'--source-sha',a.source_sha,'--permit-sha',a.permit_sha],'CPU_GENERATION_VERIFIER')
    assert rc==0,'CPU generation verifier failed'
    receipt=publish(ROOT,seal_manifest(),'GENERATION_DELIVERY')
    C.atomic(ROOT/'runtime/FINAL_ARCHIVE29.json',receipt)
    C.atomic(ROOT/'GENERATION_READY.json',dict(status='READY99_FOR_LOCAL_TWO_FRESH_CPU_SCORES',source_sha256=a.source_sha,permit_sha256=a.permit_sha,requested132=True,admitted99=True,C2_denied=True,archive=receipt))

def seal_manifest():
    late={'GENERATION_MANIFEST.sha256','runtime/PARENT_STDOUT.log','runtime/PARENT_COMPLETE.json','runtime/BASH_WAIT.json','runtime/FINAL_ARCHIVE29.json','GENERATION_READY.json'}
    paths=[str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.') and str(p.relative_to(ROOT)) not in late and 'independent_delivery' not in p.parts]
    C.atomic(ROOT/'FINAL_MANIFEST_SCOPE.json',dict(status='MAIN_DELIVERY_SNAPSHOT',late_owning_wait_and_parent_archive_receipts='Independent immutable sibling after actual owning exit, never invented in main seal',exclusions=sorted(late)))
    paths.append('FINAL_MANIFEST_SCOPE.json')
    with (ROOT/'GENERATION_MANIFEST.sha256').open('x') as f:
        for name in paths:f.write(C.sha(ROOT/name)+'  '+name+'\n')
        f.flush();os.fsync(f.fileno())
    return paths+['GENERATION_MANIFEST.sha256']

def run(a):
    C.source(ROOT,a.source_sha);assert C.sha(ROOT/'GPU_PERMIT.json')==a.permit_sha
    C.atomic(ROOT/'runtime/PARENT_STARTED.json',dict(owner=C.identity(os.getpid()),source_sha256=a.source_sha,permit_sha256=a.permit_sha))
    try:
        rc=owned([sys.executable,'-B',str(ROOT/'scripts/formal_controller.py'),'--source-sha',a.source_sha,'--permit-sha',a.permit_sha],'FORMAL_CONTROLLER')
        if rc!=0:
            C.atomic(ROOT/'runtime/PARTIAL_TERMINAL.json',dict(status='PARTIAL_WITH_VALIDATED_ARTIFACTS',controller_owning_exit=rc,full_accuracy=None,no_GPU_retry=True))
            receipt=publish(ROOT,seal_manifest(),'PARTIAL_GENERATION_DELIVERY')
            C.atomic(ROOT/'runtime/FINAL_ARCHIVE29.json',receipt)
            C.atomic(ROOT/'GENERATION_READY.json',dict(status='PARTIAL_DIAGNOSTIC_NO_FULL_ACCURACY',source_sha256=a.source_sha,permit_sha256=a.permit_sha,archive=receipt))
            return 1
        finish(a)
        C.atomic(ROOT/'runtime/PARENT_COMPLETE.json',dict(status='COMPLETE_TRUE_OWNING_CHAIN',source_sha256=a.source_sha,permit_sha256=a.permit_sha))
        return 0
    except BaseException as exc:
        C.atomic(ROOT/'runtime/PARENT_FIRST_FAILURE.json',dict(status='FAIL_CPU_OR_EXECUTION_DELIVERY',error=repr(exc),traceback=traceback.format_exc(),no_automatic_GPU_or_finisher_retry=True))
        return 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--permit-sha',required=True);raise SystemExit(run(p.parse_args()))
