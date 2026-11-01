"""CPU-only one-shot transition, only after Stage2 PASS and natural exits."""
import os
import shlex
import subprocess
import time
import traceback
from v4_common import CODE,OUT,RUNS,FREEZE,LATCH,checked,digest,emit,alive,stop,verify

FILES=['v4_persistent_common.py','v4_persistent_worker.py','v4_persistent_controller.py',
       'v4_persistent_stages.py','v4_persistent_pipeline.py','v4_terminal_report.py',
       'freeze_v4_persistent.py','v4_transition_after_portability.py']

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    receipt=OUT/'V4_CPU_TRANSITION_STARTED.json'
    assert not receipt.exists(),'NO_DUPLICATE_TRANSITION'
    freeze=checked(FREEZE); pending={str(CODE/n):digest(CODE/n) for n in FILES}
    emit(receipt,dict(pid=os.getpid(),epoch=time.time(),stage2_commit=freeze['code_commit'],pending_code_sha256=pending,
        authorized_action='append-only commit/freeze Stage3 and start serial gated persistent pipeline only after Stage2 PASS'))
    try:
        while True:
            assert not LATCH.exists(),'PORTABILITY_STOP_NO_TRANSITION'
            p=OUT/'P1_L7_PORTABILITY_GATE.json'; amendment=OUT/'HARDWARE_MAPPING_AMENDMENT.json'
            ready=p.exists() and p.with_suffix('.sha256.json').exists() and amendment.exists() and amendment.with_suffix('.sha256.json').exists()
            if ready:
                assert checked(p)['status']=='PASS' and checked(amendment)['status']=='APPROVED_PROSPECTIVE_V4_ONLY'
                tags=[p.name.removesuffix('.PROCESS_IDENTITY.json') for p in RUNS.glob('portability_*.PROCESS_IDENTITY.json')]
                no_workers=len(tags)==2 and not any(alive(t) for t in tags)
                no_controller=subprocess.run(['tmux','has-session','-t','ling_p1l7_v4_portability'],capture_output=True).returncode!=0
                if no_workers and no_controller: break
            time.sleep(5)
        verify()
        for p,h in pending.items(): assert digest(p)==h,('PENDING_SOURCE_CHANGED',p)
        assert not subprocess.check_output(['git','-C',str(CODE),'diff','--name-only'],text=True).strip(),'UNEXPECTED_TRACKED_EDITS'
        assert not subprocess.check_output(['git','-C',str(CODE),'diff','--cached','--name-only'],text=True).strip(),'UNEXPECTED_STAGED_FILES'
        assert not (OUT/'V4_PERSISTENT_CODE_FREEZE.json').exists() and not (OUT/'V4_PERSISTENT_SUPERVISOR_STARTED.json').exists()
        subprocess.run(['git','-C',str(CODE),'add',*FILES],check=True)
        subprocess.run(['git','-C',str(CODE),'-c','user.name=Codex','-c','user.email=codex@local.invalid','commit',
            '-m','V4 qualified serial full persistent pipeline with original immutable numerics'],check=True)
        python='/data/zypan/envs/ling-kda/bin/python'
        subprocess.run(['env','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=',python,'-B',str(CODE/'freeze_v4_persistent.py')],cwd=CODE,check=True)
        args=['env','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=',python,'-B','-u',str(CODE/'v4_persistent_pipeline.py')]
        command='cd '+shlex.quote(str(CODE))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(OUT/'persistent_pipeline.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s','ling_p1l7_v4_persistent',command],check=True)
        emit(OUT/'V4_CPU_TRANSITION_COMPLETE.json',dict(status='PASS',epoch=time.time(),
            new_commit=checked(OUT/'V4_PERSISTENT_CODE_FREEZE.json')['code_commit'],stage2_completed_and_no_workers=True,
            freeze_sha256=digest(OUT/'V4_PERSISTENT_CODE_FREEZE.json'),pipeline_session='ling_p1l7_v4_persistent'))
        print('V4_TRANSITION_PERSISTENT_STARTED',flush=True)
    except Exception as exc:
        # No numerical retry and no mutation of any historical STOP.
        if not (OUT/'V4_CPU_TRANSITION_STOP.json').exists():
            emit(OUT/'V4_CPU_TRANSITION_STOP.json',dict(status='STOP',reason=repr(exc),traceback=traceback.format_exc(),epoch=time.time()))
        if not LATCH.exists(): stop(repr(exc),'CPU_TRANSITION')
        raise

if __name__=='__main__': main()
