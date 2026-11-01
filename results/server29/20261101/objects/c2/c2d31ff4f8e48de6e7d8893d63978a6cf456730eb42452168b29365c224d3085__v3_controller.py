"""Minimal launcher scheduling around the unchanged V2 worker."""
import fcntl
import shlex
import subprocess
from v3_common import *
from v3_guard import proc_identity

def alive(t):
    p=RUNS/(t+'.PROCESS_IDENTITY.json')
    if not p.exists(): return False
    v=checked(p)['identity']
    try: return proc_identity(v['pid'])==v
    except FileNotFoundError: return False

def launch(gpu,mode,rep=0):
    cfg,sel,freeze=verify(); e=next(x for x in sel['documents'] if x['gpu']==gpu)
    doc=e['document']; t=tag(mode,doc,gpu,rep)
    with (RUNS/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not any((RUNS/(t+s)).exists() for s in ('.LAUNCH_AUTHORIZATION.json','.STARTED.json','.json','.STOP.json','.log')),'NO_DUPLICATE_NO_RETRY'
        if mode=='qualification':
            assert checked(RUNS/(tag('guardprobe',doc,gpu,0)+'.json'))['status']=='PASS'
            if rep==1: complete(mode,doc,gpu,0)
        if mode in ('micro','formal'):
            assert checked(OUT/'GPU_512X2_QUALIFICATION.json')['status']=='PASS'
        if mode=='formal':
            spec=checked(OUT/'FROZEN_FORMAL_32K_SPEC.json')
            assert checked(OUT/'PHYSICAL_RESOURCE_FEASIBILITY.json')['status']=='PASS' and spec['status']=='FROZEN'
            assert spec['code_commit']==freeze['code_commit']
        hw=device(gpu,cfg)
        # Prelaunch external occupancy is waiting, not a failed measurement.
        # Query/UUID/identity uncertainty remains a hard failure, never safe.
        busy={'UNAUTHORIZED_TARGET_COMPUTE_PID','UNEXPLAINED_TARGET_MEMORY_OR_UTILIZATION'}
        while not hw['exclusive'] and set(hw['failure_conditions']).issubset(busy):
            assert not LATCH.exists(),'V3_STOP_NO_RETRY'
            print('V3_AWAITING_NATURAL_DEVICE_RELEASE',gpu,flush=True)
            time.sleep(30); hw=device(gpu,cfg)
        if not hw['exclusive']:
            stop('PRELAUNCH_RESOURCE_OR_OWNERSHIP_CHECK_FAILED',mode,hardware=hw)
            raise RuntimeError('DEVICE_NOT_SAFE_NO_SIGNAL_NO_ALTERNATE')
        timeout=1800
        if mode=='micro':
            q=checked(OUT/'GPU_512X2_QUALIFICATION.json')
            timeout=max(1800,8*max(x['maximum_block_seconds'] for x in q['documents'])+300)
        if mode=='formal': timeout=checked(OUT/'FROZEN_FORMAL_32K_SPEC.json')['no_progress_seconds'][doc]
        wrapper=CODE/('v3_probe.py' if mode=='guardprobe' else 'v3_worker.py')
        session=f'ling_p1l7_v3_{mode}_gpu{gpu}_fresh{rep}'
        assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
        receipt=dict(epoch=time.time(),gpu=gpu,gpu_uuid=hw['gpu_uuid'],document=doc,mode=mode,repeat=rep,
            session=session,hardware=hw,runner_sha256=digest(CODE/'v2_worker.py'),wrapper_sha256=digest(wrapper),
            freeze_sha256=digest(FREEZE),pair_selection_sha256=digest(OLD/'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json'),
            no_progress_seconds=timeout,administrative_total_wall_cap=None,GPU_HOURS_CAP=None,FIXED_VRAM_PERCENT_CAP=None,
            formal_spec_sha256=digest(OUT/'FROZEN_FORMAL_32K_SPEC.json') if mode=='formal' else None)
        emit(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'),receipt)
        args=['env','PYTHONDONTWRITEBYTECODE=1',f'CUDA_VISIBLE_DEVICES={gpu}','PYTHONHASHSEED=0',
            'LING_GATED_RMSNORM_FIXED_CONFIG=BT16_W8','CUBLAS_WORKSPACE_CONFIG=:4096:8',
            '/data/zypan/envs/ling-kda/bin/python','-B','-u',str(wrapper),'--document',doc,'--gpu',str(gpu),'--repeat',str(rep)]
        if mode!='guardprobe': args+=['--mode',mode]
        command='cd '+shlex.quote(str(CODE))+' && exec '+shlex.join(args)+' > '+shlex.quote(str(RUNS/(t+'.log')))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
        print('V3_LAUNCHED',mode,gpu,rep,flush=True)
    return t

def wait(t):
    while True:
        assert not LATCH.exists(),'V3_STOP_NO_RETRY'
        fail=RUNS/(t+'.STOP.json'); done=RUNS/(t+'.json')
        if fail.exists(): stop('WORKER_STOP','supervision',run_stop=str(fail)); raise RuntimeError('WORKER_STOP')
        extra=RUNS/(t+'.REDUCER_DIAGNOSTIC.json')
        ready=done.exists() and done.with_suffix('.sha256.json').exists()
        if t.startswith('qualification'): ready=ready and extra.exists() and extra.with_suffix('.sha256.json').exists()
        if ready and not alive(t):
            assert checked(done)['status']=='PASS'; return checked(done),done
        if not ready:
            started=RUNS/(t+'.STARTED.json')
            if started.exists() and not alive(t):
                stop('WORKER_DISAPPEARED_WITHOUT_COMPLETION','supervision',tag=t); raise RuntimeError('UNKNOWN_WORKER_EXIT')
            receipt=checked(RUNS/(t+'.LAUNCH_AUTHORIZATION.json'))
            log=RUNS/(t+'.log')
            last_update=max(receipt['epoch'],log.stat().st_mtime if log.exists() else receipt['epoch'])
            if time.time()-last_update>receipt['no_progress_seconds']:
                stop('NO_PROGRESS_WATCHDOG_CPU_SUPERVISOR','supervision',tag=t,last_progress_epoch=last_update,
                    active_worker_not_signaled=True,user_action_required_if_kernel_cannot_reach_safe_boundary=True)
                raise RuntimeError('NO_PROGRESS_WATCHDOG_CPU_SUPERVISOR')
        time.sleep(5)

def aggregate_qualification():
    cfg,sel,freeze=verify(); rows=[]; diagnostics=[]
    for e in sel['documents']:
        a,p=complete('qualification',e['document'],e['gpu'],0); b,q=complete('qualification',e['document'],e['gpu'],1)
        assert a['numeric_fingerprint']==b['numeric_fingerprint'],'FRESH_PROCESS_EXACT_REPEAT_FAIL'
        assert a['stream']['fp_replay_parity_tokens']==b['stream']['fp_replay_parity_tokens']==512
        at,bt=tag('qualification',e['document'],e['gpu'],0),tag('qualification',e['document'],e['gpu'],1)
        ai=checked(RUNS/(at+'.PROCESS_IDENTITY.json'))['identity']; bi=checked(RUNS/(bt+'.PROCESS_IDENTITY.json'))['identity']
        assert ai!=bi and ai['pid']!=bi['pid']
        ad=checked(RUNS/(at+'.REDUCER_DIAGNOSTIC.json')); bd=checked(RUNS/(bt+'.REDUCER_DIAGNOSTIC.json'))
        assert ad['samples']==bd['samples'],'DIAGNOSTIC_FRESH_PROCESS_REPEAT_FAIL'
        diagnostics.extend(ad['samples'])
        rows.append(dict(document=e['document'],gpu=e['gpu'],gpu_uuid=a['gpu_uuid'],status='PASS',
            canonical_sha256=[digest(p),digest(q)],canonical_paths=[str(p),str(q)],fresh_process_exact=True,
            numeric_fingerprint=a['numeric_fingerprint'],persistent_integrity=a['persistent_integrity'],
            maximum_block_seconds=max(x['elapsed_seconds'] for v in (a,b) for x in v['stream']['blocks']),
            runtime_telemetry=a['runtime_telemetry']+b['runtime_telemetry']))
    emit(OUT/'GPU_512X2_QUALIFICATION.json',dict(status='PASS',documents=rows,all18_KDA_6_MLA_conv=True,
        same_gpu_live_FP_replay_exact=True,fresh_process_512x2_exact=True,engine_sha256=ENGINE_SHA,code_commit=freeze['code_commit'],
        reducer_diagnostics=diagnostics,FP32_carrier_not_physical_INT8=True))
    print('V3_BOTH_GPU_512X2_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=('launch','aggregate'))
    ap.add_argument('--gpu',type=int,choices=(3,6)); ap.add_argument('--mode',choices=('guardprobe','qualification','micro','formal'))
    ap.add_argument('--repeat',type=int,default=0); a=ap.parse_args()
    if a.action=='launch': launch(a.gpu,a.mode,a.repeat)
    else: aggregate_qualification()
