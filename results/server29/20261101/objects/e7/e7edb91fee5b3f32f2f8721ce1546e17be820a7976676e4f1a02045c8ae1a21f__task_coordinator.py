"""Offline stage dispatcher for the one frozen worker; no Torch imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

TASK = 'QWEN_GDN_AIME26_STEPQUANT6_VS_H_DUALM_2SEED_V1'
PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
STOP_SIGNAL = None


def signal_stop(signum, _frame):
    # A flag avoids an exception between Popen and registering its child.
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def read(p):
    return json.loads(Path(p).read_text())


def save(p, d, new=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if new:
        try: os.link(tmp, p)
        finally: tmp.unlink()
    else:
        os.replace(tmp, p)


def nvml():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    return [dict(pid=int(a[0]), UUID=a[1], process=a[2], memory_MiB=a[3])
            for line in p.stdout.splitlines() if line.strip()
            for a in [[s.strip() for s in line.split(',')]]]


def nvml_safe():
    try:return nvml(), None
    except Exception:return [], traceback.format_exc()


def identity(pid):
    try:
        raw = Path('/proc/'+str(pid)+'/stat').read_text()
        start = raw[raw.rfind(')')+2:].split()[19]
        cmd = Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return dict(start_tick=start, cmdline=cmd)
    except FileNotFoundError:
        return None


def worker_phase(root, mode, gpu):
    return root/('formal/worker'+str(gpu) if mode=='formal' else 'qualification/gpu'+str(gpu)+'/'+mode)


def run(args):
    root = args.root.resolve(); auth = read(args.authorization)
    if auth.get('task_id')!=TASK or auth.get('status')!='PASS' or args.mode not in auth.get('allowed_modes',[]):
        raise ValueError('Explicit matching root authorization required')
    gpus = auth['dispatch_GPUs']
    if not gpus or len(set(gpus))!=len(gpus) or any(g not in range(4) for g in gpus):
        raise ValueError('Invalid fixed physical GPU roster')
    if args.mode=='formal' and gpus!=[0,1,2,3]:
        raise ValueError('Formal execution requires all four fixed GPUs')
    phase = root/'execution'/args.mode
    if (root/'FIRST_FAILURE.json').exists():
        raise RuntimeError('Prior task failure is terminal; no automatic retry')
    if (phase/'STARTED.json').exists():
        raise RuntimeError('Refuse restarting an active/completed/failed stage')
    requested = {auth['gpu_uuid_map'][str(g)] for g in gpus}
    if any(a['UUID'] in requested for a in nvml()):
        raise RuntimeError('Requested physical GPU is occupied; no preemption')
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True,timeout=10)
    physical = {int(p[0]):p for s in rows.splitlines() for p in [[x.strip() for x in s.split(',')]]}
    for g in gpus:
        if physical[g][1]!=auth['gpu_uuid_map'][str(g)] or '3090' not in physical[g][2]:
            raise RuntimeError('Fixed RTX3090 physical identity changed')
    ledger_path = root/'GPU_RESOURCE_LEDGER.json'; ledger = read(ledger_path)
    history = ledger.get('dispatches', [])
    if any(r.get('stage')==args.mode for r in history):
        raise RuntimeError('Ledger already contains this stage; never rerun a unit')
    save(phase/'STARTED.json',dict(task_id=TASK,mode=args.mode,pid=os.getpid(),identity=identity(os.getpid()),
                                unix=time.time(),authorization_sha256=hashlib.sha256(args.authorization.read_bytes()).hexdigest()),new=True)
    jobs=[]; failure=None
    try:
        for g in gpus:
            if STOP_SIGNAL is not None:raise RuntimeError('Coordinator stop signal '+str(STOP_SIGNAL))
            logfile=phase/('gpu'+str(g)+'.log'); handle=logfile.open('xb')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            cmd=[PYTHON,str(root/'source/shared_runtime_worker.py'),'--root',str(root),'--gpu',str(g),
                 '--mode',args.mode,'--authorization',str(args.authorization.resolve())]
            started=time.monotonic()
            proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(stage=args.mode,gpu=g,gpu_uuid=auth['gpu_uuid_map'][str(g)],pid=proc.pid,
                        dispatch_unix=time.time(),dispatch_monotonic=started,identity=None,
                        occupied_seconds=0.,returncode=None,process_exited=False,NVML_released=False)
            jobs.append(dict(proc=proc,record=record,logfile=logfile,handle=handle,last_progress=started,last_size=0))
            until=time.monotonic()+2
            while record['identity'] is None and proc.poll() is None and time.monotonic()<until:
                record['identity']=identity(proc.pid)
                if record['identity'] is None:time.sleep(.01)
            if proc.poll() is None and (record['identity'] is None or str(root/'source/shared_runtime_worker.py') not in record['identity']['cmdline']):
                raise RuntimeError('Live worker identity could not be established')
        while True:
            if STOP_SIGNAL is not None:raise RuntimeError('Coordinator stop signal '+str(STOP_SIGNAL))
            now=time.monotonic(); apps=nvml(); own={j['proc'].pid for j in jobs}
            foreign=[a for a in apps if a['UUID'] in requested and a['pid'] not in own]
            if foreign: raise RuntimeError('Foreign compute appeared on reserved GPU; stop own work, never terminate foreign PID')
            for j in jobs:
                rec=j['record']; code=j['proc'].poll(); size=j['logfile'].stat().st_size
                if size!=j['last_size']:j.update(last_size=size,last_progress=now)
                rec['process_exited']=code is not None;rec['returncode']=code
                rec['NVML_released']=not any(a['pid']==rec['pid'] for a in apps)
                if 'release_monotonic' not in rec:
                    rec['occupied_seconds']=now-rec['dispatch_monotonic']
                    if rec['process_exited'] and rec['NVML_released']:rec['release_monotonic']=now
                if code is not None:
                    if code: raise RuntimeError('Worker exited nonzero; inspect log and preserved receipt: '+str(j['logfile']))
                    result=worker_phase(root,args.mode,rec['gpu'])/'RESULT.json'
                    if not result.is_file() or not read(result).get('status','').startswith('PASS'):
                        raise RuntimeError('Worker ended without valid receipt; process and log checked, no retry')
                    if args.mode=='formal' and read(result).get('units')!=30:
                        raise RuntimeError('Formal worker receipt must contain30 complete generations')
                elif now-j['last_progress']>auth.get('watchdog_seconds',1800):
                    raise RuntimeError('No worker stage/token progress within frozen watchdog interval')
            ledger.update(status='RUNNING_'+args.mode.upper(),dispatches=history+[j['record'] for j in jobs],
                          all_GPUs_released=False,own_NVML_pids_remaining=[a['pid'] for a in apps if a['pid'] in own],
                          cumulative_GPU_occupied_seconds=sum(r['occupied_seconds'] for r in history+[j['record'] for j in jobs]),
                          updated_unix=time.time())
            ledger['cumulative_GPU_hours']=ledger['cumulative_GPU_occupied_seconds']/3600
            save(ledger_path,ledger)
            if all('release_monotonic' in j['record'] for j in jobs):break
            time.sleep(2)
    except BaseException:
        failure=dict(status='FAIL_CLOSED_STOP',stage=args.mode,unix=time.time(),traceback=traceback.format_exc())
        for j in jobs:
            if j['proc'].poll() is None:
                current=identity(j['proc'].pid); expected=j['record']['identity']
                if current and expected and current['start_tick']==expected['start_tick'] and str(root/'source/shared_runtime_worker.py') in current['cmdline']:
                    os.killpg(j['proc'].pid,signal.SIGTERM)
        until=time.monotonic()+15
        while any(j['proc'].poll() is None for j in jobs) and time.monotonic()<until:time.sleep(.5)
        for j in jobs:
            if j['proc'].poll() is None:
                current=identity(j['proc'].pid);expected=j['record']['identity']
                if current and expected and current['start_tick']==expected['start_tick'] and str(root/'source/shared_runtime_worker.py') in current['cmdline']:
                    os.killpg(j['proc'].pid,signal.SIGKILL)
        if not (root/'FIRST_FAILURE.json').exists():save(root/'FIRST_FAILURE.json',failure,new=True)
    finally:
        deadline=time.monotonic()+60
        while jobs:
            apps,nvml_error=nvml_safe(); own={j['proc'].pid for j in jobs}
            if not nvml_error and all(j['proc'].poll() is not None for j in jobs) and not any(a['pid'] in own for a in apps):break
            if time.monotonic()>deadline:break
            time.sleep(1)
        now=time.monotonic();apps,nvml_error=nvml_safe();own={j['proc'].pid for j in jobs}
        for j in jobs:
            rec=j['record'];rec['process_exited']=j['proc'].poll() is not None;rec['returncode']=j['proc'].poll()
            rec['NVML_released']=not nvml_error and not any(a['pid']==rec['pid'] for a in apps)
            if 'release_monotonic' not in rec:
                rec['occupied_seconds']=now-rec['dispatch_monotonic']
                if rec['process_exited'] and rec['NVML_released']:rec['release_monotonic']=now
                else:rec['accounted_through_monotonic']=now
            j['handle'].close()
        released=all(j['record']['process_exited'] and j['record']['NVML_released'] for j in jobs)
        ledger.update(status='FAIL_CLOSED_STOP' if failure or not released else 'STAGE_COMPLETE_'+args.mode.upper(),
                      dispatches=history+[j['record'] for j in jobs],all_GPUs_released=released,
                      own_NVML_pids_remaining=sorted(own) if nvml_error else [a['pid'] for a in apps if a['pid'] in own],
                      NVML_status='UNKNOWN' if nvml_error else 'VERIFIED',NVML_error=nvml_error,updated_unix=time.time())
        ledger['cumulative_GPU_occupied_seconds']=sum(r['occupied_seconds'] for r in ledger['dispatches'])
        ledger['cumulative_GPU_hours']=ledger['cumulative_GPU_occupied_seconds']/3600
        if args.mode=='formal':
            completed={'STEPQUANT6':0,'H_DUALM':0};incomplete=0;seen=set();valid=True
            try:units={u['canonical_index']:u for u in read(root/'FROZEN_UNITS.json')['units']}
            except Exception:
                units={};failure=failure or dict(status='FAIL_CLOSED_STOP',stage=args.mode,reason='Frozen unit manifest unreadable',traceback=traceback.format_exc())
            for g in range(4):
                for p in (root/'formal'/('worker'+str(g))).glob('[0-9][0-9]_*.json'):
                    if p.name.endswith('.ATTEMPT.json'):continue
                    try:row=read(p)
                    except Exception:
                        incomplete+=1;valid=False
                        failure=failure or dict(status='FAIL_CLOSED_STOP',stage=args.mode,reason='Unreadable formal raw record: '+str(p),traceback=traceback.format_exc())
                        continue
                    key=(row.get('canonical_index'),row.get('method'));u=units.get(key[0])
                    good=(row.get('status')=='COMPLETE' and u is not None and key not in seen and
                          key[1] in completed and row.get('problem_id')==u['problem_id'] and row.get('seed')==u['seed'] and
                          g==u['gpu'] and row.get('gpu',{}).get('UUID')==auth['gpu_uuid_map'][str(g)] and
                          row.get('runtime_error') is None and row.get('nonfinite') is False)
                    seen.add(key);valid=valid and good
                    if good:completed[row['method']]+=1
                    else:incomplete+=1
            ledger['formal_generations_completed']=completed
            ledger['formal_technical_incomplete_records']=incomplete
            expected={(i,m) for i in units for m in ('STEPQUANT6','H_DUALM')}
            if not valid or incomplete or seen!=expected or completed!={'STEPQUANT6':60,'H_DUALM':60}:
                failure=failure or dict(status='FAIL_CLOSED_STOP',stage=args.mode,unix=time.time(),
                    reason='Formal completion/identity gate failed; never score incomplete/duplicate pairs')
                ledger['status']='FAIL_CLOSED_STOP'
        if (failure or not released) and not (root/'FIRST_FAILURE.json').exists():
            save(root/'FIRST_FAILURE.json',failure or dict(status='FAIL_CLOSED_STOP',stage=args.mode,reason='Resource release not verified'),new=True)
        save(ledger_path,ledger)
        save(phase/'RESULT.json',dict(status='FAIL_CLOSED_STOP' if failure or not released else 'PASS',
                                     stage=args.mode,all_GPUs_released=released,ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest()),new=True)
    if failure or not released:raise RuntimeError('Stage stopped; retain first failure and never patch-and-retry GPU')


if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,signal_stop)
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--mode',required=True,choices=('calibration','canary','fresh','resource','formal'))
    p.add_argument('--authorization',required=True,type=Path);run(p.parse_args())
