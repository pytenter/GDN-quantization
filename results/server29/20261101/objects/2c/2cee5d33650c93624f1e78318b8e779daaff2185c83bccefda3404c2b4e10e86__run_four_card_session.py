#!/usr/bin/env python3
"""One-shot offline launcher. Numerical/scientific admission stays in each controller."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())

def identity(pid):
    p=Path('/proc')/str(pid)
    stat=(p/'stat').read_text()
    return {'pid':pid,'starttick':int(stat[stat.rfind(')')+2:].split()[19]),
            'cmdline':[x.decode() for x in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}

def verify_control(root):
    freeze=json.loads((root/'SOURCE_FREEZE.json').read_text())
    control=root/'CONTROL_SOURCE_MANIFEST.json';payload=root/'PAYLOAD_BUILD_MANIFEST.json'
    for file,key in [(control,'control_manifest_sha256'),(payload,'payload_build_manifest_sha256')]:
        if hashlib.sha256(file.read_bytes()).hexdigest()!=freeze[key]:
            raise RuntimeError('Common frozen control binding changed: '+key)
    manifest=json.loads(control.read_text())
    if not manifest.get('files'):raise RuntimeError('Empty common control source roster')
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):raise RuntimeError('Control source path escape')
        data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise RuntimeError('Frozen common control changed: '+row['path'])
    if not any((root/r['path']).resolve()==Path(__file__).resolve() for r in manifest['files']):
        raise RuntimeError('Launcher absent common frozen control source')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root', required=True, type=Path)
    a=ap.parse_args();root=a.root.resolve()
    verify_control(root)
    if (root/'FIRST_FAILURE.json').exists():raise RuntimeError('Prior STOP prohibits relaunch')
    once(root/'SESSION_STARTED.json', {'status':'STARTED_ONCE','identity':identity(os.getpid()),'unix':time.time(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'GPU_time_limit':None})
    children=[];observed={}
    try:
        for number in range(4):
            card=root/('gpu'+str(number));log=(root/('gpu'+str(number)+'_controller.log')).open('xb')
            command=[sys.executable,'-B',str(card/'scripts/run_cublas_controller.py'),'--root',str(card),
                     '--run','--allow-gpu-diag','YES','--authorization',str(card/'AUTHORIZATION.json')]
            child=subprocess.Popen(command, cwd=card, stdout=log, stderr=subprocess.STDOUT)
            children.append((number,child,log))
            once(root/('COORDINATOR_'+str(number)+'_SPAWN.json'), {'command':command,'identity':identity(child.pid),'unix':time.time()})
        while len(observed)<4:
            for number,child,log in children:
                code=child.poll()
                if code is None or number in observed:continue
                receipt={'returncode':code,'pid':child.pid,'first_exit_observed_unix':time.time(),
                         'first_exit_observed_monotonic':time.monotonic()}
                once(root/('COORDINATOR_'+str(number)+'_EXIT.json'), receipt);observed[number]=receipt;log.close()
                if code and not (root/'FIRST_FAILURE.json').exists():
                    try:once(root/'FIRST_FAILURE.json', {'status':'FAIL_CLOSED_STOP','card':number,'error':'Controller exited nonzero; inspect retained log',**receipt,'no_retry':True})
                    except FileExistsError:pass
            time.sleep(1)
        failed=(root/'FIRST_FAILURE.json').exists() or any(v['returncode'] for v in observed.values())
        once(root/'SESSION_RESULT.json', {'status':'FAIL_CLOSED_STOP' if failed else 'FINITE_PHASE_B_SCHEDULE_EXITED',
            'coordinator_exit_observations':observed,'requires_independent_result_and_release_verification':True,
            'phase_D3_if_benefit_requires_separate_frozen_root':True})
        return 1 if failed else 0
    except BaseException as exc:
        if not (root/'FIRST_FAILURE.json').exists():
            try:once(root/'FIRST_FAILURE.json', {'status':'FAIL_CLOSED_STOP','error':str(exc),'unix':time.time(),'no_retry':True})
            except FileExistsError:pass
        # Exact worker cleanup remains the admitted controllers' responsibility;
        # never signal broad process names or foreign jobs here.
        for _,child,_ in children:
            if child.poll() is None:child.wait()
        raise

if __name__=='__main__':raise SystemExit(main())
