#!/usr/bin/env python3
"""Synthetic stdlib release-proof fixtures. No /proc, NVML or GPU query is run."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def run(script, output):
    spec = importlib.util.spec_from_file_location('release_builder_CPU_fixture', script)
    B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    B.query_final_inventory = lambda: (_ for _ in ()).throw(AssertionError('Real NVML query forbidden'))
    B.process_observation = lambda _: (_ for _ in ()).throw(AssertionError('Real process query forbidden'))
    B.scan_run_processes = lambda _: (_ for _ in ()).throw(AssertionError('Real process scan forbidden'))
    checks = []
    def check(name, value):
        if not value: raise RuntimeError('CPU release fixture failed: ' + name)
        checks.append(dict(check=name, status='PASS'))
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(B.encoded(value))
    def fixture(root, complete=False):
        root = root.resolve()
        payload = root/'payload'; payload.mkdir(parents=True)
        controller = dict(pid=101, starttick=10001, cmdline=['python','-B',str(payload/'scripts/restoration_controller.py'),'--root',str(payload),'--run','--allow-gpu-restore','YES','--authorization',str(payload/'HUMAN_GPU_RESTORE_AUTHORIZATION.json')])
        supervisor = dict(pid=100, starttick=10000, cmdline=['python','-B',str(root/'scripts/offline_controller_once.py'),'--root',str(root)])
        write(payload/'SOURCE_FREEZE.json',dict(task_id=B.TASK, files=[]))
        freeze=B.digest(B.stable(payload/'SOURCE_FREEZE.json'))
        write(payload/'ROOT.json',dict(task_id=B.TASK,ALLOW_GPU_RESTORE='YES'))
        write(payload/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',dict(task_id=B.TASK,ALLOW_GPU_RESTORE='YES',execution_root=str(payload),source_freeze_sha256=freeze))
        invocation=dict(task_id=B.TASK,retry=False,command=controller['cmdline'],supervisor_identity=supervisor,source_freeze_sha256=freeze,authorization_sha256=B.digest(B.stable(payload/'HUMAN_GPU_RESTORE_AUTHORIZATION.json')),monotonic=1.)
        write(root/'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json',invocation)
        write(root/'operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json',dict(command=controller['cmdline'],returncode=0 if complete else 2,before_monotonic=1.,after_monotonic=40.,wall_seconds=39.,retry=False))
        initial=root/'operations/LIVE_INITIAL_PROCESS_SNAPSHOT.json'
        write(initial,{'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json':invocation,'actual_processes':[supervisor,controller]})
        write(payload/'execution/CONTROLLER_STARTED.json',dict(task_id=B.TASK,controller_identity=controller,source_freeze_sha256=freeze,authorization_sha256=invocation['authorization_sha256']))
        units=B.UNITS if complete else B.UNITS[:1]; workers=[];events=[]
        for i,unit in enumerate(units):
            ident=dict(pid=200+i,starttick=20000+i,cmdline=['python','-B',str(payload/'worker/restoration_worker.py'),'--unit',unit])
            workers.append(ident);t=10.+i*3
            write(payload/'execution'/unit/'DISPATCH.json',dict(worker_identity=ident,dispatch_before_monotonic=t,dispatch_after_monotonic=t+.1))
            u=dict(id=unit,mode='numeric' if i<2 else 'benchmark')
            events += [dict(event='FIRST_DISPATCH',monotonic=t+.2,unit=u,worker_identity=ident,before_monotonic=t,after_monotonic=t+.1),
                       dict(event='FIRST_PROCESS_EXIT_OBSERVED',monotonic=t+1,unit=u,worker_identity=ident,observed_monotonic=t+1),
                       dict(event='FIRST_ALL_UUID_NVML_RELEASE_OBSERVED',monotonic=t+1.1,unit=u,worker_identity=ident,observed_monotonic=t+1.1),
                       dict(event='UNIT_RESERVATION_INTERVAL_BOUNDS',monotonic=t+1.2,unit_id=unit,reservation_seconds_lower=.8,reservation_seconds_upper=1.1)]
        status='COMPLETE_FINITE_RECOVERY_UNITS' if complete else 'FAIL_CLOSED_STOP'
        write(payload/'execution/CONTROLLER_RESULT.json',dict(task_id=B.TASK,status=status,completed_units=units))
        if not complete:
            write(payload/'FIRST_FAILURE.json',dict(status='FAIL_CLOSED_STOP',error='SYNTHETIC_NUMERIC_FAIL_NOT_A_REAL_GPU_RESULT'))
            events.append(dict(event='FAIL_CLOSED_STOP',monotonic=39.))
        ledger=payload/'execution/RESOURCE_EVENTS.jsonl';ledger.write_bytes(b''.join(B.encoded(x).replace(b'\n',b'')+b'\n'for x in events))
        return dict(root=root,payload=payload,initial=initial,controller=controller,supervisor=supervisor,workers=workers,ledger=ledger,output=root/'operations/RELEASE_ADMISSION.json')
    def inventory(apps=None):
        return dict(devices=[dict(physical_index=0,uuid='GPU-target',name='RTX3090'),dict(physical_index=1,uuid='GPU-other',name='RTX3090')],
                    apps=[dict(pid=999,uuid='GPU-other',memory_MiB='100')] if apps is None else apps,
                    commands=[],before_monotonic=50.,after_monotonic=50.1)
    def build(f,reader=None,inv=None,scan=None):
        return B.build(f['root'],f['initial'],f['output'],process_reader=reader or (lambda _:None),inventory_reader=inv or (lambda:inventory()),scanner=scan or (lambda _:[]),clock=lambda:50.)
    def reject(name, fn):
        try:fn()
        except (RuntimeError,FileNotFoundError):check(name,True)
        else:check(name,False)
    def change_json(path,fn):
        j=B.read(path);fn(j);write(path,j)
    with tempfile.TemporaryDirectory(prefix='qwen_release_proof_CPU_')as tmp:
        tmp=Path(tmp)
        f=fixture(tmp/'valid_stop');before=B.digest(B.stable(f['ledger']));proof=build(f)
        check('true_STOP_released_resource_admission_no_science_promotion',proof['status']=='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED'and proof['native_scientific_controller_status']=='FAIL_CLOSED_STOP')
        check('controller_supervisor_worker_full_identity_roster',len(proof['own_processes'])==3 and proof['owned_worker_identities']==f['workers']and proof['supervisor_identities']==[f['supervisor']])
        check('foreign_GPU_record_and_full_device_inventory_preserved',proof['final_NVML_apps']==inventory()['apps']and len(proof['final_NVML_devices'])==2 and proof['final_owned_NVML_records']==[])
        check('closed_native_ledger_untouched',before==B.digest(B.stable(f['ledger'])))
        check('all_evidence_relative_exactSHAbytes',all(not Path(x['path']).is_absolute()and f['root'].joinpath(x['path']).stat().st_size==x['bytes']and B.digest(B.stable(f['root']/x['path']))==x['sha256']for x in proof['evidence_receipts']))
        check('no_signals_and_no_dispatch',proof['no_signals_sent']is True and proof['GPU_dispatches']==0)
        reject('proof_noclobber_existing',lambda:build(f))
        f=fixture(tmp/'valid_complete',True);proof=build(f)
        check('complete_sixworkers_plus_controller_and_supervisor',len(proof['owned_worker_identities'])==6 and len(proof['own_processes'])==8 and proof['controller_result']['status']=='COMPLETE_FINITE_RECOVERY_UNITS')
        f=fixture(tmp/'alive');calls=[]
        reject('alive_controller_refused_before_NVML',lambda:build(f,reader=lambda p:dict(f['controller'],state='S')if p==101 else None,inv=lambda:(calls.append('NVML')or inventory())))
        check('no_NVMLquery_when_controller_alive',calls==[])
        f=fixture(tmp/'changed_image')
        reject('same_starttick_changed_cmdline_not_PIDreuse',lambda:build(f,reader=lambda p:dict(f['controller'],cmdline=['foreign-image'],state='S')if p==101 else None))
        f=fixture(tmp/'reused_safe');proof=build(f,reader=lambda p:dict(pid=p,starttick=99999,cmdline=['unrelated'],state='S'))
        check('PIDreuse_without_NVML_proves_old_identity_gone',all(x['final_process_observation']['observation']=='PID_REUSED_OLD_STARTTICK_GONE'for x in proof['own_processes']))
        f=fixture(tmp/'reuse_NVML')
        reject('PIDreuse_with_NVML_numericPID_conservatively_refused',lambda:build(f,reader=lambda p:dict(pid=p,starttick=99999,cmdline=['unrelated'],state='S'),inv=lambda:inventory([dict(pid=101,uuid='GPU-other',memory_MiB='100')])))
        f=fixture(tmp/'zombie');proof=build(f,reader=lambda p:dict(pid=p,starttick=10001 if p==101 else 10000 if p==100 else 20000,cmdline=[],state='Z'))
        check('reaped_or_zombie_exit_distinguished_from_reuse',all(x['final_process_observation']['observation']=='SAME_STARTTICK_EXITED_ZOMBIE_OR_DEAD'for x in proof['own_processes']))
        f=fixture(tmp/'wrong_GPU')
        reject('owned_PID_on_any_UUID_blocks_release',lambda:build(f,inv=lambda:inventory([dict(pid=200,uuid='GPU-unauthorized',memory_MiB='100')])))
        f=fixture(tmp/'missing_devices')
        reject('empty_full_NVML_inventory_refused',lambda:build(f,inv=lambda:dict(devices=[],apps=[])))
        f=fixture(tmp/'extra_live')
        reject('unlisted_live_same_run_process_refused',lambda:build(f,scan=lambda _:[dict(pid=888,starttick=123,cmdline=[str(f['payload']/'worker/restoration_worker.py')],state='S')]))
        f=fixture(tmp/'ledger_changed')
        def mutate_ledger():
            with f['ledger'].open('ab')as stream:stream.write(B.encoded(dict(event='UNEXPECTED_WRITE',monotonic=99.)).replace(b'\n',b'')+b'\n')
            return inventory()
        reject('ledger_change_during_final_inventory_refused',lambda:build(f,inv=mutate_ledger))
        f=fixture(tmp/'missing_exit');(f['root']/'operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json').unlink()
        reject('true_native_exit_receipt_required',lambda:build(f))
        f=fixture(tmp/'wrong_rc');change_json(f['root']/'operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json',lambda j:j.update(returncode=0))
        reject('nativeSTOP_vs_true_RC0_contradiction_refused',lambda:build(f))
        f=fixture(tmp/'wrong_initial');change_json(f['initial'],lambda j:j['actual_processes'][1].update(starttick=888))
        reject('actual_initial_controller_starttick_required',lambda:build(f))
        f=fixture(tmp/'wrong_auth');change_json(f['payload']/'HUMAN_GPU_RESTORE_AUTHORIZATION.json',lambda j:j.update(ALLOW_GPU_RESTORE='NO'))
        reject('current_source_authorization_binding_required',lambda:build(f))
        f=fixture(tmp/'partial_line');f['ledger'].write_bytes(B.stable(f['ledger']).rstrip(b'\n'))
        reject('uncommitted_resource_ledger_tail_refused',lambda:build(f))
        f=fixture(tmp/'duplicate_exit');events=[json.loads(x)for x in B.stable(f['ledger']).splitlines()]
        events.insert(2,dict(events[1]));f['ledger'].write_bytes(b''.join(json.dumps(x).encode()+b'\n'for x in events))
        reject('duplicate_first_exit_event_refused',lambda:build(f))
        f=fixture(tmp/'missing_release');events=[json.loads(x)for x in B.stable(f['ledger']).splitlines()];events=[x for x in events if x['event']!='FIRST_ALL_UUID_NVML_RELEASE_OBSERVED'];f['ledger'].write_bytes(b''.join(json.dumps(x).encode()+b'\n'for x in events))
        reject('native_all_UUID_release_event_required',lambda:build(f))
        f=fixture(tmp/'dispatch_mismatch');change_json(f['payload']/'execution/NUMERIC_FRESH0/DISPATCH.json',lambda j:j['worker_identity'].update(starttick=333))
        reject('closed_ledger_dispatch_identity_match_required',lambda:build(f))
        f=fixture(tmp/'marker_source');change_json(f['payload']/'execution/CONTROLLER_STARTED.json',lambda j:j.update(source_freeze_sha256='0'*64))
        reject('controller_started_source_binding_required',lambda:build(f))
        f=fixture(tmp/'outside_output')
        reject('proof_may_not_write_native_payload',lambda:B.build(f['root'],f['initial'],f['payload']/'evil.json',process_reader=lambda _:None,inventory_reader=lambda:inventory(),scanner=lambda _:[]))
    value=dict(status='PASS_CPU_SYNTHETIC_RELEASE_ADMISSION_BUILDER_TESTS_NOT_ACTUAL_RELEASE',checks_passed=len(checks),checks_total=len(checks),checks=checks,GPU_queries=0,NVML_queries=0,GPU_dispatches=0,process_queries=0,signals_sent=0,Torch_imports=0,source_bindings=[dict(path=script.name,bytes=script.stat().st_size,sha256=hashlib.sha256(script.read_bytes()).hexdigest()),dict(path=Path(__file__).name,bytes=Path(__file__).stat().st_size,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())],scope='Only synthetic temporary metadata, injected process identities and full foreign-preserving GPU inventory. No actual proof or snapshot was generated for the active run.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=value['status'],checks_passed=len(checks),source_bindings=value['source_bindings'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--builder',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();run(args.builder,args.output)
