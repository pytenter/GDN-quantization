#!/usr/bin/env python3
"""Finite serial GPU0 controller. No retries, elapsed caps, or foreign signals."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pace_control as C
import pace_continuation as K


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.identity = C.process_identity(os.getpid())
        self.stop_signal = None
        self.current = None
        self.intervals = []
        self.completed = []
        self.reuse_admitted = False
        self.session_start = time.time()

    def signal(self, sig, frame):
        # Do not raise between Popen and registration. Cleanup is synchronous.
        self.stop_signal = int(sig)

    def request_stop(self):
        if self.stop_signal is not None:
            return dict(kind='OWN_CONTROLLER_SIGNAL', signal=self.stop_signal)
        p = self.root / 'GLOBAL_STOP_REQUEST.json'
        if p.exists():
            value = C.read(p)
            C.require(value['task_id'] == C.TASK and value['execution_root'] == C.REMOTE and
                      value['source_manifest_sha256'] == C.sha(self.root / C.SOURCE),
                      'Stop request identity is malformed; fail closed')
            return dict(kind='GLOBAL_STOP_REQUEST', request_sha256=C.sha(p), request=value)
        return None

    def stop_evidence(self):
        try:
            return self.request_stop()
        except BaseException as e:
            return dict(kind='MALFORMED_STOP_REQUEST_FAIL_CLOSED', error=repr(e))

    def ledger(self):
        active = 0 if self.current is None else time.monotonic() - self.current['dispatch_monotonic']
        value = dict(task_id=C.TASK, execution_revision=C.REVISION, method_variant=C.METHOD_VARIANT,
                     state_scale_format=C.STATE_SCALE_FORMAT, own_gpu_uuid=C.UUID, physical_gpu=0,
                     no_GPUh_or_elapsed_caps=True, progress_watchdog_seconds=1800,
                     accounting_scope='Own Popen dispatch-to-exit-and-NVML-release wall-time accounting; includes model load and IO, not exact physical GPU active/SM time',
                     physical_GPU_active_seconds='UNRESOLVED_NOT_MEASURED',
                     intervals=self.intervals, completed_units=self.completed,
                     reused_formal_count=3 if self.reuse_admitted else 0,
                     reused_formal_ids=list(K.REUSED_IDS) if self.reuse_admitted else [],
                     current_formal_count=sum(x['stage'] == 'formal' for x in self.completed),
                     formal_total_unique=(3 if self.reuse_admitted else 0)+sum(x['stage'] == 'formal' for x in self.completed),
                     historical_costs_are_separate=True,
                     cumulative_GPU_seconds=sum(x['GPU_seconds'] for x in self.intervals) + active,
                     all_recorded_workers_released=all(x['released'] for x in self.intervals),
                     active_unit=None if self.current is None else self.current['unit'], unix=time.time())
        value['cumulative_GPUh'] = value['cumulative_GPU_seconds'] / 3600
        C.status(self.root / 'RESOURCE_LEDGER.json', value)
        return value

    def observe(self, pid):
        obs = C.query_gpu()
        C.require(not any(x['pid'] == pid and x['uuid'] != C.UUID for x in obs['apps']),
                  'Own PID used an unauthorized GPU')
        C.require(not any(x['uuid'] == C.UUID and x['pid'] != pid for x in obs['apps']),
                  'Foreign GPU0 process appeared; never preempt it')
        return obs

    def finish_child(self, proc, identity, target, failure):
        deadline = time.monotonic() + 120
        actions = []
        if proc.poll() is None:
            try:
                C.terminate_owned(proc, identity)
                actions.append('SIGTERM_EXACT_OWN_PID')
            except BaseException as e:
                actions.append('TERMINATION_REFUSED:' + repr(e))
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    C.terminate_owned(proc, identity, kill=True)
                    actions.append('SIGKILL_EXACT_OWN_PID')
                except BaseException as e:
                    actions.append('KILL_REFUSED:' + repr(e))
        observation = None
        error = None
        while time.monotonic() < deadline:
            try:
                observation = C.query_gpu()
                if C.release_state(proc.poll() is not None, observation, proc.pid, identity):
                    break
            except BaseException as e:
                error = repr(e); observation = None
            self.ledger()
            time.sleep(.5)
        released = C.release_state(proc.poll() is not None, observation, proc.pid, identity)
        current = self.current
        seconds = time.monotonic() - current['dispatch_monotonic']
        interval = dict(task_id=C.TASK, stage=current['stage'], unit=current['unit'],
                        pid=proc.pid, identity=identity, dispatch_unix=current['dispatch_unix'],
                        accounting_end_unix=time.time(), GPU_seconds=seconds, GPUh=seconds / 3600,
                        returncode=proc.poll(), released=released, cleanup_actions=actions,
                        NVML=observation, NVML_error=error,
                        physical_GPU_active_seconds='UNRESOLVED_NOT_MEASURED',
                        accounting_final=bool(released), lower_bound_only=not released,
                        failure=failure)
        C.atomic_new(target / 'RESOURCE_INTERVAL.json', interval)
        self.intervals.append(interval)
        self.current = None
        self.ledger()
        return interval

    def execute_unit(self, stage, unit):
        K.dispatch_allowed(self.root, stage, unit)
        C.require(self.request_stop() is None, 'Human stop before next dispatch')
        C.authorize(self.root); C.model_guard(self.root); C.device(); C.storage_guard(self.root)
        target = C.unit_target(self.root, stage, unit)
        C.require(not target.exists(), 'Existing/stale unit path: no restart or double counting')
        target.mkdir(parents=True, exist_ok=False)
        dispatch = dict(task_id=C.TASK, execution_revision=C.REVISION,
                        method_variant=C.METHOD_VARIANT, state_scale_format=C.STATE_SCALE_FORMAT, stage=stage, unit=unit, controller=self.identity,
                        execution_root=str(self.root), gpu=0, gpu_uuid=C.UUID,
                        source_manifest_sha256=C.sha(self.root / C.SOURCE),
                        permit_sha256=C.sha(self.root / 'GPU_PERMIT.json'))
        C.atomic_new(target / 'DISPATCH.json', dispatch)
        argv = [sys.executable, str(self.root / 'scripts/pace_worker.py'), '--root', str(self.root),
                '--stage', stage, '--unit', unit]
        identity = None
        failure = None
        proc = None
        with (target / 'worker.log').open('xb', buffering=0) as log:
            dispatch_mono, dispatch_unix = time.monotonic(), time.time()
            proc = subprocess.Popen(argv, cwd=self.root, env=C.worker_environment(), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # Register before any stop/exception is acted on, including a signal in Popen.
            self.current = dict(stage=stage, unit=unit, dispatch_monotonic=dispatch_mono,
                                dispatch_unix=dispatch_unix, pid=proc.pid)
            try:
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    try:
                        identity = C.process_identity(proc.pid)
                        C.require(identity['cmdline'] == argv, 'New child full argv differs')
                        break
                    except FileNotFoundError:
                        if proc.poll() is not None:
                            break
                        time.sleep(.02)
                C.require(identity is not None, 'Child identity was not registered')
                C.atomic_new(target / 'ACTUAL_CHILD_IDENTITY.json', identity)
                self.ledger()
                signature = None
                last_progress = time.monotonic()
                next_check = 0
                while proc.poll() is None:
                    stop = self.request_stop()
                    C.require(stop is None, 'Human requested stop: ' + repr(stop))
                    pp = target / 'PROGRESS.json'
                    if pp.exists():
                        s = pp.stat(); now_signature = (s.st_mtime_ns, s.st_size)
                        if now_signature != signature:
                            signature = now_signature; last_progress = time.monotonic()
                    C.require(time.monotonic() - last_progress <= 1800,
                              'No new real worker progress for1800seconds')
                    if time.monotonic() >= next_check:
                        self.observe(proc.pid); C.storage_guard(self.root); self.ledger()
                        next_check = time.monotonic() + 10
                    time.sleep(.5)
                C.require(proc.returncode == 0, 'Worker exited nonzero: ' + str(proc.returncode))
            except BaseException as e:
                failure = dict(error=repr(e), traceback=traceback.format_exc(), stop=self.stop_evidence())
            finally:
                interval = self.finish_child(proc, identity, target, failure)
        C.require(interval['released'], 'Worker exit/NVML release unresolved; accounting is only a lower bound')
        C.require(failure is None and interval['returncode'] == 0, 'Unit failed: ' + repr(failure))
        result = C.verify_result(self.root, stage, unit, identity)
        self.completed.append(dict(stage=stage, unit=unit, result_sha256=C.sha(target / 'RESULT.json'),
                                   interval_sha256=C.sha(target / 'RESOURCE_INTERVAL.json')))
        self.ledger()
        return result

    def run(self):
        C.authorize(self.root)
        C.require(not (self.root / 'SESSION_STARTED.json').exists() and
                  not (self.root / 'SESSION_RESULT.json').exists(), 'Session already started: no rerun')
        C.atomic_new(self.root / 'SESSION_STARTED.json', dict(task_id=C.TASK, execution_revision=C.REVISION, method_variant=C.METHOD_VARIANT,
                     state_scale_format=C.STATE_SCALE_FORMAT, identity=self.identity,
                     source_manifest_sha256=C.sha(self.root / C.SOURCE), permit_sha256=C.sha(self.root / 'GPU_PERMIT.json'),
                     unix=time.time()))
        failure = None
        try:
            C.device(); C.storage_guard(self.root); C.model_guard(self.root)
            C.atomic_new(self.root / 'EXTERNAL_PRE_GPU_FRESH_SHA.json', C.verify_external(self.root, full_model=True))
            C.validate_jobs(self.root)
            admission = K.record_admission(self.root)
            self.reuse_admitted = True
            self.execute_unit(K.GATE_STAGE, K.GATE_UNIT)
            K.admit_reused_science(self.root)
            C.atomic_new(self.root / 'validation/ACTUAL_CALIBRATED_MASK_CONTRACT.json', C.verify_head_mask(self.root))
            C.atomic_new(self.root / 'FORMAL_ADMISSION.json', dict(task_id=C.TASK,
                         execution_revision=C.REVISION, method_variant=C.METHOD_VARIANT,
                         state_scale_format=C.STATE_SCALE_FORMAT,
                         status='PASS_EXPONENT_FORMAT_GATE_REUSE3_PLUS57_ADMITTED',
                         science_freeze_sha256=C.verify_science(self.root),
                         scale_repair_gate_sha256=C.sha(self.root / 'execution/SCALE_REPAIR_GATES/RESULT.json'),
                         continuation_admission_sha256=C.sha(self.root / 'CONTINUATION_ADMISSION.json'),
                         reused_calibration_admission_sha256=C.sha(self.root / 'REUSED_CALIBRATION_ADMISSION.json'),
                         inherited_qualifications=admission['proofs'][:3],
                         reused_formal_samples=3, formal_new_samples=57, formal_total_unique=60,
                         same_mask_both_conditions=True, failed_partial_credit=0))
            for job in K.remaining_jobs(self.root):
                self.execute_unit('formal', job['id'])
            C.require(len([x for x in self.completed if x['stage'] == 'formal']) == 57 and
                      len(K.completed_rows(self.root)) == 60, 'Exact3 reused plus57 current completion missing')
        except BaseException as e:
            failure = dict(status='FAIL_CLOSED_STOP', task_id=C.TASK, error=repr(e),
                           traceback=traceback.format_exc(), stop=self.stop_evidence(),
                           completed_units=self.completed, unix=time.time(), no_automatic_retry=True)
            if not (self.root / 'FIRST_FAILURE.json').exists():
                C.atomic_new(self.root / 'FIRST_FAILURE.json', failure)
        finally:
            ledger = self.ledger()
            released = self.current is None and all(x['released'] for x in self.intervals)
            result = dict(status='GPU_COMPLETE_PACE48_REUSE3_PLUS57_FORMAL' if failure is None else 'FAIL_CLOSED_STOP',
                          task_id=C.TASK, execution_revision=C.REVISION, method_variant=C.METHOD_VARIANT,
                          state_scale_format=C.STATE_SCALE_FORMAT, controller=self.identity, failure=failure,
                          completed_units=self.completed, formal_completed=sum(x['stage'] == 'formal' for x in self.completed),
                          reused_formal_count=3 if self.reuse_admitted else 0,
                          current_formal_count=sum(x['stage']=='formal' for x in self.completed),
                          formal_total_unique=(3 if self.reuse_admitted else 0)+sum(x['stage']=='formal' for x in self.completed),
                          historical_costs_are_separate=True, uninterrupted_continuous60=False,
                          own_workers_exit_and_NVML_release=released, ledger=ledger,
                          source_manifest_sha256=C.sha(self.root / C.SOURCE), permit_sha256=C.sha(self.root / 'GPU_PERMIT.json'),
                          unix=time.time())
            C.atomic_new(self.root / 'SESSION_RESULT.json', result)
        return 0 if failure is None and released else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print('CPU-only entry: explicit --execute and exact frozen permit are required; no device API was called')
        return 0
    c = Controller(args.root)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, c.signal)
    return c.run()

if __name__ == '__main__':
    raise SystemExit(main())
