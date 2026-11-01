"""Independent CPU control-path mocks. Never imports torch or queries hardware."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def scenario(kind):
    spec = importlib.util.spec_from_file_location('independent_control_' + kind, ROOT / 'control/coordinator_4gpu.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    clock = SimpleNamespace(now=0.0)
    jobs = []
    state = dict(ledger_count=0, signaled=[])
    uuids = {str(g): 'SYNTHETIC_UUID_' + str(g) for g in range(4)}

    class Stop(RuntimeError):
        classification = 'FAIL_AND_STOP'

    def require(ok, message, classification='FAIL_AND_STOP'):
        if not ok:
            exc = Stop(message)
            exc.classification = classification
            raise exc

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / mod.TASK
        for rel in ['logs', 'analysis', 'control'] + ['qualification_gpu/gpu' + str(g) + '/analysis' for g in range(4)]:
            (root / rel).mkdir(parents=True, exist_ok=True)

        def read(path):
            return json.loads(Path(path).read_text())

        def write(rel, value, exclusive=False):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x' if exclusive else 'w') as f:
                json.dump(value, f)
            if rel == 'analysis/GPU_RESOURCE_LEDGER_LIVE.json':
                state['ledger_count'] = len(value['phases'])

        class Child:
            def __init__(self, command, **kwargs):
                self.command = command
                self.pid = 9000 + len(jobs)
                self.birth = clock.now
                self.rc = None
                self.gpu = int(command[4])
                self.phase = command[3]
                self.output = Path(command[6])
                self.label = ('gpu' + str(self.gpu) + '_' + self.phase) if self.phase.startswith('canary') else self.phase
                jobs.append(self)

            def poll(self):
                duration = 3.0 if kind == 'lane_failure' and self.label.startswith('science_') and self.label.endswith('H_C128') and self.gpu != 1 else 1.0
                if self.rc is None and clock.now - self.birth >= duration:
                    self.rc = 1 if kind == 'lane_failure' and self.label == 'science_doc2_H_C128' else 0
                    if self.rc == 0:
                        result = dict(status='PASS')
                        if self.label in mod.SCIENCE:
                            parts = self.label.split('_', 2)
                            result.update(condition=parts[2], doc=int(parts[1][3:]), consumed=32768, scores=384)
                        p = self.output / 'analysis' / ('RECEIPT_' + self.phase + '.json')
                        p.write_text(json.dumps(dict(status='PASS', result=result)))
                return self.rc

            def send_signal(self, signal):
                state['signaled'].append(self.label)
                self.rc = -15

        def identity(pid):
            child = next(x for x in jobs if x.pid == pid)
            return dict(pid=pid, start_time_ticks=str(pid), cmdline=child.command)

        def terminate(child, recorded, label):
            if child.poll() is None:
                require(recorded is not None and control.exact_child(identity(child.pid), label), 'Wrong mock child')
                child.send_signal(None)
            return dict(exact_PID_only=True)

        def probe(*args):
            if kind == 'final_query_failure' and state['ledger_count'] == 23:
                return dict(status='QUERY_FAILED', error='Synthetic final inventory failure')
            out = {}
            for g in range(4):
                apps = []
                for child in jobs:
                    if child.gpu != g:
                        continue
                    if child.poll() is None or (kind == 'release_unknown' and child.label == 'gpu1_canary_initial'):
                        apps.append(dict(pid=child.pid))
                out[str(g)] = dict(physical_GPU_ID=g, GPU_UUID=uuids[str(g)], used_MiB=100, apps=apps)
            return dict(status='PASS', GPUs=out)

        control = SimpleNamespace(PYTHON='/SYNTHETIC_PYTHON', StopExecution=Stop, require=require,
            read=read, write=write, sha=lambda p:'SYNTHETIC_SHA', source_check=lambda m:None,
            process_identity=identity, terminate_own_child=terminate)
        helper = SimpleNamespace(resident_stdio=lambda r:str(r/'logs/SYNTHETIC.log'),
            tmux_residency=lambda:dict(status='PASS_SYNTHETIC'), emit=lambda *args:None,
            frozen_documents=lambda *args:None)
        budget = dict(phase_predictions_seconds={p:1000.0 for p in mod.SCIENCE})
        mod.helpers = lambda r:(helper, control)
        mod.verify = lambda *args:({}, {}, budget, uuids)
        mod.storage = lambda *args:dict(status='PASS_SYNTHETIC')
        mod.idle = lambda *args:dict(used_MiB=0)
        mod.inventory = probe
        mod.safe_inventory = probe
        mod.canary_memory = lambda *args:dict(status='PASS_SYNTHETIC')
        mod.cross_gpu_canary = lambda *args:dict(status='PASS_SYNTHETIC')
        def adjusted(*args):
            write('analysis/FOUR_GPU_PREDICTION_ADJUSTMENT.json', dict(status='SYNTHETIC'))
            return budget, {}
        mod.adjust_after_fresh = adjusted
        mod.subprocess = SimpleNamespace(Popen=Child, STDOUT=-2)
        mod.time = SimpleNamespace(monotonic=lambda:clock.now, time=lambda:clock.now,
                                  sleep=lambda seconds:setattr(clock, 'now', clock.now+seconds))
        original_signal = mod.signal.signal
        mod.signal = SimpleNamespace(SIGTERM=15, SIGHUP=1, signal=lambda *a:None)
        try:
            result = mod.run(root)
        finally:
            del original_signal
        ledger = read(root/'gpu_resource_ledger.json')
        require(abs(ledger['GPU_HOURS_USED'] * 3600 - sum(r['reservation_seconds'] for r in ledger['phases'])) < 1e-8,
                'Aggregate duration mismatch')
        if kind == 'complete':
            require(result['status'] == 'COMPLETE_AND_STOP' and result['GPU_release_verified'] and len(jobs) == 23,
                    'All23 should complete: ' + str(result.get('coordinator_failure')))
            require(all(result['gates'][p] == 'PASS' for p in mod.UNITS), 'Missing registered unit')
        elif kind == 'lane_failure':
            require(result['status'] == 'FAIL_AND_STOP' and len(jobs) < 23 and not any('doc5' in j.label for j in jobs),
                    'Failure must stop remaining work')
            require(all(j.poll() is not None for j in jobs), 'A mock owned worker remains active')
            require(set(state['signaled']) == {'science_doc1_H_C128', 'science_doc3_H_C128', 'science_doc4_H_C128'},
                    'Failure must signal all three still-running exact owned lanes')
        elif kind == 'release_unknown':
            require(result['status'] == 'FAIL_AND_STOP' and not result['GPU_release_verified'], 'Unknown release must fail')
            require(ledger['GPU_HOURS_USED'] * 3600 >= 30, 'Unknown release lost reservation accounting')
        elif kind == 'final_query_failure':
            require(result['status'] == 'FAIL_AND_STOP' and not result['GPU_release_verified']
                    and (root/'FIRST_FAILURE.json').is_file(), 'Final query failure cannot retain COMPLETE')
        return dict(status='PASS', scenario=kind, final_status=result['status'], dispatched=len(jobs),
                    released=result['GPU_release_verified'], total_GPU_seconds=ledger['GPU_HOURS_USED'] * 3600,
                    signaled_phases=state['signaled'])


if __name__ == '__main__':
    print(json.dumps(dict(status='PASS', CPU_ONLY=True, actual_scientific_data_read=False,
        torch_imported=False, GPU_work_performed=False,
        scenarios=[scenario(k) for k in ('complete', 'lane_failure', 'release_unknown', 'final_query_failure')]), ensure_ascii=False))
