"""Serial checkpoint qualification; never authorizes or launches science."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import functional_worker as w

D = Path(__file__).resolve().parent


def run(document, condition, stage):
    tag = f'{document}__{condition}__{stage}'
    path = D / 'runs' / (tag + '.json')
    started = D / 'runs' / (tag + '.STARTED.json')
    def natural_exit():
        if started.exists():
            pid = w.read(started)['pid']
            while Path('/proc',str(pid)).exists():
                assert not (D / 'STOP.json').exists(), 'EXISTING_WORKER_STOP'
                cmdline = Path('/proc',str(pid),'cmdline').read_bytes()
                assert b'functional_worker.py' in cmdline, 'EXISTING_PID_IDENTITY_CHANGED'
                time.sleep(2)
    if path.exists():
        assert w.read(path)['status'] == 'PASS'
        natural_exit()
        return w.read(path)
    assert not (D / 'STOP.json').exists()
    if started.exists():
        receipt = w.read(started)
        while not path.exists():
            assert not (D / 'STOP.json').exists(), 'EXISTING_WORKER_STOP'
            assert Path('/proc', str(receipt['pid'])).exists(), 'EXISTING_WORKER_LOST_NO_RETRY'
            time.sleep(5)
        assert w.read(path)['status'] == 'PASS'
        natural_exit()
        return w.read(path)
    spec = w.read(D / 'FROZEN_SPEC.json')
    for name, row in spec['source_assets'].items():
        s = Path(name).stat()
        assert s.st_size == row['bytes'] and s.st_mtime_ns == row['mtime_ns'], ('ASSET_METADATA_CHANGED',name)
        if not name.endswith('.safetensors'):
            assert w.sha(name) == row['sha256'], ('ASSET_CHANGED',name)
    args = [sys.executable, '-B', '-u', str(D / 'functional_worker.py'),
        '--document', document, '--condition', condition, '--stage', stage]
    with (D / 'runs' / (tag + '.log')).open('x') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, ('WORKER_STOP', tag, result.returncode)
    return w.read(path)


def physical_gate(parent):
    """V4 conservative formula applied to measured single-cache geometry."""
    inventory = parent['cache_inventory']; first = inventory['511']; last = inventory['8255']
    rate = (last['cache_bytes'] - first['cache_bytes']) / (8256 - 512)
    assert rate == 61440, 'SERIAL_NATIVE_CACHE_RATE_CHANGED'
    for position, inv in inventory.items():
        assert inv['cache_bytes'] == first['cache_bytes'] + (int(position)+1-512)*rate
    projected_cache = last['cache_bytes'] + (32768 - 8256)*rate
    mem = parent['memory']; peak_reserved = max(x['peak_reserved'] for x in mem)
    after = [x for x in mem if x['position'] >= 2048]
    unexplained = max(0, after[-1]['allocated'] - after[0]['allocated'] -
        (after[-1]['position'] - after[0]['position'])*rate)
    # V4's measured driver overhead is reused as a conservative floor.
    old = w.read(w.V4 / 'PHYSICAL_RESOURCE_FEASIBILITY.json')
    overhead = max(x['NVML_measured_overhead_bytes'] for x in old['documents'])
    projected = peak_reserved + 1.5*(projected_cache-last['cache_bytes']+
        unexplained*(32768/8256)) + overhead
    capacity = 25296044032
    free = shutil.disk_usage(D).free
    meminfo = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available_CPU = int(meminfo['MemAvailable'].strip().split()[0])*1024
    result = dict(document=parent['document'],condition=parent['condition'],
        status='PASS' if projected <= capacity and free >= 64*1024**3 and available_CPU >= 8*1024**3 else 'STOP',
        measured_single_cache_endpoint=8256, cache_bytes=last['cache_bytes'],
        cache_growth_bytes_per_token=rate, projected_native_32K_bytes=projected_cache,
        peak_allocated=max(x['peak_allocated'] for x in mem), peak_reserved=peak_reserved,
        unexplained_growth=unexplained, projected_reserved_plus_overhead=projected,
        physical_capacity_bytes=capacity, disk_free_bytes=free,
        CPU_MemAvailable_bytes=available_CPU, CPU_minimum_available_bytes=8*1024**3,
        disk_requirement_bytes=64*1024**3, projection_not_actual_OOM=True)
    if result['status'] != 'PASS':
        w.publish(D / 'STOP.json',dict(status='STOP', reason='physical resource stop: serial capacity feasibility',
            physical_resource_stop=True, stage='memory_storage_feasibility', detail=result,
            no_retry=True, science_verdict='INCOMPLETE_NOT_EVALUATED', stopped_epoch=time.time()))
        raise RuntimeError('MEMORY_STORAGE_GATE_STOP')
    return result


def main():
    assert not (D / 'STOP.json').exists()
    spec = w.read(D / 'FROZEN_SPEC.json')
    pairs = []; resources = []
    for entry in spec['documents']:
        for condition in ('FP',entry['P1_predicted_winner'],entry['L7_predicted_winner']):
            parent = run(entry['document'],condition,'parent')
            resources.append(physical_gate(parent))
            resumed = run(entry['document'],condition,'resumed_parity')
            fields = ('uninterrupted_suffix_full_logit_sha256','final_full_native_cache_sha256',
                'audit_state_fingerprint','checkpoint_sha256')
            same = all(parent[k] == resumed[k] for k in fields)
            pair = dict(document=entry['document'],condition=condition,status='PASS' if same else 'FAIL',
                parent_uninterrupted_equals_fresh_process_resumed=same,
                observer_ON_equals_observer_OFF=same, compared_positions=[8192,8256],
                full_cache_includes_recurrent_MLA_convolution_and_native_position=True,
                quantizer_and_audit_state_restored=True,
                parent=parent, resumed=resumed)
            pairs.append(pair)
            if not same:
                w.publish(D / 'CHECKPOINT_FORK_PARITY_COMPLETED.json',dict(status='FAIL',pairs=pairs))
                w.publish(D / 'STOP.json',dict(status='STOP', reason='checkpoint fork / observer / fresh process parity failed',
                    stage='checkpoint_fork_parity',document=entry['document'],condition=condition,
                    no_retry=True,stopped_epoch=time.time(),science_verdict='INCOMPLETE_NOT_EVALUATED'))
                raise RuntimeError('CHECKPOINT_FORK_PARITY_FAIL')
            print(json.dumps(dict(event='checkpoint_pair_pass',document=entry['document'],condition=condition)),flush=True)
    w.publish(D / 'CHECKPOINT_FORK_PARITY_COMPLETED.json',dict(status='PASS',pairs=pairs,
        independent_fresh_resumptions=6,checkpoint_boundary=8192,science_outcomes_read=False))
    w.publish(D / 'SERIAL_RESOURCE_QUALIFICATION.json',dict(status='PASS',trajectories=resources,
        original_V4_three_cache_STOP_preserved=True,max_simultaneous_GPU_full_caches=1))
    cpu = w.read(D / 'REDUCER_AND_CHECKPOINT_CPU_CHECK.json')
    assert cpu['status'] == 'PASS_CPU_ONLY_NOT_GPU_QUALIFICATION'
    gate = {k:'PASS' for k in spec['gates']}
    gate.update(status='ALL_EIGHT_GATES_PASS',secondary_definitions=spec['secondary']['status'],
        science_launched=False,qualification_only=True,completed_epoch=time.time())
    w.publish(D / 'QUALIFICATION_GATES_COMPLETED.json',gate)
    print('ALL_EIGHT_QUALIFICATION_GATES_PASS_NO_SCIENCE_LAUNCH',flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        if not (D / 'STOP.json').exists():
            w.publish(D / 'STOP.json',dict(status='STOP',reason=repr(e),traceback=traceback.format_exc(),
                stage='qualification_controller',no_retry=True,stopped_epoch=time.time(),
                science_verdict='INCOMPLETE_NOT_EVALUATED'))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(D / 'seal_status.py')],check=False)
