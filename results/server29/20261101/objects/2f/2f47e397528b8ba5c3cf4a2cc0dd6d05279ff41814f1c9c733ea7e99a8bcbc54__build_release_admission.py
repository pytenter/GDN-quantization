#!/usr/bin/env python3
"""Build post-exit release evidence. Default is a no-query, no-write preview.

--run performs only process-file reads and final nvidia-smi inventory queries;
it cannot launch, signal, retry, import a model/runtime, or alter native payload.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

TASK = 'QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1'
SCHEMA = 'QWEN_RECOVERY_RELEASE_ADMISSION_V1'
UNITS = ['NUMERIC_FRESH0', 'NUMERIC_FRESH1', 'OLD_FP_R0', 'OLD_FP_R1', 'OLD_FP_R2', 'NEW_FP_R0']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def stable(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe evidence: ' + str(path))
    before = path.stat()
    require(before.st_size <= (128 if path.name == 'RESOURCE_EVENTS.jsonl' else 32) * 1024**2, 'Oversized metadata evidence')
    body = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Evidence changed while reading: ' + str(path))
    return body


def digest(body):
    return hashlib.sha256(body).hexdigest()


def read(path):
    return json.loads(stable(path))


def descriptor(path, base):
    path = Path(path).resolve(); base = Path(base).resolve()
    require(base in path.parents, 'Evidence descriptor escapes expected root')
    body = stable(path)
    return dict(path=str(path.relative_to(base)), bytes=len(body), sha256=digest(body))


def identity_key(value):
    require(isinstance(value, dict) and type(value.get('pid')) is int and value['pid'] > 0 and
            type(value.get('starttick')) is int and value['starttick'] > 0 and
            isinstance(value.get('cmdline'), list) and value['cmdline'] and
            all(isinstance(x, str) for x in value['cmdline']), 'Invalid exact process identity')
    return value['pid'], value['starttick'], tuple(value['cmdline'])


def native_identity(value):
    identity_key(value)
    return {k: value[k] for k in ['pid', 'starttick', 'cmdline']}


def process_observation(pid):
    proc = Path('/proc') / str(pid)
    try:
        before = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        cmdline = (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        after = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    require(before[19] == after[19], 'PID changed during /proc identity read')
    return dict(pid=int(pid), starttick=int(after[19]), state=after[0],
                cmdline=[x.decode() for x in cmdline] if cmdline != [b''] else [])


def exit_observation(expected, actual):
    if actual is None:
        return dict(process_exit_verified=True, observation='PID_ABSENT', actual_process=None)
    require(actual['pid'] == expected['pid'], 'Process reader returned another PID')
    if actual['starttick'] != expected['starttick']:
        return dict(process_exit_verified=True, observation='PID_REUSED_OLD_STARTTICK_GONE', actual_process=actual)
    if actual.get('state') in ['Z', 'X', 'x']:
        return dict(process_exit_verified=True, observation='SAME_STARTTICK_EXITED_ZOMBIE_OR_DEAD', actual_process=actual)
    return dict(process_exit_verified=False,
                observation='OWN_PROCESS_STILL_ALIVE' if actual['cmdline'] == expected['cmdline'] else 'SAME_PID_STARTTICK_CHANGED_IMAGE_UNRESOLVED',
                actual_process=actual)


def scan_run_processes(root):
    paths = [str(root / 'payload/scripts/restoration_controller.py'),
             str(root / 'payload/worker/restoration_worker.py'),
             str(root / 'scripts/offline_controller_once.py')]
    matches = []; inaccessible = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            actual = process_observation(int(entry.name))
            if actual is not None and any(path in actual['cmdline'] for path in paths):
                matches.append(actual)
        except FileNotFoundError:
            continue
        except PermissionError:
            inaccessible.append(int(entry.name))
    require(not inaccessible, 'Cannot inspect same-user process identities: ' + repr(inaccessible))
    return matches


def query_final_inventory():
    started = time.monotonic(); commands = []
    requests = [('devices', ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits']),
                ('apps', ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'])]
    result = {}
    for kind, command in requests:
        before = time.monotonic()
        p = subprocess.run(command, capture_output=True, text=True, timeout=10)
        commands.append(dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
                             before_monotonic=before, after_monotonic=time.monotonic()))
        require(p.returncode == 0, 'Final NVML inventory command failed: ' + kind)
        rows = []
        for line in p.stdout.splitlines():
            if not line.strip():
                continue
            fields = [x.strip() for x in line.split(',')]
            if kind == 'devices':
                require(len(fields) == 6, 'Malformed final GPU inventory')
                rows.append(dict(physical_index=int(fields[0]), uuid=fields[1], name=fields[2],
                                 memory_total_MiB=fields[3], memory_used_MiB=fields[4], memory_free_MiB=fields[5]))
            else:
                require(len(fields) == 3, 'Malformed final compute inventory')
                rows.append(dict(pid=int(fields[0]), uuid=fields[1], memory_MiB=fields[2]))
        result[kind] = rows
    require(result['devices'], 'Final full GPU device inventory is empty')
    result.update(commands=commands, before_monotonic=started, after_monotonic=time.monotonic(),
                  scope='Two read-only nvidia-smi inventories, not an atomic simultaneous hardware snapshot')
    return result


def load_closed_native(root, initial_path):
    payload = root / 'payload'
    invocation_path = root / 'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json'
    exit_path = root / 'operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json'
    require(root in initial_path.resolve().parents and payload not in initial_path.resolve().parents,
            'Initial actual process snapshot must be outside payload, inside this run')
    invocation, exited, initial = [read(p) for p in [invocation_path, exit_path, initial_path]]
    require(invocation.get('task_id') == TASK and invocation.get('retry') is False and exited.get('retry') is False,
            'Native one-shot invocation/exit receipt missing')
    require(type(exited.get('returncode')) is int and exited['before_monotonic'] <= exited['after_monotonic'] and
            math.isfinite(exited['wall_seconds']) and exited['wall_seconds'] >= 0 and
            abs(exited['wall_seconds'] - (exited['after_monotonic'] - exited['before_monotonic'])) < 1e-6,
            'Invalid actual native controller exit receipt')
    require(exited['command'] == invocation['command'], 'Native exit command differs from original invocation')
    freeze_path = payload / 'SOURCE_FREEZE.json'
    freeze_sha = digest(stable(freeze_path))
    auth_path = payload / 'HUMAN_GPU_RESTORE_AUTHORIZATION.json'
    state, auth = read(payload / 'ROOT.json'), read(auth_path)
    require(state.get('task_id') == auth.get('task_id') == TASK and state.get('ALLOW_GPU_RESTORE') == auth.get('ALLOW_GPU_RESTORE') == 'YES',
            'Independent task authorization/root binding differs')
    require(auth.get('execution_root') == str(payload) and auth.get('source_freeze_sha256') == invocation['source_freeze_sha256'] == freeze_sha and
            digest(stable(auth_path)) == invocation['authorization_sha256'], 'Actual native authorization/source identity changed')
    marker_path = payload / 'execution/CONTROLLER_STARTED.json'
    if not marker_path.is_file():
        marker_path = payload / 'execution/CONTROLLER_ATTEMPT_STARTED.json'
    marker = read(marker_path)
    require(marker.get('task_id') == TASK and marker.get('authorization_sha256') == invocation['authorization_sha256'],
            'Native controller marker task/authorization differs')
    if marker_path.name == 'CONTROLLER_STARTED.json':
        require(marker.get('source_freeze_sha256') == freeze_sha, 'Native controller marker source freeze differs')
    controller = native_identity(marker['controller_identity'])
    supervisor = native_identity(invocation['supervisor_identity'])
    require(controller['cmdline'] == invocation['command'], 'Native controller identity/command differs')
    expected_wrapper = str(root / 'scripts/offline_controller_once.py')
    require(expected_wrapper in supervisor['cmdline'], 'Supervisor command does not belong to this exact run')
    require(initial.get('operations/OFFLINE_NATIVE_INVOCATION_STARTED.json') == invocation,
            'Original actual invocation snapshot differs')
    initial_identities = {identity_key(x) for x in initial.get('actual_processes', [])}
    require(identity_key(controller) in initial_identities and identity_key(supervisor) in initial_identities,
            'Controller/supervisor lack matching actual initial process observations')
    native_path = payload / 'execution/CONTROLLER_RESULT.json'; native = read(native_path)
    require(native.get('task_id') == TASK and native.get('status') in ['COMPLETE_FINITE_RECOVERY_UNITS', 'FAIL_CLOSED_STOP'],
            'Native controller has no legal terminal receipt')
    require((native['status'] == 'COMPLETE_FINITE_RECOVERY_UNITS' and exited['returncode'] == 0) or
            (native['status'] == 'FAIL_CLOSED_STOP' and exited['returncode'] != 0), 'Native terminal status/true process returncode differ')
    ledger_path = payload / 'execution/RESOURCE_EVENTS.jsonl'; ledger_raw = stable(ledger_path)
    require(ledger_raw.endswith(b'\n'), 'Native resource ledger has an uncommitted last line')
    events = [json.loads(x) for x in ledger_raw.splitlines() if x.strip()]
    require(events and all(type(x.get('monotonic')) in [int, float] and math.isfinite(x['monotonic']) for x in events),
            'Native resource ledger time evidence absent')
    require(all(a['monotonic'] <= b['monotonic'] for a, b in zip(events, events[1:])), 'Native resource ledger time order differs')
    if native['status'] == 'FAIL_CLOSED_STOP':
        require(events[-1].get('event') == 'FAIL_CLOSED_STOP', 'STOP native ledger is not terminal/closed')
        first_path = payload / 'FIRST_FAILURE.json'; first = read(first_path)
        require(first.get('status') == 'FAIL_CLOSED_STOP', 'STOP first failure not retained')
    else:
        first_path = None
        require(not (payload / 'FIRST_FAILURE.json').exists(), 'Complete native run has contradictory failure receipt')
    dispatches = [x for x in events if x.get('event') == 'FIRST_DISPATCH']
    workers = []; seen_units = set(); seen_ids = set(); evidence = [invocation_path, exit_path, initial_path, freeze_path, auth_path, marker_path, native_path, ledger_path]
    for dispatch in dispatches:
        unit = dispatch['unit']['id']; worker = native_identity(dispatch['worker_identity'])
        require(unit in UNITS and unit not in seen_units and identity_key(worker) not in seen_ids,
                'Duplicate/unknown native worker dispatch identity')
        seen_units.add(unit); seen_ids.add(identity_key(worker)); workers.append(worker)
        receipt_path = payload / 'execution' / unit / 'DISPATCH.json'; receipt = read(receipt_path)
        require(receipt['worker_identity'] == worker and receipt['dispatch_before_monotonic'] == dispatch['before_monotonic'] and
                receipt['dispatch_after_monotonic'] == dispatch['after_monotonic'], 'Native DISPATCH and ledger differ')
        evidence.append(receipt_path)
        for kind in ['FIRST_PROCESS_EXIT_OBSERVED', 'FIRST_ALL_UUID_NVML_RELEASE_OBSERVED', 'UNIT_RESERVATION_INTERVAL_BOUNDS']:
            matches = [x for x in events if x.get('event') == kind and
                       x.get('unit', {}).get('id', x.get('unit_id')) == unit]
            require(len(matches) == 1, 'Missing/duplicate native first exit/release/reservation event: ' + unit + ':' + kind)
            if kind != 'UNIT_RESERVATION_INTERVAL_BOUNDS':
                require(matches[0].get('worker_identity') == worker, 'Native exit/release exact worker identity differs')
        observed_exit = next(x for x in events if x.get('event') == 'FIRST_PROCESS_EXIT_OBSERVED' and x.get('worker_identity') == worker)
        observed_release = next(x for x in events if x.get('event') == 'FIRST_ALL_UUID_NVML_RELEASE_OBSERVED' and x.get('worker_identity') == worker)
        require(dispatch['after_monotonic'] <= observed_exit['observed_monotonic'] <= observed_release['observed_monotonic'],
                'Native dispatch/exit/release time order differs')
    actual_dispatch = sorted(p.parent.name for p in (payload / 'execution').glob('*/DISPATCH.json'))
    require(actual_dispatch == sorted(seen_units), 'Native committed DISPATCH roster differs from closed ledger')
    if native['status'] == 'COMPLETE_FINITE_RECOVERY_UNITS':
        require([x['unit']['id'] for x in dispatches] == UNITS and native.get('completed_units') == UNITS,
                'Complete native finite unit roster differs')
    if first_path:
        evidence.append(first_path)
    return dict(payload=payload, source_freeze_sha256=freeze_sha, controller=controller, supervisor=supervisor,
                workers=workers, native=native, native_path=native_path, ledger_path=ledger_path,
                ledger_raw=ledger_raw, marker_path=marker_path, first_path=first_path, evidence=evidence,
                exit_receipt=exited, events=events)


def build(root, initial_path, output, *, process_reader=None, inventory_reader=None, scanner=None, clock=time.monotonic):
    root = Path(root).resolve(); initial_path = Path(initial_path).resolve(); output = Path(output).resolve()
    require(root in output.parents and root / 'payload' not in output.parents and not output.exists(),
            'Release proof must be new and outside payload, inside this run')
    native = load_closed_native(root, initial_path)
    reader = process_reader or process_observation; scan = scanner or scan_run_processes; inventory = inventory_reader or query_final_inventory
    expected = [(native['controller'], 'NATIVE_CONTROLLER'), (native['supervisor'], 'OFFLINE_SUPERVISOR')] + [(x, 'NATIVE_WORKER') for x in native['workers']]
    require(len({x['pid'] for x, _ in expected}) == len(expected), 'Owned numeric PID roster is ambiguous/reused across native units')
    began = clock(); before = []
    for identity, role in expected:
        observation = exit_observation(identity, reader(identity['pid']))
        require(observation['process_exit_verified'], 'Owned process is still alive or unresolved: ' + role)
        before.append(dict(identity=identity, role=role, **observation))
    before_scan = scan(root)
    require(not [x for x in before_scan if x.get('state') not in ['Z', 'X', 'x']], 'Additional live native run process remains')
    final_inventory = inventory()
    require(isinstance(final_inventory.get('apps'), list) and final_inventory.get('devices'), 'Final full actual NVML inventory incomplete')
    owned_pids = {x['pid'] for x, _ in expected}
    own_apps = [x for x in final_inventory['apps'] if x['pid'] in owned_pids]
    require(not own_apps, 'An owned numeric PID is still present on an actual NVML UUID (including PID reuse ambiguity)')
    after = []; processes = []
    for identity, role in expected:
        observation = exit_observation(identity, reader(identity['pid']))
        require(observation['process_exit_verified'], 'Owned process reappeared/alive after final inventory')
        after.append(dict(identity=identity, role=role, **observation))
        processes.append(dict(identity, role=role, process_exit_verified=True, all_UUID_NVML_release_verified=True,
                              final_process_observation=observation))
    after_scan = scan(root)
    require(not [x for x in after_scan if x.get('state') not in ['Z', 'X', 'x']], 'Native run process remains after final inventory')
    require(stable(native['ledger_path']) == native['ledger_raw'], 'Native closed resource ledger changed during release observation')
    require(read(native['native_path']) == native['native'], 'Native terminal receipt changed during release observation')
    snapshot = dict(status='ACTUAL_FINAL_PROCESS_AND_ALL_GPU_NVML_INVENTORY', before_monotonic=began,
                    after_monotonic=clock(), process_observations_before=before, process_observations_after=after,
                    additional_run_process_scan_before=before_scan, additional_run_process_scan_after=after_scan,
                    final_NVML_inventory=final_inventory, final_owned_NVML_records=own_apps,
                    all_foreign_NVML_records_preserved=[x for x in final_inventory['apps'] if x['pid'] not in owned_pids],
                    no_signals_sent=True, no_GPU_dispatch=True)
    snapshot_path = output.with_name(output.stem + '_FINAL_SNAPSHOT.json')
    require(not snapshot_path.exists(), 'Final release snapshot already exists; no overwrite')
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open('xb') as stream:
        stream.write(encoded(snapshot)); stream.flush(); os.fsync(stream.fileno())
    evidence = [descriptor(p, root) for p in native['evidence']] + [descriptor(snapshot_path, root)]
    proof = dict(schema=SCHEMA, task_id=TASK, status='PASS_ALL_OWN_PROCESSES_EXITED_ALL_UUIDS_RELEASED',
                 payload_root=str(native['payload']), execution_root=str(native['payload']),
                 source_freeze_sha256=native['source_freeze_sha256'], controller_identity=native['controller'],
                 supervisor_identity=native['supervisor'], supervisor_identities=[native['supervisor']],
                 owned_worker_identities=native['workers'], own_processes=processes,
                 controller_exit_verified=True, controller_exited=True, all_owned_processes_exited=True,
                 all_owned_GPU_resources_released=True, final_owned_NVML_records=own_apps,
                 final_NVML_apps=final_inventory['apps'], final_NVML_devices=final_inventory['devices'],
                 final_NVML_inventory=final_inventory, final_process_snapshot=descriptor(snapshot_path, root),
                 native_scientific_controller_status=native['native']['status'],
                 controller_result=dict(descriptor(native['native_path'], native['payload']), status=native['native']['status']),
                 resource_ledger=descriptor(native['ledger_path'], native['payload']),
                 first_failure=descriptor(native['first_path'], native['payload']) if native['first_path'] else None,
                 native_controller_exit_receipt=descriptor(root / 'operations/NATIVE_CONTROLLER_EXIT_RECEIPT.json', root),
                 native_invocation_receipt=descriptor(root / 'operations/OFFLINE_NATIVE_INVOCATION_STARTED.json', root),
                 evidence_receipts=evidence, no_signals_sent=True, GPU_dispatches=0,
                 release_builder_sha256=digest(stable(Path(__file__))),
                 meaning='Release admission only; native scientific status is retained, never promoted. PID-reused NVML rows conservatively block release.')
    with output.open('xb') as stream:
        stream.write(encoded(proof)); stream.flush(); os.fsync(stream.fileno())
    return proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--initial-snapshot', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args(); root = args.root.resolve()
    if not args.run:
        print(json.dumps(dict(status='PREVIEW_ONLY_NO_QUERY_NO_WRITES', task_id=TASK, root=str(root),
                              GPU_dispatches=0, NVML_queries=0, process_queries=0,
                              required_initial_snapshot='operations/LIVE_INITIAL_PROCESS_SNAPSHOT.json')))
        return
    initial = args.initial_snapshot or root / 'operations/LIVE_INITIAL_PROCESS_SNAPSHOT.json'
    output = args.output or root / 'analysis/POST_GPU_RELEASE_ADMISSION.json'
    result = build(root, initial, output)
    print(json.dumps(dict(status=result['status'], output=str(output), own_processes=len(result['own_processes']),
                         native_scientific_controller_status=result['native_scientific_controller_status'], GPU_dispatches=0)))


if __name__ == '__main__':
    main()
