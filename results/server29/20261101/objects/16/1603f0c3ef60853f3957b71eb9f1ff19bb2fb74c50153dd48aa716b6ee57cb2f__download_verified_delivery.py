#!/usr/bin/env python3
"""Finite, read-only verified small delivery from one completed task."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import tempfile
import time

TASK = 'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'
REMOTE = '/data/zypan/analyses/qwen_gdn_dual_m_equivalent_speedup_v1'
PY = '/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-S',
       '/tmp/qwen_gdn_replication_20261013.sock', 'zypan@202.38.247.55']

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ready-sha256', required=True)
    a = ap.parse_args()
    output = a.output.resolve()
    assert not output.exists(), 'Do not overwrite a previous delivery'
    command = shlex.join([PY, '-B', REMOTE + '/delivery_tools/pack_small_remote.py',
                          '--root', REMOTE, '--ready-sha256', a.ready_sha256])
    with tempfile.TemporaryFile() as transport:
        subprocess.run(SSH + [command], stdout=transport, check=True)
        transport.seek(0)
        with tarfile.open(fileobj=transport, mode='r:') as archive:
            members = archive.getmembers()
            names = [m.name for m in members]
            assert len(names) == len(set(names))
            data = {}
            for m in members:
                name = PurePosixPath(m.name)
                assert m.isfile() and not name.is_absolute() and '..' not in name.parts and m.size <= 16 * 1024**2
                assert not m.name.endswith('.trace.json')
                data[m.name] = archive.extractfile(m).read()
    assert digest(data['DELIVERY_READY.json']) == a.ready_sha256
    ready = json.loads(data['DELIVERY_READY.json'])
    assert ready['task_id'] == TASK and ready['scientific_status'] == 'PROFILE_ONLY_COMPLETE_AND_STOP'
    assert digest(data['SHA256_MANIFEST.json']) == ready['manifest']['sha256']
    manifest = json.loads(data['SHA256_MANIFEST.json'])
    rows = [r for r in manifest['all_current_task_files'] + manifest['frozen_small_external_dependencies'] if r['download']]
    for key in ('manifest', 'checksums', 'remote_verification'):
        row = ready[key]
        rows.append(dict(row, delivery_path=Path(row['path']).relative_to(REMOTE).as_posix()))
    rows.append(dict(delivery_path='DELIVERY_READY.json', bytes=len(data['DELIVERY_READY.json']), sha256=a.ready_sha256))
    expected = {r['delivery_path']: r for r in rows}
    assert set(expected) == set(data), 'Unexpected/missing file in finite delivery'
    verified = []
    for name, raw in data.items():
        r = expected[name]
        assert len(raw) == r['bytes'] and digest(raw) == r['sha256'], name
        verified.append(dict(path=name, bytes=len(raw), sha256=digest(raw)))
    proof = json.loads(data['hashes/FINAL_REMOTE_DELIVERY_VERIFICATION.json'])
    assert proof['status'] == 'PASS' and proof['large_trace_SHA_bytes_verified']
    assert proof['all_bound_process_identities_exited'] and proof['tracked_NVML_pids_absent']
    source = json.loads(data['SOURCE_MANIFEST.json'])
    for row in source['execution_sources']:
        assert digest(data[row['path']]) == row['sha256'] and len(data[row['path']]) == row['bytes']
    for row in source['external_dependencies']:
        name = 'dependency_snapshot/' + row['path'].lstrip('/')
        assert digest(data[name]) == row['sha256'] and len(data[name]) == row['bytes']
    assert digest(data['FINAL_TECHNICAL_REPORT.md']) == ready['report']['sha256']
    output.mkdir(parents=True)
    for name, raw in data.items():
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as handle:
            handle.write(raw)
        assert digest(dest.read_bytes()) == expected[name]['sha256']
    receipt = {'task_id': TASK, 'status': 'PASS', 'scientific_status': ready['scientific_status'],
               'unix': time.time(), 'remote_root': REMOTE, 'local_delivery': str(output),
               'downloaded_files': len(verified), 'all_downloaded_SHA_bytes_verified': True,
               'frozen_execution_sources_verified': len(source['execution_sources']),
               'frozen_external_dependencies_verified': len(source['external_dependencies']),
               'large_traces_downloaded': False, 'models_or_large_tensors_downloaded': False,
               'large_trace_count_remotely_verified': proof['large_trace_count'],
               'all_own_GPUs_released': True, 'GPU_hours': ready['GPU_hours'],
               'READY_sha256': a.ready_sha256, 'report_sha256': ready['report']['sha256'],
               'manifest_sha256': ready['manifest']['sha256'], 'files': sorted(verified, key=lambda x: x['path'])}
    receipt_path = output / 'LOCAL_DELIVERY_VERIFICATION.json'
    with receipt_path.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
