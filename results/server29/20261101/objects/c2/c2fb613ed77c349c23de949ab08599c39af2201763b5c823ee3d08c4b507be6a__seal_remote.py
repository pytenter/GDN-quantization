#!/usr/bin/env python3
"""CPU-only final catalog/seal. Never import GPU libraries or modify frozen sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TASK = 'QWEN_GDN_DUAL_M_EQUIVALENT_SPEEDUP_V1'
SMALL_LIMIT = 16 * 1024**2
ALLOWED = {'.json', '.jsonl', '.csv', '.md', '.txt', '.py', '.log', '.sh'}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def load(p):
    return json.loads(p.read_text())

def create(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with p.open('xb') as f:
        f.write(data)
    return {'path': str(p), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def absent_old_identity(pid, expected_tick):
    p = Path('/proc') / str(pid) / 'stat'
    if not p.exists():
        return True
    try:
        tick = p.read_text().rsplit(')', 1)[1].split()[19]
        return tick != str(expected_tick)
    except FileNotFoundError:
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest_path = root / 'SHA256_MANIFEST.json'
    assert not manifest_path.exists(), 'Never overwrite a seal'
    ledger_path = root / 'GPU_RESOURCE_LEDGER.json'
    ledger_sha = sha(ledger_path)
    ledger = load(ledger_path)
    assert ledger['status'] == 'PASS_PHASE_COMPLETE'
    assert ledger['all_own_GPUs_released'] and ledger['cumulative_GPU_hours'] <= 2
    assert len(ledger['dispatches']) == 1
    assert not (root / 'FIRST_FAILURE.json').exists()
    result = load(root / 'execution/profile/RESULT.json')
    controller = load(root / 'execution/profile/CONTROLLER_RESULT.json')
    auth = load(root / 'AUTHORIZATION_PHASE0.json')
    assert result['status'] == 'PASS' and controller['status'] == 'PASS'
    for key, name in [('source_manifest_sha256', 'SOURCE_MANIFEST.json'),
                      ('plan_sha256', 'PLAN.json'), ('reference_binding_sha256', 'REFERENCE_BINDING.json')]:
        assert sha(root / name) == auth[key] == result[key], key
    assert result['authorization_sha256'] == sha(root / 'AUTHORIZATION_PHASE0.json')
    assert controller['ledger_sha256'] == ledger_sha
    assert load(root / 'GATE_STATUS_FINAL.json')['status'] == 'PROFILE_ONLY_COMPLETE_AND_STOP'
    sources = load(root / 'SOURCE_MANIFEST.json')
    for row in sources['execution_sources']:
        p = root / row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
    external = []
    for row in sources['external_dependencies']:
        p = Path(row['path'])
        assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
        assert row['bytes'] <= SMALL_LIMIT and p.suffix in ALLOWED
        external.append(dict(row, delivery_path='dependency_snapshot/' + str(p).lstrip('/'), download=True))
    create(root / 'hashes/FINAL_EXTERNAL_DEPENDENCY_VERIFICATION.json', {
        'task_id': TASK, 'status': 'PASS', 'scope': 'Fresh CPU SHA/bytes of all frozen small external dependencies',
        'source_manifest_sha256': sha(root / 'SOURCE_MANIFEST.json'), 'checked': len(external), 'files': external,
        'unix': time.time(), 'model_or_large_tensor_downloaded': False,
    })
    excluded_names = {'SHA256_MANIFEST.json', 'DELIVERY_READY.json', 'SHA256SUMS.txt',
                      'FINAL_REMOTE_DELIVERY_VERIFICATION.json'}
    entries = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name in excluded_names or '__pycache__' in p.parts:
            continue
        assert not p.is_symlink(), 'Unexpected symlink: ' + str(p)
        rel = p.relative_to(root).as_posix()
        allowed = p.suffix in ALLOWED and p.stat().st_size <= SMALL_LIMIT and not rel.startswith('profiles/')
        entries.append({'path': rel, 'remote_path': str(p), 'bytes': p.stat().st_size,
                        'sha256': sha(p), 'download': allowed,
                        'delivery_path': rel if allowed else None,
                        'role': 'REMOTE_ONLY_LARGE_CHROME_TRACE' if rel.startswith('profiles/') else 'TASK_ASSET'})
    assert len([x for x in entries if x['role'] == 'REMOTE_ONLY_LARGE_CHROME_TRACE']) == 3
    manifest = {'task_id': TASK, 'status': 'SEALED_PROFILE_ONLY_COMPLETE_AND_STOP', 'unix': time.time(),
                'remote_root': str(root), 'all_current_task_files': entries,
                'frozen_small_external_dependencies': external,
                'source_manifest_sha256': sha(root / 'SOURCE_MANIFEST.json'), 'phase0_ledger_sha256': ledger_sha,
                'candidate_created': False, 'large_assets_retained_remote': True,
                'self_reference_policy': 'This manifest, checksums, final remote proof and DELIVERY_READY are excluded to avoid circular hashes; READY binds their exact SHA/bytes.'}
    manifest_desc = create(manifest_path, manifest)
    sums = ''.join(x['sha256'] + '  ' + x['delivery_path'] + '\n'
                   for x in entries + external if x['download'])
    sums += manifest_desc['sha256'] + '  SHA256_MANIFEST.json\n'
    sums_path = root / 'hashes/SHA256SUMS.txt'
    with sums_path.open('x') as f:
        f.write(sums)
    for row in entries + external:
        p = Path(row.get('remote_path', row['path']))
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
    tracked = []
    start = load(root / 'execution/profile/CONTROLLER_STARTED.json')
    tracked.append((start['pid'], start['identity']['start_tick']))
    tracked.extend((d['pid'], d['identity']['start_tick']) for d in ledger['dispatches'])
    assert all(absent_old_identity(*x) for x in tracked)
    nvml = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True).stdout.strip()
    live_pids = {int(line.split(',')[0]) for line in nvml.splitlines() if line.strip()}
    assert not live_pids.intersection(pid for pid, tick in tracked)
    assert sha(ledger_path) == ledger_sha
    proof_desc = create(root / 'hashes/FINAL_REMOTE_DELIVERY_VERIFICATION.json', {
        'task_id': TASK, 'status': 'PASS', 'unix': time.time(), 'manifest': manifest_desc,
        'checksums': {'path': str(sums_path), 'bytes': sums_path.stat().st_size, 'sha256': sha(sums_path)},
        'root_asset_count': len(entries), 'root_assets_fresh_independent_SHA_bytes_verified': True,
        'external_dependency_count': len(external), 'external_dependencies_fresh_SHA_bytes_verified': True,
        'frozen_execution_sources': len(sources['execution_sources']), 'frozen_execution_sources_unchanged': True,
        'large_trace_count': 3, 'large_trace_SHA_bytes_verified': True,
        'no_models_or_large_tensors_downloaded': True, 'all_bound_process_identities_exited': True,
        'tracked_processes': tracked, 'tracked_NVML_pids_absent': True, 'NVML_apps': nvml,
        'ledger_sha256': ledger_sha, 'GPU_hours': ledger['cumulative_GPU_hours'],
    })
    def desc(p):
        return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
    ready = {'task_id': TASK, 'status': 'READY_FOR_EXACT_SMALL_FILE_DOWNLOAD', 'manifest': manifest_desc,
             'remote_verification': proof_desc, 'checksums': desc(sums_path),
             'report': desc(root / 'FINAL_TECHNICAL_REPORT.md'),
             'gate_status': desc(root / 'GATE_STATUS_FINAL.json'), 'GPU_hours': ledger['cumulative_GPU_hours'],
             'all_own_GPUs_released': True, 'scientific_status': 'PROFILE_ONLY_COMPLETE_AND_STOP'}
    ready_desc = create(root / 'DELIVERY_READY.json', ready)
    print(json.dumps({'status': 'PASS', 'ready': ready_desc, 'root_asset_count': len(entries),
                      'external_count': len(external), 'large_traces_remain_remote': 3}))

if __name__ == '__main__':
    main()
