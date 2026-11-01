"""Standalone stdlib-only forensic collector. Never import experiment/GPU code.

Writes exclusively beside this script, outside the original experiment root.
Does not execute any worker, launcher, selector, loss helper, or budget guard.
"""
import ast
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data/zypan/experiment_runs/LING_KDA_L7_OBJECTIVE_REDUNDANCY_AND_PERSISTENT_VALIDATION_V1')
OUT = Path(__file__).resolve().parent
TASK = 'LING_KDA_P1_L7_PHASEB_STOP_FORENSIC_AUDIT_V1'
COMMITS = {
    'PhaseA': 'a74255b1b173d27a3e51f16dbc4cb26b46104d08',
    'CPU_analyze': 'fb2c838a01bcbc09d2c2aef9ad4e166bb9c42c48',
    'Calibration': 'c7131f3f144846ab9d3f0558210493b37a896097',
    'Pair_audit': '288bdc0c6b98098d5710c10cff41980e82148bed',
    'Persistent': 'f1fa49883d5f1f65ed47d8ddc0b37a62cbee49d1',
}
IDENTITIES = {}
CHECKS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, expected=None, group=None):
    p = Path(path); key = str(p)
    if key not in IDENTITIES:
        if p.is_file():
            before = p.stat(); value = sha(p); after = p.stat()
            IDENTITIES[key] = dict(path=key, exists=True, bytes=after.st_size,
                mtime_ns=after.st_mtime_ns, sha256=value,
                stable_while_read=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
        else:
            IDENTITIES[key] = dict(path=key, exists=False)
    item = IDENTITIES[key]
    if expected is not None:
        CHECKS.append(dict(path=key, group=group, expected_sha256=expected,
                           actual_sha256=item.get('sha256'), passed=item.get('sha256') == expected))
    return item


def read(path):
    record(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
                observed_epoch=time.time())


def original_inventory():
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and '.git' not in p.parts and '__pycache__' not in p.parts)
    return {str(p): sha(p) for p in paths}


def source(path):
    record(path)
    return Path(path).read_text(encoding='utf-8')


def main():
    assert OUT != ROOT and OUT.name == TASK
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.dont_write_bytecode
    start = time.time(); before = original_inventory()
    names = [
        'PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json', 'SOURCE_IDENTITIES.json',
        'PROSPECTIVE_DOCUMENT_SELECTION.json', 'PHASE_B_DOCUMENT_SELECTION_RULE_FREEZE.json',
        'PRIOR_DOCUMENT_PATH_IDENTITY_AUDIT.json', 'PHASE_B_PAIR_METHODS_FREEZE.json',
        'PHASE_B_CALIBRATION_CONFIG.json', 'PHASE_B_CALIBRATION_CODE_FREEZE.json',
        'PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json', 'FRESH_MODEL_WEIGHT_AUDIT.json',
        'HISTORICAL_IMMUTABILITY_BASELINE.json', 'PREMEASUREMENT_CODE_FREEZE.json',
        'PREMEASUREMENT_ASSET_SUPPLEMENT.json', 'PERSISTENT_FAIL_CLOSED_STOP.json',
        'final_stop_20261011/DELIVERABLE_MANIFEST.json',
        'final_stop_20261011/RESULT_AND_DELIVERY_HASHES.json',
        'final_stop_20261011/GPU_COST_LEDGER.json', 'GPU_COST_LEDGER.json',
        'phase_a_deliverables_20261011/GPU_COST_LEDGER.json',
    ]
    data = {n: read(ROOT/n) for n in names}
    tables = read(ROOT/'PHASE_B_CALIBRATION_MATCHED_TABLES.json')
    data['PHASE_B_CALIBRATION_MATCHED_TABLES.json'] = {
        'scores': tables['scores'], 'ranks': tables['ranks'],
        'original_table_lengths': {k: len(v) for k, v in tables.items()}}
    for p in sorted(ROOT.glob('*.sha256.json')):
        side = read(p)
        if isinstance(side, dict) and 'path' in side and 'sha256' in side:
            record(side['path'], side['sha256'], 'root_sidecar')
    delivery = data['final_stop_20261011/RESULT_AND_DELIVERY_HASHES.json']
    for path, expected in delivery.items(): record(path, expected, 'final_delivery_manifest')
    sums = source(ROOT/'final_stop_20261011/SHA256SUMS.txt')
    for line in sums.splitlines():
        expected, path = line.split('  ', 1); record(path, expected, 'final_sha256sums')
    freeze = data['PERSISTENT_CODE_AND_ASSET_FREEZE_V1.json']
    for name, expected in freeze['files'].items(): record(ROOT/name, expected, 'frozen_persistent_code')
    for path, expected in freeze['dependencies'].items(): record(path, expected, 'frozen_persistent_dependency')
    identities = data['SOURCE_IDENTITIES.json']
    for item in identities['rotations'].values(): record(item['path'], item['sha256'], 'rotation_roster')
    for path, expected in identities['source_hashes'].items(): record(path, expected, 'source_identity')
    historical = data['HISTORICAL_IMMUTABILITY_BASELINE.json']
    for row in historical['files']:
        for kind in ('canonical', 'sidecar'):
            record(row[kind]['path'], row[kind]['sha256'], 'historical_' + kind)
    for path, expected in historical['attempt04_attempt05_permanent_stop'].items():
        record(path, expected, 'permanent_attempt_stop')
    weight_audit = data['FRESH_MODEL_WEIGHT_AUDIT.json']; weight_rows = []
    for row in weight_audit['files']:
        item = record(row['path'], row['sha256'], 'fresh_full_weight_rehash')
        weight_rows.append(dict(**item, original_audit_mtime_ns=row['mtime_ns'],
            original_audit_bytes=row['bytes'], metadata_matches_original=
            item.get('mtime_ns') == row['mtime_ns'] and item.get('bytes') == row['bytes']))
    cfg = data['PHASE_B_CALIBRATION_CONFIG.json']
    tokens = {}
    model_config = read(Path(identities['model_identity']['path'])/'config.json')
    for row in data['PROSPECTIVE_DOCUMENT_SELECTION.json']['selected_documents']:
        for kind in ('token_file', 'raw_text'): record(row[kind]['path'], row[kind]['sha256'], 'prospective_input')
        vals = read(row['token_file']['path'])
        tokens[row['document_id']] = dict(count=len(vals), container_type=type(vals).__name__,
            all_integer_nonnegative=all(type(v) is int and v >= 0 for v in vals),
            minimum=min(vals), maximum=max(vals), vocab_size=model_config.get('vocab_size'),
            legal_vocab=all(type(v) is int and 0 <= v < model_config['vocab_size'] for v in vals))
    calibration = {}; runs = {}; persistent_listing = []
    for p in sorted((ROOT/'runs_b_calibration').glob('*.json')):
        value = read(p)
        if p.name.endswith('.sha256.json'):
            record(value['path'], value['sha256'], 'calibration_sidecar')
        if '.partial.' not in p.name: calibration[p.name] = value
    for p in sorted((ROOT/'runs_b_persistent').iterdir()):
        if not p.is_file(): continue
        item = record(p); persistent_listing.append(item)
        if p.suffix == '.json': runs[p.name] = read(p)
        elif p.suffix == '.log': runs[p.name] = source(p)
    static = {}
    for name in ('run_persistent_stage_v1.py', 'persistent_stage_common_v1.py',
                 'launch_persistent_stage_v1.py', 'persistent_engine_v1.py',
                 'budget_persistent_complete_v1.py', 'select_phase_b_pairs.py',
                 'select_phase_b_documents_cpu.py', 'objective.py', 'analyze_phase_a.py',
                 'PERSISTENT_EXECUTION_PROTOCOL_V1.md'):
        static[str(ROOT/name)] = source(ROOT/name)
    for path in freeze['dependencies']:
        if Path(path).suffix == '.py' and any(x in Path(path).name for x in
                ('capture', 'common', 'replay_core', 'run_direction', 'run_short_horizon', 'transition')):
            static[path] = source(path)
    torch_source = Path('/data/zypan/envs/ling-kda/lib/python3.11/site-packages/torch/__init__.py')
    text = source(torch_source); tree = ast.parse(text)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
                   ('are_deterministic_algorithms_enabled', 'use_deterministic_algorithms')]
    static[str(torch_source)] = '\n\n'.join(ast.get_source_segment(text, n) for n in definitions)
    git = dict(head=command(['git','-C',str(ROOT),'rev-parse','HEAD']),
        top_level=command(['git','-C',str(ROOT),'rev-parse','--show-toplevel']),
        status_before=command(['git','-C',str(ROOT),'status','--porcelain=v1']),
        tracked_unstaged=command(['git','-C',str(ROOT),'diff','--name-status']),
        tracked_staged=command(['git','-C',str(ROOT),'diff','--cached','--name-status']), commits={})
    for role, commit in COMMITS.items():
        git['commits'][role] = dict(commit=commit,
            existence=command(['git','-C',str(ROOT),'cat-file','-e',commit+'^{commit}']),
            tree=command(['git','-C',str(ROOT),'rev-parse',commit+'^{tree}']),
            metadata=command(['git','-C',str(ROOT),'show','-s','--format=%H%n%T%n%cI%n%s',commit]))
    git['persistent_tree_file_comparison'] = {}
    for name in freeze['files']:
        p = subprocess.run(['git','-C',str(ROOT),'show',freeze['code_commit']+':'+name], capture_output=True)
        git['persistent_tree_file_comparison'][name] = dict(returncode=p.returncode,
            committed_content_sha256=hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None,
            actual_sha256=record(ROOT/name).get('sha256'),
            equals_run_file=p.returncode == 0 and hashlib.sha256(p.stdout).hexdigest() == record(ROOT/name).get('sha256'))
    packages = {}
    for name in ('torch','nvidia-cuda-runtime-cu12','fla-core','flash-linear-attention','transformers','triton','tokenizers'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    hardware = {}
    for gpu in (3,6):
        uuid = cfg['gpu_uuids'][str(gpu)]
        hardware[str(gpu)] = dict(expected_uuid=uuid,
            gpu_query=command(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
            compute_apps=command(['nvidia-smi','-i',uuid,'--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']),
            device_holders=command(['fuser',f'/dev/nvidia{gpu}']))
    disks = {}
    scratch_candidates = [ROOT, Path('/tmp'), Path('/dev/shm'), Path('/data/zypan/scratch'), Path('/scratch'),
                          Path('/data/zypan/.triton/cache'), Path('/data/zypan/.cache/torch')]
    for key in ('TMPDIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME'):
        if os.environ.get(key): scratch_candidates.append(Path(os.environ[key]))
    for p in scratch_candidates:
        if p.exists():
            usage = shutil.disk_usage(p)
            disks[str(p)] = dict(total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free,
                filesystem_device=p.stat().st_dev, inode_stat=list(os.statvfs(p)))
        else: disks[str(p)] = dict(exists=False)
    resources = dict(observed_epoch=time.time(), kernel=os.uname().release,
        packages=packages, hardware=hardware, disks=disks,
        maintenance=dict(reboot_required=Path('/var/run/reboot-required').exists(),
                         shutdown_schedule_present=Path('/run/systemd/shutdown/scheduled').exists()),
        audit_env=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       dont_write_bytecode=sys.dont_write_bytecode, python=sys.version, executable=sys.executable),
        torch_cuda_fla_transformers_not_imported=not any(n.split('.')[0] in {'torch','fla','transformers','triton'} for n in sys.modules))
    after = original_inventory(); git['status_after'] = command(['git','-C',str(ROOT),'status','--porcelain=v1'])
    changed = [p for p in sorted(set(before)|set(after)) if before.get(p) != after.get(p)]
    output = dict(task=TASK, actual_server_evidence=True, started_epoch=start, completed_epoch=time.time(),
        original_root=str(ROOT), output_root=str(OUT), collector_sha256=sha(__file__),
        data=data, calibration=calibration, persistent=runs, persistent_listing=persistent_listing,
        static_sources=static, git=git, resources=resources, tokens=tokens, model_weights=weight_rows,
        identities=IDENTITIES, hash_checks=CHECKS,
        original_inventory_before=before, original_inventory_after=after,
        original_files_changed_or_added_or_removed=changed,
        NO_NEW_GPU_SCIENTIFIC_RUN=True, NO_NEW_32K_LABELS=True, NO_TRAINING=True,
        no_original_worker_launcher_selector_guard_import_or_execution=True, gpu_hours_this_audit=0)
    path = OUT/'READONLY_SERVER_EVIDENCE.json'
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(output, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    with (OUT/'READONLY_SERVER_EVIDENCE.sha256.json').open('x', encoding='utf-8') as f:
        json.dump(dict(path=str(path), sha256=sha(path)), f, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps(dict(status='READONLY_FORENSIC_COLLECTION_COMPLETE', output=str(path), sha256=sha(path),
        hash_checks=len(CHECKS), hash_failures=sum(not x['passed'] for x in CHECKS),
        weight_shards_rehashed=len(weight_rows), original_files_changed=changed, gpu_hours_this_audit=0,
        no_gpu_runtime_imports=resources['torch_cuda_fla_transformers_not_imported'])))


if __name__ == '__main__': main()
