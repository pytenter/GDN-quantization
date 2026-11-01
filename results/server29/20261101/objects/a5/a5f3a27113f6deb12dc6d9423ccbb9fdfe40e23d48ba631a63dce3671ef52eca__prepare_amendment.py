"""Read-only old-tree audit plus append-only engineering amendment preparation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from execution_capture import write_json

A = Path(__file__).resolve().parent
E = A.parents[1]
ROOT = E.parents[1]
OLD = '66966044c0e25eb31fc15ffb1724f48de734a926'
HEAD = '566ae8f191d1fb429561d2ed6b9c4d0a730e44c9'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for raw in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == HEAD
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT)
    assert subprocess.check_output(['git', 'rev-parse', OLD], cwd=ROOT, text=True).strip() == OLD
    hist = read(E / 'historical_assets/asset_manifest.json')
    legacy = hist['legacy_worktree']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=legacy, text=True).strip() == hist['legacy_HEAD']
    status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=legacy)
    assert status == (E / 'historical_assets/git_status_porcelain_v1.txt').read_bytes()
    for x in hist['files']:
        assert sha(x['path']) == x['sha256'] == sha(x['snapshot_path'])
    print('LEGACY_READ_ONLY_HEAD_STATUS_85_BYTES_PASS', flush=True)
    identity = read(E / 'source_identity.json')
    for kind in ['inputs', 'protected_historical_files']:
        for i, x in enumerate(identity[kind]):
            assert Path(x['path']).stat().st_size == x['bytes'] and sha(x['path']) == x['sha256'], x['path']
            if i % 100 == 0:
                print('REHASH_PROTECTED', kind, i, len(identity[kind]), flush=True)
    for x in read(E / 'premeasurement_sha256.json')['files']:
        assert sha(x['path']) == x['sha256']
    protected = []
    for p in sorted(E.rglob('*')):
        if p.is_file() and not p.is_relative_to(A) and '__pycache__' not in p.parts:
            protected.append(dict(repo_relative_path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    write_json(A / 'protected_identity.json', dict(files=protected,
        legacy_HEAD=hist['legacy_HEAD'], legacy_git_status_sha256=hashlib.sha256(status).hexdigest(),
        historical_asset_count=len(hist['files']), inherited_count=len(identity['inputs']),
        protected_parent_count=len(identity['protected_historical_files']),
        legacy_worktree_changed=False, old_stops_preserved=True))
    write_json(A / 'amendment.json', dict(EXPERIMENT='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',
        TASK_CLASS='ENGINEERING_ONLY_AMENDMENT', OLD_PREMEASUREMENT_COMMIT_SHA=OLD,
        predecessor_stop_commit=HEAD, authority='Current explicit user engineering-only amendment',
        old_failure_interpretation='Engineering/dependency retention failure. NOT evidence against observability. NOT a numeric/historical parity failure. NO Discovery outcome.',
        permitted_modes=['verify', 'smoke', 'parity', 'gate'], formal_discovery_authorized=False,
        scientific_metrics_thresholds_tolerances_model_rotations_quantizer_recurrence_docs_horizons_FP_KL_JPROP_K3_scorer='UNCHANGED',
        GPU6_FORMAL_AUTHORIZED='NO', FP_REFERENCE_PAIRING_GATE='DEVELOPMENT_SMOKE_ONLY',
        cross_hardware_FP_bitwise='FAIL_PRESERVED', FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',
        FINAL_ACTION='PASS_PREMEASUREMENT_AND_STOP_OR_FAIL_AND_STOP',
        NEXT_AUTHORITY_REQUIRED='Explicit authorization to begin formal Discovery'))
    write_json(A / 'manifest_schema.json', dict(schema='EXECUTION_BYTE_CAPTURE_V1',
        required_event_fields=['logical_name', 'module_name', 'original_absolute_path', 'preserved_artifact_path',
            'byte_size', 'sha256_capture', 'sha256_pre_execution', 'capture_stage', 'executed', 'execution_kind',
            'python_executable', 'process_identity', 'git_classification', 'source_classification'],
        closure_rule='retained_blob_SHA == capture_SHA == immediate_preexecution_SHA; observed ledger == retained event ids',
        missing_original_path_at_execution='FAIL', missing_or_corrupt_blob_after_process_exit='FAIL',
        raw_source_reconstruction='FORBIDDEN', scientific_identity_excludes_random_temp_path=True))
    with (A / 'AMENDMENT.md').open('x') as f:
        f.write('# Engineering-only executed-dependency retention amendment\n\n')
        f.write('Old preregistration, original Python scientific harness and all STOP/evidence files are byte-protected. No amend or old-tree repair.\n\n')
        f.write('Confirmed call chain: torch/distributed/nn/api/remote_module.py:30 → instantiator.instantiate_non_scriptable_remote_module_template → _do_instantiate_remote_module_template → _write → importlib.import_module. instantiator creates TemporaryDirectory and registers cleanup at exit. bridge.smoke → execution_inventory records the live path; bridge.gate rehashes that original path after smoke subprocess exit.\n\n')
        f.write('Wrapper captures raw file bytes before loader get_code, compares the returned code to that exact source, and audit exec rehashes source immediately before execution. Compile inputs and interpreter-frozen code are recorded with explicit representations; Python function-entry accounting covers already-loaded callable dependencies. Native extension creation records actual binary bytes. No source/kernel/returned-code mutation.\n\n')
        f.write('Coverage: CPython exec audit, first Python code-object entry and aggregated subsequent entry counts, threads created after installation, source/extension loaders. Interpreter/bootstrap execution before observer installation, subprocesses without this observer, implicit native shared-library dependencies and GPU machine-code execution are not claimed as execution-time Python closure. Frozen runtime/input SHA inventories remain the authority for those binaries. Observer internals/reentrant provenance I/O are the explicitly hashed trusted observer boundary, not scientific dependencies. The independent boundary ledger detects missing metadata; it is not a defense against malicious observer tampering.\n\n')
        f.write('All original gate definitions and numeric/scientific functions are unchanged and rerun in fresh processes, using new output paths. Old automatic Discovery controller is never called. FP pairing remains development-smoke-only. Random temporary paths/PIDs are provenance fields, not scientific identity.\n\n')
        f.write('After the new commit: verify → smoke0 → smoke1 → parity → closure/gate → STOP. Discovery is unconditionally prohibited.\n')
    print('ENGINEERING_AMENDMENT_PREPARATION_PASS', len(protected), flush=True)


if __name__ == '__main__':
    main()
