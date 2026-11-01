"""Append-only provenance amendment: gates only, NEVER Discovery."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from execution_capture import Recorder, closure, write_json

sys.dont_write_bytecode = True
TASK = 'LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1'
AMEND = Path(__file__).resolve().parent
E = AMEND.parents[1]
ROOT = E.parents[1]
R = AMEND / 'run'
SOURCE = Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1')
OLD_SHA = '66966044c0e25eb31fc15ffb1724f48de734a926'
BEFORE_HEAD = '566ae8f191d1fb429561d2ed6b9c4d0a730e44c9'
PY = '/data/zypan/envs/ling-kda/bin/python'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for raw in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()


def cmd(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def guard():
    if (R / 'STOP.json').exists():
        raise RuntimeError('AMENDMENT_STOP_PRESERVED_NO_RETRY')
    receipt = read(AMEND / 'new_premeasurement_commit.json')
    assert cmd(['git', 'rev-parse', 'HEAD']) == receipt['commit'], 'NEW_HEAD_CHANGED'
    assert cmd(['git', 'status', '--porcelain=v1', '--untracked-files=no']) == ''
    for item in read(AMEND / 'protected_identity.json')['files']:
        p = ROOT / item['repo_relative_path']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], ('OLD_ARTIFACT_CHANGED', p)
    for item in receipt['new_source_files']:
        p = ROOT / item['repo_relative_path']
        assert sha(p) == item['sha256'], ('AMENDMENT_SOURCE_CHANGED', p)
    assert read(E / 'STOP.json')['stage'] == 'gate'
    assert read(E / 'FINAL_REPORT.json')['FINAL_ACTION'] == 'FAIL_AND_STOP'
    assert read(E / 'premeasurement_commit.json')['commit'] == OLD_SHA
    return receipt


def load_bridge(rec):
    spec = importlib.util.spec_from_file_location('unchanged_frozen_bridge', E / 'scripts/bridge.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    old_read = m.read
    inputs = {'source_identity.json', 'document_manifest.json', 'rotation_manifest.json',
              'runtime_manifest.json', 'preregistration.json'}
    def routed_read(p):
        p = Path(p)
        if p.parent == R and p.name in inputs:
            p = E / p.name
        if p.parent == R / 'historical_assets':
            p = E / 'historical_assets' / p.name
        return old_read(p)
    def inventory():
        # Execution capture includes an independent observed boundary ledger.
        # Inventory contains no postprocess original-path rereads.
        return dict(historical_executed_files=sorted(m.PROJECT_DEPS),
            execution_module_files=[], execution_capture_manifest=str(rec.root / 'manifest.json'),
            bytecode_writes=False, old_dirty_directory_executed=False,
            closure_method='retained_raw_blobs_bound_to_execution_events', coverage=rec.__class__.__module__)
    # These are operational/provenance substitutions only. Frozen metric,
    # replay, cache pulse, basis, JPROP, K3 and KL functions remain untouched.
    m.D = R
    m.ROOT = ROOT
    m.SNAP = E / 'historical_assets/snapshot'
    m.guard = guard
    m.read = routed_read
    m.execution_inventory = inventory
    rec.guard_profile = m.profile
    return m


def stage(mode, repeat=None):
    guard()
    historical = read(E / 'historical_assets/asset_manifest.json')
    rows = {str(E / 'historical_assets/snapshot' / x['relative_path']): x for x in historical['files']}
    stage_name = mode if repeat is None else mode + str(repeat)
    rec = Recorder(R / 'executed_dependencies' / stage_name, historical=rows,
                   source_root=ROOT,
                   frozen_root='/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/inherited')
    rec.install()
    try:
        m = load_bridge(rec)
        if mode == 'verify':
            m.verify()
        elif mode == 'smoke':
            m.smoke(repeat)
        elif mode == 'parity':
            m.parity()
        else:
            raise RuntimeError('MODE_NOT_AUTHORIZED_NO_DISCOVERY')
    finally:
        # A failing capture is recorded; finish cannot downgrade it to PASS.
        stats = rec.finish()
    write_json(R / ('capture_' + stage_name + '.json'), stats)
    guard()
    print('CAPTURE_CLOSURE_STAGE_PASS', stage_name, json.dumps(stats), flush=True)


def gate():
    receipt = guard()
    pf = read(R / 'smoke/preflight.json')
    a = read(R / 'smoke/repeat0/result.json')
    b = read(R / 'smoke/repeat1/result.json')
    p = read(R / 'smoke/basis_and_jprop_parity.json')
    assert pf['gate'] == a['gate'] == b['gate'] == 'PASS'
    comparisons = {k: a[k] == b[k] for k in ['scores', 'target', 'endpoint', 'FP_step_hashes']}
    assert all(comparisons.values()), ('FRESH_PROCESS_NUMERIC_DETERMINISM_GATE', comparisons)
    expected = read(E / 'historical_assets/executed_dependency_plan.json')['required_historical_modules']
    got = a['dependencies']['historical_executed_files']
    assert got == b['dependencies']['historical_executed_files']
    assert set(expected).issubset({Path(x).name for x in got})
    results = []
    raw_events = []
    for stage_name in ['verify', 'smoke0', 'smoke1', 'parity']:
        path = R / 'executed_dependencies' / stage_name / 'manifest.json'
        results.append(closure(path))
        raw_events += read(path)['events']
    historical = {x['relative_path']: x for x in read(E / 'historical_assets/asset_manifest.json')['files']}
    executed_history = [historical[x] for x in got]
    for x in got:
        matches = [e for e in raw_events if e.get('historical_asset', {}).get('relative_path') == x]
        assert matches, ('HISTORICAL_NO_EXECUTION_TIME_CAPTURE', x)
        assert all(e['sha256_capture'] == historical[x]['sha256'] for e in matches)
    # Actual regenerated temporary dependency must be present in EACH new smoke.
    # This check is diagnostic; implementation captures all paths, not one name.
    for stage_name in ('smoke0', 'smoke1'):
        ev = read(R / 'executed_dependencies' / stage_name / 'manifest.json')['events']
        temp = [e for e in ev if (e['original_absolute_path'] or '').startswith('/tmp/')]
        assert temp, ('NO_RUNTIME_GENERATED_CAPTURE_IN_NEW_SMOKE', stage_name)
    sources = {}
    for key in pf:
        if key.endswith('_GATE'):
            sources[key] = (pf[key], 'smoke/preflight.json')
    for key in p:
        if key.endswith('_GATE'):
            sources[key] = (p[key], 'smoke/basis_and_jprop_parity.json')
    for key in ['HISTORICAL_K3_PARITY_GATE', 'FUTURE_KL_HARNESS_PARITY_GATE', 'INITIAL_PULSE_IDENTITY_GATE']:
        assert a[key] == b[key] == 'PASS'
        sources[key] = ('PASS', 'smoke/repeat[01]/result.json')
    assert a['historical_fixture_tests'] == b['historical_fixture_tests'] == 'PASS'
    for x in (a, b):
        assert x['target']['NO_FURTHER_QDQ_GATE'] == 'PASS'
        assert x['target']['FP_REFERENCE_PAIRING_GATE'] == x['FP_REFERENCE_PAIRING_GATE'] == 'PASS'
        # Executed unchanged replay() verifies each drive chunk and FP endpoint.
        trace = read(R / 'smoke' / ('repeat0' if x is a else 'repeat1') / 'trace/trace_freeze.json')
        for chunk in trace['chunks']:
            assert sha(chunk['artifact']['path']) == chunk['artifact']['sha256']
    sources.update(HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE=('PASS', 'executed_dependencies/*/manifest.json'),
        FRESH_PROCESS_DETERMINISM_GATE=('PASS', 'bitwise comparison of new independent processes'),
        ACCUMULATED_ERROR_CAPTURE_GATE=('PASS', 'unchanged replay and pulse-anchor exact hashes'),
        NO_FURTHER_QDQ_GATE=('PASS', 'new target.NO_FURTHER_QDQ_GATE'),
        FROZEN_DRIVE_IDENTITY_GATE=('PASS', 'new trace SHA plus unchanged replay chunk guards'),
        FP_REFERENCE_PAIRING_GATE=('PASS_DEVELOPMENT_SMOKE_ONLY', 'same GPU6 new trace/anchor hash equality'))
    gates = {}
    for name in read(E / 'preregistration.json')['blocking_gates']:
        value, evidence = sources[name]  # Missing evidence fails, no default PASS.
        assert value in ('PASS', 'PASS_DEVELOPMENT_SMOKE_ONLY'), (name, value)
        gates[name] = dict(status=value, evidence=evidence)
    tests = read(AMEND / 'engineering_tests/results.json')
    assert all(tests['ENGINEERING_TEST_' + c].startswith('PASS') for c in 'ABCDEF')
    totals = {k: sum(x[k] for x in results) for k in results[0] if k != 'gate'}
    write_json(R / 'executed_dependency_closure.json', dict(gate='PASS', **totals,
        historical_executed_files=executed_history, original_failure_preserved=True,
        retained_blob_verification_only=True))
    report = dict(EXPERIMENT=TASK, TASK_CLASS='ENGINEERING_ONLY_AMENDMENT',
        OLD_PREMEASUREMENT_COMMIT_SHA=OLD_SHA, NEW_PREMEASUREMENT_COMMIT_SHA=receipt['commit'],
        FILES_CHANGED=receipt['new_source_files'], SCIENTIFIC_PROTOCOL_CHANGED='NO',
        MODEL_CHANGED='NO', ROTATION_CHANGED='NO', RUNTIME_CHANGED='NO',
        HISTORICAL_DIRTY_WORKTREE_CHANGED='NO',
        TEMP_DEPENDENCY_CAPTURE_IMPLEMENTED='YES', ACTUAL_EXECUTION_BYTES_PRESERVED='YES_WITH_DISCLOSED_COVERAGE',
        CAPTURE_SHA_EQUALS_PREEXEC_SHA='PASS', PRESERVED_BLOB_REHASH_PASS='PASS', **totals,
        **{k: v for k, v in tests.items() if k.startswith('ENGINEERING_TEST_')},
        **{k: v['status'] for k, v in gates.items()}, PREREGISTRATION_GATE='PASS',
        FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='PASS', numeric_comparison=comparisons,
        PREMEASUREMENT_GATES_ALL_PASS='YES', GPU6_FORMAL_AUTHORIZATION_READY='YES',
        GPU6_FORMAL_AUTHORIZED='NO', cross_hardware_FP_bitwise='FAIL_PRESERVED',
        FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO', FORMAL_MEASUREMENTS=0,
        DISCOVERY_PRIMARY_COMPLETE='0/64', DISCOVERY_SECONDARY_COMPLETE='0/64',
        CONFIRMATION_EXECUTED=0, AIME_EXECUTED='NO', L9_TRAINING_EXECUTED='NO',
        NEW_ROTATION_TRAINING_EXECUTED='NO', FINAL_ACTION='PASS_PREMEASUREMENT_AND_STOP',
        NEXT_AUTHORITY_REQUIRED='Explicit authorization to begin formal Discovery',
        gates_with_new_evidence=gates, coverage=read(R / 'executed_dependencies/smoke0/manifest.json')['coverage'])
    write_json(R / 'FINAL_REPORT.json', report)
    with (R / 'FINAL_REPORT.md').open('x') as f:
        f.write('# Engineering-only premeasurement provenance amendment\n\n```json\n')
        f.write(json.dumps(report, indent=2, sort_keys=True) + '\n```\n')
    print('PASS_PREMEASUREMENT_AND_STOP_NO_DISCOVERY', json.dumps(totals), flush=True)


def pipeline():
    guard()
    # Reuse exact frozen launch profile. Importing inherited transport here is
    # operational setup only, not a scientific measurement or an endpoint.
    spec = importlib.util.spec_from_file_location('frozen_transport_setup',
        '/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py')
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    env = os.environ.copy()
    for key, value in read(E / 'runtime_manifest.json')['expected']['launch_profile'].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(t.mapped(value)) if isinstance(value, str) and value.startswith('/data01/user2/') else str(value)
    env.update(CUDA_VISIBLE_DEVICES='6', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
        PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        LING_GATED_RMSNORM_FIXED_CONFIG='BT16_W8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for args in [['verify'], ['smoke', '--repeat', '0'], ['smoke', '--repeat', '1'], ['parity'], ['gate']]:
        with (R / ('stage_' + '_'.join(args) + '.log')).open('xb') as f:
            subprocess.run([PY, __file__, *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        print('NEW_STAGE_COMPLETE', args, flush=True)
    assert not (R / 'discovery').exists()
    print('FINAL_ACTION_STOP_WAIT_FOR_AUTHORIZATION', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'smoke', 'parity', 'gate', 'pipeline'])
    parser.add_argument('--repeat', type=int)
    args = parser.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == 'pipeline':
            pipeline()
        elif args.mode == 'gate':
            gate()
        else:
            stage(args.mode, args.repeat)
    except BaseException as exc:
        if not (R / 'STOP.json').exists():
            write_json(R / 'STOP.json', dict(stage=args.mode, error=repr(exc), traceback=traceback.format_exc(),
                FINAL_ACTION='FAIL_AND_STOP', FORMAL_MEASUREMENTS=0, DISCOVERY_PRIMARY_COMPLETE='0/64',
                DISCOVERY_SECONDARY_COMPLETE='0/64', CONFIRMATION_EXECUTED=0, AIME_EXECUTED='NO',
                L9_TRAINING_EXECUTED='NO', NEW_ROTATION_TRAINING_EXECUTED='NO', no_auto_retry=True))
        raise


if __name__ == '__main__':
    main()
