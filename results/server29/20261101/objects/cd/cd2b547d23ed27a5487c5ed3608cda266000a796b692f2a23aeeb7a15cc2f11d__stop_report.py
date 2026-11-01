"""Append-only report of the preserved blocking-gate failure; never resumes."""
import collections, hashlib, json, os, subprocess, tarfile, time
from pathlib import Path
D=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v1/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1')
ROOT=D.parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    archive=D/'transport/stop_evidence.tar'
    assert sha(archive)=='af319f7673fb4e4bd73fda88a33141e2ab80257205681c436f4187a842e85f31'
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if member.isdir():continue
            assert member.isfile()
            p=(D/member.name).resolve();assert p.is_relative_to(D.resolve())
            payload=t.extractfile(member).read()
            if p.exists():assert p.read_bytes()==payload
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(payload)
    assert read(D/'STOP.json')['stage']=='gate'
    a=read(D/'smoke/repeat0/result.json');b=read(D/'smoke/repeat1/result.json')
    equal={k:a[k]==b[k] for k in ['scores','target','endpoint','FP_step_hashes']}
    assert all(equal.values())
    historical=read(D/'historical_assets/asset_manifest.json')
    roots=[p for p in a['dependencies']['historical_executed_files']]
    manifest={f['relative_path']:f for f in historical['files']}
    for rel in roots:assert rel in manifest and sha(manifest[rel]['snapshot_path'])==manifest[rel]['sha256']
    ephemeral=[dict(repeat=i,path=f['path'],bytes=f['bytes'],sha256=f['sha256']) for i,x in enumerate([a,b])
               for f in x['dependencies']['execution_module_files'] if f['path'].startswith('/tmp/')]
    dump(D/'historical_assets/executed_dependency_closure_FAILURE.json',dict(
        HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',HISTORICAL_ASSET_BYTE_IDENTITY_GATE='PASS',
        HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE='FAIL',
        historical_executed_files=[manifest[rel] for rel in roots],
        historical_helper_source_snapshot_identity='PASS',
        reason='Full executed-module closure could not be durably reverified: runtime-generated temporary Python module was removed at process exit. Recorded hash alone is insufficient; raw temporary module was not preserved.',
        runtime_temporary_dependencies=ephemeral,raw_temporary_source_retained=False,
        raw_execution_inventories=['smoke/repeat0/result.json','smoke/repeat1/result.json'],
        no_gate_waiver=True,no_automatic_retry=True,scientific_measurements=0))
    # Verify every frozen scientific input and protected history from its source.
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=historical['legacy_worktree'],text=True).strip()==historical['legacy_HEAD']
    assert subprocess.check_output(['git','status','--porcelain=v1'],cwd=historical['legacy_worktree'])==(D/'historical_assets/git_status_porcelain_v1.txt').read_bytes()
    for f in historical['files']:assert sha(f['path'])==f['sha256']
    source=read(D/'source_identity.json')
    for f in source['protected_historical_files']+source['inputs']:
        assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],f['path']
    for f in read(D/'premeasurement_sha256.json')['files']:assert sha(f['path'])==f['sha256'],f['path']
    dump(D/'provenance_poststop_integrity.json',dict(gate='PASS',legacy_HEAD=historical['legacy_HEAD'],
         legacy_status_byte_identical=True,legacy_asset_files=len(historical['files']),
         historical_file_classes=dict(collections.Counter(f['classification'] for f in historical['files'])),
         protected_parent_files=len(source['protected_historical_files']),inherited_inputs=len(source['inputs']),
         premeasurement_files_unchanged=True,old_worktree_files_written=0,old_commits_created=0,
         stash=False,reset=False,clean=False,delete=False,old_execution_directory_used=False))
    preflight=read(D/'smoke/preflight.json');parity=read(D/'smoke/basis_and_jprop_parity.json')
    report=dict(EXPERIMENT='LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1',
        PREMEASUREMENT_COMMIT_SHA=read(D/'premeasurement_commit.json')['commit'],
        PREREGISTRATION_GATE='PASS',HISTORICAL_DIRTY_WORKTREE_DISCLOSED='YES',
        HISTORICAL_ASSET_BYTE_IDENTITY_GATE='PASS',HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE='FAIL',
        BASIS_SEMANTICS_GATE=parity['BASIS_SEMANTICS_GATE'],HISTORICAL_K3_PARITY_GATE='PASS',
        HISTORICAL_JPROP_PARITY_GATE=parity['HISTORICAL_JPROP_PARITY_GATE'],FUTURE_KL_HARNESS_PARITY_GATE='PASS',
        FRESH_PROCESS_NUMERIC_DETERMINISM_GATE='PASS',numeric_comparison=equal,
        RUNTIME_IDENTITY_GATE=preflight['RUNTIME_IDENTITY_GATE'],HARDWARE_ASSIGNMENT_GATE=preflight['HARDWARE_ASSIGNMENT_GATE'],
        FP_REFERENCE_PAIRING_GATE='PASS_DEVELOPMENT_SMOKE_ONLY',GPU6_FORMAL_AUTHORIZED='NO',
        DISCOVERY_PRIMARY_COMPLETE='0/64',DISCOVERY_SECONDARY_COMPLETE='0/64',FORMAL_MEASUREMENTS=0,
        OBSERVABILITY_DISCOVERY_LABEL='NOT_EVALUATED_DUE_TO_BLOCKING_ENGINEERING_GATE',
        CONFIRMATION_EXECUTED=0,NEW_ROTATION_TRAINING_EXECUTED='NO',L9_TRAINING_EXECUTED='NO',AIME_EXECUTED='NO',
        FP_REFERENCES_MIXED_ACROSS_HARDWARE='NO',cross_hardware_FP_bitwise='FAIL_PRESERVED',
        ORIGINAL_PREREGISTERED_FINAL_ACTION='STOP_AFTER_DISCOVERY',FINAL_ACTION='FAIL_AND_STOP',
        failure='The gate tried to rehash /tmp/tmp53b1af21/_remote_module_non_scriptable.py after the originating fresh process exited; file no longer exists.',
        scientific_interpretation='Engineering/dependency retention failure, not evidence against observability and not a numeric or historical parity failure.',
        requested_next_authority='Engineering-only amendment to preserve actual executed temporary dependency bytes at creation/execution time, bind them to provenance, create a new premeasurement commit and rerun all gates. No change to metrics, tolerances, docs, rotations, runtime, or model.')
    dump(D/'FINAL_REPORT.json',report)
    with (D/'FINAL_REPORT.md').open('x') as f:
        f.write('# LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1 — blocking engineering STOP\n\n```json\n')
        f.write(json.dumps(report,indent=2,sort_keys=True)+'\n```\n\n')
        f.write('The legacy dirty worktree was disclosed, not represented as a clean source tree or as its Git HEAD contents. Its bytes, Git status and HEAD remain unchanged. All executed historical helper paths, SHA256 and Git classifications are listed in the failure closure artifact.\n\n')
        f.write('No Discovery outcome was computed. No failed gate was bypassed. Both original STOP records and all raw smoke/parity evidence are preserved. GPU6 has not been authorized for formal measurement. No training, AIME, free generation, or scientific protocol change occurred.\n')
    with (D/'sha256_manifest.txt').open('x') as f:
        for p in sorted(D.rglob('*')):
            if p.is_file() and p.name!='sha256_manifest.txt' and '__pycache__' not in p.parts:f.write(sha(p)+'  '+str(p.relative_to(D))+'\n')
    print('STOP_REPORT_COMPLETE_PARENT_IDENTITY_PASS',json.dumps({k:report[k] for k in ['HISTORICAL_ASSET_BYTE_IDENTITY_GATE','HISTORICAL_EXECUTED_DEPENDENCY_CLOSURE_GATE','FINAL_ACTION']}),flush=True)
if __name__=='__main__':main()
