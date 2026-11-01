"""Scoped engineering-only commit and exact clean sparse checkout transport."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from execution_capture import write_json

A = Path(__file__).resolve().parent
E = A.parents[1]
ROOT = E.parents[1]
TARGET = Path('/data/zypan/worktrees/ling-observability-provenance-amendment-v1')
REMOTE = 'zypan@202.38.247.29'
SSH = ['ssh', '-n', '-i', '/data01/user2/.ssh/id_ed25519_ling_remote', '-o', 'BatchMode=yes', REMOTE]
SCP = ['scp', '-i', '/data01/user2/.ssh/id_ed25519_ling_remote', '-o', 'BatchMode=yes']
PY = '/data/zypan/envs/ling-kda/bin/python'


def run(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for raw in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(raw)
    return h.hexdigest()


def freeze():
    assert run(['git', 'rev-parse', 'HEAD']) == '566ae8f191d1fb429561d2ed6b9c4d0a730e44c9'
    assert run(['git', 'diff', '--name-only']) == ''
    scope = str(A.relative_to(ROOT))
    expected = ['execution_capture.py', 'test_execution_capture.py', 'amendment_runner.py',
                'prepare_amendment.py', 'freeze_transport.py', 'AMENDMENT.md', 'amendment.json',
                'manifest_schema.json', 'protected_identity.json', 'engineering_tests/results.json']
    tests = json.loads((A / 'engineering_tests/results.json').read_text())
    assert all(tests['ENGINEERING_TEST_' + c].startswith('PASS') for c in 'ABCDEF')
    for row in json.loads((A / 'protected_identity.json').read_text())['files']:
        assert sha(ROOT / row['repo_relative_path']) == row['sha256']
    write_json(A / 'diff_scope_audit.json', dict(classification='ENGINEERING_PROVENANCE_ONLY',
        old_tracked_diff=[], scientific_functions_loaded_from_unchanged_frozen_bridge=True,
        changed_symbol_scope=['operational guard/paths', 'execution_inventory provenance', 'gate-only authority boundary'],
        metrics='UNCHANGED', thresholds='UNCHANGED', tolerances='UNCHANGED', model='UNCHANGED',
        rotations='UNCHANGED', quantizer='UNCHANGED', recurrent_update='UNCHANGED', sample_selection='UNCHANGED',
        horizons='UNCHANGED', FP_reference_selection='UNCHANGED', KL='UNCHANGED', J_PROP='UNCHANGED',
        K3='UNCHANGED', scientific_scorer='UNCHANGED', runtime_packages_kernels_flags='UNCHANGED',
        legacy_worktree_written=False, expected_added_files=expected))
    expected.append('diff_scope_audit.json')
    subprocess.run(['git', 'add', '-f', '--', *[scope + '/' + x for x in expected]], cwd=ROOT, check=True)
    staged = run(['git', 'diff', '--cached', '--name-only']).splitlines()
    assert set(staged) == {scope + '/' + x for x in expected}
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--stat'], cwd=ROOT, check=True)
    transport = A / 'transport'
    transport.mkdir()
    with (transport / 'engineering_only_git_diff.patch').open('xb') as f:
        subprocess.run(['git', 'diff', '--cached'], cwd=ROOT, stdout=f, check=True)
    subprocess.run(['git', 'commit', '-m', 'ling: preserve executed temporary dependency bytes for provenance closure'], cwd=ROOT, check=True)
    commit = run(['git', 'rev-parse', 'HEAD'])
    rows = [dict(repo_relative_path=scope + '/' + x, sha256=sha(A / x)) for x in expected]
    write_json(A / 'new_premeasurement_commit.json', dict(commit=commit,
        OLD_PREMEASUREMENT_COMMIT_SHA='66966044c0e25eb31fc15ffb1724f48de734a926',
        new_source_files=rows, protected_identity_sha256=sha(A / 'protected_identity.json'),
        scientific_measurements=0, task_class='ENGINEERING_ONLY_AMENDMENT'))
    print('NEW_PREMEASUREMENT_COMMIT_SHA', commit, flush=True)


def export():
    receipt = json.loads((A / 'new_premeasurement_commit.json').read_text())
    commit = receipt['commit']
    assert run(['git', 'rev-parse', 'HEAD']) == commit
    assert run(['git', 'status', '--porcelain=v1', '--untracked-files=no']) == ''
    scope = str(E.relative_to(ROOT))
    transport = A / 'transport'
    entries = run(['git', 'ls-tree', '-r', '-t', 'HEAD']).splitlines()
    objects = {commit, run(['git', 'rev-parse', 'HEAD^{tree}'])}
    for entry in entries:
        info, name = entry.split('\t', 1)
        mode, kind, oid = info.split()
        if kind == 'tree' or name.startswith(scope + '/'):
            objects.add(oid)
    with (transport / 'authority.pack').open('xb') as f:
        subprocess.run(['git', 'pack-objects', '--stdout'], cwd=ROOT,
            input=('\n'.join(sorted(objects)) + '\n').encode(), stdout=f, check=True)
    tracked = set(run(['git', 'ls-files', '--', scope]).splitlines())
    untracked = [x for x in json.loads((A / 'protected_identity.json').read_text())['files']
                 if x['repo_relative_path'] not in tracked]
    with tarfile.open(transport / 'protected_untracked.tar', 'x') as archive:
        for item in untracked:
            p = ROOT / item['repo_relative_path']
            assert sha(p) == item['sha256']
            archive.add(p, arcname=item['repo_relative_path'], recursive=False)
    subprocess.run(SSH + ['test ! -e ' + str(TARGET) + ' && mkdir -p ' + str(TARGET)], check=True)
    for path in [transport / 'authority.pack', transport / 'protected_untracked.tar', A / 'new_premeasurement_commit.json']:
        subprocess.run(SCP + [str(path), REMOTE + ':' + str(TARGET / path.name)], check=True)
    script = """import hashlib,json,pathlib,subprocess,tarfile
r=json.load(open('new_premeasurement_commit.json'))
subprocess.run(['git','init','--bare','authority.git'],check=True)
with open('authority.pack','rb') as f: subprocess.run(['git','--git-dir=authority.git','unpack-objects'],stdin=f,check=True)
pathlib.Path('authority.git/shallow').write_text(r['commit']+'\\n')
subprocess.run(['git','--git-dir=authority.git','update-ref','refs/heads/exp/ling-long-horizon-observability-bridge-v1',r['commit']],check=True)
subprocess.run(['git','--git-dir=authority.git','worktree','add','--no-checkout','checkout','exp/ling-long-horizon-observability-bridge-v1'],check=True)
subprocess.run(['git','-C','checkout','config','core.sparseCheckout','true'],check=True)
p=pathlib.Path(subprocess.check_output(['git','-C','checkout','rev-parse','--git-path','info/sparse-checkout'],text=True).strip())
p=p if p.is_absolute() else pathlib.Path('checkout')/p
p.parent.mkdir(parents=True,exist_ok=True);p.write_text('/SCOPE/\\n')
subprocess.run(['git','-C','checkout','read-tree','-mu','HEAD'],check=True)
with tarfile.open('protected_untracked.tar') as archive:
 for member in archive.getmembers():
  assert member.isfile();p=(pathlib.Path('checkout')/member.name).resolve()
  assert p.is_relative_to(pathlib.Path('checkout').resolve())
  raw=archive.extractfile(member).read()
  if p.exists(): assert p.read_bytes()==raw
  else:
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(raw)
p=pathlib.Path('checkout/SCOPE/provenance/executed_dependency_amendment_v1/new_premeasurement_commit.json')
with p.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\\n')
assert subprocess.check_output(['git','-C','checkout','status','--porcelain=v1','--untracked-files=no'],text=True).strip()==''
print('REMOTE_CLEAN_SPARSE_WORKTREE_COMMIT',r['commit'])
""".replace('SCOPE', scope)
    subprocess.run(SSH + ['cd ' + str(TARGET) + ' && ' + PY + ' -B -c ' + shlex.quote(script)], check=True)
    write_json(transport / 'receipt.json', dict(new_premeasurement_commit=commit,
        source=str(ROOT), clean_sparse_execution_directory=str(TARGET / 'checkout'),
        pack_sha256=sha(transport / 'authority.pack'), protected_untracked_sha256=sha(transport / 'protected_untracked.tar'),
        protected_untracked_count=len(untracked), old_source_and_remote_worktrees_written=False))


if __name__ == '__main__':
    import sys
    {'freeze': freeze, 'export': export}[sys.argv[1]]()
