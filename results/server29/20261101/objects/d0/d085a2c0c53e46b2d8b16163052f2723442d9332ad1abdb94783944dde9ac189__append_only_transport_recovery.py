"""Append-only transport recovery; preserve tested source bytes including EOF blanks."""
from pathlib import Path
import hashlib,json,shlex,subprocess,tarfile
R=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3')
C=R/'provenance_capsule_v3';A=C/'attempt_11_static_ast_qualification'
T=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');AUTH=Path(str(T)+'-authority')
OLD='ff3fb899f297c332a21cfc7b834b00780d1d1192'
KEY='/data01/user2/.ssh/id_ed25519_ling_remote'
SSH=['ssh','-n','-i',KEY,'-o','BatchMode=yes','zypan@202.38.247.29']
SCP=['scp','-i',KEY,'-o','BatchMode=yes']
def git(*a):return subprocess.check_output(['git',*a],cwd=R,text=True).strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def main():
    assert git('rev-parse','HEAD')==OLD
    assert not git('diff','--name-only')
    names=[]
    with tarfile.open(Path(str(R)+'-attempt11-instrumentation.tar.gz')) as tar:
        for m in tar.getmembers():
            p=R/m.name
            assert m.isfile() and not m.issym() and not m.islnk() and p.resolve().is_relative_to(A)
            assert p.read_bytes()==tar.extractfile(m).read(),('TESTED_INPUT_BYTES_CHANGED',m.name)
            names.append(m.name)
    assert set(git('diff','--cached','--name-only').splitlines())==set(names)
    for x in json.loads((A/'PROTECTED_PREVIOUS_IDENTITY.json').read_text())['files']:
        assert sha(R/x['path'])==x['sha256'],('PROTECTED_SOURCE_MISMATCH',x['path'])
    assert not (A/'STOP.json').exists()
    check=subprocess.run(['git','--no-pager','diff','--cached','--check'],cwd=R,capture_output=True)
    allowed={'fixture_closure_attempt11.py','generated_code_guard_attempt11.py','stage_attempt11.py','test_generated_guard_attempt11.py'}
    rows=check.stdout.decode().splitlines()
    assert check.returncode==2 and not check.stderr and len(rows)==4
    assert {Path(line.split(':')[0]).name for line in rows}==allowed
    assert all(line.endswith('new blank line at EOF.') for line in rows),'NON_EOF_FORMAT_ERROR'
    out=A/'freeze_transport_attempt01';out.mkdir()
    with (out/'FAILED_DIFF_CHECK.log').open('xb') as f:f.write(check.stdout+check.stderr)
    old_script=Path('/data01/user2/attempt11_qualified_inputs/freeze_attempt11_inputs.py')
    for name,p in [('failed_freezer.py',old_script),('append_only_transport_recovery.py',Path(__file__))]:
        with (out/name).open('xb') as f:f.write(p.read_bytes())
    emit(out/'RECOVERY_AUDIT.json',dict(
        previous_transport_attempt='FAIL_PRESERVED',previous_commit=OLD,
        failure='git diff --check rejected four EOF blank-line formatting warnings after staging tested bytes',
        exact_correction='Commit and mirror the exact tested bytes; do not format/rewrite captured or tested inputs. Preserve failed command output and freezer source.',
        regression_tests=dict(all254_tested_archive_files_exact='PASS',all10745_protected_files_exact='PASS',
                              only_four_exact_EOF_warnings='PASS',no_other_staged_files='PASS'),
        source_format_check='FAIL_PRESERVED_NOT_A_SCIENTIFIC_OR_RUNTIME_GATE',
        executable_source_changed=False,tested_bytes_rewritten=False,SCIENTIFIC_PROTOCOL_CHANGED='NO',
        TOLERANCE_RELAXED='NO',ADVANCEMENT_RULE_CHANGED='NO',FORMAL_MEASUREMENTS=0,GPU6_FORMAL_AUTHORIZED='NO'))
    extra=[str(p.relative_to(R)) for p in sorted(out.iterdir())]
    audit_tar=Path(str(R)+'-attempt11-freeze-transport-recovery.tar.gz')
    with tarfile.open(audit_tar,'x:gz') as tar:
        for n in extra:tar.add(R/n,arcname=n,recursive=False)
    subprocess.run(SCP+[str(audit_tar),'zypan@202.38.247.29:'+str(AUTH/audit_tar.name)],check=True)
    deployment="""from pathlib import Path
import tarfile
R=Path('@T@');A=R/'provenance_capsule_v3/attempt_11_static_ast_qualification/freeze_transport_attempt01'
with tarfile.open('@TAR@') as tar:
 for m in tar.getmembers():
  p=R/m.name;assert m.isfile() and not m.issym() and not m.islnk() and p.resolve().is_relative_to(A)
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(tar.extractfile(m).read())
""".replace('@T@',str(T)).replace('@TAR@',str(AUTH/audit_tar.name))
    subprocess.run(SSH+['/data/zypan/envs/ling-kda/bin/python -B -c '+shlex.quote(deployment)],check=True)
    names+=extra
    subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b''.join(n.encode()+b'\0' for n in names),cwd=R,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(names)
    subprocess.run(['git','commit','-q','-m','ling: freeze attempt11 qualified synthetic inputs preserving exact bytes'],cwd=R,check=True)
    commit=git('rev-parse','HEAD');assert not git('status','--porcelain=v1')
    objects={OLD,commit,git('rev-parse','HEAD^{tree}')}
    for line in git('ls-tree','-r','-t','HEAD').splitlines():
        spec,path=line.split('\t',1);mode,kind,oid=spec.split()
        if kind=='tree' or path.startswith('provenance_capsule_v3/'):objects.add(oid)
    pack=Path(str(R)+'-attempt11-qualified-inputs.pack')
    with pack.open('xb') as f:subprocess.run(['git','pack-objects','--stdout','--quiet'],input=('\n'.join(sorted(objects))+'\n').encode(),cwd=R,stdout=f,check=True)
    subprocess.run(SCP+[str(pack),'zypan@202.38.247.29:'+str(AUTH/pack.name)],check=True)
    journal=Path(str(R)+'-attempt11-qualified-mirror.json')
    emit(journal,dict(old=OLD,commit=commit,files=[dict(path=n,sha256=sha(R/n)) for n in names]))
    subprocess.run(SCP+[str(journal),'zypan@202.38.247.29:'+str(AUTH/journal.name)],check=True)
    code="""from pathlib import Path
import hashlib,json,subprocess
T=Path('@T@');A=Path('@AUTH@');journal=A/'@JOURNAL@'
assert hashlib.sha256(journal.read_bytes()).hexdigest()=='@SHA@'
payload=json.loads(journal.read_text());old=payload['old'];commit=payload['commit'];names=[]
def git(*a):return subprocess.check_output(['git',*a],cwd=T,text=True).strip()
assert git('rev-parse','HEAD')==old
for row in payload['files']:
 assert hashlib.sha256((T/row['path']).read_bytes()).hexdigest()==row['sha256'];names.append(row['path'])
with (A/'@PACK@').open('rb') as f:subprocess.run(['git','--git-dir='+str(A/'repository.git'),'unpack-objects','-q'],stdin=f,check=True)
subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b''.join(n.encode()+b'\\0' for n in names),cwd=T,check=True)
assert git('write-tree','--missing-ok')==git('rev-parse',commit+'^{tree}')
subprocess.run(['git','update-ref','refs/heads/exp/ling-long-horizon-observability-bridge-v3',commit,old],cwd=T,check=True)
assert not git('status','--porcelain=v1')
print('ATTEMPT11_INPUT_COMMIT_NOT_V3_QUALIFICATION',commit,flush=True)
""".replace('@T@',str(T)).replace('@AUTH@',str(AUTH)).replace('@JOURNAL@',journal.name).replace('@SHA@',sha(journal)).replace('@PACK@',pack.name)
    subprocess.run(SSH+['/data/zypan/envs/ling-kda/bin/python -B -c '+shlex.quote(code)],check=True)
if __name__=='__main__':main()
