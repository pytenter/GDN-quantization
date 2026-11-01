"""Append-only two-end Git freeze; sparse target receives exact Git objects."""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess, tarfile
S=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3')
T=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');A=Path(str(T)+'-authority')
KEY='/data01/user2/.ssh/id_ed25519_ling_remote'
SCP=['scp','-i',KEY,'-o','BatchMode=yes'];SSH=['ssh','-n','-i',KEY,'-o','BatchMode=yes','zypan@202.38.247.29']
def git(*a):return subprocess.check_output(['git',*a],cwd=S,text=True).strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('old');parser.add_argument('message');x=parser.parse_args()
    assert git('rev-parse','HEAD')==x.old and not git('status','--porcelain=v1')
    archive=Path(str(S)+'-'+x.label+'.tar.gz')
    assert not archive.exists()
    subprocess.run(SCP+['zypan@202.38.247.29:'+str(A/(x.label+'.tar.gz')),str(archive)],check=True)
    names=[]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p=S/member.name
            assert p.resolve().is_relative_to(S) and member.isfile() and not member.issym() and not member.islnk()
            assert member.name.startswith(('provenance_capsule_v3/scientific_reproducibility_v1/','experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/'))
            data=tar.extractfile(member).read()
            if p.exists():assert p.read_bytes()==data,('NO_OVERWRITE',member.name)
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(data)
            names.append(member.name)
    assert not git('diff','--name-only')
    subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=('\0'.join(names)+'\0').encode(),cwd=S,check=True)
    assert set(git('diff','--cached','--name-only').splitlines()).issubset(names)
    assert git('diff','--cached','--name-only')
    subprocess.run(['git','diff','--cached','--check'],cwd=S,check=True)
    subprocess.run(['git','commit','-q','-m',x.message],cwd=S,check=True)
    commit=git('rev-parse','HEAD');assert not git('status','--porcelain=v1')
    objects={x.old,commit,git('rev-parse','HEAD^{tree}')}
    for line in git('ls-tree','-r','-t','HEAD').splitlines():
        spec,path=line.split('\t',1);_,kind,oid=spec.split()
        if kind=='tree' or path.startswith(('provenance_capsule_v3/','experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1/')):objects.add(oid)
    pack=Path(str(S)+'-'+x.label+'.pack')
    with pack.open('xb') as f:subprocess.run(['git','pack-objects','--stdout'],input=('\n'.join(sorted(objects))+'\n').encode(),cwd=S,stdout=f,check=True)
    journal=Path(str(S)+'-'+x.label+'.journal.json')
    with journal.open('x') as f:json.dump(dict(old=x.old,commit=commit,names=names,files=[dict(path=n,sha256=sha(S/n)) for n in names]),f)
    for p in [pack,journal]:subprocess.run(SCP+[str(p),'zypan@202.38.247.29:'+str(A/p.name)],check=True)
    code="""from pathlib import Path
import hashlib,json,subprocess
T=Path('@T@');A=Path('@A@');j=json.loads((A/'@J@').read_text());old=j['old'];commit=j['commit']
def git(*a):return subprocess.check_output(['git',*a],cwd=T,text=True).strip()
assert git('rev-parse','HEAD')==old
for row in j['files']:assert hashlib.sha256((T/row['path']).read_bytes()).hexdigest()==row['sha256']
with (A/'@P@').open('rb') as f:subprocess.run(['git','--git-dir='+str(A/'repository.git'),'unpack-objects'],stdin=f,check=True)
subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=('\\0'.join(j['names'])+'\\0').encode(),cwd=T,check=True)
assert git('write-tree','--missing-ok')==git('rev-parse',commit+'^{tree}')
subprocess.run(['git','update-ref','refs/heads/exp/ling-long-horizon-observability-bridge-v3',commit,old],cwd=T,check=True)
assert not git('status','--porcelain=v1')
print('TWO_END_BYTE_IDENTICAL_CLEAN_COMMIT',commit,flush=True)
""".replace('@T@',str(T)).replace('@A@',str(A)).replace('@J@',journal.name).replace('@P@',pack.name)
    subprocess.run(SSH+['/data/zypan/envs/ling-kda/bin/python -B -c '+shlex.quote(code)],check=True)
    print('FREEZE_COMPLETE',x.label,commit,flush=True)
if __name__=='__main__':main()
