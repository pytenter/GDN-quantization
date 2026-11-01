"""Commit unqualified engineering inputs only, mirror exact objects to sparse target."""
from pathlib import Path
import shlex, subprocess, tarfile
R=Path('/data01/user2/worktrees/ling-long-horizon-observability-bridge-v3');C=R/'provenance_capsule_v3'
T=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3');AUTH=Path(str(T)+'-authority')
KEY='/data01/user2/.ssh/id_ed25519_ling_remote'
SSH=['ssh','-n','-i',KEY,'-o','BatchMode=yes','zypan@202.38.247.29'];SCP=['scp','-i',KEY,'-o','BatchMode=yes']
OLD='ff3fb899f297c332a21cfc7b834b00780d1d1192'
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
def main():
    assert git('rev-parse','HEAD')==OLD and not git('status','--porcelain=v1')
    archive=Path(str(R)+'-attempt11-instrumentation.tar.gz')
    subprocess.run(SCP+['zypan@202.38.247.29:'+str(AUTH/'attempt11-instrumentation.tar.gz'),str(archive)],check=True)
    names=[]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            p=R/member.name
            assert p.resolve().is_relative_to(C/'attempt_11_static_ast_qualification') and member.isfile() and not member.issym() and not member.islnk()
            assert not p.exists(),'NO_OVERWRITE'
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(tar.extractfile(member).read())
            names.append(member.name)
    assert not git('diff','--name-only')
    subprocess.run(['git','add','-f','--',*names],cwd=R,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(names)
    subprocess.run(['git','--no-pager','diff','--cached','--check'],cwd=R,check=True)
    subprocess.run(['git','commit','-q','-m','ling: freeze attempt11 static AST audit association engineering correction'],cwd=R,check=True)
    commit=git('rev-parse','HEAD');assert not git('status','--porcelain=v1')
    objects={OLD,commit,git('rev-parse','HEAD^{tree}')}
    for line in git('ls-tree','-r','-t','HEAD').splitlines():
        spec,path=line.split('\t',1);mode,kind,oid=spec.split()
        if kind=='tree' or path.startswith('provenance_capsule_v3/'):objects.add(oid)
    pack=Path(str(R)+'-attempt11-instrumentation.pack')
    with pack.open('xb') as f:subprocess.run(['git','pack-objects','--stdout'],input=('\n'.join(sorted(objects))+'\n').encode(),cwd=R,stdout=f,check=True)
    subprocess.run(SCP+[str(pack),'zypan@202.38.247.29:'+str(AUTH/'attempt11-instrumentation.pack')],check=True)
    code="""from pathlib import Path
import subprocess
T=Path('@TARGET@');A=Path('@AUTH@');old='@OLD@';commit='@COMMIT@';names=@NAMES@
def git(*a):return subprocess.check_output(['git',*a],cwd=T,text=True).strip()
assert git('rev-parse','HEAD')==old
with (A/'attempt11-instrumentation.pack').open('rb') as f:subprocess.run(['git','--git-dir='+str(A/'repository.git'),'unpack-objects'],stdin=f,check=True)
subprocess.run(['git','add','-f','--',*names],cwd=T,check=True)
assert git('write-tree','--missing-ok')==git('rev-parse',commit+'^{tree}')
subprocess.run(['git','update-ref','refs/heads/exp/ling-long-horizon-observability-bridge-v3',commit,old],cwd=T,check=True)
assert not git('status','--porcelain=v1')
print('ATTEMPT11_INPUT_COMMIT_NOT_QUALIFICATION',commit,flush=True)
""".replace('@TARGET@',str(T)).replace('@AUTH@',str(AUTH)).replace('@OLD@',OLD).replace('@COMMIT@',commit).replace('@NAMES@',repr(names))
    subprocess.run(SSH+['/data/zypan/envs/ling-kda/bin/python -B -c '+shlex.quote(code)],check=True)
    print('SOURCE_TARGET_ATTEMPT11_INPUTS_FROZEN',commit,flush=True)
if __name__=='__main__':main()


