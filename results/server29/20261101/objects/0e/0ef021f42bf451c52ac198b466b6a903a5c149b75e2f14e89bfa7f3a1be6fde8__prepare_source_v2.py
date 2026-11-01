"""Read-only protected identity audit, append-only V2 preparation. No commit."""
import json
import os
from pathlib import Path
import subprocess
from capsule_v2 import sha, write

A=Path(__file__).resolve().parent;E=A.parents[1];ROOT=E.parents[1]


def read(p):return json.loads(Path(p).read_text())


def legacy_check():
    h=read(E/'historical_assets/asset_manifest.json');k=h['legacy_worktree']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=k,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=k)
    assert head==h['legacy_HEAD']
    assert status==(E/'historical_assets/git_status_porcelain_v1.txt').read_bytes()
    for x in h['files']:
        assert sha(x['path'])==x['sha256']==sha(x['snapshot_path']),x['path']
    return dict(HEAD=head,status_sha256=__import__('hashlib').sha256(status).hexdigest(),
        historical_files=len(h['files']),bytes_unchanged=True)


def main():
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='c9c8e3199e0d017b9712809b77d8f16cbd07f165'
    assert subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()==''
    legacy=legacy_check();print('LEGACY_85_FILES_HEAD_STATUS_PASS',flush=True)
    identity=read(E/'source_identity.json')
    for kind in ('inputs','protected_historical_files'):
        for i,x in enumerate(identity[kind]):
            assert sha(x['path'])==x['sha256'],x['path']
            if i%100==0:print('PROTECTED_REHASH',kind,i,len(identity[kind]),flush=True)
    protected=[]
    for p in sorted(E.rglob('*')):
        if p.is_file() and not p.is_relative_to(A) and p.name!='capsule_runner_v2.py' and '__pycache__' not in p.parts:
            protected.append(dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p)))
    write(A/'protected_identity_before.json',dict(legacy=legacy,files=protected,
        inherited_inputs=len(identity['inputs']),protected_parent_files=len(identity['protected_historical_files']),
        source_head='c9c8e3199e0d017b9712809b77d8f16cbd07f165',scientific_protocol_changed=False))
    write(A/'architecture_freeze.json',dict(AMENDMENT_SHA256=sha(A/'AMENDMENT_V2.md'),
        FORMAL_MEASUREMENTS_BEFORE_AMENDMENT=0,authority='Current explicit V2 instruction',
        allowed_modes=['verify','smoke','parity'],GPU6_FORMAL_AUTHORIZED=False,
        source_files=[dict(name=p.name,sha256=sha(p)) for p in sorted(A.glob('*.py'))],
        architecture_tests=read(A/'engineering_tests/results.json'),
        architecture_frozen_before_real_dependency_discovery=True))
    print('V2_ARCHITECTURE_SOURCE_PRESERVATION_FREEZE_PASS',len(protected),flush=True)


if __name__=='__main__':main()
