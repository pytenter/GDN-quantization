"""Publish immutable new source, or start its one authorized offline chain."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/zypan/analyses/qwen_gdn_pace48_aime26_seed1_60_v1_scale_repair_20261025'
ARCHIVE = '/data/zypan/qwen_gdn_pace48_archive/QWEN_GDN_PACE48_AIME26_SEED1_60_V1_SCALE_REPAIR_20261025'
TMUX = 'qwen-pace48-scale-repair-gpu0-20261025'
TRANSPORT_SHA = 'db0ea71612d633b0fc33feef1d95c42e273ab2bd9ffc971b8d7e4ddc97f74f40'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transport():
    path = ROOT / 'operations/remote.py'
    assert sha(path) == TRANSPORT_SHA
    spec = importlib.util.spec_from_file_location('pinned_scale_publish_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_final_cpu():
    # Recheck the final two-fresh CPU/file identities before publishing or even
    # querying the selected device. This does not execute any test or GPU code.
    path = ROOT / 'operations/freeze_static_once.py'
    sys.path.insert(0, str(ROOT / 'src'))
    spec = importlib.util.spec_from_file_location('scale_publish_frozen_cpu_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_final_cpu_admission(ROOT)


def publish(module, source_sha, permit_sha):
    source = json.loads((ROOT / 'source_manifest.json').read_text())
    names = [row['path'] for row in source['files']] + ['source_manifest.json', 'GPU_PERMIT.json']
    assert len(names) == len(set(names)) and source['count'] == len(source['files'])
    rows = []
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz', dereference=True) as archive:
        for name in names:
            p = Path(name)
            assert not p.is_absolute() and '..' not in p.parts and str(p) == name and name != '.'
            path = ROOT / name
            current = path
            while current != ROOT:
                assert not current.is_symlink()
                current = current.parent
            assert path.is_file()
            rows.append(dict(path=name, bytes=path.stat().st_size, sha256=sha(path)))
            archive.add(path, arcname=name, recursive=False)
    program = r'''
import base64,hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
r=Path(REMOTE)
assert r.is_dir() and not r.is_symlink()
assert not any((r/n).exists() for n in ['GPU_PERMIT.json','SESSION_STARTED.json','OFFLINE_PARENT_STARTED.json','SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json','SCALE_REPAIR_FROZEN_PUBLICATION_CLAIM.json','DELIVERY_MANIFEST.json'])
expected={x['path']:x for x in ROWS}
assert len(expected)==len(ROWS)
def safe(name):
 p=Path(name);assert not p.is_absolute() and '..'not in p.parts and str(p)==name and name!='.' and not any(x in name for x in ['\x00','\r','\n'])
 q=r/p;current=q
 while current!=r:
  assert not current.is_symlink(),'Existing ancestor/leaf symlink is forbidden'
  current=current.parent
 assert not q.exists() or q.is_file(),'Existing nonregular publication target forbidden'
 return q
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(DATA)),mode='r:gz') as archive:
 members=archive.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile() and m.name in expected and m.size==expected[m.name]['bytes']
  safe(m.name)
  with archive.extractfile(m) as f:assert hashlib.sha256(f.read()).hexdigest()==expected[m.name]['sha256']
 claim=r/'SCALE_REPAIR_FROZEN_PUBLICATION_CLAIM.json'
 with claim.open('x') as f:
  json.dump(dict(status='CLAIMED_ONCE_FROZEN_PUBLICATION',source_sha256=SOURCE_SHA,permit_sha256=PERMIT_SHA,unix=time.time(),no_automatic_retry=True),f,sort_keys=True);f.flush();os.fsync(f.fileno())
 # Replace each staged file with a newly owned inode. Truncating an existing
 # hardlink could otherwise modify a historical asset outside this new root.
 for m in members:
  q=safe(m.name);q.parent.mkdir(parents=True,exist_ok=True)
  tmp=q.with_name(q.name+'.publication.'+str(os.getpid()))
  with archive.extractfile(m) as src,tmp.open('xb') as dst:
   for block in iter(lambda:src.read(4194304),b''):dst.write(block)
   dst.flush();os.fsync(dst.fileno())
  os.replace(tmp,q)
for row in ROWS:
 p=r/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
sys.path.insert(0,str(r/'src'))
import pace_control as C
assert C.sha(r/C.SOURCE)==SOURCE_SHA and C.sha(r/'GPU_PERMIT.json')==PERMIT_SHA
C.authorize(r)
sys.path.insert(0,str(r/'operations'))
import freeze_static_once as F
F.validate_final_cpu_admission(r)
print(json.dumps(dict(status='PASS_FROZEN_NEW_SOURCE_PUBLICATION_NO_GPU_OR_CHILD',count=len(ROWS),source_sha256=SOURCE_SHA,permit_sha256=PERMIT_SHA,GPU_calls=0,old_assets_modified=False)))
'''
    code = '\n'.join(name + '=' + repr(value) for name, value in
                     [('REMOTE', REMOTE), ('DATA', base64.b64encode(payload.getvalue()).decode()),
                      ('ROWS', rows), ('SOURCE_SHA', source_sha), ('PERMIT_SHA', permit_sha)])
    return module.run(code+'\n'+program, role='55',
                      python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=180)


def launch(module, source_sha, permit_sha):
    # The archive host is CPU-only. Do not require a model path or query its GPUs.
    archive = module.run('''
import json,os
from pathlib import Path
p=Path('''+repr(ARCHIVE)+''')
assert not p.exists() and p.parent.is_dir()
s=os.statvfs(p.parent);free=s.f_bavail*s.f_frsize
assert free>=8*1024**3
print(json.dumps(dict(status='PASS_CPU_ARCHIVE29_TARGET_ABSENT',free_bytes=free,GPU_calls=0)))
''', role='29', python='/usr/bin/python3', timeout=30)
    program = r'''
import json,os,subprocess,sys,time
from pathlib import Path
r=Path(REMOTE)
sys.path.insert(0,str(r/'src'))
import pace_control as C
assert C.sha(r/C.SOURCE)==SOURCE_SHA and C.sha(r/'GPU_PERMIT.json')==PERMIT_SHA
C.authorize(r);C.model_guard(r);C.storage_guard(r)
sys.path.insert(0,str(r/'operations'))
import freeze_static_once as F
F.validate_final_cpu_admission(r)
obs=C.device()
assert not any((r/n).exists() for n in ['SESSION_STARTED.json','OFFLINE_PARENT_STARTED.json','SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json','SESSION_RESULT.json','DELIVERY_MANIFEST.json'])
assert subprocess.run(['tmux','has-session','-t',TMUX],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
claim=dict(status='CLAIMED_ONCE_NEW_AUTHORIZED_PACE_REPAIR_CHAIN',source_manifest_sha256=SOURCE_SHA,permit_sha256=PERMIT_SHA,tmux=TMUX,NVML_before=obs,unix=time.time(),no_automatic_launch_retry=True)
C.atomic_new(r/'SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json',claim)
argv=['tmux','new-session','-d','-s',TMUX,'--','/bin/sh',str(r/'scripts/run_offline.sh')]
completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
receipt=dict(status='UNIQUE_LAUNCH_COMMAND_OBSERVED',argv=argv,returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,source_manifest_sha256=SOURCE_SHA,permit_sha256=PERMIT_SHA,unix=time.time(),claim_sha256=C.sha(r/'SCALE_REPAIR_UNIQUE_LAUNCH_CLAIM.json'))
if completed.returncode==0:
 pane=subprocess.run(['tmux','list-panes','-t',TMUX,'-F','#{pane_pid}'],capture_output=True,text=True)
 receipt['initial_pane_query']=dict(returncode=pane.returncode,stdout=pane.stdout,stderr=pane.stderr)
 receipt['initial_pane_identities']=[]
 for value in pane.stdout.splitlines():
  try:receipt['initial_pane_identities'].append(C.process_identity(int(value)))
  except FileNotFoundError:receipt['initial_pane_identities'].append(dict(pid=int(value),state='ALREADY_EXITED_REQUIRES_REAL_PARENT_RECEIPTS'))
  except (OSError,ValueError) as exc:receipt['initial_pane_identities'].append(dict(observed_pane=value,state='IDENTITY_UNRESOLVED_REQUIRES_REAL_PARENT_RECEIPTS',error=repr(exc)))
C.atomic_new(r/'UNIQUE_OFFLINE_LAUNCH_RECEIPT.json',receipt)
print(json.dumps(receipt))
assert completed.returncode==0,'Unique launch failed; retain claim, no automatic retry'
'''
    code = '\n'.join(name+'='+repr(value) for name, value in
                     [('REMOTE', REMOTE), ('TMUX', TMUX), ('SOURCE_SHA', source_sha), ('PERMIT_SHA', permit_sha)])
    receipt = module.run(code+'\n'+program, role='55',
                         python='/data/ydai/miniconda3/envs/bitdecode/bin/python', timeout=60)
    return dict(archive29_admission=archive, launch=receipt)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--publish-frozen-once', action='store_true')
    group.add_argument('--launch-once', action='store_true')
    args = parser.parse_args()
    validate_final_cpu()
    name = 'FROZEN_SCALE_REPAIR_PUBLICATION_RECEIPT_LOCAL.json' if args.publish_frozen_once else 'UNIQUE_OFFLINE_LAUNCH_RECEIPT_LOCAL.json'
    assert not (ROOT/'operations'/name).exists(), 'Existing local once-only receipt; do not repeat'
    source_sha, permit_sha = sha(ROOT/'source_manifest.json'), sha(ROOT/'GPU_PERMIT.json')
    module = transport()
    result = publish(module, source_sha, permit_sha) if args.publish_frozen_once else launch(module, source_sha, permit_sha)
    with (ROOT/'operations'/name).open('x') as stream:
        json.dump(result, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
