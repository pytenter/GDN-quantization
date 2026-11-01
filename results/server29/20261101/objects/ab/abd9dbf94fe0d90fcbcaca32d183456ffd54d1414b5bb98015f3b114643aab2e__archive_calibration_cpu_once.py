"""CPU-only once-only durable29 calibration evidence, before long formal run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only archive needs empty CUDA visibility')
    sys.path.insert(0,str(root/'src'));import execution_control as C
    C.require(C.read(root/'calibration/RAW_CLONE_VERIFICATION.json')['status']=='PASS_ALL_ACTUAL_NONAIME_RAW_SHA_BYTES',
              'Actual complete raw clone verification required')
    C.require(C.read(root/'CALIBRATION_PARAMETER_SELECTION.json')['status']=='PASS_CAL_ONLY_FINITE_GRID_SELECTION',
              'Actual CAL-only parameter selection required')
    final='/data/zypan/qwen_gdn_wscale_rowbal_archive/PRE_FORMAL_CALIBRATION_'+C.TASK
    C.atomic_new(root/'calibration/ARCHIVE29_CLAIM.json',dict(identity=C.process_identity(os.getpid()),
        archive=final,scope='Shared nonAIME calibration only, not AIME qualification/formal completion',GPU_calls=0))
    expected={x['path']:x for x in C.read(root/'CALIBRATION_SOURCE_FREEZE.json')['files']}
    expected.update({x['path']:x for x in C.read(root/'calibration/RAW_CLONE_VERIFICATION.json')['files']})
    paths=set(expected)
    paths.update(['CALIBRATION_SOURCE_FREEZE.json','CALIBRATION_PERMIT.json','TRANSPORTED_QUERY_OMEGA.json',
        'CALIBRATION_PARAMETER_SELECTION.json','calibration/FP_COLLECTION_MANIFEST.json',
        'calibration/FP_COLLECTION_EXIT_RELEASE.json','calibration/INDEPENDENT_FP_EXIT_RAW_AUDIT.json',
        'calibration/PARENT_FAILURE.json','calibration/PARENT_STARTED.json','calibration/PARENT_DISPATCH.json',
        'calibration/RAW_CLONE_VERIFICATION.json','calibration/execution/STARTED.json',
        'calibration/execution/DRIVER_IDENTITY.json','calibration/execution/PREFLIGHT_LIBRARY.json',
        'calibration/execution/POSTFLIGHT_LIBRARY.json',
        'operations/CALIBRATION_OMEGA_INDEPENDENT_CPU_RECOMPUTE.json',
        'operations/archive_calibration_cpu_once.py'])
    rows=[]
    for name in sorted(paths):
        p=root/name;C.require(p.is_file() and not p.is_symlink(),'Unsafe calibration archive file')
        row=dict(path=name,bytes=p.stat().st_size,sha256=C.sha(p))
        if name in expected:
            C.require(row['bytes']==expected[name]['bytes'] and row['sha256']==expected[name]['sha256'],
                      'Previously frozen/verified archive input changed:'+name)
        rows.append(row)
    manifest=dict(status='SHARED_NONAIME_CALIBRATION_EVIDENCE_ONLY',task_id=C.TASK,files=rows,
        count=len(rows),bytes=sum(x['bytes'] for x in rows),formal_generations=0,
        original_CPU_parent_failure_preserved=True,source_freeze_sha256=C.sha(root/'CALIBRATION_SOURCE_FREEZE.json'))
    C.atomic_new(root/'calibration/ARCHIVE29_MANIFEST.json',manifest)
    manifest_sha=C.sha(root/'calibration/ARCHIVE29_MANIFEST.json');stage=final+'.staging_'+manifest_sha[:16]
    socket='/tmp/qwen_replication_archive29.sock';host='zypan@202.38.247.29'
    def remote(code,*args):
        p=subprocess.run(['ssh','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=15',host,
            shlex.join(['/usr/bin/python3','-u','-B','-c',code,*args])],capture_output=True,text=True)
        C.require(p.returncode==0,'CPU calibration archive operation failed:'+p.stderr[-4000:])
        return json.loads(p.stdout)
    remote('''import sys,json,os
from pathlib import Path
final=Path(sys.argv[1]);stage=Path(sys.argv[2]);assert not final.exists() and not stage.exists()
s=os.statvfs('/data/zypan');assert s.f_bavail*s.f_frsize>=120*1024**3
stage.mkdir(parents=True);print(json.dumps(dict(status='RESERVED_EMPTY_NEW_STAGE')))
''',final,stage)
    names=sorted(paths|{'calibration/ARCHIVE29_MANIFEST.json'})
    files_path=root/'calibration/ARCHIVE29_FILES.txt'
    with files_path.open('x') as stream:stream.write(''.join(n+'\n' for n in names))
    p=subprocess.run(['rsync','-a','--files-from='+str(files_path),'-e',
        shlex.join(['ssh','-S',socket,'-o','BatchMode=yes']),str(root)+'/',host+':'+stage+'/'],
        capture_output=True,text=True)
    C.require(p.returncode==0,'CPU calibration transfer failed:'+p.stderr[-4000:])
    verification=remote('''import sys,json,hashlib,os,ctypes
from pathlib import Path
stage=Path(sys.argv[1]);final=Path(sys.argv[2]);expected=sys.argv[3]
p=stage/'calibration/ARCHIVE29_MANIFEST.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
m=json.loads(p.read_text());count=0
for row in m['files']:
 p=stage/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 assert h.hexdigest()==row['sha256'];count+=1
 fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
v=dict(status='PASS_INDEPENDENT29_NONAIME_RAW_SOURCE_SHA_BYTES_NOREPLACE',count=count,bytes=m['bytes'],manifest_sha256=expected,archive=str(final),AIME_generations=0)
seal=stage/'CALIBRATION_EVIDENCE_SEAL.json'
with seal.open('x') as stream:stream.write(json.dumps(v,sort_keys=True,separators=(',',':'))+'\\n');stream.flush();os.fsync(stream.fileno())
for folder in sorted([p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True)+[stage]:
 fd=os.open(folder,os.O_RDONLY);os.fsync(fd);os.close(fd)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
assert fn(-100,os.fsencode(stage),-100,os.fsencode(final),1)==0,os.strerror(ctypes.get_errno())
fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(v))
''',stage,final,manifest_sha)
    C.atomic_new(root/'calibration/ARCHIVE29_VERIFICATION.json',verification)
    print(json.dumps(verification))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
