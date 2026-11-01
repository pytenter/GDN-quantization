#!/usr/bin/env python3
"""Once-only independent CPU engineering STOP delivery, never original finisher."""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback
import engineering_stop_common as C
from engineering_stop_common import require,read,canonical,sha,write_new,checked_file,safe
TASK=C.TASK;TERMINAL=C.TERMINAL;ROOT=C.ORIGINAL;WORK=C.WORK;ARCHIVE=C.ARCHIVE
STAGE=ARCHIVE+'.partial_once';CONTROLS=set(C.CONTROLS)
SSH=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','zypan@202.38.247.29']
def save(name,obj):C.save(WORK/name,obj)
def identity():return C.identity(os.getpid())
def check_tools():
 require(Path(__file__).resolve()==WORK/'operations/publish_engineering_stop_recovery_v1.py','Wrong independent CPU work root')
 m=read(WORK/'operations/ENGINEERING_STOP_RECOVERY_TOOL_MANIFEST.json')
 for x in m['files']:
  p=WORK/safe(x['path']);require(sha(p)==x['sha256']and p.stat().st_size==x['bytes'],'Independent CPU tool bytes changed')
 require(len(m['files'])==3,'Expected common/publisher/independent verifier identities')
def live_paths():
 out={}
 for fd in(1,2):
  n=os.readlink('/proc/self/fd/'+str(fd))
  if n.startswith('/'):
   p=Path(n).resolve();require(not p.is_relative_to(ROOT),'Publisher log must never write original root')
   if p.is_relative_to(WORK):out[str(p.relative_to(WORK))]='Exact live publisher output; supplemental after exit'
 return out
REMOTE_COMMON = r'''
from pathlib import Path
import base64,ctypes,errno,hashlib,json,os,socket,sys,tempfile,time
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_new(p,b):
 with Path(p).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def rename_noreplace(src,dst):
 fn=ctypes.CDLL(None,use_errno=True).renameat2
 fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 if fn(-100,os.fsencode(src),-100,os.fsencode(dst),1):raise OSError(ctypes.get_errno(),'renameat2_RENAME_NOREPLACE')
def inventory(r):
 out=set()
 for p in r.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():out.add(str(p.relative_to(r)))
  else:assert p.is_dir()
 return out
def exact_dirs(r,names):
 allowed={'.'}
 for n in names:allowed.update(str(p)for p in Path(n).parents)
 actual={str(p.relative_to(r))for p in r.rglob('*')if p.is_dir()}
 assert actual<=allowed,'Unknown empty archive directory'
'''


def on29(code):
    result = subprocess.run(SSH + ["exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -"],
                            input=code, text=True, capture_output=True)
    require(result.returncode == 0, "Archive29 CPU step failed, retain stage/no retry: " + result.stderr[-6000:])
    return json.loads(result.stdout)


def preflight():
    code = REMOTE_COMMON + "\narchive=Path(" + repr(ARCHIVE) + ");stage=Path(" + repr(STAGE) + ")\n" + r'''
base=archive.parent;base.mkdir(parents=True,exist_ok=True)
assert not archive.exists() and not stage.exists(), 'Existing final/stage; never overwrite or retry blindly'
s=os.statvfs(base);free=s.f_bavail*s.f_frsize;assert free>=120*1024**3
with tempfile.TemporaryDirectory(prefix='engineering_stop_noreplace_',dir=str(base)) as d:
 p=Path(d);a=p/'src';b=p/'existing';c=p/'new';a.write_bytes(b'SOURCE');b.write_bytes(b'KEEP')
 try:rename_noreplace(a,b)
 except OSError as e:assert e.errno==errno.EEXIST
 else:raise AssertionError('NOREPLACE overwritten')
 assert a.read_bytes()==b'SOURCE' and b.read_bytes()==b'KEEP'
 rename_noreplace(a,c);assert c.read_bytes()==b'SOURCE' and not a.exists()
print(json.dumps(dict(status='PASS_ARCHIVE20GiB_PLUS100GiB_HEADROOM_AND_ATOMIC_NOREPLACE',host=socket.gethostname(),free_bytes=free)))
'''
    return on29(code)



def build_rows(originals,excluded_live):
 rows=[dict(path='original_attempt/'+x['path'],source_path=x['source_path'],bytes=x['bytes'],sha256=x['sha256'])for x in originals]
 excluded=set(C.CONTROLS)|{'PUBLISHER_RESULT.json','PUBLISHER_FAILURE.json','ATOMIC_PUBLICATION_RECEIPT.json'}|set(excluded_live)
 eng,links=C.inventory(WORK);require(not links,'No new symlink allowed in independent CPU work root')
 for x in eng:
  if x['path']not in excluded:rows.append(dict(path='engineering_delivery/'+x['path'],source_path=x['source_path'],bytes=x['bytes'],sha256=x['sha256']))
 rows.sort(key=lambda x:x['path']);require(len(rows)==len({x['path']for x in rows}),'Duplicate archive path')
 require(sum(x['bytes']for x in rows)<20*1024**3,'Engineering STOP package exceeds20GiB bound')
 return rows,excluded
def initialize(manifest):
    code = REMOTE_COMMON + "\nr=Path(" + repr(STAGE) + ");final=Path(" + repr(ARCHIVE) + ")\nmanifest_bytes=" + repr(canonical(manifest)) + "\n" + r'''
assert not r.exists() and not final.exists()
s=os.statvfs(r.parent);assert s.f_bavail*s.f_frsize>=120*1024**3
r.mkdir();fsyncdir(r.parent);write_new(r/'SHA256_MANIFEST.json',manifest_bytes)
m=json.loads(manifest_bytes);write_new(r/'SHA256SUMS',''.join(x['sha256']+'  '+x['path']+'\n' for x in m['files']).encode());fsyncdir(r)
print(json.dumps(dict(status='PASS_NEW_STAGE_AND_EXACT_MANIFEST_CONTROLS',manifest_sha256=sha(r/'SHA256_MANIFEST.json'),sums_sha256=sha(r/'SHA256SUMS'))))
'''
    return on29(code)


def transfer(rows):
    receiver = REMOTE_COMMON + "\nr=Path(" + repr(STAGE) + ")\n" + r'''
import tarfile
m=json.loads((r/'SHA256_MANIFEST.json').read_text());expected={x['path']:x for x in m['files']};seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as stream:
 for member in stream:
  name=member.name;assert name in expected and name not in seen and member.isfile() and member.size==expected[name]['bytes']
  rel=Path(name);assert not rel.is_absolute() and all(x not in ('.','..') for x in rel.parts)
  p=r/rel;assert p.resolve().is_relative_to(r.resolve());p.parent.mkdir(parents=True,exist_ok=True)
  source=stream.extractfile(member);assert source is not None;count=0
  with p.open('xb') as target:
   for block in iter(lambda:source.read(1024*1024),b''):target.write(block);count+=len(block)
   target.flush();os.fsync(target.fileno())
  assert count==expected[name]['bytes'];seen.add(name)
assert seen==set(expected)
print(json.dumps(dict(status='PASS_ALL_UNCOMPRESSED_SNAPSHOT_MEMBERS_RECEIVED',count=len(seen))))
'''
    proc = subprocess.Popen(SSH + ["exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -c " + shlex.quote(receiver)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|", dereference=True) as stream:
            for row in rows:
                path = Path(row["source_path"])
                before = path.stat()
                require(checked_file(path)["sha256"] == row["sha256"] and before.st_size == row["bytes"],
                        "Source changed before publication transfer:" + str(path))
                stream.add(path, arcname=row["path"], recursive=False)
                after = path.stat()
                require((before.st_size, before.st_ino, before.st_mtime_ns) == (after.st_size, after.st_ino, after.st_mtime_ns)
                        and sha(path) == row["sha256"], "Source changed during transfer:" + str(path))
        proc.stdin.close()
        proc.stdin = None
        stdout, stderr = proc.communicate()
        require(proc.returncode == 0, "Partial archive transfer failed/no retry: " + stderr.decode(errors="replace")[-6000:])
        result = json.loads(stdout)
        require(result["count"] == len(rows), "Archive receiver count differs")
        return result
    except BaseException:
        if proc.stdin is not None:
            proc.stdin.close()
            proc.stdin = None
        proc.communicate()
        raise



def seal_publish(expected):
 code=REMOTE_COMMON+'\nr=Path('+repr(STAGE)+');final=Path('+repr(ARCHIVE)+')\nexpected='+repr(expected)+'\nsource_sha='+repr(C.SOURCE_SHA)+';task='+repr(TASK)+'\n'+r"""
assert not final.exists()
assert sha(r/'SHA256_MANIFEST.json')==expected['manifest_sha256'] and sha(r/'SHA256SUMS')==expected['sums_sha256']
m=json.loads((r/'SHA256_MANIFEST.json').read_text());assert m['task_id']==task and m['terminal']=='FAIL_CLOSED_STOP'
assert m['engineering_archive'] is True and m['original_science_seal_created'] is False
assert m['formal_completed']==m['qualification_completed']==0 and m['planned_formal']==60
assert (r/'SHA256SUMS').read_bytes()==''.join(x['sha256']+'  '+x['path']+'\n'for x in m['files']).encode()
verified=[]
for x in m['files']:
 p=r/x['path'];assert not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(r.resolve())
 a=p.stat();h=sha(p);b=p.stat();assert(a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
 assert h==x['sha256'] and b.st_size==x['bytes'];verified.append(dict(path=x['path'],bytes=b.st_size,sha256=h))
by={x['path']:x for x in verified};assert inventory(r)==set(by)|{'SHA256_MANIFEST.json','SHA256SUMS'};exact_dirs(r,set(by)|{'SHA256_MANIFEST.json','SHA256SUMS'})
assert sha(r/'original_attempt/SOURCE_FREEZE.json')==source_sha
sources=json.loads((r/'original_attempt/SOURCE_FREEZE.json').read_text())['files'];assert len(sources)==153
for x in sources:
 y=by['original_attempt/'+x['path']];assert y['bytes']==x['bytes'] and y['sha256']==x['sha256']
links=json.loads((r/'engineering_delivery/ORIGINAL_SYMLINK_DESCRIPTORS.json').read_text())
assert len(links['links'])==m['symlink_descriptor_count'] and all(x['followed'] is False and x['copied_as_link'] is False for x in links['links'])
assert not any((r/'original_attempt'/n).exists()for n in('FINAL_SEAL_RECEIPT.json','FINAL_ARCHIVE_VERIFICATION.json','SHA256_MANIFEST.json','SHA256SUMS','FINISHER_RESULT.json'))
assert sha(r/'original_attempt/FINISHER_FAILURE.json')==m['original_finisher_failure_sha256']
assert sha(r/'engineering_delivery/evidence/original_local_reader/LOCAL_DOWNLOADER_FAILURE.json')==m['original_reader_failure_sha256']
verification=dict(status='PASS_INDEPENDENT_ARCHIVE29_EVERY_REGULAR_FILE_FRESH_SHA_BYTES_AND_SYMLINK_DESCRIPTORS',task_id=task,terminal='FAIL_CLOSED_STOP',engineering_archive=True,host=socket.gethostname(),monotonic=time.monotonic(),count=len(verified),bytes=sum(x['bytes']for x in verified),files=verified,source_count=153,source_freeze_sha256=source_sha,manifest_sha256=sha(r/'SHA256_MANIFEST.json'),symlink_descriptor_count=len(links['links']),no_symlink_followed=True)
write_new(r/'FINAL_ARCHIVE_VERIFICATION.json',canonical(verification))
seal=dict(status='PASS_ENGINEERING_STOP_RECOVERY_ATOMIC_NOREPLACE_ARCHIVE',task_id=task,terminal='FAIL_CLOSED_STOP',engineering_archive=True,original_science_seal_created=False,original_finisher_failure_replaced=False,formal_completed=0,qualification_completed=0,planned_formal=60,complete_accuracy=None,GPU_retries=0,archive_root=str(final),atomic='renameat2_RENAME_NOREPLACE',source_freeze_sha256=source_sha,manifest_sha256=sha(r/'SHA256_MANIFEST.json'),SHA256SUMS_sha256=sha(r/'SHA256SUMS'),archive_verification_sha256=sha(r/'FINAL_ARCHIVE_VERIFICATION.json'),symlink_descriptor_count=len(links['links']))
write_new(r/'FINAL_SEAL_RECEIPT.json',canonical(seal));names=['SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']
assert inventory(r)==set(by)|set(names);exact_dirs(r,set(by)|set(names))
for x in verified:assert (r/x['path']).stat().st_size==x['bytes']and sha(r/x['path'])==x['sha256']
fsyncdir(r);rename_noreplace(r,final);fsyncdir(final.parent)
controls={}
for n in names:
 b=(final/n).read_bytes();controls[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
print(json.dumps(dict(status='PASS_INDEPENDENT_ENGINEERING_STOP_ATOMIC_ARCHIVE',archive_root=str(final),count=len(verified),bytes=verification['bytes'],symlink_descriptor_count=len(links['links']),controls=controls)))
"""
 return on29(code)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 if not a.execute:print(json.dumps(dict(status='PLAN_ONLY',archive=ARCHIVE,source153=C.SOURCE_SHA,original_mutations=0,GPU_dispatches=0)));return 0
 require(os.environ.get('CUDA_VISIBLE_DEVICES')==''and sys.dont_write_bytecode,'CPU-only and -B required');check_tools()
 require(not any((WORK/n).exists()for n in set(C.CONTROLS)|{'PUBLISHER_STARTED.json','PUBLISHER_RESULT.json','PUBLISHER_FAILURE.json','ARCHIVE_PREFLIGHT.json','ATOMIC_PUBLICATION_RECEIPT.json'}),'Existing independent publication attempt; preserve/no retry')
 me=identity();live=live_paths();save('PUBLISHER_STARTED.json',dict(identity=me,monotonic=time.monotonic(),CPU_only=True,original_root_writes=0,GPU_dispatches=0,excluded_live=live))
 try:
  guard=C.original_guard();save('ORIGINAL_SOURCE_PROCESS_STOP_READONLY_GUARD.json',guard)
  originals,links=C.inventory(ROOT);require(links,'Expected actual engineering negative-fixture symlinks were not inventoried')
  save('ORIGINAL_REGULAR_FILE_CATALOG.json',dict(original_root=str(ROOT),files=originals,count=len(originals),bytes=sum(x['bytes']for x in originals),excluded_regular_paths=[]))
  save('ORIGINAL_SYMLINK_DESCRIPTORS.json',dict(status='PASS_ALL_ORIGINAL_LINKS_LSTAT_READLINK_NEVER_FOLLOWED',original_root=str(ROOT),links=links,count=len(links),links_copied_or_followed=0,original_links_deleted_or_rewritten=False))
  report=dict(status='TRUE_TECHNICAL_STOP_0_OF_60_NO_ACCURACY',task_id=TASK,source_freeze_sha256=C.SOURCE_SHA,source_count=153,qualification_completed=0,formal_completed=0,planned_formal=60,complete_accuracy=None,scientific_runtime_qualification='FAILED_UNEXPECTED_MAPPED_CONTROLLED_LIBRARY_AFTER_MODEL_LOADED',original_finisher='TRUE_FAILURE_ON_ENGINEERING_FIXTURE_SYMLINK_CATALOG',original_reader='TRUE_FAILURE_WAITING_FOR_FINISHER_SUCCESS',original_science_seal_exists=False,independent_engineering_archive_only=True,old_seals_unchanged=True,GPU_or_finisher_restarts=0,FP32_State_is_packed_INT8=False,EXACT_GPUh='UNRESOLVED')
  save('HONEST_ENGINEERING_STOP_REPORT.json',report)
  pf=preflight();save('ARCHIVE_PREFLIGHT.json',pf);rows,excluded=build_rows(originals,live)
  raw=[x for x in rows if x['path'].lower().endswith(C.RAW_SUFFIXES)]
  manifest=dict(status='FROZEN_INDEPENDENT_ENGINEERING_STOP_REGULAR_FILES_AND_LINK_DESCRIPTORS',task_id=TASK,terminal=TERMINAL,engineering_archive=True,original_science_seal_created=False,original_root=str(ROOT),engineering_work_root=str(WORK),original_source_freeze_sha256=C.SOURCE_SHA,source_count=153,formal_completed=0,qualification_completed=0,planned_formal=60,files=rows,count=len(rows),bytes=sum(x['bytes']for x in rows),original_regular_count=len(originals),symlink_descriptor_count=len(links),symlinks_never_followed=True,publisher_identity=me,original_finisher_failure_sha256=C.FINISHER_FAILURE_SHA,original_reader_failure_sha256=C.READER_FAILURE_SHA,remote_only_raw=dict(count=len(raw),bytes=sum(x['bytes']for x in raw),files=raw),excluded_exact_engineering_paths=sorted(excluded),original_regular_file_exclusions=[],nested_basename_exclusions=False)
  initial=initialize(manifest);received=transfer(rows)
  again,again_links=C.inventory(ROOT);require(again==originals and again_links==links,'Original root changed after stopped snapshot; retain stage, no automatic retry')
  published=seal_publish(initial);controls={}
  for name in C.CONTROLS:
   x=published['controls'][name];b=base64.b64decode(x['base64'],validate=True);require(len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256'],'Exact archive-control bytes differ');write_new(WORK/name,b);controls[name]=dict(bytes=len(b),sha256=sha(WORK/name))
  summary=dict(status=published['status'],publisher_identity=me,task_id=TASK,terminal=TERMINAL,archive_root=ARCHIVE,count=len(rows),bytes=manifest['bytes'],source_count=153,symlink_descriptor_count=len(links),raw_count=len(raw),raw_bytes=sum(x['bytes']for x in raw),formal_completed=0,planned_formal=60,complete_accuracy=None,controls=controls,original_science_seal_created=False,old_root_writes=0,GPU_or_finisher_restarts=0,transfer=received,local_verification='NOT_RUN')
  save('ATOMIC_PUBLICATION_RECEIPT.json',summary);save('PUBLISHER_RESULT.json',dict(status='PASS_ENGINEERING_STOP_ARCHIVE_PUBLICATION_ONLY',identity=me,controls=controls,original_science_seal_created=False,monotonic=time.monotonic()));print(json.dumps(summary,sort_keys=True),flush=True);return 0
 except BaseException as e:
  if not(WORK/'PUBLISHER_FAILURE.json').exists():save('PUBLISHER_FAILURE.json',dict(status='REAL_CPU_ENGINEERING_DELIVERY_FAILURE_PRESERVE_NO_AUTORETRY',identity=me,error=repr(e),traceback=traceback.format_exc(),original_root_writes=0,GPU_dispatches=0))
  raise
if __name__=='__main__':raise SystemExit(main())
