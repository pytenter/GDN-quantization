#!/usr/bin/env python3
"""Independent one-shot read-only Mac delivery of engineering STOP recovery."""
from pathlib import Path
import argparse,base64,ctypes,errno,hashlib,json,os,shlex,subprocess,sys,tempfile,traceback
import engineering_stop_common as C
TASK=C.TASK;TERMINAL=C.TERMINAL;ROOT=Path(__file__).resolve().parents[1]
REMOTE=str(C.ORIGINAL);WORK=str(C.WORK);ARCHIVE=C.ARCHIVE;SOURCE_SHA=C.SOURCE_SHA
CONTROLS=C.CONTROLS;RAW_SUFFIXES=C.RAW_SUFFIXES
TEXT_SUFFIXES=('.py','.json','.jsonl','.csv','.tsv','.txt','.md','.sh','.log','.diff','.yaml','.yml','.toml','.sha256','.html')
PYTHON55='/data/ydai/miniconda3/envs/bitdecode/bin/python'
SSH55=['ssh','-S','/tmp/qwen_gdn_replication_20261013.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','zypan@202.38.247.55']
SSH29=['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','zypan@202.38.247.29']
require=C.require;load=C.read;canonical=C.canonical;sha=C.sha;write_new=C.write_new;safe=C.safe
def save(name,obj):C.save(ROOT/'operations'/name,obj)
def text_eligible(row):
 n=row['path'].lower();return not n.endswith(RAW_SUFFIXES)and(n.endswith(TEXT_SUFFIXES)or Path(n).name in('sha256sums','license','notice'))
def remote_cpu(role, code):
    env = "exec env CUDA_VISIBLE_DEVICES= USE_TORCH=0 USE_TF=0 PYTHONDONTWRITEBYTECODE=1 "
    if role == "55":
        command = SSH55 + [env + shlex.quote(PYTHON55) + " -B -"]
    else:
        inner = SSH29 + [env + "/usr/bin/python3 -B -"]
        command = SSH55 + ["exec " + " ".join(shlex.quote(x) for x in inner)]
    result = subprocess.run(command, input=code, text=True, capture_output=True)
    require(result.returncode == 0, "Read-only CPU transport/verification failed, retained/no retry:" + role + ":" + result.stderr[-6000:])
    return json.loads(result.stdout)


REMOTE_HELPERS = r'''
from pathlib import Path
import base64,hashlib,importlib.util,json,os,socket,subprocess,sys,time
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(name):
 p=Path(name);assert name and not p.is_absolute() and str(p)==name and all(x not in ('.','..') for x in p.parts)
 assert '\n' not in name and '\r' not in name
 return p
def inventory(r):
 rows=set()
 for p in r.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():rows.add(str(p.relative_to(r)))
  else:assert p.is_dir()
 return rows
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();return dict(pid=pid,starttick=int(s[s.rfind(')')+2:].split()[19]),cmdline=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
 except FileNotFoundError:return None
'''



def audit_code(role,manifest_sha,seal_sha):
 code=REMOTE_HELPERS+'\nrole='+repr(role)+';archive=Path('+repr(ARCHIVE)+');work=Path('+repr(WORK)+');original=Path('+repr(REMOTE)+')\nr=work if role=="55"else archive\ncontrols='+repr(CONTROLS)+';source_sha='+repr(SOURCE_SHA)+';task='+repr(TASK)+'\nexpected_manifest_sha='+repr(manifest_sha)+';expected_seal_sha='+repr(seal_sha)+'\n'+r"""
data={n:(r/n).read_bytes()for n in controls};assert hashlib.sha256(data['SHA256_MANIFEST.json']).hexdigest()==expected_manifest_sha and hashlib.sha256(data['FINAL_SEAL_RECEIPT.json']).hexdigest()==expected_seal_sha
m=json.loads(data['SHA256_MANIFEST.json']);seal=json.loads(data['FINAL_SEAL_RECEIPT.json']);v=json.loads(data['FINAL_ARCHIVE_VERIFICATION.json'])
assert m['task_id']==seal['task_id']==task and m['terminal']==seal['terminal']=='FAIL_CLOSED_STOP'
assert m['engineering_archive'] is True and seal['engineering_archive'] is True and m['original_science_seal_created'] is False and seal['original_science_seal_created'] is False
assert m['formal_completed']==m['qualification_completed']==seal['formal_completed']==seal['qualification_completed']==0 and m['planned_formal']==seal['planned_formal']==60
assert seal['complete_accuracy'] is None and seal['GPU_retries']==0 and seal['original_finisher_failure_replaced'] is False
assert seal['status']=='PASS_ENGINEERING_STOP_RECOVERY_ATOMIC_NOREPLACE_ARCHIVE'and seal['atomic']=='renameat2_RENAME_NOREPLACE'and seal['archive_root']==str(archive)
assert seal['manifest_sha256']==v['manifest_sha256']==expected_manifest_sha
assert seal['SHA256SUMS_sha256']==hashlib.sha256(data['SHA256SUMS']).hexdigest()and seal['archive_verification_sha256']==hashlib.sha256(data['FINAL_ARCHIVE_VERIFICATION.json']).hexdigest()
assert seal['source_freeze_sha256']==v['source_freeze_sha256']==m['original_source_freeze_sha256']==source_sha
assert v['status']=='PASS_INDEPENDENT_ARCHIVE29_EVERY_REGULAR_FILE_FRESH_SHA_BYTES_AND_SYMLINK_DESCRIPTORS'
rows=m['files'];assert len(rows)==m['count']==len({x['path']for x in rows})and sum(x['bytes']for x in rows)==m['bytes']
assert data['SHA256SUMS']==''.join(x['sha256']+'  '+x['path']+'\n'for x in rows).encode()
def asset(name):
 rel=safe(name)
 if role=='29':return archive/rel
 bases={'original_attempt':original,'engineering_delivery':work};assert rel.parts[0]in bases
 p=bases[rel.parts[0]].joinpath(*rel.parts[1:]);assert p.resolve().is_relative_to(bases[rel.parts[0]].resolve());return p
actual=[]
for x in rows:
 p=asset(x['path']);assert role=='29'or str(p)==x['source_path'];assert p.is_file()and not p.is_symlink()
 a=p.stat();h=sha(p);b=p.stat();assert(a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
 assert h==x['sha256']and b.st_size==x['bytes'];actual.append(dict(path=x['path'],bytes=b.st_size,sha256=h))
assert actual==v['files']and len(actual)==v['count']and sum(x['bytes']for x in actual)==v['bytes'];by={x['path']:x for x in actual}
assert sha(asset('original_attempt/SOURCE_FREEZE.json'))==source_sha
sources=read(asset('original_attempt/SOURCE_FREEZE.json'))['files'];assert len(sources)==153
for x in sources:
 y=by['original_attempt/'+x['path']];assert y['bytes']==x['bytes']and y['sha256']==x['sha256']
links=read(asset('engineering_delivery/ORIGINAL_SYMLINK_DESCRIPTORS.json'));assert len(links['links'])==m['symlink_descriptor_count']==seal['symlink_descriptor_count']
assert all(x['followed'] is False and x['copied_as_link'] is False and x['target_lstat_probe']['followed_symlinks'] is False for x in links['links'])
assert sha(asset('original_attempt/FINISHER_FAILURE.json'))==m['original_finisher_failure_sha256']
assert sha(asset('engineering_delivery/evidence/original_local_reader/LOCAL_DOWNLOADER_FAILURE.json'))==m['original_reader_failure_sha256']
origbase=asset('original_attempt/SOURCE_FREEZE.json').parent
assert not any((origbase/n).exists()for n in('FINISHER_RESULT.json','SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json'))
session=read(asset('original_attempt/SESSION_RESULT.json'));assert session['status']=='FAIL_CLOSED_STOP'and session['formal_completed']==session['qualification_completed']==0 and session['attempted_units']==1
assert session['own_release_all_PASS']and session['all_own_processes_released']
external=[];models=[];guard=None;publisher_exit=None
if role=='29':
 assert inventory(archive)==set(by)|set(controls),'Archive regular-file extras rejected; no links are permitted'
 allowed={'.'}
 for n in set(by)|set(controls):allowed.update(str(p)for p in Path(n).parents)
 assert {str(p.relative_to(archive))for p in archive.rglob('*')if p.is_dir()}<=allowed,'Unknown empty archive directory'
else:
 pub=m['publisher_identity'];cur=identity(pub['pid']);assert cur!=pub;publisher_exit=dict(expected=pub,current=cur,exact_identity_exited=True)
 result=read(work/'PUBLISHER_RESULT.json');assert result['status']=='PASS_ENGINEERING_STOP_ARCHIVE_PUBLICATION_ONLY'and result['identity']==pub
 assert read(work/'ATOMIC_PUBLICATION_RECEIPT.json')['status']=='PASS_INDEPENDENT_ENGINEERING_STOP_ATOMIC_ARCHIVE'
 sys.path.insert(0,str(work/'operations'));import engineering_stop_common as C
 guard=C.original_guard();files,currentlinks=C.inventory(original)
 frozen=read(work/'ORIGINAL_REGULAR_FILE_CATALOG.json')['files'];assert files==frozen and currentlinks==links['links']
 assert len(files)==m['original_regular_count']
 toolmanifest=read(work/'operations/ENGINEERING_STOP_RECOVERY_TOOL_MANIFEST.json')
 for x in toolmanifest['files']:
  p=work/x['path'];assert sha(p)==x['sha256']and p.stat().st_size==x['bytes']
 bindings=read(original/'analysis/EXTERNAL_BINDINGS.json')['files'];assert len(bindings)==len({x['path']for x in bindings})==61
 for x in bindings:
  p=Path(x['path']);a=p.stat();h=sha(p);b=p.stat();assert(a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
  assert h==x['sha256']and b.st_size==x['bytes'];external.append(dict(path=x['path'],bytes=b.st_size,sha256=h))
 for x in read(original/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
  s=Path(x['path']).stat();assert(s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns']);models.append(dict(path=x['path'],bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
 assert len(models)==17
print(json.dumps(dict(status='PASS_INDEPENDENT_READONLY_ENGINEERING_STOP_FULL_REGULAR_SHA_BYTES_SYMLINK_DESCRIPTORS',role=role,task_id=task,terminal='FAIL_CLOSED_STOP',engineering_archive=True,original_science_seal_created=False,formal_completed=0,qualification_completed=0,planned_formal=60,complete_accuracy=None,count=len(actual),bytes=sum(x['bytes']for x in actual),files=actual,source_count=153,source_freeze_sha256=source_sha,source_files=sources,symlink_descriptors=links['links'],external_bound_file_fresh_SHA=external,model_stat=models,actual_runtime_closure_qualified=False,original_process_guard=guard,publisher_exit=publisher_exit,archive_extras_rejected=role=='29',controls={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())for n,b in data.items()}),sort_keys=True))
"""
 return code
def fetch_code(rows):
    code = REMOTE_HELPERS + "\nr=Path(" + repr(ARCHIVE) + ")\nrows=" + repr(rows) + "\n" + r'''
out=[]
for x in rows:
 p=r/safe(x['path']);assert p.resolve().is_relative_to(r.resolve()) and not p.is_symlink()
 assert p.stat().st_size==x['bytes'] and x['bytes']<=32*1024**2
 b=p.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 out.append(dict(path=x['path'],bytes=len(b),sha256=x['sha256'],base64=base64.b64encode(b).decode()))
print(json.dumps(out))
'''
    return code


def rename_exclusive(source, final):
    require(sys.platform == "darwin", "This final local publication implementation requires macOS renamex_np")
    lib = ctypes.CDLL(None, use_errno=True)
    fn = lib.renamex_np
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(os.fsencode(source), os.fsencode(final), 4):
        raise OSError(ctypes.get_errno(), "renamex_np_RENAME_EXCL")


def atomic_preflight():
    with tempfile.TemporaryDirectory(prefix="p1_cancelled_atomic_", dir=str(ROOT)) as d:
        p = Path(d)
        a, b, c = p / "source", p / "existing", p / "new"
        a.write_bytes(b"SOURCE")
        b.write_bytes(b"KEEP")
        try:
            rename_exclusive(a, b)
        except OSError as error:
            require(error.errno == errno.EEXIST, "Unexpected local atomic negative-test failure")
        else:
            raise RuntimeError("Local atomic negative test overwritten existing file")
        require(a.read_bytes() == b"SOURCE" and b.read_bytes() == b"KEEP", "Atomic rejection changed files")
        rename_exclusive(a, c)
        require(c.read_bytes() == b"SOURCE" and not a.exists(), "Atomic positive test failed")



def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--manifest-sha256');ap.add_argument('--seal-sha256');a=ap.parse_args()
 if not a.execute:print(json.dumps(dict(status='PLAN_ONLY_NO_REMOTE_MUTATIONS',archive=ARCHIVE,formal_completed=0,complete_accuracy=None)));return 0
 require(ROOT.name==C.ATTEMPT and a.manifest_sha256 and a.seal_sha256,'Actual local task and archive-control pins required')
 tm=load(ROOT/'operations/ENGINEERING_STOP_RECOVERY_TOOL_MANIFEST.json')
 for x in tm['files']:
  p=ROOT/safe(x['path']);require(p.is_file()and sha(p)==x['sha256']and p.stat().st_size==x['bytes'],'Independent local tool identity changed')
 stage=ROOT/'engineering_stop_recovery_v1_delivery.partial_once';final=ROOT/'engineering_stop_recovery_v1_delivery'
 require(not stage.exists()and not final.exists()and not(ROOT/'operations/ENGINEERING_STOP_RECOVERY_LOCAL_STARTED.json').exists(),'Existing local recovery; preserve/no retry')
 save('ENGINEERING_STOP_RECOVERY_LOCAL_STARTED.json',dict(pid=os.getpid(),manifest_sha256=a.manifest_sha256,seal_sha256=a.seal_sha256,remote_mutations=0,GPU_dispatches=0))
 try:
  atomic_preflight();x=remote_cpu('55',audit_code('55',a.manifest_sha256,a.seal_sha256));y=remote_cpu('29',audit_code('29',a.manifest_sha256,a.seal_sha256))
  require(x['files']==y['files']and x['controls']==y['controls']and x['symlink_descriptors']==y['symlink_descriptors'],'55/29 bytes/catalog/link descriptors differ')
  require(x['publisher_exit']['exact_identity_exited']and len(x['external_bound_file_fresh_SHA'])==61 and len(x['model_stat'])==17,'Final process/dependency/model proof incomplete')
  small=[z for z in y['files']if text_eligible(z)];excluded=[z for z in y['files']if not text_eligible(z)]
  require(sum(z['bytes']for z in small)<2*1024**3 and all(z['bytes']<=32*1024**2 for z in small),'Text delivery bound exceeded')
  stage.mkdir();batch=[];size=0;batches=[]
  for row in small:
   if batch and size+row['bytes']>4*1024**2:batches.append(batch);batch=[];size=0
   batch.append(row);size+=row['bytes']
  if batch:batches.append(batch)
  for batch in batches:
   got=remote_cpu('29',fetch_code(batch));require([z['path']for z in got]==[z['path']for z in batch],'Transfer roster differs')
   for row,z in zip(batch,got):
    b=base64.b64decode(z['base64'],validate=True);require(len(b)==row['bytes']==z['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']==z['sha256'],'Transfer bytes differ');write_new(stage/safe(row['path']),b)
  for n,z in y['controls'].items():
   b=base64.b64decode(z['base64'],validate=True);require(len(b)==z['bytes']and hashlib.sha256(b).hexdigest()==z['sha256'],'Original29 control bytes differ');write_new(stage/n,b)
  expected={z['path']for z in small}|set(CONTROLS);actual=set()
  for p in stage.rglob('*'):
   require(not p.is_symlink(),'No symlink may reach local delivery')
   if p.is_file():actual.add(str(p.relative_to(stage)))
  require(actual==expected,'Local file inventory differs')
  for row in small:require((stage/row['path']).stat().st_size==row['bytes']and sha(stage/row['path'])==row['sha256'],'Local fresh byteSHA differs')
  for n,z in y['controls'].items():require(sha(stage/n)==z['sha256']and(stage/n).stat().st_size==z['bytes'],'Original control local bytes differ')
  require(not any(n.lower().endswith(RAW_SUFFIXES)for n in actual),'Raw/model download forbidden');rename_exclusive(stage,final)
  result=dict(status='PASS_LEGAL_ENGINEERING_STOP_RECOVERY_LOCAL_SMALL_FILE_DELIVERY',task_id=TASK,terminal='FAIL_CLOSED_STOP',original_science_seal_created=False,original_finisher_failure_replaced=False,formal_completed=0,qualification_completed=0,planned_formal=60,complete_accuracy=None,local_delivery=str(final),local_files=len(actual),remote_regular_files=len(y['files']),source_count=153,source_freeze_sha256=SOURCE_SHA,symlink_descriptors_count=len(y['symlink_descriptors']),original_symlinks_followed_or_modified=0,raw_or_model_downloads=0,excluded_remote_files=excluded,external_bound61_fresh_SHA=True,model17_stat=True,actual_runtime_closure_qualified=False,GPU_or_finisher_restarts=0,remote_mutations=0,controls={n:{k:v for k,v in z.items()if k!='base64'}for n,z in y['controls'].items()})
  save('ENGINEERING_STOP_RECOVERY_LOCAL_DELIVERY_VERIFICATION.json',result)
  for proof in(x,y):
   for z in proof['controls'].values():z.pop('base64')
  save('ENGINEERING_STOP_RECOVERY_FINAL_INDEPENDENT_READONLY_VERIFICATION.json',dict(status='PASS_FRESH55_29_ALL_REGULAR_BYTES_SYMLINK_DESCRIPTORS_SOURCE153_EXTERNAL61_MODEL17_EXACT_EXIT',compute55=x,archive29=y))
  save('ENGINEERING_STOP_RECOVERY_FINAL_DELIVERY_SUMMARY.json',{k:v for k,v in result.items()if k!='excluded_remote_files'})
  print(json.dumps({k:v for k,v in result.items()if k!='excluded_remote_files'},sort_keys=True),flush=True);return 0
 except BaseException as e:
  p=ROOT/'operations/ENGINEERING_STOP_RECOVERY_LOCAL_FAILURE.json'
  if not p.exists():write_new(p,canonical(dict(status='REAL_READONLY_ENGINEERING_STOP_DELIVERY_FAILURE_RETAIN_NO_AUTORETRY',error=repr(e),traceback=traceback.format_exc(),remote_mutations=0,GPU_dispatches=0)))
  raise
if __name__=='__main__':raise SystemExit(main())
