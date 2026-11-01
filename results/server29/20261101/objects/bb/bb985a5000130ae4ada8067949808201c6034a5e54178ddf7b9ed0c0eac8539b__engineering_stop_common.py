"""Independent CPU delivery helpers; no Torch, CUDA, signalling or old writes."""
from pathlib import Path
import hashlib,json,os,stat,subprocess

TASK='QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1'
ATTEMPT=TASK+'_DIRECTORY_REPAIR_RERUN_20261022'
ORIGINAL=Path('/data/zypan/analyses/'+ATTEMPT.lower())
WORK=Path(str(ORIGINAL)+'_engineering_stop_recovery_v1')
ARCHIVE='/data/zypan/qwen_oscar_resident_rotation_efficient_qualification_archive/'+ATTEMPT.lower()+'_engineering_stop_recovery_v1'
SOURCE_SHA='f0c27e48442e924b1a49fa254c4f262f49be85da00a3ec52c3073d7e73e77d5c'
FINISHER_FAILURE_SHA='6cd4b54058c5058adbb769ba474374bef852f838bc9302188674a4974989e8c9'
READER_FAILURE_SHA='352456fc34924b3a1320bc800132b37f97b6e0336045f2f89a44d9f42dddbdbd'
TERMINAL='FAIL_CLOSED_STOP'
CONTROLS=('SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json')
RAW_SUFFIXES=('.bin','.npz','.npy','.f32','.pt','.pth','.safetensor','.safetensors','.ckpt','.onnx','.gguf','.pyc','.pyo')

def require(ok,message):
 if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_new(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def save(p,v):write_new(p,canonical(v))
def safe(n):
 p=Path(n);require(isinstance(n,str)and n and not p.is_absolute()and str(p)==n and all(x not in('.','..','')for x in n.split('/'))and not any(x in n for x in('\n','\r','\0')),'Unsafe archive path');return p
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
  if f[0]in('Z','X','x'):return None
  return dict(pid=int(pid),starttick=int(f[19]),cmdline=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'))
 except(FileNotFoundError,ProcessLookupError):return None
 except PermissionError:
  require((Path('/proc')/str(pid)).stat().st_uid!=os.geteuid(),'Own identity unreadable');return None
def checked_file(p):
 p=Path(p);a=p.lstat();require(stat.S_ISREG(a.st_mode),'Only regular file accepted, no symlink following:'+str(p))
 h=sha(p);b=p.lstat();require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'Changing regular file:'+str(p))
 return dict(source_path=str(p),bytes=b.st_size,sha256=h)
def target_probe(link,target):
 # Lexical normalization followed by lstat of each component; never follow a
 # link or open a target. This records actual missing targets without traversal.
 dest=Path(os.path.normpath(target if os.path.isabs(target)else str(link.parent/target)))
 cur=Path(dest.anchor)
 for part in dest.parts[1:]:
  cur=cur/part
  try:s=cur.lstat()
  except FileNotFoundError:return dict(path=str(dest),dangling=True,first_missing=str(cur),followed_symlinks=False)
  if stat.S_ISLNK(s.st_mode):return dict(path=str(dest),dangling=None,unresolved='TARGET_CHAIN_IS_SYMLINK_NOT_FOLLOWED',first_link=str(cur),followed_symlinks=False)
  if cur!=dest and not stat.S_ISDIR(s.st_mode):return dict(path=str(dest),dangling=True,first_non_directory=str(cur),followed_symlinks=False)
 return dict(path=str(dest),dangling=False,followed_symlinks=False)
def inventory(root):
 """All regular bytes and all link descriptors. No symlink is traversed."""
 root=Path(root);files=[];links=[]
 def walk(directory):
  for e in sorted(os.scandir(directory),key=lambda e:e.name):
   p=Path(e.path);s=e.stat(follow_symlinks=False);rel=str(p.relative_to(root));safe(rel)
   if stat.S_ISLNK(s.st_mode):
    target=os.readlink(p)
    links.append(dict(path=rel,target=target,lstat=dict(mode=s.st_mode,device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,uid=s.st_uid,gid=s.st_gid,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns),target_lstat_probe=target_probe(p,target),copied_as_link=False,followed=False))
   elif stat.S_ISDIR(s.st_mode):walk(p)
   else:
    require(stat.S_ISREG(s.st_mode),'Unsupported special file:'+str(p));files.append(dict(path=rel,**checked_file(p)))
 walk(root);files.sort(key=lambda x:x['path']);links.sort(key=lambda x:x['path']);return files,links
def original_guard():
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only environment required')
 require(sha(ORIGINAL/'SOURCE_FREEZE.json')==SOURCE_SHA,'Original153 source identity changed')
 f=read(ORIGINAL/'SOURCE_FREEZE.json');require(len(f['files'])==153,'Expected exact original153 sources')
 for x in f['files']:
  row=checked_file(ORIGINAL/safe(x['path']));require(row['bytes']==x['bytes']and row['sha256']==x['sha256'],'Frozen source changed:'+x['path'])
 raw=[x for x in f['files']if x['path'].lower().endswith(RAW_SUFFIXES)]
 require(len(raw)==25 and sum(x['bytes']for x in raw)==176591622,'Original25 raw source assets changed')
 session=read(ORIGINAL/'SESSION_RESULT.json');task=read(ORIGINAL/'TASK_RESULT.json')
 require(session['status']==task['status']==TERMINAL and session['formal_completed']==task['actual_formal_completed']==0 and session['qualification_completed']==0 and session['attempted_units']==1,'Original real STOP0/60 differs')
 require(session['own_release_all_PASS']and session['all_own_processes_released'],'Original own release unresolved')
 require(sha(ORIGINAL/'FINISHER_FAILURE.json')==FINISHER_FAILURE_SHA and not(ORIGINAL/'FINISHER_RESULT.json').exists(),'Original finisher failure changed or falsely replaced')
 require(not any((ORIGINAL/n).exists()for n in CONTROLS),'Original failed science must remain without fabricated science seal')
 require(sha(WORK/'evidence/original_local_reader/LOCAL_DOWNLOADER_FAILURE.json')==READER_FAILURE_SHA,'Original local reader failure not preserved')
 identities=[read(ORIGINAL/n)['identity']for n in('CONTROLLER_STARTED.json','FINISHER_STARTED.json')];intervals=[]
 for folder in('qualification','formal'):
  for p in sorted((ORIGINAL/folder).glob('*/STARTED.json')):identities.append(read(p)['identity'])
  for p in sorted((ORIGINAL/folder).glob('*/RESOURCE_INTERVAL.json')):
   x=read(p);require(x['source_freeze_sha256']==SOURCE_SHA and x['release']=='PASS','Original unit release/source invalid');identities.append(x['identity']);intervals.append(dict(path=str(p.relative_to(ORIGINAL)),sha256=sha(p),unit=x['unit'],exit_returncode=x['exit_returncode'],release=x['release']))
 require(len(intervals)==1 and intervals[0]['exit_returncode']!=0,'Expected one original failed and released unit')
 unique={json.dumps(x,sort_keys=True):x for x in identities};observed=[]
 for x in unique.values():
  cur=identity(x['pid']);require(cur!=x,'Original exact own process is still alive');observed.append(dict(expected=x,current=cur,exact_identity_exited=True))
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   x=identity(int(p.name));require(x is None or not any(a.startswith(str(ORIGINAL/'scripts')+'/')for a in x['cmdline']),'Own original scientific process remains')
 p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20);require(p.returncode==0,p.stderr)
 apps=[]
 for line in p.stdout.splitlines():
  if not line.strip():continue
  a=[x.strip()for x in line.split(',')];x=identity(int(a[1]));require(x is None or not any(v==str(ORIGINAL)or v.startswith(str(ORIGINAL)+'/')for v in x['cmdline']),'Own original NVML owner remains');apps.append(dict(uuid=a[0],pid=int(a[1]),identity=x))
 return dict(status='PASS_REAL153_STOP_ZERO_AND_RELEASED_OWN_IDENTITIES_READONLY',source_count=153,source_sha256=SOURCE_SHA,raw_source_count=25,raw_source_bytes=176591622,formal_completed=0,planned_formal=60,qualification_completed=0,exact_process_exit=observed,intervals=intervals,NVML=dict(status='PASS_NO_OWN_NVML_FOREIGN_ALLOWED',apps=apps),original_science_seal_exists=False,original_finisher_failure_sha256=FINISHER_FAILURE_SHA,original_reader_failure_sha256=READER_FAILURE_SHA,full9B_qualified=False,scientific_failure_rejudged=False)
