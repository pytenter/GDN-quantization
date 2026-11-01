"""Lossless archive-backed byte I/O. Stdlib only; no model/GPU imports.

Remote publication is atomic and no-clobber: temp xb, fsync, independent
reread SHA, link(temp, final), unlink(temp). No local large raw copy exists.
"""
import contextlib,hashlib,json,os,select,shlex,subprocess,time
from pathlib import Path,PurePosixPath

def require(ok,message):
 if not ok:raise RuntimeError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def load_archive_config(root):
 p=Path(root)/'ARCHIVE_CONFIG.json';raw=p.read_bytes();c=json.loads(raw)
 require(c.get('task_id')=='QWEN_GDN_FAST_RUNTIME_QUALIFICATION_V1' and bool(c.get('execution_run_id')),'Archive task/run identity missing')
 require(c.get('host')=='zypan@202.38.247.29','Wrong archive host')
 require(c.get('control_path')=='/tmp/qwen_replication_archive29.sock','Wrong archive SSH control path')
 require(c.get('card_key') in {'gpu0','gpu1','gpu2','gpu3'},'Invalid archive card key')
 a=PurePosixPath(c['archive_root']);require(a.is_absolute() and '..' not in a.parts and '\n' not in str(a),'Unsafe archive root')
 require(int(c['required_quota_bytes'])>=200*1024**3,'Archive per-card reservation below200GiB')
 require(int(c['global_reservation_bytes'])>=800*1024**3 and int(c['global_headroom_bytes'])>=100*1024**3,'Archive global reservation/headroom too small')
 require(10<=float(c.get('io_timeout_seconds',300))<=3600,'Invalid bounded archive I/O timeout')
 return dict(c,config_sha256=digest(raw),config_path=str(p))

_PUT=r'''import hashlib,json,os,sys,uuid
from pathlib import Path
p=Path(sys.argv[1]);expected=int(sys.argv[2]);wanted=sys.argv[3];headroom=int(sys.argv[4]);p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():raise RuntimeError('Archive target already exists')
v=os.statvfs(p.parent)
if v.f_bavail*v.f_frsize<expected+headroom:raise RuntimeError('Archive headroom admission failed')
t=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
n=0
with t.open('xb') as f:
 while True:
  b=sys.stdin.buffer.read(1048576)
  if not b:break
  n+=len(b)
  if n>expected:raise RuntimeError('Archive stream too long')
  f.write(b)
 f.flush();os.fsync(f.fileno())
if n!=expected:raise RuntimeError('Archive stream truncated')
h=hashlib.sha256();a=t.stat()
with t.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
z=t.stat()
if h.hexdigest()!=wanted or (a.st_ino,a.st_size,a.st_mtime_ns)!=(z.st_ino,z.st_size,z.st_mtime_ns):raise RuntimeError('Archive independent reread failed')
os.link(t,p);t.unlink();d=os.open(p.parent,os.O_RDONLY)
try:os.fsync(d)
finally:os.close(d)
print(json.dumps(dict(path=str(p),bytes=n,sha256=h.hexdigest(),independent_reread=True,atomic_no_clobber_published=True)))
'''
_GET=r'''import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);a=p.stat();h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
z=p.stat()
if (a.st_ino,a.st_size,a.st_mtime_ns)!=(z.st_ino,z.st_size,z.st_mtime_ns):raise RuntimeError('Archive changed before read')
sys.stdout.buffer.write((json.dumps(dict(path=str(p),bytes=z.st_size,sha256=h.hexdigest()))+'\n').encode());sys.stdout.buffer.flush()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(b)
sys.stdout.buffer.flush();q=p.stat()
if (z.st_ino,z.st_size,z.st_mtime_ns)!=(q.st_ino,q.st_size,q.st_mtime_ns):raise RuntimeError('Archive changed during read')
'''
_PREFLIGHT=r'''import json,os,sys
from pathlib import Path
c=json.loads(sys.argv[1]);p=Path(c['archive_root']);p.mkdir(parents=True,exist_ok=True);card=p/c['card_key'];card.mkdir(exist_ok=True)
v=os.statvfs(p);free=v.f_bavail*v.f_frsize;used=sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
remaining=max(0,int(c['global_reservation_bytes'])-used);need=remaining+int(c['global_headroom_bytes'])
if free<need:raise RuntimeError('Archive global remaining reservation/headroom unavailable')
print(json.dumps(dict(status='PASS',available_bytes=free,required_quota_bytes=c['required_quota_bytes'],global_reservation_bytes=c['global_reservation_bytes'],global_headroom_bytes=c['global_headroom_bytes'],existing_run_bytes=used,remaining_global_reservation_bytes=remaining,remote_card_root=str(card))))
'''

class SSHArchiveTransport:
 def __init__(self,config):self.config=config;self.timeout=float(config.get('io_timeout_seconds',300))
 def _command(self,code,args):
  command=shlex.join([self.config.get('remote_python','python3'),'-B','-c',code]+[str(x) for x in args])
  return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S',self.config['control_path'],self.config['host'],command]
 def _start(self,code,args,write=False):
  return subprocess.Popen(self._command(code,args),stdin=subprocess.PIPE if write else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
 def put_immutable(self,path,chunks,expected_bytes,expected_sha256):
  p=self._start(_PUT,[path,expected_bytes,expected_sha256,self.config['global_headroom_bytes']],True)
  try:
   os.set_blocking(p.stdin.fileno(),False)
   n=0;h=hashlib.sha256()
   for block in chunks:
    view=memoryview(block);h.update(view);n+=len(view)
    while view:
     require(select.select([], [p.stdin.fileno()], [],self.timeout)[1],'Archive write inactivity timeout')
     try:count=os.write(p.stdin.fileno(),view[:1048576])
     except (BlockingIOError,InterruptedError):continue
     require(count>0,'Archive write pipe closed');view=view[count:]
   require(n==expected_bytes and h.hexdigest()==expected_sha256,'Local archive stream identity changed')
   p.stdin.close();p.stdin=None;out,err=p.communicate(timeout=self.timeout)
   require(p.returncode==0,'Archive write failed: '+err.decode(errors='replace')[:2000]);r=json.loads(out)
   require(r['bytes']==expected_bytes and r['sha256']==expected_sha256 and r['independent_reread'] and r['atomic_no_clobber_published'],'Archive write receipt mismatch')
   return r
  except BaseException:
   p.kill();p.wait();raise
 def open_verified(self,path,expected_bytes,expected_sha256):return _RemoteReader(self,path,expected_bytes,expected_sha256)
 def preflight(self):
  p=self._start(_PREFLIGHT,[json.dumps(self.config,sort_keys=True)])
  try:out,err=p.communicate(timeout=self.timeout)
  except BaseException:p.kill();p.wait();raise
  require(p.returncode==0,'Archive preflight failed: '+err.decode(errors='replace')[:2000]);return json.loads(out)

class _RemoteReader:
 def __init__(self,transport,path,expected_bytes,expected_sha256):
  self.transport=transport;self.p=transport._start(_GET,[path]);self.n=0;self.h=hashlib.sha256();self.expected_bytes=expected_bytes;self.expected_sha=expected_sha256;self.eof=False;self.closed=False
  try:
   header=bytearray()
   while not header.endswith(b'\n'):
    block=self._pipe_read(1);require(block,'Archive header truncated');header.extend(block);require(len(header)<=16384,'Archive header excessive')
   r=json.loads(header);require(r['bytes']==expected_bytes and r['sha256']==expected_sha256,'Archive fresh independent identity mismatch')
  except BaseException:self._abort();raise
 def _pipe_read(self,n):
  require(select.select([self.p.stdout.fileno()],[],[],self.transport.timeout)[0],'Archive read inactivity timeout')
  return os.read(self.p.stdout.fileno(),min(n,1048576))
 def read(self,n):
  require(type(n) is int and n>=0,'Bounded archive read size required');out=bytearray()
  while len(out)<n and not self.eof:
   block=self._pipe_read(n-len(out))
   if not block:self.eof=True;break
   out.extend(block);self.n+=len(block);self.h.update(block)
   require(self.n<=self.expected_bytes,'Archive trailing bytes beyond receipt')
  return bytes(out)
 def tell(self):return self.n
 def _abort(self):
  if not self.closed:
   if self.p.poll() is None:self.p.kill()
   self.p.wait();self.p.stdout.close();self.p.stderr.close();self.closed=True
 def __enter__(self):return self
 def __exit__(self,typ,value,tb):
  if typ is not None:self._abort();return False
  try:
   if not self.eof:require(not self.read(1),'Archive unexpected trailing byte')
   require(self.n==self.expected_bytes and self.h.hexdigest()==self.expected_sha,'Archive streamed SHA/bytes mismatch')
   self.p.wait(timeout=self.transport.timeout);err=self.p.stderr.read();require(self.p.returncode==0,'Archive read process failed: '+err.decode(errors='replace')[:2000])
  finally:self._abort()

TRANSPORT_FACTORY=SSHArchiveTransport
def archive_preflight(root,card=None):
 c=load_archive_config(root)
 if card is not None:
  key=card if isinstance(card,str) else ('gpu'+str(card.get('physical_index',card.get('index'))))
  require(key==c['card_key'],'Archive card mapping mismatch')
 r=TRANSPORT_FACTORY(c).preflight();require(r.get('status')=='PASS','Archive preflight not PASS');return dict(r,archive_config_sha256=c['config_sha256'])
def store_for_unit(unit_dir):
 root=Path(unit_dir).parents[2]
 if not (root/'ARCHIVE_CONFIG.json').is_file():return None
 c=load_archive_config(root);return ArchiveStore(c,TRANSPORT_FACTORY(c),Path(unit_dir).relative_to(root))
class ArchiveStore:
 def __init__(self,config,transport,unit_relative):self.config=config;self.transport=transport;self.unit_relative=PurePosixPath(str(unit_relative));require('..' not in self.unit_relative.parts,'Unsafe unit archive path')
 def path(self,name):
  require(Path(name).name==name,'Unsafe archive file name');return str(PurePosixPath(self.config['archive_root'])/self.config['card_key']/self.unit_relative/name)
 def put(self,name,raws):
  n=sum(len(x) for x in raws);h=hashlib.sha256()
  for x in raws:h.update(x)
  r=self.transport.put_immutable(self.path(name),iter(raws),n,h.hexdigest());return dict(r,host=self.config['host'],archive_config_sha256=self.config['config_sha256'],card_key=self.config['card_key'])
 def open(self,receipt):
  require(receipt['host']==self.config['host'] and receipt['card_key']==self.config['card_key'],'Archive receipt host/card mismatch')
  require(receipt['archive_config_sha256']==self.config['config_sha256'],'Archive config identity changed')
  require(str(PurePosixPath(receipt['path']).parent)==str(PurePosixPath(self.config['archive_root'])/self.config['card_key']/self.unit_relative),'Archive receipt path out of unit scope')
  return self.transport.open_verified(receipt['path'],receipt['bytes'],receipt['sha256'])
def open_receipt_raw(directory,item,failure=False):
 key='failure_raw_archive' if failure else 'raw_archive'
 if key in item:
  a=store_for_unit(directory);require(a is not None,'Archive config missing for remote raw');return a.open(item[key])
 return (Path(directory)/item['failure_raw_file' if failure else 'raw_file']).open('rb')
def archive_progress(directory,token,receipt,role):
 p=Path(directory)/'ARCHIVE_PROGRESS.jsonl';raw=json.dumps(dict(token=token,role=role,unix=time.time(),receipt=receipt),sort_keys=True,separators=(',',':')).encode()+b'\n'
 fd=os.open(p,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
 try:os.write(fd,raw);os.fsync(fd)
 finally:os.close(fd)

def enumerate_archive_objects(root):
 """Metadata-only expected-object catalog, never reads large raw on this host."""
 rows={}
 for p in sorted(Path(root).rglob('token_*.json')):
  item=json.loads(p.read_text())
  for key in ('raw_archive','failure_raw_archive'):
   if key not in item:continue
   x=dict(item[key],local_token_receipt=str(p),role=key);identity=(x['host'],x['path'])
   if identity in rows:require((rows[identity]['bytes'],rows[identity]['sha256'])==(x['bytes'],x['sha256']),'Conflicting archived-object receipts')
   else:rows[identity]=x
 return list(rows.values())

_FINAL_VERIFY=r'''import hashlib,json,os,sys,uuid
from pathlib import Path
v=json.load(sys.stdin);rows=[]
for x in v['objects']:
 p=Path(x['path']);a=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 z=p.stat();stable=(a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 if not stable or z.st_size!=x['bytes'] or h.hexdigest()!=x['sha256']:raise RuntimeError('Final archive raw identity mismatch:'+str(p))
 rows.append(dict(path=str(p),bytes=z.st_size,sha256=h.hexdigest(),fresh_independent_stream_SHA=True))
 print(json.dumps(dict(event='ARCHIVE_FINAL_HASH_PROGRESS',path=str(p),bytes=z.st_size,sha256=h.hexdigest())),flush=True)
result=dict(status='PASS_FRESH_REMOTE_RAW_SHA_BYTES',objects=rows)
if v['publish_catalog']:
 p=Path(v['catalog_path']);raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n').encode();t=p.with_name(p.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
 with t.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.link(t,p);t.unlink();d=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(d)
 finally:os.close(d)
 check=p.read_bytes()
 if check!=raw:raise RuntimeError('Archive catalog reread changed')
 result['catalog']=dict(path=str(p),bytes=len(check),sha256=hashlib.sha256(check).hexdigest(),atomic_no_clobber_published=True)
print(json.dumps(result))
'''
def final_verify_archive(root,publish_catalog=True):
 """One SSH batch per per-card config; freshly streams every expected raw SHA.

Called only once at final seal when publish_catalog=True. Existing catalogs
are immutable and a repeated publishing call is rejected, never overwritten.
"""
 objects=enumerate_archive_objects(root);groups={}
 for x in objects:
  directory=Path(x['local_token_receipt']).parent;c=load_archive_config(directory.parents[2]);require(x['archive_config_sha256']==c['config_sha256'],'Final archive config mismatch')
  g=groups.setdefault(c['config_sha256'],dict(config=c,objects=[]));g['objects'].append(x)
 out=[]
 for g in groups.values():
  c=g['config'];transport=TRANSPORT_FACTORY(c)
  payload=dict(objects=[{k:x[k] for k in ('path','bytes','sha256')} for x in g['objects']],publish_catalog=publish_catalog,catalog_path=str(PurePosixPath(c['archive_root'])/c['card_key']/'ASSET_CATALOG.json'))
  if hasattr(transport,'verify_objects'):r=transport.verify_objects(payload)
  else:
   p=transport._start(_FINAL_VERIFY,[],True)
   try:
    sending=memoryview(json.dumps(payload,sort_keys=True).encode());os.set_blocking(p.stdin.fileno(),False)
    while sending:
     require(select.select([],[p.stdin.fileno()],[],transport.timeout)[1],'Final archive catalog input inactivity timeout')
     try:sent=os.write(p.stdin.fileno(),sending[:1048576])
     except (BlockingIOError,InterruptedError):continue
     require(sent>0,'Final archive catalog input pipe closed');sending=sending[sent:]
    p.stdin.close();p.stdin=None
    pending=bytearray();r=None
    while True:
     require(select.select([p.stdout.fileno()],[],[],transport.timeout)[0],'Final archive hash inactivity timeout')
     block=os.read(p.stdout.fileno(),1048576)
     if not block:break
     pending.extend(block);require(len(pending)<=16000000,'Final archive result exceeds bounded metadata buffer')
     while b'\n' in pending:
      line,_,rest=pending.partition(b'\n');pending=bytearray(rest);value=json.loads(line)
      if value.get('event')!='ARCHIVE_FINAL_HASH_PROGRESS':r=value
    require(not pending,'Final archive result truncated');p.wait(timeout=transport.timeout);err=p.stderr.read()
   except BaseException:p.kill();p.wait();raise
   finally:p.stdout.close();p.stderr.close()
   require(p.returncode==0 and r is not None,'Final archive verification failed:'+err.decode(errors='replace')[:2000])
  require(r['status']=='PASS_FRESH_REMOTE_RAW_SHA_BYTES' and len(r['objects'])==len(g['objects']),'Final archive verification receipt invalid');out.append(dict(r,card_key=c['card_key'],archive_config_sha256=c['config_sha256']))
 return dict(status='PASS_FRESH_REMOTE_ARCHIVE_VERIFICATION',objects=len(objects),cards=out,large_raw_downloaded=False)
