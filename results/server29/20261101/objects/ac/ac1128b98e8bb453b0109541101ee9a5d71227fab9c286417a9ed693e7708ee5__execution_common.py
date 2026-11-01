"""CPU identity, immutable source verification and original model pins."""
import hashlib,json,os,stat,subprocess,time
from pathlib import Path
UUIDS=['GPU-a78176ed-f97f-c54f-6201-5a01b570f422','GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3','GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31','GPU-30b07aba-a728-7649-4899-e09d33f8bc75']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
    cmd=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
    after=(p/'stat').read_text();w=after[after.rfind(')')+2:].split()
    assert int(v[19])==int(w[19]),'PID epoch changed during identity observation'
    return dict(pid=pid,ppid=int(w[1]),starttick=int(w[19]),state=w[0],cmdline=cmd)
def same_process(a,b):return all(a[k]==b[k] for k in ['pid','ppid','starttick','cmdline'])
def apps():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[]
    for line in p.stdout.splitlines():
        v=[x.strip() for x in line.split(',')]
        if len(v)==3:rows.append(dict(pid=int(v[0]),uuid=v[1],memory_MiB=v[2]))
    return rows
def source(root,expected):
    if not __debug__:raise RuntimeError('Python optimization would disable source/control assertions')
    root=Path(root);p=root/'FORMAL_SOURCE_MANIFEST.json';assert sha(p)==expected,'Source manifest identity changed'
    for r in json.loads(p.read_text())['files']:
        q=root/r['path'];assert q.is_file() and q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'Frozen source/input changed: '+r['path']
def model_pins(root,full=False):
    rows=json.loads((Path(root)/'analysis/MODEL_TOKENIZER_FRESH_SHA.json').read_text())['files'];out=[]
    for r in rows:
        p=Path(r['path']);s=p.stat();assert s.st_size==r['bytes'],'Model byte size changed'
        if full:assert sha(p)==r['sha256'],'Model/tokenizer bytes changed'
        out.append(dict(path=str(p),bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,expected_sha256=r['sha256'],fresh_full_SHA=full))
    return out
def regular_maps():
    rows=[];seen=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(None,5)
        if len(fields)!=6 or not fields[5].startswith('/'):continue
        name=fields[5]
        if name.endswith(' (deleted)'):
            if name.startswith(('/dev/','/memfd:')):continue
            raise RuntimeError('Deleted regular mapped identity unresolved: '+name)
        p=Path(name)
        try:s=p.stat()
        except FileNotFoundError:raise RuntimeError('Mapped path vanished: '+name)
        if not stat.S_ISREG(s.st_mode):continue
        key=(s.st_dev,s.st_ino)
        if key in seen:continue
        seen.add(key);rows.append(dict(path=name,bytes=s.st_size,device=s.st_dev,inode=s.st_ino,sha256=sha(p)))
    return rows
