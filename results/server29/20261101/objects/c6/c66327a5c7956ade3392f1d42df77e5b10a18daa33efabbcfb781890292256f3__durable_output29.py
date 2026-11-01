"""CPU-only, once-per-closed-unit SHA/byte publication to fixed server29 storage.

This module does not import torch, inspect GPU devices, launch workers, retry
network writes, delete scratch, or modify prior scientific archives.
"""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

TASK = 'QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1'
REVISION = TASK + '_REUSE48_ARCHIVE29_20261025'
REMOTE_ROOT = '/dev/shm/qwen_wscale_rowbal_reuse48_archive29_20261025'
ARCHIVE_ROOT = '/data/zypan/qwen_gdn_wscale_rowbal_archive/' + REVISION
LIVE_ROOT = '/data/zypan/qwen_gdn_wscale_rowbal_archive/LIVE_' + REVISION
SOCKET = '/tmp/qwen_replication_archive29.sock'
HOST = 'zypan@202.38.247.29'
_MANIFEST = 'DURABLE29_MANIFEST.json'
_RECEIPT = 'DURABLE29_RECEIPT.json'
_CLAIM = 'DURABLE29_STARTED.json'
_THREAD_LOCK = threading.Lock()


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    with Path(path).open('rb') as stream:
        return json.load(stream)


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(_canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def _identity(pid):
    p = Path('/proc') / str(pid)
    return dict(pid=pid, starttick=int((p/'stat').read_text().rsplit(')', 1)[1].split()[19]),
                cmdline=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v])


def _configuration(root):
    root = Path(root).resolve()
    _require(str(root) == REMOTE_ROOT, 'Durable29 exact authorized compute root required')
    grant = _read(root/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json')
    metadata = _read(root/'ROOT.json')
    permit = _read(root/'GPU_PERMIT.json')
    _require(grant['status'] == 'AUTHORIZED_ONE_NEW_INDEPENDENT_REUSE48_STORAGE29_CONTINUATION'
             and grant['new_remote_root'] == REMOTE_ROOT and grant['task_id'] == TASK
             and grant['archive29_root'] == ARCHIVE_ROOT and grant['live_archive29_root'] == LIVE_ROOT
             and grant['server29_CPU_only_no_GPU_queries_or_work'] is True,
             'Durable29 human grant/root/storage scope differs')
    _require(metadata['remote_root'] == REMOTE_ROOT and metadata['archive_root'] == ARCHIVE_ROOT
             and metadata['live_archive_root'] == LIVE_ROOT,
             'Durable29 ROOT storage scope differs')
    source = _sha(root/'SOURCE_FREEZE.json')
    _require(permit['execution_root'] == REMOTE_ROOT and permit['task_id'] == TASK
             and permit['source_freeze_sha256'] == source,
             'Durable29 source/permit root differs')
    return dict(task_id=TASK, execution_revision=REVISION, compute_root=REMOTE_ROOT,
                live_root=LIVE_ROOT, archive_root=ARCHIVE_ROOT,
                source_freeze_sha256=source, permit_sha256=_sha(root/'GPU_PERMIT.json'),
                human_authorization_sha256=_sha(root/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'))


def _collect(target):
    rows = []
    for p in sorted(Path(target).rglob('*')):
        _require(not p.is_symlink(), 'Unit publication refuses symlink: ' + str(p))
        if not p.is_file() or p.name in (_MANIFEST, _RECEIPT):
            continue
        before = p.stat()
        digest = _sha(p)
        after = p.stat()
        _require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                 (after.st_ino, after.st_size, after.st_mtime_ns),
                 'Closed unit changed during SHA: ' + str(p))
        rows.append(dict(path=p.relative_to(target).as_posix(), bytes=after.st_size,
                         sha256=digest))
    _require(rows, 'Empty closed-unit publication refused')
    return rows


# This exact CPU program is used in production and exercised by isolated CPU
# tests. It accepts only the one fixed live root; tests virtualize that Path.
REMOTE_PROGRAM = r'''
import base64,ctypes,hashlib,json,os,sys
from pathlib import Path
LIVE='/data/zypan/qwen_gdn_wscale_rowbal_archive/LIVE_QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_REUSE48_ARCHIVE29_20261025'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def new(p,x):
 with p.open('xb') as f:f.write(canonical(x)+b'\n');f.flush();os.fsync(f.fileno())
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def rename_noreplace(stage,final):
 lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2
 fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 rc=fn(-100,os.fsencode(stage),-100,os.fsencode(final),1)
 require(rc==0,'renameat2 NOREPLACE failed: '+os.strerror(ctypes.get_errno()));fsyncdir(final.parent)
cfg=json.loads(base64.b64decode(sys.argv[2]));action=sys.argv[1]
require(action in ('reserve','seal'),'Unsupported CPU archive action')
require(cfg['live_root']==LIVE and cfg['compute_root']=='/dev/shm/qwen_wscale_rowbal_reuse48_archive29_20261025','Exact authorized live/compute root differs')
unit=cfg['unit'];require(unit and '/' not in unit and unit not in ('.','..') and all(c.isalnum() or c in '_-' for c in unit),'Unsafe unit')
root=Path(LIVE);ident={k:cfg[k] for k in ('task_id','execution_revision','compute_root','live_root','archive_root','source_freeze_sha256','permit_sha256','human_authorization_sha256')}
if not root.exists():
 root.mkdir(parents=True);new(root/'LIVE_IDENTITY.json',ident);(root/'units').mkdir();(root/'staging').mkdir();fsyncdir(root)
require(root.is_dir() and not root.is_symlink() and json.loads((root/'LIVE_IDENTITY.json').read_bytes())==ident,'Live root identity differs')
final=root/'units'/unit;stage=root/'staging'/(unit+'_'+cfg['manifest_sha256'][:16])
if action=='reserve':
 require(not final.exists() and not stage.exists(),'Already published/staged unit; no duplicate or automatic retry')
 s=os.statvfs(root);require(s.f_bavail*s.f_frsize>=120*1024**3,'Durable29 reservation headroom below120GiB')
 stage.mkdir();fsyncdir(stage.parent);print(json.dumps(dict(status='NEW_UNIT_STAGE_RESERVED',stage=str(stage),final=str(final))))
else:
 require(stage.is_dir() and not stage.is_symlink(),'Unit stage absent/unsafe')
 mpath=stage/'DURABLE29_MANIFEST.json';require(sha(mpath)==cfg['manifest_sha256'],'Unit manifest SHA differs')
 m=json.loads(mpath.read_bytes());require(m['binding']==ident and m['unit']==unit and m['formal']==cfg['formal'],'Unit manifest scope differs')
 expected={x['path'] for x in m['files']}|{'DURABLE29_MANIFEST.json'};actual=set()
 for p in stage.rglob('*'):
  require(not p.is_symlink(),'Symlink in unit archive')
  if p.is_file():actual.add(p.relative_to(stage).as_posix())
 require(actual==expected,'Unit archive exact file inventory differs')
 for x in m['files']:
  p=stage/x['path'];require(p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Unit archived SHA/byte mismatch:'+x['path'])
  with p.open('rb') as f:os.fsync(f.fileno())
 receipt=dict(status='PASS_DURABLE29_CLOSED_UNIT',unit=unit,formal=cfg['formal'],manifest_sha256=cfg['manifest_sha256'],tree_sha256=hashlib.sha256(canonical(m['files'])).hexdigest(),files=len(m['files']),bytes=sum(x['bytes'] for x in m['files']),remote_root=str(final),source_freeze_sha256=cfg['source_freeze_sha256'],permit_sha256=cfg['permit_sha256'],actual_exit_returncode=m['actual_exit_returncode'],release=m['release'],server29_GPU_calls=0,rename_policy='renameat2_NOREPLACE')
 new(stage/'DURABLE29_RECEIPT.json',receipt);fsyncdir(stage);rename_noreplace(stage,final);print(json.dumps(receipt))
'''


def _remote(action, cfg):
    encoded = base64.b64encode(_canonical(cfg)).decode('ascii')
    cmd = ['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           HOST, shlex.join(['/usr/bin/python3', '-u', '-B', '-c', REMOTE_PROGRAM,
                            action, encoded])]
    p = subprocess.run(cmd, capture_output=True, timeout=300)
    _require(p.returncode == 0, 'Durable29 ' + action + ' failed: ' + p.stderr.decode(errors='replace'))
    return json.loads(p.stdout)


def publish_unit(root, target, *, unit, formal=False):
    """Publish one exited/released unit once; any error prevents continuation."""
    root, target = Path(root).resolve(), Path(target).resolve()
    _require(type(formal) is bool and isinstance(unit, str) and
             re.fullmatch(r'[A-Za-z0-9_-]+', unit) is not None,
             'Durable29 unit/formal identity invalid')
    _require(target == root/('formal' if formal else 'execution')/unit,
             'Durable29 target must be exact current unit directory')
    with _THREAD_LOCK:
        lock = root/'operations/DURABLE29_PUBLISH.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            cfg = _configuration(root)
            _require(not (target/_CLAIM).exists() and not (target/_RECEIPT).exists(),
                     'Durable29 publication already claimed; no automatic repeat')
            roster = _read(root/('FORMAL_JOB_MANIFEST.json' if formal else 'GPU_EXECUTION_PLAN.json'))
            if formal:
                jobs = roster['jobs']
            else:
                from qualification_reuse import remaining_units
                jobs = remaining_units(roster)
            matches = [j for j in jobs if j['id'] == unit]
            _require(len(matches) == 1, 'Durable29 unit absent/duplicate in exact current roster')
            interval, dispatch = _read(target/'RESOURCE_INTERVAL.json'), _read(target/'DISPATCH.json')
            _require(interval['unit'] == unit and dispatch['unit'] == unit and
                     dispatch['formal'] is formal and dispatch['gpu'] == matches[0]['gpu'] and
                     dispatch['permit_sha256'] == cfg['permit_sha256'],
                     'Durable29 closed interval/dispatch/permit differs')
            _require(interval['release'] == 'PASS', 'Unit must actually release before durable publication')
            child = interval['identity']
            try:
                live = _identity(child['pid']) == child
            except (FileNotFoundError, ProcessLookupError):
                live = False
            _require(not live, 'Unit still alive; refuse mutable live outputs')
            _new(target/_CLAIM, dict(status='STARTED_ONCE_DURABLE29_CPU_PUBLICATION',
                                    publisher_identity=_identity(os.getpid()), unit=unit,
                                    formal=formal, binding=cfg, monotonic=time.monotonic()))
            rows = _collect(target)
            manifest = dict(status='CLOSED_UNIT_SHA_BYTE_MANIFEST', binding=cfg, unit=unit,
                            formal=formal, actual_exit_returncode=interval['exit_returncode'],
                            release=interval['release'], dispatch_sha256=_sha(target/'DISPATCH.json'),
                            resource_interval_sha256=_sha(target/'RESOURCE_INTERVAL.json'), files=rows)
            _new(target/_MANIFEST, manifest)
            cfg = dict(cfg, unit=unit, formal=formal, manifest_sha256=_sha(target/_MANIFEST))
            reserved = _remote('reserve', cfg)
            _require(reserved['status'] == 'NEW_UNIT_STAGE_RESERVED', 'Durable29 stage reservation differs')
            sshcmd = shlex.join(['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15'])
            p = subprocess.run(['rsync', '-a', '-e', sshcmd, str(target)+'/',
                                HOST+':'+reserved['stage']+'/'], capture_output=True, timeout=2400)
            _require(p.returncode == 0, 'Durable29 transfer failed: ' + p.stderr.decode(errors='replace'))
            receipt = _remote('seal', cfg)
            _require(receipt['status'] == 'PASS_DURABLE29_CLOSED_UNIT' and
                     receipt['manifest_sha256'] == cfg['manifest_sha256'] and
                     receipt['unit'] == unit and receipt['permit_sha256'] == cfg['permit_sha256'],
                     'Durable29 verified receipt identity differs')
            _require(_collect(target) == rows, 'Closed unit changed during publication')
            _new(target/_RECEIPT, receipt)
            return receipt
