"""Once-only CPU archive publication, dynamic inventory and exact control bytes.

No Torch, GPU, credentials, old-root writes, retry, or model download. The29
archive independently hashes every received file, rejects unknown members,
and publishes using real renameat2(RENAME_NOREPLACE). Original29 JSON bytes
are mirrored to55 verbatim. Every exclusion is an exact root-relative path.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

CONTROLS = {"SHA256_MANIFEST.json", "SHA256SUMS", "FINAL_ARCHIVE_VERIFICATION.json", "FINAL_SEAL_RECEIPT.json"}
POST_SEAL = {"FINISHER_RESULT.json", "FINISHER_FAILURE.json", "operations/ATOMIC_ARCHIVE_PUBLICATION.json"}
ARCHIVE_BASE = Path("/data/zypan/qwen_oscar_resident_rotation_efficient_qualification_archive")
SSH = ["ssh", "-S", "/tmp/qwen_replication_archive29.sock", "-o", "BatchMode=yes",
       "-o", "ControlMaster=no", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=6", "zypan@202.38.247.29"]
RAW_SUFFIXES = {".bin", ".npz", ".npy", ".pt", ".pth", ".f32", ".safetensors"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def safe_relative(name):
    value = PurePosixPath(name)
    return (isinstance(name, str) and bool(name) and not value.is_absolute()
            and not any(part in ("..", ".") for part in name.split("/"))
            and "\n" not in name and "\r" not in name and "\0" not in name)


def live_descriptor_exclusions(root):
    root = Path(root).resolve()
    result = {}
    for fd in (1, 2):
        target = os.readlink("/proc/self/fd/" + str(fd))
        if not target.startswith("/"):
            continue
        path = Path(target).resolve()
        if path.is_relative_to(root):
            relative = path.relative_to(root).as_posix()
            require(relative not in CONTROLS | {"FINISHER_STARTED.json", "TASK_RESULT.json"},
                    "Finisher live output must not target a science/seal receipt")
            result[relative] = "Exact live fd%d, preserved separately after true finisher exit" % fd
    return result


def catalog(root, live_exclusions):
    root = Path(root).resolve()
    exclusions = CONTROLS | POST_SEAL | set(live_exclusions)
    rows = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        require(not path.is_symlink(), "Symlink refused in science catalog: " + name)
        if path.is_dir():
            continue
        require(path.is_file() and safe_relative(name), "Unsafe/nonregular science member: " + name)
        if name in exclusions:
            continue
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        require((before.st_size, before.st_ino, before.st_mtime_ns) ==
                (after.st_size, after.st_ino, after.st_mtime_ns), "Science changed during catalog: " + name)
        rows.append(dict(path=name, bytes=before.st_size, sha256=digest))
    require(rows and sum(row["bytes"] for row in rows) <= 20 * 1024 ** 3,
            "Empty science or publication exceeds20GiB bound")
    require(len({row["path"] for row in rows}) == len(rows), "Duplicate science filename")
    return rows, sorted(exclusions)


def verify_rows(root, rows):
    root = Path(root).resolve()
    for row in rows:
        require(safe_relative(row["path"]), "Unsafe catalog path")
        path = root / row["path"]
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
                and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Science bytes changed after snapshot: " + row["path"])


def decode_control(record):
    data = base64.b64decode(record["base64"], validate=True)
    require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
            "Archive original control bytes/SHA differ")
    return data


REMOTE_COMMON = r'''
from pathlib import Path
import base64,ctypes,errno,hashlib,json,os,socket,sys,tarfile,tempfile,time
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_new(p,b):
 with Path(p).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def fsyncdir(p):
 d=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(d)
 finally:os.close(d)
def rename_noreplace(a,b):
 fn=ctypes.CDLL(None,use_errno=True).renameat2
 fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 if fn(-100,os.fsencode(a),-100,os.fsencode(b),1):raise OSError(ctypes.get_errno(),'renameat2_RENAME_NOREPLACE')
def safe(n):
 p=Path(n)
 return bool(n) and not p.is_absolute() and all(x not in ('..','.') for x in n.split('/')) and '\n' not in n and '\r' not in n and '\x00' not in n
def inventory(r,expected):
 allowed_dirs={'.'}
 for n in expected:
  assert safe(n)
  allowed_dirs.update(str(p) for p in Path(n).parents)
 found=set()
 for p in r.rglob('*'):
  assert not p.is_symlink(),'symlink archive member'
  n=str(p.relative_to(r))
  if p.is_file():found.add(n)
  else:assert p.is_dir() and n in allowed_dirs,'unexpected directory/nonregular member'
 assert found==set(expected),'unknown or missing archive file'
 return found
'''


def on29(code):
    result = subprocess.run(SSH + ["/usr/bin/python3", "-"], input=code, text=True, capture_output=True)
    require(result.returncode == 0, "Archive29 step failed; preserve stage/no retry: " + result.stderr[-5000:])
    return json.loads(result.stdout)


def paths(root):
    archive = ARCHIVE_BASE / Path(root).name
    return archive, archive.with_name(archive.name + ".partial_once")


def preflight_program(archive, stage):
    return REMOTE_COMMON + "\narchive=Path(" + repr(str(archive)) + ");stage=Path(" + repr(str(stage)) + ")\n" + r'''
archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists() and not stage.exists(),'no overwrite or implicit stage retry'
s=os.statvfs(archive.parent);free=s.f_bavail*s.f_frsize
assert free>=120*1024**3,'20GiB archive plus100GiB free headroom required'
with tempfile.TemporaryDirectory(prefix='resident_actual_atomic_',dir=str(archive.parent)) as d:
 p=Path(d);a=p/'source';b=p/'existing';c=p/'new'
 a.write_bytes(b'ORIGINAL_SOURCE');b.write_bytes(b'EXISTING_KEEP')
 try:rename_noreplace(a,b)
 except OSError as e:assert e.errno==errno.EEXIST
 else:raise AssertionError('NOREPLACE overwrote an existing destination')
 assert a.read_bytes()==b'ORIGINAL_SOURCE' and b.read_bytes()==b'EXISTING_KEEP'
 rename_noreplace(a,c);assert not a.exists() and c.read_bytes()==b'ORIGINAL_SOURCE'
print(json.dumps(dict(status='PASS_ACTUAL29_RENAMEAT2_POSITIVE_NEGATIVE_AND_SPACE',host=socket.gethostname(),free_bytes=free)))
'''


def initialize_program(archive, stage, manifest):
    return (REMOTE_COMMON + "\nr=Path(" + repr(str(stage)) + ");final=Path(" + repr(str(archive)) + ")\n"
            + "raw=" + repr(canonical(manifest)) + "\n" + r'''
assert not r.exists() and not final.exists()
m=json.loads(raw);s=os.statvfs(r.parent)
assert m['bytes']<=20*1024**3 and s.f_bavail*s.f_frsize>=120*1024**3
r.mkdir(exist_ok=False);fsyncdir(r.parent)
write_new(r/'SHA256_MANIFEST.json',raw)
write_new(r/'SHA256SUMS',''.join(x['sha256']+'  '+x['path']+'\n' for x in m['files']).encode())
fsyncdir(r)
print(json.dumps(dict(manifest_sha256=sha(r/'SHA256_MANIFEST.json'),sums_sha256=sha(r/'SHA256SUMS'))))
''')


def receiver_program(stage):
    return REMOTE_COMMON + "\nr=Path(" + repr(str(stage)) + ")\n" + r'''
m=json.loads((r/'SHA256_MANIFEST.json').read_text());expected={x['path']:x for x in m['files']};seen=set()
assert len(expected)==len(m['files'])
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  name=member.name
  assert safe(name) and name in expected and name not in seen and member.isfile()
  assert member.size==expected[name]['bytes']
  p=r/name;assert p.resolve().is_relative_to(r.resolve())
  p.parent.mkdir(parents=True,exist_ok=True);source=archive.extractfile(member);assert source is not None
  total=0
  with p.open('xb') as dest:
   for b in iter(lambda:source.read(1024*1024),b''):dest.write(b);total+=len(b)
   dest.flush();os.fsync(dest.fileno())
  assert total==expected[name]['bytes'];seen.add(name)
assert seen==set(expected)
print(json.dumps(dict(status='PASS_UNCOMPRESSED_RECEIVER_ALL_EXPECTED',count=len(seen))))
'''


def transfer(root, stage, rows):
    command = SSH + ["exec /usr/bin/python3 -c " + shlex.quote(receiver_program(stage))]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|", dereference=True) as archive:
            for row in rows:
                path = Path(root) / row["path"]
                before = path.stat()
                require(before.st_size == row["bytes"] and sha(path) == row["sha256"], "Source changed before transfer")
                archive.add(path, arcname=row["path"], recursive=False)
                after = path.stat()
                require((before.st_size, before.st_ino, before.st_mtime_ns) ==
                        (after.st_size, after.st_ino, after.st_mtime_ns) and sha(path) == row["sha256"],
                        "Source changed during transfer")
        process.stdin.close(); process.stdin = None
        stdout, stderr = process.communicate()
        require(process.returncode == 0, "Archive stream failed; preserve partial: " + stderr.decode(errors="replace")[-5000:])
        result = json.loads(stdout)
        require(result["count"] == len(rows), "Archive received count differs")
        return result
    except BaseException:
        if process.stdin is not None:
            process.stdin.close(); process.stdin = None
        process.communicate()
        raise


def publish_program(archive, stage, initial, manifest):
    return (REMOTE_COMMON + "\nr=Path(" + repr(str(stage)) + ");final=Path(" + repr(str(archive)) + ")\n"
            + "initial=" + repr(initial) + "\nexpected=" + repr(manifest) + "\n" + r'''
assert not final.exists()
assert sha(r/'SHA256_MANIFEST.json')==initial['manifest_sha256'] and sha(r/'SHA256SUMS')==initial['sums_sha256']
m=json.loads((r/'SHA256_MANIFEST.json').read_text());assert m==expected
names={x['path'] for x in m['files']};assert len(names)==m['count']==len(m['files'])
assert sum(x['bytes'] for x in m['files'])==m['bytes']<=20*1024**3
assert sha(r/'SOURCE_FREEZE.json')==m['source_freeze_sha256']
assert (r/'SHA256SUMS').read_bytes()==''.join(x['sha256']+'  '+x['path']+'\n' for x in m['files']).encode()
verified=[]
for x in m['files']:
 p=r/x['path'];assert safe(x['path']) and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];verified.append(x)
inventory(r,names|{'SHA256_MANIFEST.json','SHA256SUMS'})
verification=dict(status='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_FRESH_SHA_AND_BYTES',
 host=socket.gethostname(),task_id=m['task_id'],execution_attempt=m['execution_attempt'],terminal=m['terminal'],
 count=len(verified),bytes=m['bytes'],files=verified,source_freeze_sha256=m['source_freeze_sha256'],
 manifest_sha256=sha(r/'SHA256_MANIFEST.json'),monotonic=time.monotonic())
write_new(r/'FINAL_ARCHIVE_VERIFICATION.json',canonical(verification))
seal=dict(status='PASS_ATOMIC_NOREPLACE_SEALED_SCIENCE',task_id=m['task_id'],execution_attempt=m['execution_attempt'],
 terminal=m['terminal'],formal_completed=m['formal_completed'],planned_formal=60,archive_root=str(final),
 source_freeze_sha256=m['source_freeze_sha256'],manifest_sha256=sha(r/'SHA256_MANIFEST.json'),
 SHA256SUMS_sha256=sha(r/'SHA256SUMS'),archive_verification_sha256=sha(r/'FINAL_ARCHIVE_VERIFICATION.json'),
 atomic='renameat2_RENAME_NOREPLACE',old_science_terminations_unchanged=True)
write_new(r/'FINAL_SEAL_RECEIPT.json',canonical(seal))
controls={'SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json'}
inventory(r,names|controls)
for x in verified:assert (r/x['path']).stat().st_size==x['bytes'] and sha(r/x['path'])==x['sha256']
fsyncdir(r);rename_noreplace(r,final);fsyncdir(final.parent)
out={}
for n in sorted(controls):
 b=(final/n).read_bytes();out[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
print(json.dumps(dict(status='PASS_INDEPENDENT29_ATOMIC_NOREPLACE_PUBLICATION',archive_root=str(final),
 count=m['count'],bytes=m['bytes'],terminal=m['terminal'],controls=out)))
''')


def publish(root, *, task_id, terminal, formal_completed, finisher_identity):
    root = Path(root).resolve()
    require(not any((root / name).exists() for name in CONTROLS), "Science seal already exists; no republish")
    archive, stage = paths(root)
    live = live_descriptor_exclusions(root)
    preflight = on29(preflight_program(archive, stage))
    write_new(root / "operations/ARCHIVE_PUBLICATION_PREFLIGHT.json", canonical(preflight))
    rows, exclusions = catalog(root, live)
    raw = [row for row in rows if Path(row["path"]).suffix.lower() in RAW_SUFFIXES]
    manifest = dict(task_id=task_id, execution_attempt=root.name, terminal=terminal,
        source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"), formal_completed=formal_completed,
        count=len(rows), bytes=sum(row["bytes"] for row in rows), files=rows,
        raw_remote_only_files=raw, raw_remote_only_count=len(raw),
        excluded_exact_relative_paths=exclusions, excluded_live_descriptors=live,
        exclusion_scope="exact root-relative paths only; nested original seal/metadata are included",
        finisher_identity=finisher_identity)
    initial = on29(initialize_program(archive, stage, manifest))
    received = transfer(root, stage, rows)
    verify_rows(root, rows)
    result = on29(publish_program(archive, stage, initial, manifest))
    controls = {}
    require(set(result["controls"]) == CONTROLS, "Archive returned a different control set")
    for name in sorted(CONTROLS):
        data = decode_control(result["controls"][name])
        write_new(root / name, data)
        controls[name] = dict(bytes=len(data), sha256=sha(root / name))
    receipt = dict(status=result["status"], archive_root=result["archive_root"],
        terminal=terminal, formal_completed=formal_completed, science_count=len(rows),science_bytes=manifest["bytes"],
        dynamic_source_count=len(json.loads((root / "SOURCE_FREEZE.json").read_text())["files"]),
        raw_remote_only_count=len(raw), controls=controls, transfer=received,
        exact_original29_control_bytes_mirrored=True, local_small_delivery_still_required=True)
    write_new(root / "operations/ATOMIC_ARCHIVE_PUBLICATION.json", canonical(receipt))
    return receipt
