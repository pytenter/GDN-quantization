#!/usr/bin/env python3
"""Once-only independent CPU archive recovery after an identified tool-basename gate failure.

Default is a read-only plan. Execute only after the separately reviewed prepare,
two fresh original FrozenV4 CPU passes and partial report have genuinely exited.
No old finisher/SESSION, GPU dispatch, scientific rerun, retry or overwrite.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import traceback


ROOT = Path("/data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_gpu_continuation_20261022")
P1 = Path("/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022")
WORK = ROOT / "operations/P1_CANCELLED_CPU_DELIVERY"
TASK = "QWEN_GDN_OSCAR_P1_HUMAN_CANCELLED_PARTIAL_DELIVERY_20261022"
TERMINAL = "HUMAN_CANCELLED_P1_PARTIAL_15_OF_30_C1_S3_TECHNICAL_STOP_ZERO"
SOURCE_SHA = "3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14"
PREPARE_TOOL_SHA = "ddc35bd8b0ce7c3c0daef460185295f4928a726262fd04f70cf82bff5f4c99c8"
ARCHIVE = "/data/zypan/qwen_oscar_sym32_unified_v2_archive/" + TASK
STAGE = ARCHIVE + ".partial_once"
CONTROLS = {"SHA256_MANIFEST.json", "SHA256SUMS", "FINAL_ARCHIVE_VERIFICATION.json", "FINAL_SEAL_RECEIPT.json"}
TOOL_MANIFEST = "P1_CANCELLED_PUBLICATION_RECOVERY_V1_TOOL_MANIFEST.json"
FAILED_PUBLISHER_SHA = "a2b459a5b37165cb8fa56cd44c1d392cb8c9eb245d5d4e6bd8047607b9e0d2f9"
FAILED_RECEIPT_SHA = "987bb28eb3db6cbec70045fb5757f73a8cf5f545b09e9ff2696e693278b24377"
ACTUAL_STOP_TOOL_SHA = "42db6200363c8ee723444f7fdb243be22a54b8cd2c319f7398ddcdf980a3b070"
UNEXECUTED_STOP_CANDIDATE_SHA = "83f1f01dcb2bfc915a4f859b01dd475110cc105b6b443d453d956b8eeae7e83a"
RAW_SUFFIXES = (".bin", ".npz", ".npy", ".f32", ".pt", ".pth", ".safetensor", ".safetensors", ".ckpt", ".onnx", ".gguf")
SSH = ["ssh", "-S", "/tmp/qwen_replication_archive29.sock", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
       "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "zypan@202.38.247.29"]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def save(name, obj):
    write_new(WORK / name, canonical(obj))


def identity():
    raw = Path("/proc/self/stat").read_text()
    return dict(pid=os.getpid(), starttick=int(raw[raw.rfind(")") + 2:].split()[19]),
                cmdline=Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))


def safe(name):
    path = Path(name)
    require(isinstance(name, str) and name and not path.is_absolute() and str(path) == name
            and all(x not in (".", "..") for x in path.parts) and "\n" not in name and "\r" not in name,
            "Unsafe manifest path")
    return path


def checked_file(path):
    require(not path.is_symlink() and path.is_file(), "Only regular independent snapshot files accepted:" + str(path))
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_ino, before.st_mtime_ns) == (after.st_size, after.st_ino, after.st_mtime_ns),
            "File changing during CPU snapshot:" + str(path))
    return dict(source_path=str(path), bytes=after.st_size, sha256=digest)


def module():
    path = ROOT / "operations/p1_cancelled_delivery.py"
    require(sha(path) == PREPARE_TOOL_SHA, "Original separately reviewed partial scoring tool changed")
    spec = importlib.util.spec_from_file_location("independent_p1_partial_validation", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def validate_partial(args):
    m = module()
    manifest, stop = m.manifest_and_stop(args, WORK)
    control, validate = m.original_bindings()
    released = m.own_exit_check(control, stop)
    require(m.checked_payloads(control, validate, stop) == manifest["records"], "Original legal15 inventory drifted")
    result = load(WORK / "CPU_PARTIAL_SCORE_AND_REPORT_RESULT.json")
    require(result["status"] == "PASS_TWO_FRESH_CPU_PARTIAL15_SCORE_REPORT_ONLY"
            and result["task_id"] == TASK and result["terminal"] == TERMINAL
            and result["legal_complete"] == 15 and result["planned_P1"] == 30 and result["full30_accuracy"] is None
            and result["C1_S3_formal_counts"] == {"C1": 0, "S3": 0} and result["incomplete_scored"] == 0,
            "Actual partial-only science report is incomplete or makes full30 claims")
    processes = [load(WORK / f"SCORE_PASS_{i}_STARTED.json")["identity"] for i in (0, 1)]
    processes.append(load(WORK / "REPORT_STARTED.json")["identity"])
    require(processes[0] != processes[1] and all(not control.same_process(x) for x in processes),
            "Separate CPU scorers/report must genuinely exit before publication")
    a, b, c = [load(WORK / name) for name in ("SCORE_PASS_0.json", "SCORE_PASS_1.json", "FROZEN_V4_PARTIAL_SCORES.json")]
    require(a == b == c and len(a["scores"]) == 15 and a["cancelled_samples_scored"] == 0
            and a["full30_accuracy"] is None and a["legal_manifest_sha256"] == args.expected_manifest_sha256,
            "Two original FrozenV4 CPU score passes are not identical actual legal15")
    require(sha(ROOT / "operations/P1_AUTHORIZED_STOP/STOP_FAILURE.json") == stop["prior_stop_failure_preserved_sha256"],
            "Original stop-tool failure evidence missing or changed")
    require(stop["tool_sha256"] == ACTUAL_STOP_TOOL_SHA
            and sha(ROOT / "operations/complete_authorized_p1_stop_kernel310_reviewed.py") == stop["tool_sha256"],
            "Actual executed reviewed stop-recovery tool identity changed")
    require(sha(ROOT / "operations/complete_authorized_p1_stop_kernel310.py") == UNEXECUTED_STOP_CANDIDATE_SHA,
            "Original unexecuted stop candidate must also remain preserved")
    require(sha(ROOT / "operations/publish_p1_cancelled_cpu.py") == FAILED_PUBLISHER_SHA
            and sha(WORK / "PUBLISHER_FAILURE.json") == FAILED_RECEIPT_SHA,
            "Previous independent CPU publisher failure/source must remain byte-identical")
    previous = load(WORK / "PUBLISHER_FAILURE.json")
    previous_started = load(WORK / "PUBLISHER_STARTED.json")["identity"]
    require(previous["identity"] == previous_started and not control.same_process(previous_started),
            "Previous failed CPU publisher must actually exit")
    require(previous["error"] == "RuntimeError('Actual executed stop-recovery tool identity changed')",
            "Only the audited exact CPU path-binding failure can enter this recovery")
    require(not (WORK / "ARCHIVE_PREFLIGHT.json").exists() and not (WORK / "PUBLISHER_RESULT.json").exists(),
            "Unexpected original publication progress; do not blindly reuse or repeat an archive attempt")
    originals = load(WORK / "ORIGINAL_P1_FULL_FILE_CATALOG.json")
    actual = m.original_file_catalog()
    require(actual == originals["files"], "Original full P1 tree changed since legal snapshot; no implicit resnapshot/retry")
    current = os.statvfs(ROOT)
    require(current.f_bavail * current.f_frsize >= 8 * 1024 ** 3, "Compute disk headroom below8GiB")
    return dict(status="PASS_PARTIAL15_TWO_FRESH_ORIGINAL_FROZEN_V4_AUTHORIZED_STOP_READONLY",
                source_freeze_sha256=SOURCE_SHA, source_count=622, external_fresh_SHA_count=61, model_stat_count=17,
                actual_complete=15, originally_planned_P1=30, correct_on_completed15=result["correct_on_completed15"],
                full30_accuracy=None, C1_S3_formal_counts={"C1": 0, "S3": 0}, incomplete_scored=0,
                legal_manifest_sha256=args.expected_manifest_sha256, released=released, CPU_processes=processes), actual


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
with tempfile.TemporaryDirectory(prefix='p1_cancelled_noreplace_',dir=str(base)) as d:
 p=Path(d);a=p/'src';b=p/'existing';c=p/'new';a.write_bytes(b'SOURCE');b.write_bytes(b'KEEP')
 try:rename_noreplace(a,b)
 except OSError as e:assert e.errno==errno.EEXIST
 else:raise AssertionError('NOREPLACE overwritten')
 assert a.read_bytes()==b'SOURCE' and b.read_bytes()==b'KEEP'
 rename_noreplace(a,c);assert c.read_bytes()==b'SOURCE' and not a.exists()
print(json.dumps(dict(status='PASS_ARCHIVE20GiB_PLUS100GiB_HEADROOM_AND_ATOMIC_NOREPLACE',host=socket.gethostname(),free_bytes=free)))
'''
    return on29(code)


def live_paths():
    out = {}
    for fd in (1, 2):
        name = os.readlink(f"/proc/self/fd/{fd}")
        if name.startswith("/"):
            path = Path(name).resolve()
            require(not path.is_relative_to(P1), "Publisher log must never write inside original P1 root")
            if path.is_relative_to(WORK):
                out[str(path.relative_to(WORK))] = "Exact live independent CPU publisher output; supplemental after exit"
    return out


def build_rows(originals, excluded_live):
    rows = []
    for old in originals:
        relative = safe(old["path"])
        row = checked_file(P1 / relative)
        require(row["bytes"] == old["bytes"] and row["sha256"] == old["sha256"], "Original snapshot drifted")
        rows.append(dict(path="original_P1/" + str(relative), **row))
    excluded = CONTROLS | {"PUBLICATION_RECOVERY_V1_RESULT.json", "PUBLICATION_RECOVERY_V1_FAILURE.json", "PUBLICATION_RECOVERY_V1_ATOMIC_RECEIPT.json"} | set(excluded_live)
    for path in sorted(WORK.rglob("*")):
        require(not path.is_symlink(), "Unsafe partial work symlink")
        if path.is_dir():
            continue
        relative = str(path.relative_to(WORK))
        if relative not in excluded:
            rows.append(dict(path="cancelled_delivery/" + str(safe(relative)), **checked_file(path)))
    # Retain actual stop attempts/failures and exact tools, including the failure
    # before successful recovery. No broad operations/ snapshot while GPU work runs.
    selected = []
    for path in sorted((ROOT / "operations").iterdir()):
        name = path.name
        if name == "P1_CANCELLED_CPU_DELIVERY":
            continue
        if name.startswith(("P1_", "STOP_P1_", "STOP_TOOL_", "stop_p1_", "freeze_p1_", "complete_authorized_p1_")):
            require(not path.is_symlink(), "Unsafe stop operations symlink")
            selected.extend([x for x in path.rglob("*") if x.is_file()] if path.is_dir() else [path])
    selected.extend([ROOT / "operations/p1_cancelled_delivery.py", ROOT / "operations/publish_p1_cancelled_cpu.py", Path(__file__).resolve()])
    seen = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        relative = str(path.relative_to(ROOT / "operations"))
        rows.append(dict(path="stop_operations/" + str(safe(relative)), **checked_file(path)))
    rows.sort(key=lambda x: x["path"])
    require(len({x["path"] for x in rows}) == len(rows), "Duplicate archive names")
    require({"cancelled_delivery/PUBLICATION_RECOVERY_V1_STARTED.json", "cancelled_delivery/PUBLICATION_RECOVERY_V1_PREFLIGHT.json",
             "stop_operations/P1_AUTHORIZED_STOP/STOP_FAILURE.json",
             "stop_operations/P1_AUTHORIZED_STOP_COMPLETION/USER_STOP_RECEIPT.json",
             "cancelled_delivery/PUBLISHER_FAILURE.json",
             "stop_operations/complete_authorized_p1_stop_kernel310.py",
             "stop_operations/complete_authorized_p1_stop_kernel310_reviewed.py",
             "stop_operations/publish_p1_cancelled_cpu.py"}.issubset({x["path"] for x in rows}),
            "Required actual cancellation/startup/failure evidence missing")
    require(sum(x["bytes"] for x in rows) < 20 * 1024 ** 3, "Partial archive exceeds20GiB bound")
    return rows, excluded


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
    code = REMOTE_COMMON + "\nr=Path(" + repr(STAGE) + ");final=Path(" + repr(ARCHIVE) + ")\nexpected=" + repr(expected) + "\n"
    code += "task=" + repr(TASK) + ";terminal=" + repr(TERMINAL) + ";source_sha=" + repr(SOURCE_SHA) + "\n" + r'''
assert not final.exists()
assert sha(r/'SHA256_MANIFEST.json')==expected['manifest_sha256'] and sha(r/'SHA256SUMS')==expected['sums_sha256']
m=json.loads((r/'SHA256_MANIFEST.json').read_text());assert m['task_id']==task and m['terminal']==terminal
assert m['legal_complete']==15 and m['originally_planned_P1']==30 and m['full30_accuracy'] is None
assert m['C1_S3_formal_counts']==dict(C1=0,S3=0) and m['incomplete_scored']==0
assert (r/'SHA256SUMS').read_bytes()==''.join(x['sha256']+'  '+x['path']+'\n' for x in m['files']).encode()
verified=[]
for x in m['files']:
 p=r/x['path'];assert not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(r.resolve())
 a=p.stat();h=sha(p);b=p.stat();assert (a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
 assert h==x['sha256'] and b.st_size==x['bytes'];verified.append(dict(path=x['path'],bytes=b.st_size,sha256=h))
by={x['path']:x for x in verified};assert inventory(r)==set(by)|{'SHA256_MANIFEST.json','SHA256SUMS'}
assert sha(r/'original_P1/SOURCE_FREEZE.json')==source_sha
source=json.loads((r/'original_P1/SOURCE_FREEZE.json').read_text());assert len(source['files'])==622
for x in source['files']:
 row=by['original_P1/'+x['path']];assert row['bytes']==x['bytes'] and row['sha256']==x['sha256']
score=[json.loads((r/'cancelled_delivery'/n).read_text()) for n in ('SCORE_PASS_0.json','SCORE_PASS_1.json','FROZEN_V4_PARTIAL_SCORES.json')]
assert score[0]==score[1]==score[2] and len(score[0]['scores'])==15
report=json.loads((r/'cancelled_delivery/CPU_PARTIAL_SCORE_AND_REPORT_RESULT.json').read_text())
assert report['full30_accuracy'] is None and report['legal_complete']==15 and report['planned_P1']==30
assert not (r/'original_P1/SESSION_RESULT.json').exists() and not (r/'original_P1/FINALIZER_STARTED.json').exists()
verification=dict(status='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_FRESH_SHA_AND_BYTES',task_id=task,terminal=terminal,
 host=socket.gethostname(),monotonic=time.monotonic(),count=len(verified),bytes=sum(x['bytes'] for x in verified),
 source_freeze_sha256=source_sha,source_count=622,manifest_sha256=sha(r/'SHA256_MANIFEST.json'),files=verified)
write_new(r/'FINAL_ARCHIVE_VERIFICATION.json',canonical(verification))
seal=dict(status='PASS_HUMAN_CANCELLED_PARTIAL15_ARCHIVE_SEALED',task_id=task,terminal=terminal,
 legal_complete=15,originally_planned_P1=30,correct_on_completed15=report['correct_on_completed15'],full30_accuracy=None,
 C1_S3_formal_counts=dict(C1=0,S3=0),incomplete_scored=0,GPU_reruns=0,old_SESSION_fabricated=False,
 archive_root=str(final),atomic='renameat2_RENAME_NOREPLACE',source_freeze_sha256=source_sha,
 manifest_sha256=sha(r/'SHA256_MANIFEST.json'),SHA256SUMS_sha256=sha(r/'SHA256SUMS'),
 archive_verification_sha256=sha(r/'FINAL_ARCHIVE_VERIFICATION.json'),old_technical_failures_preserved=True)
write_new(r/'FINAL_SEAL_RECEIPT.json',canonical(seal))
names=['SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']
assert inventory(r)==set(by)|set(names)
for x in verified:assert (r/x['path']).stat().st_size==x['bytes'] and sha(r/x['path'])==x['sha256']
fsyncdir(r);rename_noreplace(r,final);fsyncdir(final.parent)
controls={}
for n in names:
 b=(final/n).read_bytes();controls[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
print(json.dumps(dict(status='PASS_INDEPENDENT_ATOMIC_NOREPLACE_P1_CANCELLED_PARTIAL_ARCHIVE',task_id=task,terminal=terminal,
 archive_root=str(final),count=len(verified),bytes=verification['bytes'],controls=controls)))
'''
    return on29(code)


def check_self(expected):
    path = Path(__file__).resolve()
    require(path == ROOT / "operations/publish_p1_cancelled_cpu_recovery_v1.py" and expected and sha(path) == expected,
            "Explicit independent publisher selfSHA/root required")
    manifest = load(ROOT / "operations" / TOOL_MANIFEST)
    rows = [x for x in manifest["files"] if x["path"] == "operations/publish_p1_cancelled_cpu_recovery_v1.py"]
    require(len(rows) == 1 and rows[0]["sha256"] == expected and rows[0]["bytes"] == path.stat().st_size,
            "Publication tool manifest identity mismatch")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--expected-self-sha256")
    ap.add_argument("--expected-manifest-sha256")
    args = ap.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PLAN_ONLY_NO_MUTATION",task_id=TASK,terminal=TERMINAL,archive=ARCHIVE,
             prerequisites="Actual partial15 validation/two fresh original CPU score/report exited",GPU_dispatches=0)))
        return 0
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("USE_TORCH") == "0"
            and os.environ.get("USE_TF") == "0" and sys.dont_write_bytecode, "Explicit CPU-only environment and -B required")
    check_self(args.expected_self_sha256)
    require(WORK.is_dir() and args.expected_manifest_sha256, "Prepared scored partial delivery root/manifest required")
    names = CONTROLS | {"PUBLICATION_RECOVERY_V1_STARTED.json", "PUBLICATION_RECOVERY_V1_RESULT.json", "PUBLICATION_RECOVERY_V1_FAILURE.json", "PUBLICATION_RECOVERY_V1_ATOMIC_RECEIPT.json", "PUBLICATION_RECOVERY_V1_PREFLIGHT.json"}
    require(not any((WORK / n).exists() for n in names), "Once-only publication already attempted; preserve/no retry")
    publisher = identity()
    excluded_live = live_paths()
    save("PUBLICATION_RECOVERY_V1_STARTED.json",dict(identity=publisher,task_id=TASK,terminal=TERMINAL,monotonic=time.monotonic(),
         self_sha256=args.expected_self_sha256,GPU_dispatches=0,old_root_writes=0,excluded_live_exact_paths=excluded_live))
    try:
        admission, originals = validate_partial(args)
        remote_preflight = preflight()
        save("PUBLICATION_RECOVERY_V1_PREFLIGHT.json",dict(CPU_admission=admission,archive29_preflight=remote_preflight))
        rows, excluded = build_rows(originals, excluded_live)
        raw = [x for x in rows if x["path"].lower().endswith(RAW_SUFFIXES)]
        manifest = dict(status="FROZEN_ACTUAL_HUMAN_CANCELLED_P1_PARTIAL15_SNAPSHOT",task_id=TASK,terminal=TERMINAL,
             files=rows,count=len(rows),bytes=sum(x["bytes"] for x in rows),legal_complete=15,originally_planned_P1=30,
             correct_on_completed15=admission["correct_on_completed15"],full30_accuracy=None,C1_S3_formal_counts={"C1":0,"S3":0},
             incomplete_scored=0,GPU_reruns=0,original_P1_root=str(P1),CPU_work_root=str(WORK),
             original_source_freeze_sha256=SOURCE_SHA,source_count=622,legal_manifest_sha256=args.expected_manifest_sha256,
             publisher_identity=publisher,publication_attempt="INDEPENDENT_CPU_RECOVERY_V1_ACTUAL_REVIEWED_TOOL_PATH",
             previous_failed_CPU_publisher_sha256=FAILED_PUBLISHER_SHA,previous_CPU_failure_receipt_sha256=FAILED_RECEIPT_SHA,
             actual_executed_stop_tool_sha256=ACTUAL_STOP_TOOL_SHA,
             publication_control_records=dict(started="PUBLICATION_RECOVERY_V1_STARTED.json",result="PUBLICATION_RECOVERY_V1_RESULT.json",atomic="PUBLICATION_RECOVERY_V1_ATOMIC_RECEIPT.json"),
             remote_only_raw=dict(count=len(raw),bytes=sum(x["bytes"] for x in raw),files=raw),
             exact_work_exclusions=sorted(excluded),nested_basename_exclusions=False,
             complete_original_P1_catalog_included=True,all_original_stop_failures_preserved=True)
        initial = initialize(manifest)
        transferred = transfer(rows)
        published = seal_publish(initial)
        controls = {}
        for name in sorted(CONTROLS):
            record = published["controls"][name]
            data = base64.b64decode(record["base64"],validate=True)
            require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], "Exact29 control bytes differ")
            write_new(WORK / name,data)
            controls[name] = dict(bytes=len(data),sha256=sha(WORK / name))
        summary = dict(status=published["status"],task_id=TASK,terminal=TERMINAL,publisher_identity=publisher,
             archive_root=ARCHIVE,count=len(rows),bytes=manifest["bytes"],raw_count=len(raw),raw_bytes=sum(x["bytes"] for x in raw),
             legal_complete=15,originally_planned_P1=30,full30_accuracy=None,controls=controls,transfer=transferred,
             exact29_original_control_bytes_mirrored=True,old_root_writes=0,GPU_reruns=0,independent_local_verification="NOT_RUN")
        save("PUBLICATION_RECOVERY_V1_ATOMIC_RECEIPT.json",summary)
        save("PUBLICATION_RECOVERY_V1_RESULT.json",dict(status="PASS_CPU_PARTIAL15_ARCHIVE_PUBLICATION_ONLY",identity=publisher,controls=controls,
             terminal=TERMINAL,monotonic=time.monotonic(),GPU_dispatches=0,local_delivery="NOT_RUN"))
        print(json.dumps(summary,ensure_ascii=False,sort_keys=True),flush=True)
        return 0
    except BaseException as error:
        if not (WORK / "PUBLICATION_RECOVERY_V1_FAILURE.json").exists():
            save("PUBLICATION_RECOVERY_V1_FAILURE.json",dict(status="PUBLICATION_FAILED_RETAIN_STAGE_AND_EVIDENCE_NO_AUTORETRY",identity=publisher,
                 error=repr(error),traceback=traceback.format_exc(),archive=ARCHIVE,stage=STAGE,old_root_writes=0,GPU_dispatches=0))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
