#!/usr/bin/env python3
"""Once-only CPU publication of RESOURCE_BLOCKED_NO_DISPATCH.

No Torch import, GPU context, launcher, old-root mutation, retries, or overwrite.
The archive machine independently hashes all bytes and creates all four seal
controls. Their exact original bytes are mirrored back, never reserialized.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import traceback


TASK = "QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1"
TERMINAL = "RESOURCE_BLOCKED_NO_DISPATCH"
SOURCE_FREEZE_SHA256 = "faa812e121ffd2987ded1d3b88460e4ece274939c7161c6e34848fdff83b0c08"
ARCHIVE = "/data/zypan/qwen_oscar_resident_rotation_efficient_qualification_archive/" + TASK
STAGE = ARCHIVE + ".partial_once"
CONTROL_NAMES = {"SHA256_MANIFEST.json", "SHA256SUMS", "FINAL_ARCHIVE_VERIFICATION.json", "FINAL_SEAL_RECEIPT.json"}
STARTED = "operations/RESOURCE_BLOCKED_CPU_PUBLISHER_STARTED.json"
RESULT = "operations/RESOURCE_BLOCKED_CPU_PUBLISHER_RESULT.json"
FAILURE = "operations/RESOURCE_BLOCKED_CPU_PUBLISHER_FAILURE.json"
PUBLICATION = "operations/RESOURCE_BLOCKED_ATOMIC_PUBLICATION_RECEIPT.json"
PREFLIGHT = "operations/RESOURCE_BLOCKED_ARCHIVE_PREFLIGHT.json"
SSH = ["ssh", "-S", "/tmp/qwen_replication_archive29.sock", "-o", "BatchMode=yes",
       "-o", "ControlMaster=no", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=6", "zypan@202.38.247.29"]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def process_identity():
    raw = Path("/proc/self/stat").read_text()
    return dict(pid=os.getpid(), starttick=int(raw[raw.rfind(")") + 2:].split()[19]),
                cmdline=Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0"))


def on29(code):
    result = subprocess.run(SSH + ["/usr/bin/python3", "-"], input=code, text=True, capture_output=True)
    require(result.returncode == 0, "Archive29 CPU step failed; inspect retained stage, never retry blindly: " +
            result.stderr[-5000:] + result.stdout[-1000:])
    return json.loads(result.stdout)


def live_stdout_stderr(root):
    """Exclude only this running publisher's exact log paths, not operations/."""
    found = {}
    for descriptor in (1, 2):
        name = os.readlink(f"/proc/self/fd/{descriptor}")
        if not name.startswith("/"):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(root):
            relative = str(path.relative_to(root))
            require(relative not in CONTROL_NAMES and relative not in {STARTED, PREFLIGHT},
                    "Publisher output must not overwrite a seal/admission record")
            found[relative] = f"live publisher fd{descriptor}; preserve as supplemental operation evidence after exit"
    return found


def validate_cpu_terminal(root):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Publisher must run with CUDA_VISIBLE_DEVICES empty")
    result = load(root / "TASK_RESULT.json")
    require(result["task_id"] == TASK and result["status"] == TERMINAL,
            "Only the actual resource-blocked terminal can be published")
    require(result["actual_GPU_dispatches"] == result["actual_formal_completed"] == 0 and result["planned_formal"] == 60,
            "Cannot publish a fabricated no-dispatch terminal")
    require(load(root / "GPU_PERMIT.json")["ALLOW_GPU"] == "NO", "Resource-blocked root must retain NO_GPU permit")
    compare = load(root / "qualification/CPU_QUALIFICATION_COMPARISON.json")
    require(compare["status"] == "PASS_TWO_FRESH_PROCESSES_PER_INDEPENDENT_CPU_SUITE"
            and compare["math_checks_each"] == 4033 and compare["integration_checks_each"] == 310
            and compare["GPU_qualified"] is False and compare["full9B_qualified"] is False,
            "Independent CPU suites must not be represented as GPU qualification")
    math = [load(root / f"qualification/CPU_MATH_QUALIFICATION_{i}.json") for i in (2, 3)]
    integration = [load(root / f"qualification/CPU_RESIDENT_INTEGRATION_{i}.json") for i in (1, 2)]
    for record in math:
        require(record["status"] == "PASS_CPU_ONLY_RESIDENT_MATH_AND_ACTUAL_TORCH128"
                and len(record["checks"]) == 4033 and all(x["status"] == "PASS" for x in record["checks"])
                and record["GPU_dispatches"] == 0 and record["CUDA_kernel_qualified"] is False,
                "Actual CPU math receipt not complete")
    require(math[0]["process"]["pid"] != math[1]["process"]["pid"], "Math receipts must be two fresh processes")
    for record in integration:
        require(record["status"] == "PASS_CPU_ACTUAL_GDN_FORWARD_RESIDENT_INTEGRATION"
                and len(record["checks"]) == 310 and all(x["status"] == "PASS" for x in record["checks"])
                and record["new_gpu_dispatches"] == record["new_aime_generations"] == 0,
                "Actual CPU GDN integration receipt not complete")
    require(not (root / "GPU_RUN_STARTED.json").exists(), "New GPU run exists despite blocked terminal")
    for unit in load(root / "QUALIFICATION_JOB_MANIFEST.json")["units"]:
        require(unit["status"] == "NOT_RUN" and not (root / "qualification" / unit["id"]).exists(),
                "A GPU qualification unit was attempted")
    formal = root / "formal"
    require(not formal.exists() or not any(formal.rglob("*")), "Unexpected formal output in blocked task")
    freeze = load(root / "SOURCE_FREEZE.json")
    require(digest(root / "SOURCE_FREEZE.json") == SOURCE_FREEZE_SHA256 and len(freeze["files"]) == 68,
            "Original68 scientific source freeze identity changed")
    for row in freeze["files"]:
        path = (root / row["path"]).resolve()
        require(path.is_relative_to(root) and not (root / row["path"]).is_symlink(), "Source path escapes new root")
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "Frozen source changed:" + row["path"])
    external = load(root / "analysis/EXTERNAL_BINDINGS.json")["files"]
    require(len(external) == 61, "Expected exact61 inherited external bindings")
    for row in external:
        path = Path(row["path"])
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "External binding changed:" + str(path))
    models = load(root / "analysis/MODEL_TOKENIZER_FRESH_SHA.json")["files"]
    require(len(models) == 17, "Expected exact17 model/tokenizer stat identities")
    for row in models:
        stat = Path(row["path"]).stat()
        require((stat.st_size, stat.st_ino, stat.st_mtime_ns) == (row["bytes"], row["inode"], row["mtime_ns"]),
                "Model/tokenizer stat changed:" + row["path"])
    available = os.statvfs(root)
    require(available.f_bavail * available.f_frsize >= 8 * 1024 ** 3, "Compute /data headroom below8GiB")
    # Read-only NVML and process inspection; no driver context or signalling.
    sys.path.insert(0, str(root / "vendor"))
    import inherited_device_guards as guards
    observation = guards.query_gpu()
    own, exited_during_readonly_observation = [], []
    for app in observation["apps"]:
        try:
            identity = guards.process_identity(app["pid"])
        except FileNotFoundError:
            exited_during_readonly_observation.append(app["pid"])
            continue
        if str(root) in " ".join(identity.get("cmdline", [])):
            own.append(dict(app=app, identity=identity))
    require(not own, "New-task GPU process exists despite CPU-only blocked terminal")
    return dict(status="PASS_CPU_RESOURCE_BLOCKED_SOURCE61_EXTERNAL17_MODEL_STAT_NO_OWN_GPU",
                source_freeze_sha256=digest(root / "SOURCE_FREEZE.json"), source_count=len(freeze["files"]),
                math_processes=[x["process"] for x in math], math_checks_each=4033, integration_checks_each=310,
                external_fresh_SHA_count=61, model_stat_count=17, observed_NVML=observation,
                own_GPU_processes=own, exited_during_readonly_observation=exited_during_readonly_observation,
                CUDA_context_created=False, old_root_writes=0)


REMOTE_COMMON = '''
from pathlib import Path
import base64, ctypes, errno, hashlib, json, os, socket, stat, sys, tempfile, time
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\\n'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_new(p,data):
 with Path(p).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def rename_noreplace(src,dst):
 lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2
 fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
 if fn(-100,os.fsencode(src),-100,os.fsencode(dst),1):raise OSError(ctypes.get_errno(),'renameat2_RENAME_NOREPLACE')
def inventory(r):
 out=set()
 for p in r.rglob('*'):
  assert not p.is_symlink(), 'Archive contains symlink'
  if p.is_file():out.add(str(p.relative_to(r)))
  else:assert p.is_dir(), 'Archive contains unsupported nonregular member'
 return out
'''


def archive_preflight():
    code = REMOTE_COMMON + "\narchive=Path(" + repr(ARCHIVE) + ");stage=Path(" + repr(STAGE) + ")\n" + '''
base=archive.parent;base.mkdir(parents=True,exist_ok=True)
assert not archive.exists() and not stage.exists(), 'No overwrite or implicit stage retry'
space=os.statvfs(base);free=space.f_bavail*space.f_frsize
assert free>=120*1024**3, 'Need20GiB publication budget plus100GiB headroom'
with tempfile.TemporaryDirectory(prefix='resident_blocked_atomic_test_',dir=str(base)) as d:
 p=Path(d);a=p/'source';b=p/'existing';c=p/'new';a.write_bytes(b'SOURCE_ORIGINAL');b.write_bytes(b'EXISTING_KEEP')
 try:rename_noreplace(a,b)
 except OSError as e:assert e.errno==errno.EEXIST
 else:raise AssertionError('Atomic NOREPLACE negative test unexpectedly overwrote')
 assert a.read_bytes()==b'SOURCE_ORIGINAL' and b.read_bytes()==b'EXISTING_KEEP'
 rename_noreplace(a,c);assert c.read_bytes()==b'SOURCE_ORIGINAL' and not a.exists()
print(json.dumps(dict(status='PASS_ACTUAL_ATOMIC_NOREPLACE_POSITIVE_NEGATIVE_AND_SPACE',host=socket.gethostname(),free_bytes=free,archive=str(archive),stage=str(stage))))
'''
    return on29(code)


def science_catalog(root, excluded_live):
    exact_exclusions = CONTROL_NAMES | {RESULT, FAILURE, PUBLICATION} | set(excluded_live)
    rows = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        require(not path.is_symlink(), "Symlinks are not science files:" + relative)
        if path.is_dir():
            continue
        require(path.is_file(), "Unsupported nonregular source:" + relative)
        if relative in exact_exclusions:
            continue
        require("\n" not in relative and "\r" not in relative and not relative.startswith("/"), "Unsafe manifest filename")
        rows.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest(path)))
    require(rows and sum(x["bytes"] for x in rows) < 20 * 1024 ** 3, "Science exceeds20GiB publication bound")
    raw = [x for x in rows if Path(x["path"]).suffix.lower() == ".npz"]
    require(len(raw) == 25 and sum(x["bytes"] for x in raw) == 176591622,
            "Expected exact25 frozen remote-only NPZ assets/176591622bytes")
    require(STARTED in {x["path"] for x in rows} and PREFLIGHT in {x["path"] for x in rows},
            "Publisher startup/preflight must precede sealed snapshot")
    return rows, raw, exact_exclusions


def initialize_stage(manifest):
    payload = canonical(manifest).decode()
    code = REMOTE_COMMON + "\nr=Path(" + repr(STAGE) + ");final=Path(" + repr(ARCHIVE) + ")\n"
    code += "manifest_bytes=" + repr(payload.encode()) + "\n"
    code += '''
assert not r.exists() and not final.exists()
space=os.statvfs(r.parent);free=space.f_bavail*space.f_frsize
m=json.loads(manifest_bytes);assert free>=100*1024**3+max(20*1024**3,m['bytes'])
r.mkdir(exist_ok=False);fsyncdir(r.parent)
write_new(r/'SHA256_MANIFEST.json',manifest_bytes)
sums=''.join(x['sha256']+'  '+x['path']+'\\n' for x in m['files']).encode()
write_new(r/'SHA256SUMS',sums);fsyncdir(r)
print(json.dumps(dict(status='NEW_STAGE_AND_EXACT_CONTROLS_CREATED',manifest_sha256=sha(r/'SHA256_MANIFEST.json'),sums_sha256=sha(r/'SHA256SUMS'))))
'''
    return on29(code)


def transfer_uncompressed(root, rows):
    receiver = REMOTE_COMMON + "\nr=Path(" + repr(STAGE) + ")\n" + '''
import tarfile
m=json.loads((r/'SHA256_MANIFEST.json').read_text());expected={x['path']:x for x in m['files']};seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  name=member.name
  assert name in expected and name not in seen and member.isfile() and member.size==expected[name]['bytes']
  relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts and '.' not in relative.parts
  p=r/relative;assert p.resolve().is_relative_to(r.resolve());p.parent.mkdir(parents=True,exist_ok=True)
  source=archive.extractfile(member);assert source is not None
  size=0
  with p.open('xb') as dest:
   for block in iter(lambda:source.read(1024*1024),b''):dest.write(block);size+=len(block)
   dest.flush();os.fsync(dest.fileno())
  assert size==expected[name]['bytes'];seen.add(name)
assert seen==set(expected)
print(json.dumps(dict(status='UNCOMPRESSED_TAR_RECEIVED_ALL_EXPECTED_FILES',count=len(seen))))
'''
    command = SSH + ["exec /usr/bin/python3 -c " + shlex.quote(receiver)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|", dereference=True) as stream:
            for row in rows:
                path = root / row["path"]
                before = path.stat()
                require(before.st_size == row["bytes"] and digest(path) == row["sha256"], "Snapshot changed before transfer:" + row["path"])
                stream.add(path, arcname=row["path"], recursive=False)
                after = path.stat()
                require((before.st_size, before.st_ino, before.st_mtime_ns) ==
                        (after.st_size, after.st_ino, after.st_mtime_ns) and digest(path) == row["sha256"],
                        "Snapshot changed during transfer:" + row["path"])
        proc.stdin.close()
        proc.stdin = None
        stdout, stderr = proc.communicate()
        require(proc.returncode == 0, "Archive transfer failed, retain partial and do not retry: " + stderr.decode(errors="replace")[-5000:])
        result = json.loads(stdout)
        require(result["count"] == len(rows), "Receiver file count differs")
        return result
    except BaseException:
        if proc.stdin is not None:
            proc.stdin.close()
            proc.stdin = None
        proc.communicate()
        raise


def verify_seal_publish29(initial_controls, source_sha):
    code = REMOTE_COMMON + "\nr=Path(" + repr(STAGE) + ");final=Path(" + repr(ARCHIVE) + ")\n"
    code += "expected_initial=" + repr(initial_controls) + "\nsource_sha=" + repr(source_sha) + "\n"
    code += "task=" + repr(TASK) + ";terminal=" + repr(TERMINAL) + "\n" + '''
assert not final.exists()
assert sha(r/'SHA256_MANIFEST.json')==expected_initial['manifest_sha256']
assert sha(r/'SHA256SUMS')==expected_initial['sums_sha256']
m=json.loads((r/'SHA256_MANIFEST.json').read_text());verified=[]
assert m['task_id']==task and m['terminal']==terminal and sha(r/'SOURCE_FREEZE.json')==source_sha
assert (r/'SHA256SUMS').read_bytes()==''.join(x['sha256']+'  '+x['path']+'\\n' for x in m['files']).encode()
for x in m['files']:
 p=r/x['path'];assert not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(r.resolve())
 actual=sha(p);size=p.stat().st_size
 assert size==x['bytes'] and actual==x['sha256'];verified.append(dict(path=x['path'],bytes=size,sha256=actual))
assert inventory(r)=={x['path'] for x in verified}|{'SHA256_MANIFEST.json','SHA256SUMS'}
verification=dict(status='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_FRESH_SHA_AND_BYTES',host=socket.gethostname(),
 monotonic=time.monotonic(),task_id=task,terminal=terminal,count=len(verified),bytes=sum(x['bytes'] for x in verified),
 manifest_sha256=sha(r/'SHA256_MANIFEST.json'),source_freeze_sha256=source_sha,files=verified)
write_new(r/'FINAL_ARCHIVE_VERIFICATION.json',canonical(verification))
seal=dict(status='PASS_RESOURCE_BLOCKED_NO_DISPATCH_SEALED',task_id=task,terminal=terminal,
 actual_GPU_dispatches=0,actual_formal_completed=0,planned_formal=60,GPU_qualification='NOT_RUN',GPU_performance='NOT_RUN',
 CPU_math='TWO_FRESH4033_PASS',CPU_GDN_integration='TWO_FRESH310_PASS',accuracy=None,TPS=None,
 archive_root=str(final),atomic='renameat2_RENAME_NOREPLACE',source_freeze_sha256=source_sha,
 manifest_sha256=sha(r/'SHA256_MANIFEST.json'),SHA256SUMS_sha256=sha(r/'SHA256SUMS'),
 archive_verification_sha256=sha(r/'FINAL_ARCHIVE_VERIFICATION.json'),old_experiment_terminations_unchanged=True)
write_new(r/'FINAL_SEAL_RECEIPT.json',canonical(seal))
names=['SHA256_MANIFEST.json','SHA256SUMS','FINAL_ARCHIVE_VERIFICATION.json','FINAL_SEAL_RECEIPT.json']
assert inventory(r)=={x['path'] for x in verified}|set(names)
# Fresh check after control creation, before the single atomic publication.
for x in verified:assert (r/x['path']).stat().st_size==x['bytes'] and sha(r/x['path'])==x['sha256']
fsyncdir(r);rename_noreplace(r,final);fsyncdir(final.parent)
controls={}
for name in names:
 data=(final/name).read_bytes();controls[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode())
print(json.dumps(dict(status='PASS_INDEPENDENT_ATOMIC_NOREPLACE_RESOURCE_BLOCKED_PUBLICATION',archive_root=str(final),
 terminal=terminal,count=len(verified),bytes=verification['bytes'],source_freeze_sha256=source_sha,controls=controls)))
'''
    return on29(code)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    require(root == Path("/data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1"),
            "Publisher only accepts the exact independent55 task root")
    require(not any((root / n).exists() for n in CONTROL_NAMES | {STARTED, RESULT, FAILURE, PUBLICATION, PREFLIGHT}),
            "Once-only publication already attempted; inspect and preserve, no automatic retry")
    # Independent CPU delivery code is bound separately from the immutable
    # 68-file scientific/GPU source freeze; neither freeze nor old seals change.
    tool_manifest = load(root / "operations/READINESS_DELIVERY_TOOL_MANIFEST.json")
    self_rows = [row for row in tool_manifest["files"]
                 if row["path"] == "operations/publish_resource_blocked_cpu.py"]
    require(len(self_rows) == 1, "Independent publisher identity missing from delivery-tool manifest")
    self_path = root / self_rows[0]["path"]
    require(self_path.resolve() == Path(__file__).resolve() and self_path.stat().st_size == self_rows[0]["bytes"]
            and digest(self_path) == self_rows[0]["sha256"], "Independent CPU publisher selfSHA/bytes changed")
    identity = process_identity()
    excluded_live = live_stdout_stderr(root)
    save(root, STARTED, dict(identity=identity, monotonic=time.monotonic(), task_id=TASK, terminal=TERMINAL,
                            GPU_dispatches=0, original_tasks_untouched=True, excluded_live_paths=excluded_live))
    try:
        admission = validate_cpu_terminal(root)
        preflight = archive_preflight()
        save(root, PREFLIGHT, dict(local_CPU_admission=admission, archive_preflight=preflight))
        rows, raw, exclusions = science_catalog(root, excluded_live)
        manifest = dict(status="FROZEN_RESIDENT_ROTATION_CPU_RESOURCE_BLOCKED_SNAPSHOT", task_id=TASK,
                        terminal=TERMINAL, count=len(rows), bytes=sum(x["bytes"] for x in rows), files=rows,
                        actual_GPU_dispatches=0, actual_formal_completed=0, planned_formal=60,
                        source_freeze_sha256=admission["source_freeze_sha256"],
                        remote_only_raw_assets=dict(count=len(raw), bytes=sum(x["bytes"] for x in raw), files=raw),
                        excluded_exact_paths=sorted(exclusions), excluded_live_paths=excluded_live,
                        control_exclusion_scope="Exact top-level paths only; nested original metadata are included",
                        publisher_identity=identity)
        initial_controls = initialize_stage(manifest)
        transfer = transfer_uncompressed(root, rows)
        published = verify_seal_publish29(initial_controls, admission["source_freeze_sha256"])
        controls = {}
        for name in sorted(CONTROL_NAMES):
            record = published["controls"][name]
            raw_control = base64.b64decode(record["base64"], validate=True)
            require(len(raw_control) == record["bytes"] and hashlib.sha256(raw_control).hexdigest() == record["sha256"],
                    "Archive control response byte identity mismatch:" + name)
            with (root / name).open("xb") as handle:
                handle.write(raw_control)
                handle.flush()
                os.fsync(handle.fileno())
            controls[name] = dict(bytes=len(raw_control), sha256=digest(root / name))
        summary = dict(status=published["status"], task_id=TASK, terminal=TERMINAL, archive_root=ARCHIVE,
                       science_files=len(rows), science_bytes=manifest["bytes"], raw_NPZ_count=len(raw),
                       raw_NPZ_bytes=sum(x["bytes"] for x in raw), source_count=admission["source_count"],
                       source_freeze_sha256=admission["source_freeze_sha256"], controls=controls,
                       exact_original29_control_bytes_mirrored=True, transfer=transfer,
                       GPU_dispatches=0, formal_completed=0, planned_formal=60,
                       publisher_identity=identity, independent_local_delivery_still_required=True)
        save(root, PUBLICATION, summary)
        save(root, RESULT, dict(status="PASS_CPU_RESOURCE_BLOCKED_ARCHIVE_PUBLICATION_ONLY", identity=identity,
                               monotonic=time.monotonic(), archive_root=ARCHIVE, controls=controls,
                               GPU_dispatches=0, scientific_GPU_success_claimed=False))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        if not (root / FAILURE).exists():
            save(root, FAILURE, dict(status="CPU_PUBLICATION_FAILED_RETAIN_STAGE_AND_EVIDENCE_NO_AUTORETRY",
                 identity=identity, error=repr(error), traceback=traceback.format_exc(), GPU_dispatches=0,
                 archive_root=ARCHIVE, partial_stage=STAGE, original_tasks_untouched=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
