#!/usr/bin/env python3
"""Independent Mac CPU read-only verification after actual CPU publication recovery.

Never writes remotely, calls an old finisher, starts CUDA, downloads model/raw
Tensor bytes, concatenates trajectories, retries a transfer or overwrites a seal.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import traceback


TASK = "QWEN_GDN_OSCAR_P1_HUMAN_CANCELLED_PARTIAL_DELIVERY_20261022"
TERMINAL = "HUMAN_CANCELLED_P1_PARTIAL_15_OF_30_C1_S3_TECHNICAL_STOP_ZERO"
ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/data/zypan/analyses/qwen_gdn_oscar_resident_rotation_efficient_qualification_v1_gpu_continuation_20261022"
P1 = "/data/zypan/analyses/qwen_gdn_oscar_sym32_p1_c1_s3_unified_recovery_aime26_v2_rerun_20261022"
WORK = REMOTE + "/operations/P1_CANCELLED_CPU_DELIVERY"
ARCHIVE = "/data/zypan/qwen_oscar_sym32_unified_v2_archive/" + TASK
SOURCE_SHA = "3955f03f2eaa56f83245a85ccad1c56d7d98b70eaa046154a1d95f5b79a7fa14"
PREPARE_SHA = "ddc35bd8b0ce7c3c0daef460185295f4928a726262fd04f70cf82bff5f4c99c8"
CONTROLS = ("SHA256_MANIFEST.json", "SHA256SUMS", "FINAL_ARCHIVE_VERIFICATION.json", "FINAL_SEAL_RECEIPT.json")
RAW_SUFFIXES = (".bin", ".npz", ".npy", ".f32", ".pt", ".pth", ".safetensor", ".safetensors", ".ckpt", ".onnx", ".gguf")
TEXT_SUFFIXES = (".py", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".sh", ".log", ".diff", ".yaml", ".yml", ".toml", ".sha256", ".html")
PYTHON55 = "/data/ydai/miniconda3/envs/bitdecode/bin/python"
SSH55 = ["ssh", "-S", "/tmp/qwen_gdn_replication_20261013.sock", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
         "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", "zypan@202.38.247.55"]
SSH29 = ["ssh", "-S", "/tmp/qwen_replication_archive29.sock", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
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
    write_new(ROOT / "operations" / name, canonical(obj))


def safe(name):
    p = Path(name)
    require(isinstance(name, str) and name and not p.is_absolute() and str(p) == name
            and all(x not in (".", "..") for x in p.parts) and "\n" not in name and "\r" not in name,
            "Unsafe relative science path")
    return p


def text_eligible(row):
    name = row["path"].lower()
    if name.endswith(RAW_SUFFIXES):
        return False
    return name.endswith(TEXT_SUFFIXES) or Path(name).name in ("sha256sums", "license", "notice")


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


def audit_code(role, manifest_sha, seal_sha):
    code = REMOTE_HELPERS + "\nrole=" + repr(role) + ";archive=Path(" + repr(ARCHIVE) + ");work=Path(" + repr(WORK) + ");p1=Path(" + repr(P1) + ");continuation=Path(" + repr(REMOTE) + ")\n"
    code += "r=work if role=='55' else archive\ncontrols=" + repr(CONTROLS) + ";task=" + repr(TASK) + ";terminal=" + repr(TERMINAL) + "\n"
    code += "expected_manifest_sha=" + repr(manifest_sha) + ";expected_seal_sha=" + repr(seal_sha) + ";source_sha=" + repr(SOURCE_SHA) + ";prepare_sha=" + repr(PREPARE_SHA) + "\n" + r'''
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('USE_TORCH')==os.environ.get('USE_TF')=='0'
data={n:(r/n).read_bytes() for n in controls}
assert hashlib.sha256(data['SHA256_MANIFEST.json']).hexdigest()==expected_manifest_sha
assert hashlib.sha256(data['FINAL_SEAL_RECEIPT.json']).hexdigest()==expected_seal_sha
m=json.loads(data['SHA256_MANIFEST.json']);seal=json.loads(data['FINAL_SEAL_RECEIPT.json']);v=json.loads(data['FINAL_ARCHIVE_VERIFICATION.json'])
assert m['task_id']==seal['task_id']==task and m['terminal']==seal['terminal']==terminal
assert m['legal_complete']==seal['legal_complete']==15 and m['originally_planned_P1']==seal['originally_planned_P1']==30
assert m['full30_accuracy'] is None and seal['full30_accuracy'] is None
assert m['C1_S3_formal_counts']==seal['C1_S3_formal_counts']==dict(C1=0,S3=0)
assert m['incomplete_scored']==seal['incomplete_scored']==0 and m['GPU_reruns']==seal['GPU_reruns']==0
assert seal['status']=='PASS_HUMAN_CANCELLED_PARTIAL15_ARCHIVE_SEALED' and seal['old_SESSION_fabricated'] is False
assert seal['archive_root']==str(archive) and seal['atomic']=='renameat2_RENAME_NOREPLACE'
assert m['original_source_freeze_sha256']==seal['source_freeze_sha256']==source_sha
assert seal['manifest_sha256']==expected_manifest_sha
assert seal['SHA256SUMS_sha256']==hashlib.sha256(data['SHA256SUMS']).hexdigest()
assert seal['archive_verification_sha256']==hashlib.sha256(data['FINAL_ARCHIVE_VERIFICATION.json']).hexdigest()
assert v['status']=='PASS_INDEPENDENT_ARCHIVE29_EVERY_FILE_FRESH_SHA_AND_BYTES' and v['source_freeze_sha256']==source_sha and v['source_count']==622
rows=m['files'];assert len(rows)==m['count']==len({x['path'] for x in rows}) and sum(x['bytes'] for x in rows)==m['bytes']
assert data['SHA256SUMS']==''.join(x['sha256']+'  '+x['path']+'\n' for x in rows).encode()
actual=[]
for x in rows:
 rel=safe(x['path'])
 if role=='29':p=archive/rel
 else:
  parts=rel.parts;assert len(parts)>1
  bases={'original_P1':p1,'cancelled_delivery':work,'stop_operations':continuation/'operations'}
  assert parts[0] in bases;p=bases[parts[0]].joinpath(*parts[1:])
  assert str(p)==x['source_path'] and p.resolve().is_relative_to(bases[parts[0]].resolve())
 assert not p.is_symlink() and p.is_file()
 a=p.stat();h=sha(p);b=p.stat();assert (a.st_size,a.st_ino,a.st_mtime_ns)==(b.st_size,b.st_ino,b.st_mtime_ns)
 assert b.st_size==x['bytes'] and h==x['sha256'];actual.append(dict(path=x['path'],bytes=b.st_size,sha256=h))
assert actual==v['files'] and len(actual)==v['count'] and sum(x['bytes'] for x in actual)==v['bytes']
by={x['path']:x for x in actual}
def asset(name):
 if role=='29':return archive/safe(name)
 p=safe(name);bases={'original_P1':p1,'cancelled_delivery':work,'stop_operations':continuation/'operations'}
 return bases[p.parts[0]].joinpath(*p.parts[1:])
assert sha(asset('original_P1/SOURCE_FREEZE.json'))==source_sha
assert sha(asset('stop_operations/complete_authorized_p1_stop_kernel310_reviewed.py'))=='42db6200363c8ee723444f7fdb243be22a54b8cd2c319f7398ddcdf980a3b070'
assert sha(asset('stop_operations/complete_authorized_p1_stop_kernel310.py'))=='83f1f01dcb2bfc915a4f859b01dd475110cc105b6b443d453d956b8eeae7e83a'
assert sha(asset('stop_operations/publish_p1_cancelled_cpu.py'))=='a2b459a5b37165cb8fa56cd44c1d392cb8c9eb245d5d4e6bd8047607b9e0d2f9'
assert sha(asset('cancelled_delivery/PUBLISHER_FAILURE.json'))=='987bb28eb3db6cbec70045fb5757f73a8cf5f545b09e9ff2696e693278b24377'
sources=read(asset('original_P1/SOURCE_FREEZE.json'))['files'];assert len(sources)==622
for x in sources:
 y=by['original_P1/'+x['path']];assert y['sha256']==x['sha256'] and y['bytes']==x['bytes']
scores=[read(asset('cancelled_delivery/'+n)) for n in ('SCORE_PASS_0.json','SCORE_PASS_1.json','FROZEN_V4_PARTIAL_SCORES.json')]
assert scores[0]==scores[1]==scores[2] and len(scores[0]['scores'])==15
assert scores[0]['full30_accuracy'] is None and scores[0]['cancelled_samples_scored']==0
report=read(asset('cancelled_delivery/CPU_PARTIAL_SCORE_AND_REPORT_RESULT.json'))
assert report['legal_complete']==15 and report['planned_P1']==30 and report['full30_accuracy'] is None
assert not (asset('original_P1/SOURCE_FREEZE.json').parent/'SESSION_RESULT.json').exists()
assert not (asset('original_P1/SOURCE_FREEZE.json').parent/'FINALIZER_STARTED.json').exists()
legal=read(asset('cancelled_delivery/P1_LEGAL_PAYLOAD_MANIFEST.json'))
assert len(legal['records'])==legal['complete']==15 and legal['never_started_P1_count']==12 and len(legal['cancelled_units'])==3
assert sha(asset('cancelled_delivery/P1_LEGAL_PAYLOAD_MANIFEST.json'))==m['legal_manifest_sha256']
external=[];models=[];release=None;publisher_exit=None;extras=[]
if role=='29':assert inventory(archive)==set(by)|set(controls), 'Archive extra files rejected'
else:
 expected_old={str(Path(x['path']).relative_to('original_P1')) for x in rows if x['path'].startswith('original_P1/')}
 assert inventory(p1)==expected_old, 'Original P1 tree changed after actual cancellation snapshot'
 pub=m['publisher_identity'];now=identity(pub['pid']);assert now!=pub
 publisher_exit=dict(expected=pub,current=now,exact_identity_exited=True)
 attempt=m['publication_control_records'];assert m['publication_attempt']=='INDEPENDENT_CPU_RECOVERY_V1_ACTUAL_REVIEWED_TOOL_PATH'
 assert attempt==dict(started='PUBLICATION_RECOVERY_V1_STARTED.json',result='PUBLICATION_RECOVERY_V1_RESULT.json',atomic='PUBLICATION_RECOVERY_V1_ATOMIC_RECEIPT.json')
 pubresult=read(work/attempt['result']);assert pubresult['status']=='PASS_CPU_PARTIAL15_ARCHIVE_PUBLICATION_ONLY' and pubresult['identity']==pub
 assert pubresult['terminal']==terminal
 publication=read(work/attempt['atomic']);assert publication['publisher_identity']==pub and publication['status']=='PASS_INDEPENDENT_ATOMIC_NOREPLACE_P1_CANCELLED_PARTIAL_ARCHIVE'
 tool=continuation/'operations/p1_cancelled_delivery.py';assert sha(tool)==prepare_sha
 spec=importlib.util.spec_from_file_location('independent_cancelled_P1_final_readonly',tool);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 control,validate=module.original_bindings()
 stop=module.stop_receipt(work/'evidence/USER_STOP_RECEIPT.json',legal['stop_receipt_original_sha256'])
 release=module.own_exit_check(control,stop)
 assert module.checked_payloads(control,validate,stop)==legal['records']
 for path,row in control.bindings(p1).items():
  p=Path(path);before=p.stat();h=sha(p);after=p.stat();assert (before.st_size,before.st_ino,before.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns)
  assert h==row['sha256'] and after.st_size==row['bytes'];external.append(dict(path=path,bytes=after.st_size,sha256=h))
 assert len(external)==61
 for x in read(p1/'analysis/MODEL_TOKENIZER_FRESH_SHA.json')['files']:
  s=Path(x['path']).stat();assert (s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])
  models.append(dict(path=x['path'],bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
 assert len(models)==17
 for n in ('SCORE_PASS_0_STARTED.json','SCORE_PASS_1_STARTED.json','REPORT_STARTED.json'):
  ident=read(work/n)['identity'];assert identity(ident['pid'])!=ident
 extras=sorted(inventory(work)-{str(Path(x['path']).relative_to('cancelled_delivery')) for x in rows if x['path'].startswith('cancelled_delivery/')} - set(controls))
print(json.dumps(dict(status='PASS_READONLY_FULL_ARCHIVE_SHA_BYTES_ACTUAL_PARTIAL15',role=role,task_id=task,terminal=terminal,
 host=socket.gethostname(),monotonic=time.monotonic(),count=len(actual),bytes=sum(x['bytes'] for x in actual),files=actual,
 source_count=622,source_freeze_sha256=source_sha,source_files=sources,external_fresh_SHA=external,model_stat=models,
 publisher_exit=publisher_exit,release=release,archive_extras_rejected=role=='29',work_supplemental_paths=extras,
 correct_on_completed15=report['correct_on_completed15'],full30_accuracy=None,C1_S3_formal_counts=dict(C1=0,S3=0),
 controls={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()) for n,b in data.items()}),sort_keys=True))
'''
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--expected-self-sha256")
    ap.add_argument("--expected-archive-manifest-sha256")
    ap.add_argument("--expected-final-seal-sha256")
    args = ap.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PLAN_ONLY_NO_REMOTE_OR_LOCAL_MUTATION",task_id=TASK,archive=ARCHIVE,
             raw_Tensor_downloads=0,full30_accuracy=None)))
        return 0
    require(ROOT == Path("/Users/pytenter/Documents/临时项目1/QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1_GPU_CONTINUATION_20261022"), "Exact local continuation root required")
    path = Path(__file__).resolve()
    require(args.expected_self_sha256 and sha(path) == args.expected_self_sha256, "Explicit reviewed downloader selfSHA required")
    tm = load(ROOT / "operations/P1_CANCELLED_LOCAL_RECOVERY_V1_TOOL_MANIFEST.json")
    bound = [x for x in tm["files"] if x["path"] == "operations/download_verify_p1_cancelled_delivery_recovery_v1.py"]
    require(len(bound) == 1 and bound[0]["sha256"] == args.expected_self_sha256 and bound[0]["bytes"] == path.stat().st_size, "Independent local tool manifest mismatch")
    require(args.expected_archive_manifest_sha256 and args.expected_final_seal_sha256, "Actual published controls must be explicitly bound")
    stage, final = ROOT / "P1_cancelled_partial_delivery.partial_once", ROOT / "P1_cancelled_partial_delivery"
    require(not stage.exists() and not final.exists() and not (ROOT / "operations/LOCAL_P1_CANCELLED_RECOVERY_V1_STARTED.json").exists(), "Existing attempt/stage/final; retain/no retry")
    save("LOCAL_P1_CANCELLED_RECOVERY_V1_STARTED.json",dict(task_id=TASK,pid=os.getpid(),self_sha256=args.expected_self_sha256,
         expected_manifest_sha256=args.expected_archive_manifest_sha256,expected_seal_sha256=args.expected_final_seal_sha256,
         remote_writes=0,GPU_dispatches=0,raw_downloads=0))
    try:
        atomic_preflight()
        audit55 = remote_cpu("55",audit_code("55",args.expected_archive_manifest_sha256,args.expected_final_seal_sha256))
        audit29 = remote_cpu("29",audit_code("29",args.expected_archive_manifest_sha256,args.expected_final_seal_sha256))
        require(audit55["files"] == audit29["files"] and audit55["controls"] == audit29["controls"], "55/29 fresh file metadata or original control bytes differ")
        require(audit55["publisher_exit"]["exact_identity_exited"] and len(audit55["external_fresh_SHA"]) == 61 and len(audit55["model_stat"]) == 17,
                "Independent publisher exit/external/model check incomplete")
        rows = audit29["files"]
        downloadable = [x for x in rows if text_eligible(x)]
        excluded = [x for x in rows if not text_eligible(x)]
        require(not any(x["path"].lower().endswith(RAW_SUFFIXES) for x in downloadable), "Raw/model download attempted")
        require(sum(x["bytes"] for x in downloadable) < 2 * 1024 ** 3 and all(x["bytes"] <= 32 * 1024 ** 2 for x in downloadable), "Local small-report bounds exceeded")
        stage.mkdir(exist_ok=False)
        batch, size = [], 0
        batches = []
        for row in downloadable:
            if batch and size + row["bytes"] > 4 * 1024 ** 2:
                batches.append(batch)
                batch, size = [], 0
            batch.append(row)
            size += row["bytes"]
        if batch:
            batches.append(batch)
        for batch in batches:
            downloaded = remote_cpu("29", fetch_code(batch))
            require([x["path"] for x in downloaded] == [x["path"] for x in batch], "Download response roster differs")
            for row, receipt in zip(batch, downloaded):
                data = base64.b64decode(receipt["base64"], validate=True)
                require(len(data) == row["bytes"] == receipt["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"] == receipt["sha256"], "Downloaded bytes differ")
                write_new(stage / safe(row["path"]),data)
        for name, record in audit29["controls"].items():
            data = base64.b64decode(record["base64"],validate=True)
            require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], "Original29 control byte mismatch")
            write_new(stage / name,data)
        local_expected = {x["path"] for x in downloadable} | set(CONTROLS)
        actual = set()
        for p in stage.rglob("*"):
            require(not p.is_symlink(), "Unexpected local symlink")
            if p.is_file():
                actual.add(str(p.relative_to(stage)))
        require(actual == local_expected, "Local delivery missing or extra files")
        for row in downloadable:
            p = stage / row["path"]
            require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "Fresh local SHA/bytes mismatch")
        for name, record in audit29["controls"].items():
            require((stage / name).stat().st_size == record["bytes"] and sha(stage / name) == record["sha256"], "Local original control mismatch")
        rename_exclusive(stage,final)
        verification = dict(status="PASS_LOCAL_HUMAN_CANCELLED_P1_PARTIAL15_NONRAW_DELIVERY",task_id=TASK,terminal=TERMINAL,
             delivery=str(final),legal_complete=15,originally_planned_P1=30,correct_on_completed15=audit55["correct_on_completed15"],
             full30_accuracy=None,C1_S3_formal_counts={"C1":0,"S3":0},incomplete_scored=0,local_files=len(actual),
             remote_files_fresh_SHA=audit55["count"],source_count=622,source_freeze_sha256=SOURCE_SHA,
             external_fresh_SHA_count=61,model_stat_count=17,remote_only_files=excluded,raw_Tensor_downloads=0,
             remote_writes=0,GPU_reruns=0,controls={n:{k:v for k,v in x.items() if k!='base64'} for n,x in audit29["controls"].items()},
             original29_controls_byte_identical_to55_and_local=True,atomic_local="renamex_np_RENAME_EXCL")
        save("LOCAL_P1_CANCELLED_DELIVERY_VERIFICATION.json",verification)
        for audit in (audit55,audit29):
            for record in audit["controls"].values():
                record.pop("base64")
        save("P1_CANCELLED_FINAL_INDEPENDENT_READONLY_VERIFICATION.json",dict(status="PASS_ALL55_29_SOURCE622_EXTERNAL61_MODEL17_ACTUAL_PARTIAL15",compute55=audit55,archive29=audit29))
        save("P1_CANCELLED_FINAL_DELIVERY_SUMMARY.json",dict(status="PASS_LEGAL_HUMAN_CANCELLED_P1_PARTIAL15_DELIVERY",task_id=TASK,terminal=TERMINAL,
             legal_complete=15,originally_planned_P1=30,correct_on_completed15=verification["correct_on_completed15"],full30_accuracy=None,
             C1_S3_formal_counts={"C1":0,"S3":0},archive=ARCHIVE,local_delivery=str(final),GPU_reruns=0,old_terminations_unchanged=True))
        print(json.dumps(verification,ensure_ascii=False,sort_keys=True),flush=True)
        return 0
    except BaseException as error:
        failure = ROOT / "operations/LOCAL_P1_CANCELLED_RECOVERY_V1_FAILURE.json"
        if not failure.exists():
            write_new(failure,canonical(dict(status="READONLY_DELIVERY_FAILED_RETAIN_EVIDENCE_NO_AUTORETRY",error=repr(error),traceback=traceback.format_exc(),
                 stage=str(stage),final=str(final),remote_writes=0,GPU_dispatches=0)))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
