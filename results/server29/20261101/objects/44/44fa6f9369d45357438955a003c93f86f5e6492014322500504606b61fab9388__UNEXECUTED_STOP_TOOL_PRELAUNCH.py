"""Independent CPU-only STOP evidence publisher; never reruns science or finalizer.

Preserves original failed controller/finalizer receipts and incomplete RAM references.
The archive contains all persistent experiment files and a SHA catalog of RAM-only
engineering references. These references remain untouched on55, never continuation.
"""
import os
import sys
import json
import time
import hashlib
import shlex
import subprocess
import traceback
from pathlib import Path

R = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(R / 'src'))
import execution_control as C

ARCHIVE = C.read(R / 'ROOT.json')['archive_root']
HOST = 'zypan@202.38.247.29'
SOCKET = '/tmp/qwen_replication_archive29.sock'

def put(name, obj):
    p = R / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def text(name, value):
    with (R / name).open('x') as f:
        f.write(value); f.flush(); os.fsync(f.fileno())

def archive_call(code, *args):
    command = ['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', '-o',
               'ConnectTimeout=15', HOST,
               shlex.join(['/usr/bin/python3', '-u', '-B', '-c', code, *args])]
    p = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)

started_name = 'operations/INDEPENDENT_STOP_PUBLISHER_STARTED.json'
assert not (R / started_name).exists(), 'Already started: inspect stage, never blindly repeat'
assert not (R / 'SHA256_MANIFEST.json').exists()
assert not (R / 'FINAL_TASK_STATUS.json').exists()
put(started_name, dict(identity=C.process_identity(os.getpid()),
                      monotonic=time.monotonic(), CPU_only=True,
                      GPU_retries=0, original_finalizer_reruns=0))
try:
    print('Independent CPU STOP verification started.', flush=True)
    assert C.sha(R / 'SOURCE_FREEZE.json') == '84a60e2fa0bde4443bca1f3401355df8259a8301d97d42f9802790aa9010044f'
    C.verify_sources(R); C.model_guard(R)
    ledger = C.read(R / 'RESOURCE_LEDGER.json')
    assert len(ledger['intervals']) == 12
    assert all(x['release'] == 'PASS' for x in ledger['intervals'])
    assert sum(x['exit_returncode'] == 0 for x in ledger['intervals']) == 9
    identities = [x['identity'] for x in ledger['intervals']]
    identities += [C.read(R / n)['identity'] for n in
                   ['GPU_RUN_STARTED.json', 'FINALIZER_STARTED.json']]
    assert not any(C.same_process(x) for x in identities)
    nvml = C.query_gpu()
    assert not any(x['pid'] in {i['pid'] for i in identities} for x in nvml['apps'])
    assert not list((R / 'formal').glob('*/DISPATCH.json'))
    assert not list((R / 'formal').glob('*/RESULT.json'))
    assert not (R / 'SESSION_RESULT.json').exists()
    failures = []
    for p in sorted((R / 'execution').glob('*/FIRST_FAILURE.json')):
        x = C.read(p)
        assert 'Temporary RAM reference margin below20GiB' in x['error']
        failures.append(dict(path=str(p.relative_to(R)), sha256=C.sha(p), data=x))
    assert len(failures) == 3
    assert "monotonic" in (R / 'controller.log').read_text()
    finalizer_failure = C.read(R / 'FINALIZER_FAILURE.json')
    assert 'SESSION_RESULT.json' in finalizer_failure['error']
    progress = C.read(R / 'FORMAL_PROGRESS.json')
    assert progress['completed'] == 0 and not any(progress['branch_alive'].values())

    # No deletion, relocation or rewriting of unpaired reference files.
    ref = C.reference_root()
    ram_rows = []
    by = {}
    for i, p in enumerate(sorted(ref.rglob('*'))):
        if not p.is_file():
            continue
        assert not p.is_symlink()
        before = p.stat(); h = C.sha(p); after = p.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        row = dict(path=str(p), bytes=after.st_size, sha256=h,
                   inode=after.st_ino, mtime_ns=after.st_mtime_ns)
        ram_rows.append(row)
        bucket = p.relative_to(ref).parts[0]
        v = by.setdefault(bucket, dict(files=0, bytes=0))
        v['files'] += 1; v['bytes'] += row['bytes']
        if len(ram_rows) % 100 == 0:
            print('RAM engineering reference SHA %d files.' % len(ram_rows), flush=True)
    assert len(ram_rows) == 1313 and sum(x['bytes'] for x in ram_rows) == 113347957760
    ram = dict(status='RETAINED_UNPAIRED_VOLATILE_ENGINEERING_REFERENCES_NOT_SCIENCE_CONTINUATION',
               root=str(ref), files=ram_rows, by_folder=by,
               total_bytes=sum(x['bytes'] for x in ram_rows), file_count=len(ram_rows),
               retained_on_compute=True, deleted_files=0,
               archive_payload_contains_raw_references=False,
               archive_contains_SHA_catalog=True,
               limitation='RAM-only references remain volatile. They are not a permanent archive or completed fresh-process qualification. The persistent scientific root is archived under its frozen20GiB bound.')
    put('operations/STOP_RAM_REFERENCE_SHA_CATALOG.json', ram)
    external = C.verify_external(R)
    print('Fresh actual61 external SHA complete.', flush=True)
    assert len(external) == 61
    C.verify_sources(R); C.model_guard(R)
    failure_chain = dict(
        primary='THREE_PARALLEL_FULL_TENSOR_REFERENCES_EXHAUST_SHARED_RAM_20GIB_MARGIN',
        primary_actual_errors=failures,
        secondary='FAILURE_EVENT_DUPLICATE_MONOTONIC_KEY_CRASHED_CONTROLLER_FAILURE_HANDLER',
        tertiary='ORIGINAL_FINALIZER_STOPPED_BECAUSE_CONTROLLER_SESSION_RESULT_WAS_NOT_WRITTEN',
        distinction='This is a resource planning and failure-reporting implementation failure, not evidence that UH is mathematically wrong or that a completed9B FP equivalence failed.',
        admitted_qualification_units=79, dispatched=12, passed=9, failed_or_cancelled=3,
        formal_counts={'P1':0,'C1':0,'S3':0}, formal_total=90,
        C1_S3_real9B_FP_equivalence='NOT_RUN', CPU_384_head_contract='PASS_PREVIOUSLY_FROZEN',
        original_controller_restarted=False, original_finalizer_restarted=False,
        GPU_retries=0, science_source_edits=0)
    put('operations/INDEPENDENT_STOP_RECONSTRUCTION.json', dict(
        status='PASS_EVIDENCE_RECONSTRUCTION_NOT_ORIGINAL_CONTROLLER_SESSION',
        failure_chain=failure_chain, original_missing_SESSION_RESULT_preserved=True,
        original_FINALIZER_FAILURE_sha256=C.sha(R/'FINALIZER_FAILURE.json'),
        original_FIRST_FAILURE_sha256=C.sha(R/'FIRST_FAILURE.json'),
        source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'), source_count=608,
        external_fresh_SHA_files=external, model_stat_count=17,
        own_exact_identities=identities, own_all_identities_exited=True,
        own_NVML_absent=True, NVML=nvml, ledger=ledger,
        RAM_catalog_sha256=C.sha(R/'operations/STOP_RAM_REFERENCE_SHA_CATALOG.json')))
    status = 'PARTIAL_RESULTS_PRESERVED_AND_STOP'
    put('FINAL_TASK_STATUS.json', dict(status=status, completed={'P1':0,'C1':0,'S3':0},
        formal_completed=0, formal_total=90, qualification_passed=9,
        qualification_dispatched=12, own_GPU_processes_released=True,
        source_freeze_sha256=C.sha(R/'SOURCE_FREEZE.json'),
        origin='INDEPENDENT_CPU_STOP_EVIDENCE_DELIVERY_AFTER_ORIGINAL_FINALIZER_FAILED',
        original_failures_unchanged=True, original_SESSION_RESULT_not_fabricated=True,
        branch_failures_preserved=True, GPU_retries=0))
    report = '''# Qwen/GDN OSCAR Sym32 Unified V2: honest STOP delivery

Status: PARTIAL_RESULTS_PRESERVED_AND_STOP.
New formal AIME generations:0/90. P1=0/30,C1=0/30,S3=0/30.
Accuracy, paired rescue/lost and interaction are unavailable; no empty scientific CSV or invented score is produced. No new P0 was run.

The three UUIDs genuinely started. Nine qualification units passed: the three unchanged-codec micro checks and P1 32/128-token checks on all three cards. The three512-token R0 units did not complete. C1/S3 real9B FP equivalence, later qualifications and all formal samples were NOT RUN. CPU384-head numerical PASS cannot substitute for these missing GPU gates.

The primary failure was aggregate resource planning: three simultaneous full-tensor RAM references consumed the shared /dev/shm allowance until the unchanged20GiB margin guard fired.1313 incomplete reference files totaling113347957760bytes remain untouched in dedicated RAM. Their fresh SHA/bytes and original paths are cataloged. They are unpaired, volatile engineering comparison references, not continuation state or durable archived scientific payload. Their raw bytes are not copied into the20GiB-bounded scientific archive. No claim of permanent reference preservation or completed fresh-process comparison is made.

The controller then failed in its error-reporting path because its event helper supplied monotonic twice. That propagated to a global queue error and prevented SESSION_RESULT.json from being written. The original once-chained finalizer exited on that missing receipt. All original source, worker failures, controller/finisher logs and FINALIZER_FAILURE remain unchanged. This independent CPU evidence publisher reconstructs the observed STOP; it does not forge an original controller SESSION_RESULT, restart a worker/finalizer, or patch frozen science.

The capacity planning and error-handler defects are implementation failures. They do not demonstrate an incorrect UH transform, because real9B UH equivalence never ran. No tolerance, matrix, sampler, question roster, cap or GPU assignment was changed after dispatch. No GPU retry or trajectory stitching occurred. All12 observed worker intervals have actual release PASS;9 exited0, the failed/cancelled intervals preserve their real exit codes. Exact coordinator, workers and original finalizer are absent from processes and ownNVML.

V1 remains QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP with0/90. Its complete frozen Cq was reused without GPU recollection. The V2 full384 unique-head M0-M3 CPU attribution and principled FP64 wholeUH/singleFP32 cast are preserved as CPU findings only. Maximum canonical deployment diagonal relative error1.6890838568648173e-7; V1construction maximum2.7452575492717835e-6. These are diagnostic matrix findings, not AIME results. V1's actual checker used FP64C evaluation; the fullFP32 checker was an additional V2 diagnostic.

P1 was designed to be independent of U. In this execution a shared controller failure terminated the pipeline before that independent branch could complete. It is not reported as successful independent P1 delivery. No further execution is authorized by this STOP evidence tool.

The persistent scientific root is independently SHA/bytes verified on29 and published with atomic RENAME_NOREPLACE. Model/raw tensors are not downloaded to Mac. Local small-file verification and a further independent audit must PASS before final delivery is claimed. The separate old INT8 experiment onGPU0 is untouched. Resource time is from same-host monotonic bounds; exact physicalGPUh remains UNRESOLVED.
'''
    text('FINAL_SCIENTIFIC_REPORT.md', report)
    rows = []
    excluded = {'SHA256_MANIFEST.json','SHA256SUMS.txt','FINAL_ARCHIVE_VERIFICATION.json',
                'FINAL_SEAL_RECEIPT.json','operations/INDEPENDENT_STOP_PUBLISHER.log'}
    for p in sorted(R.rglob('*')):
        if not p.is_file():
            continue
        assert not p.is_symlink()
        rel = str(p.relative_to(R))
        if rel in excluded:
            continue
        rows.append(dict(path=rel, bytes=p.stat().st_size, sha256=C.sha(p)))
    assert sum(x['bytes'] for x in rows) <= 20*1024**3
    put('SHA256_MANIFEST.json', dict(task_id=C.TASK,status=status,files=rows,
        count=len(rows),total_bytes=sum(x['bytes'] for x in rows),
        scope='All persistent experiment files including original failures; RAM-only engineering references are SHA-cataloged and remain untouched on55.'))
    text('SHA256SUMS.txt', ''.join(x['sha256']+'  '+x['path']+'\n' for x in rows))
    stage = ARCHIVE+'.staging_'+C.sha(R/'SHA256_MANIFEST.json')[:16]
    archive_call('''import sys,json,os
from pathlib import Path
final=Path(sys.argv[1]);stage=Path(sys.argv[2]);assert not final.exists() and not stage.exists()
s=os.statvfs('/data/zypan');assert s.f_bavail*s.f_frsize>=120*1024**3
stage.mkdir(parents=True);print(json.dumps(dict(status='NEW_STAGE_RESERVED')))
''', ARCHIVE, stage)
    sshcmd=shlex.join(['ssh','-S',SOCKET,'-o','BatchMode=yes'])
    p=subprocess.run(['rsync','-a','--exclude','/operations/INDEPENDENT_STOP_PUBLISHER.log',
                      '-e',sshcmd,str(R)+'/',HOST+':'+stage+'/'],capture_output=True,text=True,timeout=2400)
    assert p.returncode==0,p.stderr
    print('Independent archive verification and no-clobber publication.',flush=True)
    published=archive_call('''import sys,json,hashlib,os,ctypes
from pathlib import Path
stage=Path(sys.argv[1]);final=Path(sys.argv[2]);m=json.loads((stage/'SHA256_MANIFEST.json').read_text())
for x in m['files']:
 p=stage/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],x['path']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=dict(status='PASS_INDEPENDENT29_ALL_FILE_SHA_BYTES_ATOMIC_SEAL',count=len(m['files']),total_bytes=m['total_bytes'],manifest_sha256=sha(stage/'SHA256_MANIFEST.json'),archive_root=str(final),scope=m['scope'])
raw=json.dumps(v,indent=2,sort_keys=True)+chr(10);(stage/'FINAL_ARCHIVE_VERIFICATION.json').write_text(raw)
s=dict(status='PASS_FINAL_SEAL',manifest_sha256=v['manifest_sha256'],archive_verification_sha256=sha(stage/'FINAL_ARCHIVE_VERIFICATION.json'),scientific_status=m['status'])
sealraw=json.dumps(s,indent=2,sort_keys=True)+chr(10);(stage/'FINAL_SEAL_RECEIPT.json').write_text(sealraw)
lib=ctypes.CDLL(None,use_errno=True);fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
assert fn(-100,os.fsencode(stage),-100,os.fsencode(final),1)==0,os.strerror(ctypes.get_errno())
fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(verification_text=raw,seal_text=sealraw)))
''',stage,ARCHIVE)
    text('FINAL_ARCHIVE_VERIFICATION.json',published['verification_text'])
    text('FINAL_SEAL_RECEIPT.json',published['seal_text'])
    print('PASS: independent honest STOP archive published; no GPU retry.',flush=True)
except BaseException as exc:
    put('operations/INDEPENDENT_STOP_PUBLISHER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),GPU_retries=0))
    raise
