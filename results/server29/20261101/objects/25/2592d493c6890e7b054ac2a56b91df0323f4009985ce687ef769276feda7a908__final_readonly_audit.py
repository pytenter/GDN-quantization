"""Independent final audit. Remote actions are strictly reads; evidence is local."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import time

R = Path(__file__).resolve().parents[1]
CONFIG = json.loads((R / 'ROOT.json').read_text())
OUT = R / 'operations' / ('FINAL_INDEPENDENT_AUDIT_' + time.strftime('%Y%m%dT%H%M%S'))
OUT.mkdir()
ARCHIVE_CODE = r'''
import sys,json,hashlib,time
from pathlib import Path
R=Path(sys.argv[1])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
m=json.loads((R/'SHA256_MANIFEST.json').read_text());seal=json.loads((R/'FINAL_SEAL_RECEIPT.json').read_text());receipt=json.loads((R/'FINAL_ARCHIVE_VERIFICATION.json').read_text())
assert sha(R/'SHA256_MANIFEST.json')==seal['manifest_sha256']==receipt['manifest_sha256']
assert sha(R/'FINAL_ARCHIVE_VERIFICATION.json')==seal['archive_verification_sha256']
rows=[]
for x in m['files']:
 p=R/x['path'];h=sha(p);s=p.stat();rows.append(dict(path=x['path'],sha256=h,bytes=s.st_size,match=h==x['sha256'] and s.st_size==x['bytes'] and not p.is_symlink()))
print(json.dumps(dict(status='PASS' if all(x['match'] for x in rows) else 'FAIL',manifest_sha256=sha(R/'SHA256_MANIFEST.json'),seal=seal,receipt=receipt,receipt_sha256=sha(R/'FINAL_ARCHIVE_VERIFICATION.json'),rows=rows,remote_monotonic=time.monotonic())))
'''
CODE = r'''
import sys,json,hashlib,time,subprocess,shlex
from pathlib import Path
R=Path(sys.argv[1]);sys.path.insert(0,str(R/'src'));import execution_control as C
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def emit(kind,**data):print(json.dumps(dict(kind=kind,remote_monotonic=time.monotonic(),**data)),flush=True)
m=json.loads((R/'SHA256_MANIFEST.json').read_text());freeze=json.loads((R/'SOURCE_FREEZE.json').read_text());rows=[]
for x in m['files']:
 p=R/x['path'];h=sha(p);s=p.stat();rows.append(dict(path=x['path'],sha256=h,bytes=s.st_size,match=h==x['sha256'] and s.st_size==x['bytes'] and not p.is_symlink()))
actual={x['path']:x for x in rows};sources=[]
for x in freeze['files']:
 p=R/x['path'];h=sha(p);size=p.stat().st_size;sources.append(dict(path=x['path'],sha256=h,bytes=size,match=h==x['sha256'] and size==x['bytes'],included_in_final_manifest=x['path'] in actual))
history=json.loads((R/'analysis/HISTORICAL_P0_INPUT_MANIFEST.json').read_text()) if (R/'analysis/HISTORICAL_P0_INPUT_MANIFEST.json').exists() else dict(root='',files=[]);hist=[]
for x in history['files']:
 p=Path(history['root'])/x['path'];h=sha(p);size=p.stat().st_size;hist.append(dict(path=str(p),sha256=h,bytes=size,match=h==x['sha256'] and size==x['bytes']))
provenance=[]
for n in ['CPU_ATTRIBUTION_SOURCE_FREEZE.json','P1_EXECUTABLE_IDENTITY.json','V1_IMMUTABILITY_AND_REUSE_AUDIT.json','evidence/group32/delivery/SHA256_MANIFEST.json','evidence/group32/delivery/TASK_RESULT.json','V2_NUMERICAL_QUALIFICATION_CONTRACT.json','CANONICAL_UH_MATRICES_AND_SHA.json']:
 p=R/n
 if p.exists():provenance.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p)))
ledger=json.loads((R/'RESOURCE_LEDGER.json').read_text());identities=[x['identity'] for x in ledger['intervals']]
for n in ['GPU_RUN_STARTED.json','FINALIZER_STARTED.json','CALIBRATION_RUN_STARTED.json']:
 if (R/n).exists():identities.append(json.loads((R/n).read_text())['identity'])
alive=[x for x in identities if C.same_process(x)];nvml=C.query_gpu();own_pids={x['pid'] for x in identities};own_nvml=[x for x in nvml['apps'] if x['pid'] in own_pids]
model=[]
for x in json.loads((R/'analysis/MODEL_TOKENIZER_FRESH_SHA.json').read_text())['files']:
 s=Path(x['path']).stat();model.append(dict(path=x['path'],bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,match=(s.st_size,s.st_ino,s.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])))
v1root=Path(json.loads((R/'ROOT.json').read_text())['v1_remote_root']);v1manifest=json.loads((v1root/'SHA256_MANIFEST.json').read_text());v1rows=[]
for x in v1manifest['files']:
 p=v1root/x['path'];v1rows.append(dict(path=x['path'],match=p.stat().st_size==x['bytes'] and sha(p)==x['sha256']))
assert all(x['match'] for x in v1rows) and sha(v1root/'SHA256_MANIFEST.json')=='bd07019f5c81fee86c9e26d2f5452f801ead7c4bb12f54ea7057cebbf85e1efe'
assert json.loads((v1root/'FINAL_TASK_STATUS.json').read_text())['status']=='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP'
emit('COMPUTE_AUDIT',V1_fresh_immutable_rows=v1rows,V1_terminal='QUERY_EIGENBASIS_QUALIFICATION_FAIL_CLOSED_STOP',rows=rows,source_rows=sources,source_freeze_sha256=sha(R/'SOURCE_FREEZE.json'),manifest_sha256=sha(R/'SHA256_MANIFEST.json'),model_stat_rows=model,history_rows=hist,provenance=provenance,identities=identities,identities_still_live=alive,own_NVML=own_nvml,NVML=nvml,ledger=ledger,final_status=json.loads((R/'FINAL_TASK_STATUS.json').read_text()),seal=json.loads((R/'FINAL_SEAL_RECEIPT.json').read_text()),compute_receipt=json.loads((R/'FINAL_ARCHIVE_VERIFICATION.json').read_text()),compute_receipt_sha256=sha(R/'FINAL_ARCHIVE_VERIFICATION.json'))
bindings=C.bindings(R);emit('EXTERNAL_START',count=len(bindings),bytes=sum(x['bytes'] for x in bindings.values()))
for path,x in bindings.items():
 p=Path(path);before=p.stat();h=sha(p);after=p.stat();emit('EXTERNAL_ROW',path=path,bytes=after.st_size,sha256=h,match=h==x['sha256'] and after.st_size==x['bytes'] and (before.st_ino,before.st_mtime_ns)==(after.st_ino,after.st_mtime_ns))
p=subprocess.run(['ssh','-S','/tmp/qwen_replication_archive29.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.29',shlex.join(['/usr/bin/python3','-u','-B','-c',sys.argv[2],sys.argv[3]])],capture_output=True,text=True,timeout=300);assert p.returncode==0,p.stderr
emit('ARCHIVE_AUDIT',data=json.loads(p.stdout));emit('COMPLETE')
'''
command = ['ssh', '-S', CONFIG['ssh_control_path'], '-o', 'BatchMode=yes', '-o',
           'ConnectTimeout=15', 'zypan@202.38.247.55',
           shlex.join([CONFIG['remote_python'], '-u', '-B', '-c', CODE,
                       CONFIG['remote_root'], ARCHIVE_CODE, CONFIG['archive_root']])]
compute = None
archive = None
external = []
complete = False
with (OUT / 'SSH.stderr').open('w') as error, (OUT / 'STREAM.jsonl').open('x') as log:
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error, text=True)
    (OUT / 'STARTED.json').write_text(json.dumps(dict(pid=p.pid, unix=time.time(), remote_mutations=0, GPU_dispatches=0), indent=2) + '\n')
    for line in p.stdout:
        log.write(line)
        log.flush()
        x = json.loads(line)
        if x['kind'] == 'COMPUTE_AUDIT':
            compute = x
            print('Fresh compute manifest, source, process/release and history audit received.', flush=True)
        elif x['kind'] == 'EXTERNAL_START':
            print('Fresh external audit: %d files, %d bytes.' % (x['count'], x['bytes']), flush=True)
        elif x['kind'] == 'EXTERNAL_ROW':
            external.append(x)
            if len(external) % 10 == 0:
                print('External SHA: %d/61.' % len(external), flush=True)
        elif x['kind'] == 'ARCHIVE_AUDIT':
            archive = x['data']
            print('Independent archive all-file SHA/bytes audit received.', flush=True)
        elif x['kind'] == 'COMPLETE':
            complete = True
    rc = p.wait()
assert rc == 0 and complete and compute and archive, ('Incomplete independent read-only audit', rc)
assert compute['rows'] and all(x['match'] for x in compute['rows'])
assert compute['source_rows'] and all(x['match'] and x['included_in_final_manifest'] for x in compute['source_rows'])
assert compute['source_freeze_sha256'] == hashlib.sha256((R/'SOURCE_FREEZE.json').read_bytes()).hexdigest()
assert len(external) == 61 and all(x['match'] for x in external)
assert len(compute['model_stat_rows']) == 17 and all(x['match'] for x in compute['model_stat_rows'])
assert all(x['match'] for x in compute['history_rows'])
assert len(compute['ledger']['intervals']) >= 1
assert all(x['release'] == 'PASS' for x in compute['ledger']['intervals'])
if compute['final_status']['status']=='FORMAL_COMPLETE_90_90_AND_STOP':
 assert len(compute['ledger']['intervals'])==169 and all(x['exit_returncode']==0 for x in compute['ledger']['intervals'])
assert not compute['identities_still_live'] and not compute['own_NVML']
assert archive['status'] == 'PASS' and len(archive['rows']) == len(compute['rows']) and all(x['match'] for x in archive['rows'])
assert archive['manifest_sha256'] == compute['manifest_sha256'] == compute['seal']['manifest_sha256']
assert archive['seal'] == compute['seal'] and archive['receipt'] == compute['compute_receipt']
assert archive['receipt_sha256'] == compute['seal']['archive_verification_sha256']
assert compute['final_status']['status'] in ['FORMAL_COMPLETE_90_90_AND_STOP','P1_COMPLETE_U_BRANCH_FAILED_AND_STOP','ROTATION_FP_FUNCTIONAL_QUALIFICATION_FAIL_CLOSED_STOP','V2_NUMERICAL_CONTRACT_FAIL_CLOSED_STOP','PARTIAL_RESULTS_PRESERVED_AND_STOP']
by_archive = {x['path']: x for x in archive['rows']}
source_paths = {x['path'] for x in compute['source_rows']}
raw_extensions = {'.bin', '.npz', '.npy', '.pt', '.pth', '.safetensors', '.f32'}
tensors = [x for x in compute['rows'] if Path(x['path']).suffix.lower() in raw_extensions]
for x in tensors:
    y = by_archive[x['path']]
    assert (x['bytes'], x['sha256']) == (y['bytes'], y['sha256'])
result = dict(status='PASS_FINAL_INDEPENDENT_READONLY_SCIENCE_AND_ARCHIVE_AUDIT',
              source_count=len(compute['source_rows']), actual_external_fresh_SHA_count=61, model_stat_count=17,
              compute_and_archive_fresh_manifest_file_count=len(compute['rows']), paired_tensor_count=len(tensors),
              newly_created_tensor_count=sum(x['path'].startswith('calibration/') or x['path'] not in source_paths for x in tensors),V1_original585_fresh_immutable_verified=len(compute['V1_fresh_immutable_rows'])==585,
              historical_input_records=len(compute['history_rows']), all_observed_units_released=True,observed_unit_intervals=len(compute['ledger']['intervals']),
              all_exact_own_process_identities_exited=True, own_NVML_empty=True,
              source_freeze_sha256=compute['source_freeze_sha256'], manifest_sha256=compute['manifest_sha256'],
              archive_receipt_sha256=archive['receipt_sha256'],
              compute_receipt_mirror_sha256=compute['compute_receipt_sha256'],
              receipt_mirror_semantically_equal_to_exact_sealed_archive_original=True,
              receipt_serialization_difference_preserved=False, science_seal_rewritten=False,
              frozen_legacy_metadata_absent_from_science_manifest=[x for x in compute['source_rows'] if not x['included_in_final_manifest']],
              remote_mutations=0, GPU_retries=0, unix=time.time())
(OUT / 'COMPUTE_AUDIT.json').write_text(json.dumps(compute, indent=2) + '\n')
(OUT / 'ARCHIVE_AUDIT.json').write_text(json.dumps(archive, indent=2) + '\n')
(OUT / 'EXTERNAL_FRESH_SHA.json').write_text(json.dumps(external, indent=2) + '\n')
(OUT / 'TENSOR_DOUBLE_SHA.json').write_text(json.dumps(tensors, indent=2) + '\n')
(OUT / 'RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n')
(R / 'operations/FINAL_INDEPENDENT_READONLY_VERIFICATION.json').write_text(json.dumps(dict(result, evidence_path=str(OUT)), indent=2) + '\n')
print(json.dumps(dict(result, evidence_path=str(OUT))), flush=True)
