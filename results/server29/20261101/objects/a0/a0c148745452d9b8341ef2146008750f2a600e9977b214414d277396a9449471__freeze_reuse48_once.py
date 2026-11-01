"""Freeze and publish one CPU-qualified continuation. No GPU dispatch."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('staging',ROOT/'operations/reuse48_stage.py')
staging=importlib.util.module_from_spec(spec);spec.loader.exec_module(staging)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def new(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    assert ROOT.name=='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1_REUSE48_ARCHIVE29_20261025'
    assert not any((ROOT/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json'])
    admission=read(ROOT/'analysis/FINAL_REUSE48_CPU_ADMISSION.json')
    assert admission['status']=='PASS_FINAL_REUSE48_CPU_AND_STORAGE_ADMISSION' and admission['GPU_calls']==0
    for row in admission['current_source_pins']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    reuse=read(ROOT/'QUALIFICATION_REUSE_ADMISSION.json')
    assert reuse['qualification_reused']==48 and reuse['qualification_new']==4 and reuse['formal_new']==90
    assert reuse['dual_host_fresh_review_sha256']==sha(ROOT/reuse['dual_host_fresh_review_path'])
    for row in reuse['source_math_unchanged']:
        assert (ROOT/row['path']).stat().st_size==row['bytes'] and sha(ROOT/row['path'])==row['sha256']
    excluded={'operations/read_reuse48_snapshot_once.py','operations/deliver_reuse48_terminal_cpu_once.py','tests/reuse48_terminal_cpu.py','operations/CURRENT_REUSE48_HANDOFF.json','operations/REUSE48_MONITOR_BINDING.json'}
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        assert not p.is_symlink(),str(p)
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo'):continue
        name=p.relative_to(ROOT).as_posix()
        if name in excluded:continue
        assert p.suffix in {'.py','.sh','.json','.jsonl','.md','.txt','.log','.patch','.llir','.ttir','.ttgir','.ptx'},name
        rows.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
    assert len({x['path'] for x in rows})==len(rows)
    batches=[];current=[];size=0
    for row in rows:
        if current and size+row['bytes']>2*1024**2:batches.append(current);current=[];size=0
        current.append(ROOT/row['path']);size+=row['bytes']
    if current:batches.append(current)
    receipts=[staging.stage(batch) for batch in batches]
    new(ROOT/'operations/FINAL_REUSE48_PREFREEZE_STAGING.json',dict(status='PASS',batches=receipts,source_files=len(rows),GPU_calls=0))
    source=dict(task_id='QWEN_GDN_SYM8_G32_WSCALE_ROWBAL_AIME26_SEED1_V1',execution_revision=ROOT.name,
        status='FROZEN_ONCE_REUSE48_PLUS4_THEN90_ARCHIVE29',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        count=len(rows),bytes=sum(x['bytes'] for x in rows),files=rows,qualifications_current=4,qualification_reused=48,
        qualifications_required=52,formal_new_samples=90,old_formal_credit=0,no_GPU_retry=True,
        original_byteview_source_sha256=staging.SOURCE,original_failed_exit_not_reclassified=True,
        shared_calibration_reused_unchanged=True,source_hotpatch_after_STARTED=False)
    new(ROOT/'SOURCE_FREEZE.json',source)
    permit=read(ROOT/'provenance/BYTEVIEW_TRUE_STOP_METADATA/GPU_PERMIT.json')
    permit.update(execution_root=staging.REMOTE,execution_revision=ROOT.name,source_freeze_sha256=sha(ROOT/'SOURCE_FREEZE.json'),
        scope='ONE_NEW_INDEPENDENT_REUSE48_PLUS4_THEN90_ARCHIVE29_ONCE',old_qualification_credit=48,fresh_qualification_count=4,
        old_formal_credit=0,reuse48_authorization_sha256=sha(ROOT/'operations/HUMAN_REUSE48_ARCHIVE29_AUTHORIZATION.json'),
        qualification_reuse_admission_sha256=sha(ROOT/'QUALIFICATION_REUSE_ADMISSION.json'),
        durable29_live_root=read(ROOT/'ROOT.json')['live_archive_root'])
    new(ROOT/'GPU_PERMIT.json',permit)
    payload={n:(ROOT/n).read_text() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json']}
    code='ROOT='+repr(staging.REMOTE)+'\npayload='+repr(payload)+'\n'+r'''
from pathlib import Path
import json,hashlib,os,sys
r=Path(ROOT);assert not r.is_symlink()
assert not any((r/n).exists() for n in ['SOURCE_FREEZE.json','GPU_PERMIT.json','GPU_RUN_STARTED.json','SESSION_RESULT.json'])
source=json.loads(payload['SOURCE_FREEZE.json']);assert source['count']==len(source['files']) and source['qualifications_current']==4 and source['qualification_reused']==48
for x in source['files']:
 p=r/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
catalog=json.loads((r/'calibration/RAW_CLONE_VERIFICATION.json').read_text());assert catalog['count']==4323
for x in catalog['files']:
 p=r/x['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
sys.path.insert(0,str(r/'src'));import execution_control as C
C.model_guard(r);storage=C.storage_guard(r);devices=[C.device(g) for g in C.AUTHORIZED_GPUS]
assert not list((r/'execution').glob('*/DISPATCH.json')) and not list((r/'formal').glob('*/DISPATCH.json'))
for n,raw in payload.items():
 p=r/n;tmp=p.with_name(n+'.publish_'+str(os.getpid()))
 with tmp.open('xb') as f:f.write(raw.encode());f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink()
fd=os.open(r,os.O_RDONLY);os.fsync(fd);os.close(fd)
C.authorize(r,r/'GPU_PERMIT.json')
from qualification_reuse import admit_reused48
proof=admit_reused48(r,C.read(r/'GPU_EXECUTION_PLAN.json'))
assert proof['count']==48 and proof['new_GPU_intervals']==0
print(json.dumps(dict(status='PASS_ACTUAL_NEW_FROZEN_SOURCE_PERMIT_AUTHORIZE_AND_ORIGIN48_NO_DISPATCH',source_sha256=C.sha(r/'SOURCE_FREEZE.json'),permit_sha256=C.sha(r/'GPU_PERMIT.json'),files=source['count'],bytes=source['bytes'],storage=storage,selected_devices=devices,reused48=proof,GPU_dispatches=0)))
'''
    compile(code,'<validated-freeze-publication>','exec')
    result=staging.transport.run(code,role='55',python=staging.PYTHON,timeout=120)
    assert result['source_sha256']==sha(ROOT/'SOURCE_FREEZE.json') and result['permit_sha256']==sha(ROOT/'GPU_PERMIT.json')
    new(ROOT/'operations/FROZEN_REUSE48_PUBLICATION_RECEIPT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected_devices','reused48']}))

if __name__=='__main__':main()
