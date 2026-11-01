"""Independent CPU delivery recovery; never reruns the failed helper or GPU work."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'operations'))
from transport import run

SOURCE = 'f4a2224d206f9b33999b2352ceee0b87e3053d308cba3dd62a54a97453d26427'
REMOTE = '/data/zypan/analyses/pace48_lqunroll_dp32_throughput_v1/releases/ASSET_FIX_V2'
NAME = 'CPU_TERMINAL_DELIVERY_RECOVERY_20261029'
DEST = ROOT / '_local' / NAME

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def atomic(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '_local/READER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    failed = ROOT / '_local/independent_delivery'
    assert (failed / 'CLAIM.json').exists() and (failed / 'FIRST_FAILURE.json').exists()
    assert 'SyntaxError' in (failed / 'FIRST_FAILURE.json').read_text()
    assert not (DEST / 'CLAIM.json').exists(), 'Recovery is already claimed; no automatic repeat'
    plan = dict(
        scope='New independent CPU recovery after remote compilation failure; no GPU or finisher rerun',
        original_failed_claim_sha256=sha(failed / 'CLAIM.json'),
        original_first_failure_sha256=sha(failed / 'FIRST_FAILURE.json'),
        original_frozen_monitor_sha256=sha(ROOT / 'operations/monitor_dp32.py'),
        recovery_script_sha256=sha(Path(__file__)), source_sha256=SOURCE,
        output_files=['PLAN.json', 'CLAIM.json', 'REMOTE_REVIEW.json', 'LOCAL_TEXT_SHA.json', 'RESULT.json'],
        remote_audit_sibling=REMOTE + '_' + NAME,
        corrections=['Use a raw remote script so newline in manifest is valid Python',
                     'Include explicitly declared textual .log artifacts in the text allowlist'],
        original_claim_and_frozen_source_preserved=True,
        extra_review_still_required=True)
    code = r'''
import sys,json,os,time,stat,hashlib
from pathlib import Path
r=Path(REMOTE)
sys.path.insert(0,str(r/'src'))
import execution_common as C
assert C.sha(r/'SOURCE_MANIFEST.json')==SOURCE
for x in json.loads((r/'SOURCE_MANIFEST.json').read_text())['files']:
 p=r/x['path'];assert p.stat().st_size==x['bytes'] and C.sha(p)==x['sha256'],x['path']
assert not (r/'MANIFEST.sha256').exists(), 'Original manifest must not already be sealed'
assert not (r/'runtime/INDEPENDENT_DELIVERY_CLAIM.json').exists(), 'Original failed remote program must not have executed'
assert (r/'runtime/PARENT_COMPLETE.json').exists()
bash=json.loads((r/'runtime/BASH_WAIT.json').read_text())
assert type(bash['actual_Bash_wait_integer']) is int and bash['actual_Bash_wait_integer']==0
assert bash['source_sha256']==SOURCE
parent=json.loads((r/'runtime/UNIQUE_PARENT_STARTED.json').read_text())
receipts=[]
identities=[parent['owner'],parent['bash']]
for p in (r/'runtime').glob('*/OWNING_EXIT_AND_RELEASE.json'):
 x=json.loads(p.read_text())
 assert x['release_observed'] is True and type(x['owning_Popen_exit']) is int and x['owning_Popen_exit']==0
 if 'source_sha256' in x:assert x['source_sha256']==SOURCE
 else:assert x.get('GPU_credit')==0 and x.get('scope')=='Pre-freeze CPU control/report test only'
 identities.append(x['child'])
 receipts.append(dict(path=str(p.relative_to(r)),sha256=C.sha(p),owning_exit=x['owning_Popen_exit']))
assert {x['path'].split('/')[1] for x in receipts}=={'CPU_SELFTEST','THROUGHPUT_SUITE','CPU_FINAL_REPORT'}
for ident in identities:
 try:cur=C.identity(ident['pid'])
 except FileNotFoundError:cur=None
 assert cur is None or cur['starttick']!=ident['starttick'],'Exact owning epoch still present'
uuid='GPU-30b07aba-a728-7649-4899-e09d33f8bc75'
nvml=C.apps()
assert not any(x['uuid']==uuid and any(str(x['pid'])==str(i['pid']) for i in identities) for x in nvml)
maps={}
for n in ['LOADED_REGULAR_MAPPED_SHA.json','POSTFLIGHT_REGULAR_MAPPED_SHA.json']:
 for x in json.loads((r/n).read_text())['files']:maps[(x['device'],x['inode'])]=x
for x in maps.values():
 p=Path(x['path']);s=p.stat()
 assert stat.S_ISREG(s.st_mode) and s.st_size==x['bytes'] and s.st_ino==x['inode'] and s.st_dev==x['device'] and C.sha(p)==x['sha256'],x['path']
model=C.model_pins(r,full=True)
measurements=sorted((r/'measurements').glob('*.json'))
assert len(measurements)==18
result=json.loads((r/'RESULT.json').read_text())
assert result['all_requested_benchmarks_completed'] is True and result['new_AIME_samples']==0
audit=dict(status='PASS_INDEPENDENT_CPU_RECOVERY_SOURCE_MODEL_MAPS_OWNING_AND_18_MEASUREMENTS',
 source_sha256=SOURCE,receipts=receipts,actual_regular_files=len(maps),model=model,
 scientific_status=result['status'],measurements=len(measurements),no_GPU_launch_or_foreign_signals=True,
 original_failed_claim_preserved=True,extra_review_pending=True,original_failure=ORIGINAL_FAILURE)
sibling=Path(REMOTE+'_'+NAME)
sibling.mkdir(exist_ok=False)
C.atomic(sibling/'REVIEW.json',audit)
allowed={'.py','.sh','.txt','.json','.jsonl','.md','.csv','.log'}
files=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name.endswith('.lock') or p.name.startswith('.'):continue
 assert p.suffix in allowed,str(p)
 files.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=C.sha(p)))
with (r/'MANIFEST.sha256').open('x') as f:
 for x in files:f.write(x['sha256']+'  '+x['path']+'\n')
 f.flush();os.fsync(f.fileno())
files.append(dict(path='MANIFEST.sha256',bytes=(r/'MANIFEST.sha256').stat().st_size,sha256=C.sha(r/'MANIFEST.sha256')))
with (sibling/'MANIFEST.sha256').open('x') as f:
 f.write(C.sha(sibling/'REVIEW.json')+'  REVIEW.json\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(audit=audit,files=files,sibling=str(sibling),sibling_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=C.sha(p)) for p in sorted(sibling.iterdir())])))
'''
    # Validate the exact generated remote text before any remote claim or mutation.
    code = 'REMOTE='+repr(REMOTE)+'\nSOURCE='+repr(SOURCE)+'\nNAME='+repr(NAME)+'\nORIGINAL_FAILURE='+repr(plan)+'\n'+code
    compile(code, '<independent-cpu-recovery>', 'exec')
    if '--validate-script-only' in sys.argv:
        print(json.dumps(dict(status='PASS_LOCAL_AND_EXACT_REMOTE_CODE_COMPILE',claim_created=False)))
        return
    atomic(DEST / 'PLAN.json', plan)
    atomic(DEST / 'CLAIM.json', dict(pid=os.getpid(), client_time=time.time(), plan_sha256=sha(DEST / 'PLAN.json')))
    try:
        review = run(code, timeout=None)
        atomic(DEST / 'REMOTE_REVIEW.json', review)
        rows=[]
        for group,remote_base,local_base in [('primary',REMOTE,ROOT),('recovery',review['sibling'],DEST/'REMOTE_SIBLING')]:
            for row in review['files' if group=='primary' else 'sibling_files']:
                rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
                p=local_base/rel
                if p.exists():assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
                else:
                    payload=run('import json,base64,hashlib\nfrom pathlib import Path\np=Path('+repr(remote_base)+')/'+repr(str(rel))+'\nb=p.read_bytes();print(json.dumps(dict(data=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest())))',timeout=None)
                    b=base64.b64decode(payload['data'])
                    assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']==payload['sha256']
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
                rows.append(dict(**row,group=group,actual_local_path=str(p)))
        # Independently parse and rehash the sealed primary manifest on the Mac.
        lines=(ROOT/'MANIFEST.sha256').read_text().splitlines()
        assert len(lines)==len(review['files'])-1
        for line in lines:
            digest,name=line.split('  ',1);assert sha(ROOT/name)==digest
        atomic(DEST/'LOCAL_TEXT_SHA.json',dict(status='PASS_ALL_PRIMARY_AND_RECOVERY_TEXT_SHA',files=rows,no_tensor_model_environment_or_compiled_binary_download=True))
        result=dict(status='PASS_NEW_INDEPENDENT_CPU_RECOVERY_55_MAC_DELIVERY',scientific_status=review['audit']['scientific_status'],files=len(rows),source_sha256=SOURCE,manifest_sha256=sha(ROOT/'MANIFEST.sha256'),extra_review_pending=True,original_failed_claim_preserved=True)
        atomic(DEST/'RESULT.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic(DEST/'FIRST_FAILURE.json',dict(error=repr(exc),no_automatic_repeat=True))
        raise

if __name__=='__main__':main()
