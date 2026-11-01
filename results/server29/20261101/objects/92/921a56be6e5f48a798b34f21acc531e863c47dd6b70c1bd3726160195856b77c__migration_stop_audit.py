"""Read-only diagnosis of the failed parent membership gate. No measurements."""
import concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
D=Path('/data01/user2/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
identity=json.loads((D/'provenance/parent_identity.json').read_text())
membership=[]
for tree in identity['trees']:
    now={str(p) for p in Path(tree['root']).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    old=set(tree['paths'])
    membership.append(dict(root=tree['root'],before_count=len(old),now_count=len(now),added=sorted(now-old),removed=sorted(old-now)))
def verify(i):
    p=Path(i['path'])
    if not p.is_file():return dict(path=i['path'],status='MISSING',expected_sha256=i['sha256'])
    n=p.stat().st_size;h=sha(p)
    if n!=i['bytes'] or h!=i['sha256']:return dict(path=i['path'],status='CHANGED',expected_sha256=i['sha256'],observed_sha256=h,expected_bytes=i['bytes'],observed_bytes=n)
    return None
changed=[];checked=0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(verify,identity['files']):
        checked+=1
        if result is not None:changed.append(result)
        if checked%1000==0:print('PROTECTED_IDENTITY_AUDIT',checked,flush=True)
report=dict(gate='FAIL',failed_gate='PARENT_PROTECTED_TREE_MEMBERSHIP_IDENTITY',membership=membership,
    original_files_checked=checked,original_files_changed=changed,
    original_file_byte_identity='PASS' if not changed else 'FAIL',parent_membership_gate_not_relaxed=True,
    premeasurement_failure_sha256=sha(D/'run/PREMEASUREMENT_freeze_FAILURE_None.json'),
    formal_confirmation_measurements=0,formal_workers_launched=0,
    two3090_development_qualifications='PASS_GPU3_AND_GPU5',cross_hardware_FP_bitwise='FAIL_RETAINED_NOT_MIXED',
    external_files_deleted_or_moved=False,external_processes_terminated=False,
    next_action_requires_user_direction='May not ignore newly added scientific rotation/run objects in a protected parent tree without a separate explicit integrity-scope recovery authorization.',
    FINAL_ACTION='STOP_BEFORE_FORMAL_MEASUREMENT',time_unix=time.time())
out=D/'provenance/3090_MIGRATION_PRESTART_STOP.json'
with out.open('x') as s:json.dump(report,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')
print('PRESTART_STOP_AUDIT_COMPLETE',checked,'changed',len(changed),'added',sum(len(m['added']) for m in membership),'removed',sum(len(m['removed']) for m in membership),flush=True)
