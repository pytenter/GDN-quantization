"""Append-only user-authorized scope recovery; no scientific implementation edits."""
from inherit import *
import json,time

def load(p):return json.loads(Path(p).read_text())
def emit(p,x):
    p=D/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as s:json.dump(x,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')

assert not REMOTE and not (D/'run/STARTED.json').exists()
assert not list((D/'raw/confirmation').glob('*.json'))
assert c.command(['git','rev-parse','HEAD'])=='ecc792a35b4d2d45b0b0f2eccf9d0b1c1481ef49'
stop=load(D/'provenance/3090_MIGRATION_PRESTART_STOP.json')
assert stop['original_file_byte_identity']=='PASS' and stop['original_files_checked']==12516
root='/data01/user2/worktrees/ling-recurrent-dense-l6-l7-v1/experiments/LING_L6_L7_DATAADAPTIVE_ROTATION_V1'
tags=['geo_L6_x0p5','geo_L6_x0p75','geo_L6_x1p25']
exact=['rotations/GEO_L6_x0p5_final_rotation.pt','rotations/GEO_L6_x0p75_final_rotation.pt','rotations/GEO_L6_x1p25_final_rotation.pt']
exact+=['runs/wave_'+t+'.log' for t in tags]
allow=dict(exact_paths=exact,directory_prefixes=['runs/'+t+'/' for t in tags])
for tree in stop['membership']:
    for path in tree['added']:
        assert tree['root']==root
        rel=str(Path(path).relative_to(root))
        assert rel in exact or any(rel.startswith(p) for p in allow['directory_prefixes'])
emit('protocol/parent_integrity_scope_recovery_v1.json',dict(
    amendment='LING_LONG_HORIZON_PROPAGATION_CONFIRM_PARENT_INTEGRITY_SCOPE_RECOVERY_V1',
    scope='EXTERNAL_ADDITIONS_ONLY_ORIGINAL_FILE_IDENTITY_UNCHANGED',user_authorization='继续启动',
    preceding_question='是否允许将这些外部新增文件单独审计、排除出本实验的保护范围，保留原文件的严格SHA256检查，然后继续启动？',
    original_protected_inventory=c.row(D/'provenance/parent_identity.json'),
    historical_prestart_stop=c.row(D/'provenance/3090_MIGRATION_PRESTART_STOP.json'),
    original_protected_file_count=12516,external_allowlist={root:allow},
    original_files_removed_or_modified='ALWAYS_FAIL',other_new_paths='ALWAYS_FAIL',
    external_additions='AUDIT_BEFORE_AND_AFTER; NEVER_LOAD_AS_SCIENTIFIC_INPUTS',
    existing_inputs_or_snapshot_replaced=False,previous_failure_erased=False,
    hardware_assignment_changed=False,scientific_functions_changed=False,metrics_or_statistics_changed=False,
    holdout_documents_changed=False,rotation_bank_changed=False,formal_measurements_before_recovery=0,
    NEW_TRAINING='NO',NEW_AIME_GENERATION=0,FINAL_ACTION_AFTER_CONFIRMATION='STOP',time_unix=time.time()))
print('USER_AUTHORIZED_SCOPE_RECOVERY_PREPARED; FORMAL_MEASUREMENTS=0',flush=True)
