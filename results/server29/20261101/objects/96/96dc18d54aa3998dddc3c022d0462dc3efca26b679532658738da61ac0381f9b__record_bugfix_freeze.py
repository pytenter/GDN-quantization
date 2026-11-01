"""Actual committed single surgical patch, proof and reused PASS receipt."""
import hashlib
import json
import subprocess
from pathlib import Path
from audit_prepare_bugfix import REPO,ROOT,OLD,BUG,read,save,sha,text

commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert read(BUG/'static_source_audit.json')['STATIC_SOURCE_AUDIT']=='PASS'
paths=sorted(p for p in (BUG/'scripts').glob('*') if p.is_file() and p.suffix in ['.py','.sh'])
paths.extend(sorted(p for p in (BUG/'protocol').glob('*') if p.is_file()))
hashes={}
for path in paths:
    content=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO)
    assert hashlib.sha256(content).hexdigest()==sha(path)
    hashes[str(path.relative_to(BUG))]=sha(path)
save(BUG/'bugfix_commit_receipt.json',dict(bugfix_commit_sha=commit,code_sha256=hashes,
    surgical_diff_sha256=sha(BUG/'source_diff.patch'),static_audit_sha256=sha(BUG/'static_source_audit.json'),
    old_failure_snapshot_sha256=sha(BUG/'failed_attempt_snapshot.json'),scientific_commit=False,
    production_mathematical_operations_changed=0))
save(BUG/'extended_capture_freeze.json',dict(freeze_commit=commit,
    policy_sha256=sha(BUG/'protocol/extended_capture_policy.json'),script_sha256=sha(BUG/'scripts/capture_extended_drive.py'),
    scientific_premeasurement_commit=False,surgical_variable_fix=True))
save(BUG/'long_qualification_freeze.json',dict(freeze_commit=commit,code_sha256=hashes,
    rules_sha256=sha(BUG/'protocol/remaining_engineering_gate_policy.json'),
    policy_unchanged=True,scientific_premeasurement_commit=False,reused_G11_256=True))
save(BUG/'continuation_control_freeze.json',dict(controller_engineering_commit=commit,code_sha256=hashes,
    scientific_preregistration_completed=False,execution='Resume2048 only; G12; full premeasurement commit; unchanged sequential Discovery'))
status=dict(BUG_CLASSIFICATION='IMPLEMENTATION_VARIABLE_SHADOWING',BUGFIX_SCOPE='VARIABLE_RENAME_OR_SCOPE_CORRECTION_ONLY',
    FAILED_ATTEMPT_PRESERVED='YES',SCIENTIFIC_MEASUREMENT_BEFORE_FIX=0,G0_G10_PREVIOUS_PASS_PRESERVED='YES',
    G11_256_TOKEN_STATUS='PASS',G11_2048_TOKEN_PREVIOUS_STATUS='FAIL_IMPLEMENTATION_BUG',
    G11_2048_TOKEN_RETEST_STATUS='NOT_STARTED',G12_STATUS='NOT_STARTED',PREMEASUREMENT_COMMIT_SHA=None,
    DISCOVERY_COMPLETE='0/16',FINAL_ACTION='STATIC_AUDIT_PASS_RESUME_AUTHORIZED_G11_2048_ONLY',BUGFIX_COMMIT_SHA=commit)
save(BUG/'BUGFIX_STATUS_AT_FREEZE.json',status)
text(BUG/'BUGFIX_REPORT.md','# Authorized surgical variable-shadowing bugfix\n\n'+
    'Classification: IMPLEMENTATION_VARIABLE_SHADOWING. The original Tensor argument key was rebound by the metadata loop at line77. '+
    "After the loop its value was the string 'beta', and unchanged production dispatch at line81 received that string.\n\n"+
    'Correction: only the loop label and its three metadata uses become field_name. key is never assigned in the wrapper and remains the original argument Tensor, with unchanged value/dtype/shape/device/dataflow. '+
    'The entire source AST is identical after alpha-normalizing field_name back to key. No production mathematical operation changed.\n\n'+
    'Old failing path: '+str(OLD/'scripts/capture_extended_drive.py')+'\n\n'+
    'New version: '+str(BUG/'scripts/capture_extended_drive.py')+'\n\n'+
    'Actual bugfix commit SHA: '+commit+'\n\n'+
    'The original failed capture directory, source, logs, exitcodes, reports and all existing artifacts remain at their original paths and SHA values. '+
    'Only new output paths are routed to this amendment namespace to avoid overwriting those records; all non-path policy fields are identical. '+
    'G0–G10 and the three256-token runs are reused byte-identically, not rerun. Only failed G11 2048 capture/replay qualification resumes. '+
    'This engineering commit is not the full scientific PREMEASUREMENT_COMMIT_SHA. Any different/new failure stops without another patch or retry.\n\n'+
    '## Exact source diff\n\n```diff\n'+(BUG/'source_diff.patch').read_text()+'```\n\n'+
    '## Required status before GPU retest\n\n```json\n'+json.dumps(status,indent=2)+'\n```\n')
files=[p for p in sorted(BUG.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS']
lines=[sha(p)+'  '+str(p.relative_to(BUG))+'\n' for p in files]
for row in read(BUG/'failed_attempt_snapshot.json')['files']:
    lines.append(row['sha256']+'  '+str(Path(row['path']).relative_to(ROOT/'amendments')) .replace('STORAGE_ACCOUNTING_CORRECTION_V1/','../STORAGE_ACCOUNTING_CORRECTION_V1/',1)+'\n')
text(BUG/'SHA256SUMS',''.join(lines))
for line in (BUG/'SHA256SUMS').read_text().splitlines():
    expected,rel=line.split(None,1);assert sha((BUG/rel).resolve())==expected
print('SURGICAL_BUGFIX_STATIC_AUDIT_AND_ACTUAL_COMMIT_FROZEN',flush=True)
