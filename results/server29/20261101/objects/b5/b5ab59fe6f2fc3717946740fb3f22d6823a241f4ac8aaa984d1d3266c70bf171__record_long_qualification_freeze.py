"""Actual Git commit receipts before fresh-process qualification results."""
import hashlib
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,REPO,read,save,sha
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
names=['capture_extended_drive.py','persistent_anchor_replay.py','qualify_long_and_streaming.py',
    'prepare_long_qualification.py','record_long_qualification_freeze.py','run_long_qualification.sh']
paths=[AMEND/'scripts'/n for n in names]+[AMEND/'protocol/extended_capture_policy.json']
hashes={}
for path in paths:
    content=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO)
    assert hashlib.sha256(content).hexdigest()==sha(path)
    hashes[str(path.relative_to(AMEND))]=sha(path)
rules=AMEND/'protocol/remaining_engineering_gate_policy.json'
save(AMEND/'long_qualification_freeze.json',dict(freeze_commit=commit,code_sha256=hashes,
    rules_sha256=sha(rules),policy_unchanged=True,scientific_premeasurement_commit=False))
save(AMEND/'extended_capture_freeze.json',dict(freeze_commit=commit,
    policy_sha256=sha(AMEND/'protocol/extended_capture_policy.json'),
    script_sha256=sha(AMEND/'scripts/capture_extended_drive.py'),scientific_premeasurement_commit=False))
print('LONG_QUALIFICATION_FROZEN',flush=True)
