"""Append-only actual code receipt for G5-G10, frozen numerical policy unchanged."""
import hashlib
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,REPO,save,sha
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
paths=[AMEND/'scripts/isolated_replay_engineering.py',AMEND/'scripts/run_short_qualification.sh',AMEND/'protocol/remaining_engineering_gate_policy.json']
for path in paths:
    content=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO)
    assert hashlib.sha256(content).hexdigest()==sha(path)
save(AMEND/'short_qualification_freeze.json',dict(engineering_freeze_commit=commit,
    replay_engineering_script_sha256=sha(paths[0]),run_script_sha256=sha(paths[1]),
    remaining_engineering_policy_sha256=sha(paths[2]),numerical_policy_changed=False,
    full_scientific_premeasurement_freeze=False,PREMEASUREMENT_COMMIT_SHA=None))
print('SHORT_QUALIFICATION_FREEZE_CREATED',flush=True)
