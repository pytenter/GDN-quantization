"""Record the actual engineering-freeze commit; never label it full prereg."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,POLICY,REPO,read,save,sha

commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
source=AMEND/'scripts/capture_original_drive.py'
for path in [source,POLICY,AMEND/'protocol/remaining_engineering_gate_policy.json']:
    rel=str(path.relative_to(REPO))
    sealed=subprocess.check_output(['git','show',commit+':'+rel],cwd=REPO)
    import hashlib
    assert hashlib.sha256(sealed).hexdigest()==sha(path),'committed freeze differs from current file'
save(AMEND/'capture_freeze_receipt.json',dict(engineering_freeze_commit=commit,
    capture_script_sha256=sha(source),capture_policy_sha256=sha(POLICY),
    remaining_engineering_policy_sha256=sha(AMEND/'protocol/remaining_engineering_gate_policy.json'),
    full_scientific_premeasurement_freeze=False,PREMEASUREMENT_COMMIT_SHA=None,
    record_role='ENGINEERING_FREEZE_ONLY; scientific capture/replay not authorized yet'))
print('ENGINEERING_FREEZE_RECEIPT_CREATED',flush=True)
