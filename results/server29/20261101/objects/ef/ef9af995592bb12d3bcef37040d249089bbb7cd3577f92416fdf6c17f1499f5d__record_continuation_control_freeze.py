"""Record actual committed controller identity; no launch or GPU action."""
import hashlib
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from capture_original_drive import AMEND,REPO,save,sha
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
hashes={}
for name in ['scientific_lifecycle.py','continue_after_qualification.sh','record_continuation_control_freeze.py']:
    path=AMEND/'scripts'/name
    content=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO)
    assert hashlib.sha256(content).hexdigest()==sha(path)
    hashes[str(path.relative_to(AMEND))]=sha(path)
save(AMEND/'continuation_control_freeze.json',dict(controller_engineering_commit=commit,
    code_sha256=hashes,scientific_preregistration_completed=False,
    execution='Wait forG11/G12 PASS; actual full premeasurement commit; sequential Doc1..4 lifecycle; any new failure STOP'))
print('CONTINUATION_CONTROL_FROZEN',flush=True)
