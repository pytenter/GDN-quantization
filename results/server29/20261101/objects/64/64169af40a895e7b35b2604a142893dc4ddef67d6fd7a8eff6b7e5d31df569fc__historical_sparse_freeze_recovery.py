"""Finish an interrupted exact-byte two-end freeze; no scientific changes."""
from pathlib import Path
import hashlib,json,subprocess
R=Path('/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3')
A=Path(str(R)+'-authority')
J=A/'ling-long-horizon-observability-bridge-v3-scientific-repro-historical-premeasurement.journal.json'
j=json.loads(J.read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
assert git('rev-parse','HEAD')==j['old']
for row in j['files']:
    assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256']
receipt=A/'PREMEASUREMENT_SPARSE_FREEZE_ENGINEERING_RECOVERY.json'
with receipt.open('x') as f:
    json.dump(dict(old=j['old'],commit=j['commit'],failure='Git add rejected new experiment paths outside sparse checkout',correction='Exact journal-scoped git add --sparse; no checkout/source/science bytes changed',SCIENTIFIC_PROTOCOL_CHANGED='NO',FORMAL_MEASUREMENTS=0,journal_sha256=hashlib.sha256(J.read_bytes()).hexdigest()),f,indent=2)
subprocess.run(['git','add','--sparse','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=('\0'.join(j['names'])+'\0').encode(),cwd=R,check=True)
assert git('write-tree','--missing-ok')==git('rev-parse',j['commit']+'^{tree}')
subprocess.run(['git','update-ref','refs/heads/exp/ling-long-horizon-observability-bridge-v3',j['commit'],j['old']],cwd=R,check=True)
assert not git('status','--porcelain=v1')
print('EXACT_BYTE_SPARSE_FREEZE_RECOVERY_PASS',j['commit'],flush=True)
