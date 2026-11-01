"""Seal the actual scheduling commit before any new GPU qualification."""
import os
import subprocess
from pathlib import Path
import sys
REPO=Path('/data/zypan/worktrees/qwen-gdn-frozen-drive-anchor-mechanism-v1')
ROOT=REPO/'experiments/QWEN_GDN_FROZEN_DRIVE_ANCHOR_MECHANISM_V1'
PAR=ROOT/'amendments/FOUR_GPU_ANCHOR_PARALLELISM_V1'
sys.path.insert(0,str(PAR/'scripts'))
from capture_original_drive import sha,read,save

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in sorted(PAR.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    for p in files:
        blob=subprocess.check_output(['git','show',commit+':'+str(p.relative_to(REPO))],cwd=REPO)
        import hashlib
        assert hashlib.sha256(blob).hexdigest()==sha(p),'Scheduling commit is incomplete'
    save(PAR/'controller_commit_receipt.json',dict(scheduling_commit_sha=commit,
        files={str(p.relative_to(PAR)):sha(p) for p in files},scientific_premeasurement_commit=False,
        any_new_failure='FAIL_AND_STOP'))
    files.append(PAR/'controller_commit_receipt.json')
    with (PAR/'SHA256SUMS').open('x') as f:
        for p in files:f.write(sha(p)+'  '+str(p.relative_to(PAR))+'\n')
    (PAR/'SHA256SUMS').chmod(0o444)
    print('SCHEDULING_FREEZE_COMMIT '+commit,flush=True)

if __name__=='__main__':main()
