"""Wait for the existing tmux job, then seal its actual CPU results once."""
import os
import pathlib
import subprocess
import sys
import time
D=pathlib.Path(os.environ['LING_PROSPECTIVE_RESULTS'])
while True:
    terminal=(D/'PROSPECTIVE_COMPLETE.json').exists() or (D/'STOP.json').exists()
    running=subprocess.run(['tmux','has-session','-t','ling_prospective_4doc_v1'],capture_output=True).returncode==0
    if terminal and not running:break
    time.sleep(10)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
raise SystemExit(subprocess.run([sys.executable,'-B','-u',str(D/'finalize_cpu.py')],env=env).returncode)
