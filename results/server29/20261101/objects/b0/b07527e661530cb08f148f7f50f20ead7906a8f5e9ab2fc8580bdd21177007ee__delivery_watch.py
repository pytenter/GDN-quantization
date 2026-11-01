"""Finish CPU delivery after the exact two-key tmux controller terminates."""
import os
import subprocess
import sys
import time
from pathlib import Path

D=Path(os.environ["LING_CROSS_RESULTS"])
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
while True:
    terminal=(D/"CROSS_COMPLETE.json").exists() or (D/"STOP.json").exists()
    active=subprocess.run(["tmux","has-session","-t","ling_mixed_cross_v1"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    if terminal and not active:break
    time.sleep(5)
subprocess.run([sys.executable,"-B","-u",str(D/"finalize_results_cpu.py")],check=True)
