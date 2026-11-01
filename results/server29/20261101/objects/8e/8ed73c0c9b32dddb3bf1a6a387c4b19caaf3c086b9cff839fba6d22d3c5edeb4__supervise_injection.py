#!/usr/bin/env python3
"""Monitor immutable Phase-2 units, then run Phase-3 feasibility and Phase-4 analysis."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_HADAMARD_ERROR_INJECTION_FUNCTIONAL_ANALYSIS_V1')
PYTHON = '/data/ydai/miniconda3/envs/bitdecode/bin/python'


def save(obj):
    path = OUT/'analysis/supervisor_status.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sessions():
    names = []
    for index in range(4):
        name = f'injection_formal{index}'
        status = subprocess.run(['tmux', 'has-session', '-t', name], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0
        if status:
            names.append(name)
    return names


def status(label, count, active, detail=None):
    save({'status': label, 'completed_units': count, 'target_units': 60,
          'active_tmux': active, 'disk_free_gib': shutil.disk_usage(OUT).free/2**30,
          'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'detail': detail})


def run_script(script, logfile, env=None):
    with (OUT/'analysis'/logfile).open('w', encoding='utf-8') as stream:
        result = subprocess.run([PYTHON, script], cwd=str(OUT), env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    while True:
        paths = sorted((OUT/'analysis').glob('shard_*/*.json'))
        count = len(paths)
        active = sessions()
        if count > 60:
            status('FAILED_UNIT_COUNT', count, active)
            return
        if count == 60:
            break
        if not active:
            status('FAILED_SHARD_EXIT', count, active,
                   'At least one formal shard exited before all 60 unit files existed; inspect formal logs.')
            return
        status('CAPTURING', count, active)
        time.sleep(60)
    status('LOCAL_CAPTURE_COMPLETE', 60, active)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    status('FULL_LOGITS_FEASIBILITY_RUNNING', 60, active)
    if run_script('full_logits_feasibility.py', 'full_logits_feasibility.log', env) != 0:
        status('FULL_LOGITS_FEASIBILITY_FAILED', 60, sessions(), 'Inspect full_logits_feasibility.log')
        return
    feasibility = json.loads((OUT/'analysis/full_logits_feasibility.json').read_text(encoding='utf-8'))
    if feasibility['status'] == 'FULL_LOGITS_PARITY_FAILED':
        status('FULL_LOGITS_PARITY_FAILED', 60, sessions(), 'Full-path parity needs diagnosis; no analysis run')
        return
    if feasibility['status'] == 'FEASIBLE_FOR_FULL_CLEAN59':
        status('FULL_LOGITS_FEASIBLE_NEEDS_FORMAL_RUN', 60, sessions(),
               'Do not substitute pilot metrics; full clean59 logit capture required before final analysis')
        return
    if feasibility['status'] != 'NOT_FEASIBLE_FULL_CLEAN59':
        status('UNKNOWN_FEASIBILITY_STATUS', 60, sessions(), feasibility['status'])
        return
    status('ANALYZING_CLEAN59', 60, sessions())
    if run_script('analyze_injection.py', 'analysis.log') != 0:
        status('ANALYSIS_FAILED', 60, sessions(), 'Inspect analysis.log; preserve prior outputs')
        return
    status('COMPLETE', 60, sessions(), 'Exact terminal summary in analysis.log and final_report.md')


if __name__ == '__main__':
    main()
