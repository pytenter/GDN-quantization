"""Two fresh target CPU processes per changed-control contract; no CUDA/model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('DETACH', 'scripts/test_profiler_detach_lifecycle_cpu.py', 21, False, False),
    ('GUARD', 'scripts/test_directory_preparation_cpu.py', 74, False, True),
    ('NATIVE', 'scripts/test_runtime_native_closure_cpu.py', 47, True, False),
    ('CONTROL', 'scripts/test_resident_control_cpu.py', 79, False, True),
    ('EXIT', 'scripts/test_child_exit_transition_cpu.py', 30, True, False),
    ('FINISHER', 'scripts/test_resident_finisher_cpu.py', 45, True, False),
]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (ROOT / 'SOURCE_FREEZE.json').exists()
    started = ROOT / 'operations/PROFILER_REPAIR_CPU_V2_SUITE_STARTED.json'
    with started.open('x') as f:
        json.dump(dict(pid=os.getpid(), argv=sys.argv, monotonic=time.monotonic(), CUDA_VISIBLE_DEVICES=''), f)
    rows = []
    for case, script, count, root_argument, evidence_argument in CASES:
        for repeat in range(2):
            output = ROOT / 'analysis' / ('PROFILER_REPAIR_CPU_V2_' + case + '_R' + str(repeat) + '.json')
            argv = [sys.executable, '-u', '-B', str(ROOT / script), '--output', str(output)]
            if root_argument:
                argv += ['--root', str(ROOT)]
            with tempfile.TemporaryDirectory(prefix='resident-profiler-repair-cpu-') as temp:
                if evidence_argument:
                    argv += ['--evidence-root', str(Path(temp) / 'fixtures')]
                with (ROOT / 'operations' / ('PROFILER_REPAIR_CPU_V2_' + case + '_R' + str(repeat) + '.log')).open('xb') as log:
                    process = subprocess.Popen(argv, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=log)
                    code = process.wait()
                result = json.loads(output.read_text()) if output.exists() else None
                row = dict(case=case, repeat=repeat, pid=process.pid, actual_wait_returncode=code, result=result,
                           negative_fixtures_outside_science_root=True)
                rows.append(row)
                assert code == 0 and result and result.get('check_count', result.get('count')) == count, row
    assert 'torch' not in sys.modules
    with (ROOT / 'analysis/PROFILER_REPAIR_CPU_V2_TWO_FRESH_SUITE.json').open('x') as f:
        json.dump(dict(status='PASS_TWO_FRESH_TARGET_CPU_PROFILER_GUARD_NATIVE_CONTROL_EXIT_FINISHER',
             runs=rows, tests_each={name: count for name, _, count, _, _ in CASES}, GPU_dispatches=0,
             Torch_imported=False, model_forwards=0), f, sort_keys=True, indent=2)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        with (ROOT / 'operations/PROFILER_REPAIR_CPU_V2_SUITE_FAILURE.json').open('x') as f:
            json.dump(dict(error=repr(error), traceback=traceback.format_exc(), pid=os.getpid(),
                 GPU_dispatches=0, all_prior_CPU_receipts_preserved=True), f, sort_keys=True, indent=2)
        raise
