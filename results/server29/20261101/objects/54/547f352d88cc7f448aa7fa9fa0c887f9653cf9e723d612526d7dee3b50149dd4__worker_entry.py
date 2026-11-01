"""I/O-only entry to the byte-identical qualified worker; no numerical patches."""
import argparse
from pathlib import Path
import runpy
import sys

TASK = 'QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
PHASES = ('canary_initial', 'canary_fresh') + tuple(
    f'science_doc{d}_{c}' for d in range(1, 6)
    for c in ('FP_NATIVE', 'H_C128', 'H_DUAL_M'))
ROOT = Path(__file__).resolve().parents[1]


def dispatch(phase, root=ROOT, executor=runpy.run_path):
    root = Path(root).resolve()
    if root.name != TASK or phase not in PHASES:
        raise ValueError('Wrong task root or phase; only the frozen 17 phases are allowed')
    worker = root / 'src/worker.py'
    if not worker.is_file():
        raise ValueError('The unchanged qualified worker is absent')
    # core.ROOT and worker.ROOT are naturally derived from this new src location.
    # OS /proc/cmdline remains this entry + phase for exact-PID supervision.
    sys.path.insert(0, str(root / 'src'))
    sys.argv = [str(worker), phase]
    return executor(str(worker), run_name='__main__')


def self_test():
    import tempfile
    saved_argv, saved_path = list(sys.argv), list(sys.path)
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / TASK
            (root / 'src').mkdir(parents=True)
            root = root.resolve()
            (root / 'src/worker.py').write_text('# synthetic metadata-only worker fixture\n')
            seen = []
            def mock(path, run_name):
                seen.append((path, run_name, list(sys.argv), sys.path[0]))
            dispatch('science_doc5_H_DUAL_M', root, mock)
            assert seen == [(str(root / 'src/worker.py'), '__main__',
                             [str(root / 'src/worker.py'), 'science_doc5_H_DUAL_M'], str(root / 'src'))]
            for phase in ('science_doc6_H_C128', 'science_doc1_DUAL_W', 'restart'):
                try:
                    dispatch(phase, root, mock)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Unregistered phase accepted')
        return dict(status='PASS', phase_count=len(PHASES),
                    scope='Synthetic dispatch metadata only; no worker/model/Torch/CUDA executed')
    finally:
        sys.argv, sys.path = saved_argv, saved_path


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', nargs='?')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        dispatch(args.phase)
