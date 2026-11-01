"""Two planned fresh final CPU integration admissions; never a GPU launcher."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write('\n')


def pins():
    names = [p for directory in ('src', 'tests', 'scripts', 'operations', 'vendor', 'protocol')
             for p in sorted((ROOT / directory).rglob('*.py')) if '__pycache__' not in p.parts]
    names += [ROOT / 'scripts/run_offline.sh']
    return [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                 sha256=sha(p)) for p in names]


def native_cpu_reference():
    path = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
    expected = '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a'
    assert path.is_file() and sha(path) == expected
    assert sha(ROOT / 'vendor/modeling_qwen3_5.py') == expected
    rows = json.loads((ROOT / 'analysis/EXTERNAL_PYTHON_BINDINGS.json').read_text())['files']
    selected = [x for x in rows if Path(x['path']).resolve() == path.resolve()]
    assert len(selected) == 1 and selected[0]['sha256'] == expected
    assert path.stat().st_size == selected[0]['bytes']
    return dict(path=str(path), bytes=path.stat().st_size, sha256=expected,
                scope='Pinned actual installed native helper AST on CPU; no module import, weights or GPU credit')


def identity(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttick=int(stat[19]), parent_pid=int(stat[1]),
                cmdline=[x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x])


def identity_state(expected):
    if expected is None:
        return dict(status='UNRESOLVED_NO_REGISTERED_IDENTITY', identity_gone=False)
    try:
        actual = identity(expected['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return dict(status='GONE', identity_gone=True)
    if actual['starttick'] != expected['starttick']:
        return dict(status='PID_REUSED_FOREIGN_NOT_SIGNALLED', identity_gone=True, actual=actual)
    return dict(status='OWNED_ALIVE' if actual == expected else 'SAME_STARTTICK_IDENTITY_CHANGED_UNRESOLVED',
                identity_gone=False, actual=actual)


def finish_owned_cpu(proc, expected, argv):
    actions = []
    if proc.poll() is None:
        actual = identity(proc.pid)
        if expected is None:
            assert actual['parent_pid'] == os.getpid() and actual['cmdline'] == argv
            expected = actual
        assert actual == expected and actual['parent_pid'] == os.getpid(), 'Refuse foreign CPU child signal'
        proc.terminate(); actions.append('SIGTERM_EXACT_OWN_CPU_PID_STARTTICK_ARGV')
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            assert identity(proc.pid) == expected and proc.poll() is None, 'Refuse foreign CPU child kill'
            proc.kill(); actions.append('SIGKILL_EXACT_OWN_CPU_PID_STARTTICK_ARGV')
            proc.wait(timeout=5)
    return expected, actions


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not any((ROOT / n).exists() for n in ('GPU_PERMIT.json', 'SESSION_STARTED.json',
                                                'OFFLINE_PARENT_STARTED.json'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        env.pop(key, None)
    before = pins()
    native = native_cpu_reference()
    runs = []
    # Fixed two-pass plan, not a retry. Any failure stops the plan immediately.
    for number in (1, 2):
        destination = ROOT / 'engineering' / ('FINAL_CPU_INTEGRATION_fresh' + str(number))
        destination.mkdir(parents=True, exist_ok=False)
        steps = []
        for test in ('test_pace_continuation_cpu.py', 'test_pace_repair_delivery_cpu.py',
                     'test_pace_scale_diagnostics.py', 'test_pace_runtime_cpu.py',
                     'test_pace_math.py'):
            output = destination / (test + '.log')
            # Each test runs in a fresh interpreter. Torch CPU is permitted;
            # CUDA initialization/device moves are explicitly rejected.
            code = """
import json,os,runpy,sys
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
torch.set_num_threads(1)
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA operation attempted by CPU-only final admission')
torch.cuda._lazy_init=forbidden
torch.cuda.init=forbidden
torch.cuda.set_device=forbidden
torch.Tensor.cuda=forbidden
torch.nn.Module.cuda=forbidden
sys.argv=[TEST]
if TEST.endswith('test_pace_math.py'):
    sys.argv += ['--device','cpu','--report-dir',REPORT,
                 '--native-source',NATIVE_SOURCE,'--native-source-sha256',NATIVE_SHA]
runpy.run_path(TEST,run_name='__main__')
""".replace('TEST', repr(str(ROOT / 'tests' / test))).replace('REPORT', repr(str(destination / 'math'))).replace('NATIVE_SOURCE', repr(native['path'])).replace('NATIVE_SHA', repr(native['sha256']))
            started = time.time()
            expected = None; error = None; cleanup_error = None; actions = []
            with output.open('xb') as stream:
                argv = [sys.executable, '-B', '-c', code]
                proc = subprocess.Popen(argv, cwd=ROOT,
                                        env=env, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        try:
                            candidate = identity(proc.pid)
                            assert candidate['cmdline'] == argv and candidate['parent_pid'] == os.getpid()
                            expected = candidate
                            break
                        except FileNotFoundError:
                            if proc.poll() is not None:
                                break
                            time.sleep(.01)
                    assert expected is not None, 'Actual CPU child identity not registered'
                    proc.wait(timeout=600)
                except BaseException as exc:
                    error = repr(exc)
                    try:
                        expected, actions = finish_owned_cpu(proc, expected, argv)
                    except BaseException as cleanup_exc:
                        cleanup_error = repr(cleanup_exc)
                returncode = proc.poll()
            state = identity_state(expected)
            row = dict(test=test, returncode=returncode, actual_owning_wait=True,
                       actual_exited=returncode is not None, identity=expected,
                       identity_gone=state['identity_gone'], identity_state=state,
                       error=error, cleanup_error=cleanup_error, cleanup_actions=actions,
                       elapsed_seconds=time.time()-started, log_sha256=sha(output),
                       status='PASS' if returncode == 0 and error is None and cleanup_error is None
                       and state['identity_gone'] else 'FAIL')
            steps.append(row)
            save(destination / (test + '.exit.json'), row)
            assert row['status'] == 'PASS', 'Final CPU test failed: ' + test
        assert pins() == before, 'Final CPU admitted source changed during run'
        report = dict(status='PASS', returncode=0, fresh_interpreters=len(steps), steps=steps,
                      files_before=before, files_after=pins(), GPU_calls=0,
                      native_CPU_reference=native,
                      model_loads=0, real_model_AIME_forwards=0,
                      mocks_do_not_count_as_GPU_qualification=True)
        save(destination / 'RESULT.json', report)
        runs.append(dict(number=number, path=str(destination / 'RESULT.json'),
                         sha256=sha(destination / 'RESULT.json'), status='PASS', returncode=0))
        print(json.dumps(runs[-1]), flush=True)
    save(ROOT / 'operations/TARGET_TWO_FRESH_FINAL_CPU_INTEGRATION.json',
         dict(status='PASS_TWO_FRESH_TARGET_FINAL_CPU_INTEGRATION', runs=runs,
              GPU_calls=0, files=before))


if __name__ == '__main__':
    main()
