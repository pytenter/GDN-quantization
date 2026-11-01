"""Host-only physical-device mapping and globally serialized checkpoint I/O.

The qualified worker source runs unchanged via runpy. Its configure function
changes only the first physical CUDA_VISIBLE_DEVICES assertion; every remaining
AST statement, including seeds/determinism/dtype/device0, is identical.
"""
import argparse
import ast
import copy
import fcntl
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASK = 'QWEN_GDN_DUAL_AXIS_PROSPECTIVE_REPLICATION_5DOC_V1'
CORE_SHA = 'aa338b8b83cde62b97c94fd205d1c62999f783e7ec3242cbb813dc877e01f575'
WORKER_SHA = 'e04ec85092ee4f2207699366b2aaa61e1e71038e728c9234477b5b2c21794773'
SCIENCE = tuple(f'science_doc{d}_{c}' for d in range(1, 6) for c in ('FP_NATIVE','H_C128','H_DUAL_M'))
ASSIGNMENT = {1:0, 2:1, 3:2, 4:3, 5:0}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_ast(original_source, physical):
    original = ast.parse(original_source).body[0]
    candidate = copy.deepcopy(original)
    first = candidate.body[0]
    need(isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
         and isinstance(first.value.func, ast.Name) and first.value.func.id == 'check', 'Unexpected original configure first assertion')
    comparison = first.value.args[0]
    need(isinstance(comparison, ast.Compare) and len(comparison.comparators) == 1
         and isinstance(comparison.comparators[0], ast.Constant) and comparison.comparators[0].value == '0',
         'Original physical mapping assertion differs')
    comparison.comparators[0].value = str(physical)
    dump = lambda node:ast.dump(node, include_attributes=False)
    need([dump(n) for n in original.body[1:]] == [dump(n) for n in candidate.body[1:]],
         'A seed, numerical configuration, dtype, or logical-device statement changed')
    restored = copy.deepcopy(candidate)
    restored.body[0].value.args[0].comparators[0].value = '0'
    need(dump(restored) == dump(original), 'More than the physical host assertion changed')
    return candidate, dict(original_configure_AST_sha256=hashlib.sha256(dump(original).encode()).hexdigest(),
                          mapped_configure_AST_sha256=hashlib.sha256(dump(candidate).encode()).hexdigest(),
                          all_statements_after_first_AST_identical=True,
                          sole_source_diff="first assertion comparator '0' -> physical GPU index string",
                          original_math_seed_TF32_determinism_dtype_logical_cuda0_unchanged=True)


def install_checkpoint_hook(worker, lockfile, record, on_hook=None):
    """One main-call hook, removed before main/model/configure executes."""
    need(sys.gettrace() is None, 'Unexpected pre-existing Python trace before entry adapter')
    def trace(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'main' and Path(frame.f_code.co_filename) == worker:
            original = frame.f_globals['checkpoint']
            @functools.wraps(original)
            def serialized(*args, **kwargs):
                started = time.monotonic()
                with lockfile.open('a+b') as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    acquired = time.monotonic()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        record.append(dict(lock_wait_seconds=acquired-started,
                                           original_function_wall_seconds=time.monotonic()-acquired))
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            frame.f_globals['checkpoint'] = serialized
            sys.settrace(None)
            if on_hook:
                on_hook()
            return None
        return trace
    sys.settrace(trace)


def execute(phase, physical, uuid, output_root):
    master = ROOT.resolve(); output_root = Path(output_root).resolve()
    need(master.name == TASK and physical in (0,1,2,3), 'Wrong task/device identity')
    canary = phase in ('canary_initial', 'canary_fresh')
    need(canary or phase in SCIENCE, 'Unregistered worker phase')
    expected_output = master / 'qualification_gpu' / f'gpu{physical}' if canary else master
    need(output_root == expected_output, 'Unregistered output scope')
    if not canary:
        need(ASSIGNMENT[int(phase.split('_',2)[1][3:])] == physical, 'Frozen document/GPU assignment changed')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == str(physical), 'Visible physical device mapping changed')
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    matching = [[v.strip() for v in row.split(',')] for row in rows.splitlines()
                if row.split(',')[0].strip() == str(physical)]
    need(len(matching) == 1 and matching[0][1] == uuid and 'RTX 3090' in matching[0][2]
         and matching[0][3] == '24576', 'Physical GPU UUID/type/capacity mismatch')
    need(digest(master/'src/core.py') == CORE_SHA and digest(master/'src/worker.py') == WORKER_SHA,
         'Qualified scientific source bytes changed')
    for rel in ('source_manifest.json','preregistration.json'):
        need(digest(output_root/rel) == digest(master/rel), 'Canary copied source/protocol identity differs')
    sys.path.insert(0, str(master/'src'))
    import core as K
    need(Path(K.__file__).resolve() == master/'src/core.py', 'Wrong core module loaded')
    candidate, audit = configure_ast(inspect.getsource(K.configure), physical)
    K.ROOT, K.UUID = output_root, uuid
    exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[])),
                 str(master/'src/core.py'), 'exec'), K.__dict__)
    mapped = K.configure
    def configure():
        need(K.torch.cuda.device_count() == 1, 'Worker must expose exactly one logical device')
        mapped()
        K.save(f'analysis/DEVICE_ADAPTER_{phase}.json', dict(audit, status='PASS', physical_GPU_ID=physical,
               GPU_UUID=uuid, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], logical_device='cuda:0',
               source_manifest_sha256=K.sha(master/'source_manifest.json'),
               scientific_core_source_sha256=CORE_SHA, worker_source_sha256=WORKER_SHA,
               output_root=str(output_root), physical_identity_independently_NVML_verified=True))
    K.configure = configure
    worker = master/'src/worker.py'; lockfile = master/'control/CHECKPOINT_IO.lock'
    waits = []; hooked = []
    install_checkpoint_hook(worker, lockfile, waits, lambda:hooked.append(True))
    sys.argv = [str(worker), phase]
    try:
        runpy.run_path(str(worker), run_name='__main__')
        need(hooked == [True] and sys.gettrace() is None, 'Entry main-call hook was not removed exactly once')
    finally:
        sys.settrace(None)
        K.save(f'analysis/CHECKPOINT_LOCK_{phase}.json', dict(status='MEASURED', phase=phase,
               physical_GPU_ID=physical, GPU_UUID=uuid, original_worker_source_sha256=WORKER_SHA,
               global_lock=str(lockfile), checkpoint_function_body_unchanged=True,
               lock_scope='Complete original snapshot/save/replace/hash/catalog/optional-restore function',
               global_retention='ONE task-wide rolling CURRENT; not one retained checkpoint per lane',
               main_call_hook_count=len(hooked), entry_trace_removed_before_main=True,
               calls=waits, total_lock_wait_seconds=sum(r['lock_wait_seconds'] for r in waits)))


def self_test():
    import tempfile
    source = "def configure():\n check(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','mapping')\n random.seed(20261015)\n torch.manual_seed(20261015)\n torch.cuda.set_device(0)\n"
    for device in range(4):
        node, audit = configure_ast(source, device)
        assert audit['all_statements_after_first_AST_identical'] and node.body[0].value.args[0].comparators[0].value == str(device)
    with tempfile.TemporaryDirectory() as td:
        worker = Path(td)/'worker.py'; lock = Path(td)/'checkpoint.lock'; record=[]; hooks=[]
        text = "def checkpoint(x):\n return x+1\ndef main():\n assert sys.gettrace() is None\n assert checkpoint(2)==3\nmain()\n"
        install_checkpoint_hook(worker, lock, record, lambda:hooks.append(True))
        try:
            exec(compile(text, str(worker), 'exec'), dict(sys=sys))
        finally:
            sys.settrace(None)
        assert hooks == [True] and len(record) == 1
    return dict(status='PASS', scope='Synthetic AST and Python function hook only; no Torch/model/CUDA/NVML',
                four_host_mappings_only_first_assertion_diff=True, main_call_hook_removed_before_main=True,
                original_checkpoint_called_once_under_host_lock=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase', nargs='?')
    parser.add_argument('physical', nargs='?', type=int); parser.add_argument('uuid', nargs='?')
    parser.add_argument('output_root', nargs='?'); parser.add_argument('--self-test', action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        execute(args.phase,args.physical,args.uuid,args.output_root)
