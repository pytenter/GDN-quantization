#!/usr/bin/env python3
"""CPU-only real filesystem/fake CUDA tests; no runtime model/device import."""
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types

RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / 'drafts/worker.py'
BASE = RUN.parent / 'execution_payload/worker/restoration_worker.py'
spec = importlib.util.spec_from_file_location('observation_worker_cpu_test', SOURCE)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
checks = []


def ok(name, value):
    assert value, name
    checks.append(dict(name=name, status='PASS'))


def reject(name, fn, text=None):
    try:
        fn()
    except RuntimeError as exc:
        assert text is None or text in str(exc), (name, str(exc))
        checks.append(dict(name=name, status='PASS'))
    else:
        raise AssertionError('Expected rejection: ' + name)


def fields():
    return [dict(name=n, dtype='torch.float32', shape=[2], stride=[1], data=b'\x00\x00\x80\x3f' * 2) for n in W.SuccessActivationWriter.NAMES]


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): self.now += .001; return self.now


class FakeCuda:
    def __init__(self):
        self.calls = []; self.allocated = 10; self.reserved = 20; self.pa = 30; self.pr = 40; self.marker = object()
    def memory_allocated(self): return self.allocated
    def memory_reserved(self): return self.reserved
    def max_memory_allocated(self): return self.pa
    def max_memory_reserved(self): return self.pr
    def reset_peak_memory_stats(self, *args, **kw):
        self.calls.append(('reset', args, kw)); self.pa = self.allocated; self.pr = self.reserved; return self.marker
    def synchronize(self): self.calls.append(('sync',))


with tempfile.TemporaryDirectory(prefix='worker_observation_cpu_', dir=RUN / 'tests') as temporary:
    root = Path(temporary); target = root / 'execution/NUMERIC_FRESH0'; target.mkdir(parents=True)
    real_statvfs = W.os.statvfs
    W.os.statvfs = lambda _p: types.SimpleNamespace(f_bavail=2**40, f_frsize=1)
    try:
        layers = [x for x in range(32) if x % 4 != 3]
        progress = []
        writer = W.SuccessActivationWriter(root, target, 'NUMERIC_FRESH0', layers, 32, 8 * 1024**3, 10 * 1024**3, {'fixture': 'synthetic_only'}, lambda phase, details: progress.append((phase, details)))
        f = fields(); saved = copy.deepcopy(f)
        rec = writer.capture(0, layers[0], 'chunk', f, None)
        ok('capture_does_not_modify_input_serialized_snapshots', f == saved)
        ok('capture_keeps_all_nine_raw_fields_in_order', tuple(x['name'] for x in rec['tensors']) == W.SuccessActivationWriter.NAMES)
        raw = (root / rec['raw_path']).read_bytes()
        ok('whole_binary_SHA_bytes_match_actual_file', hashlib.sha256(raw).hexdigest() == rec['raw_sha256'] and len(raw) == rec['raw_bytes'])
        ok('all_field_offsets_SHA_bytes_match', all(hashlib.sha256(raw[x['offset']:x['offset'] + x['bytes']]).hexdigest() == x['sha256'] for x in rec['tensors']))
        ok('capture_scope_not_mathematical_PASS', rec['status'] == 'RAW_CAPTURE_COMPLETE_NOT_MATH_VERDICT')
        ok('contiguous_serialization_distinguished_from_source_stride', rec['serialization'] == 'contiguous_C_order_native_dtype' and rec['source_stride_is_metadata_not_serialized_stride'])
        reject('duplicate_capture_rejected', lambda: writer.capture(0, layers[0], 'chunk', fields(), None), 'out-of-order')
        reject('skipped_layer_rejected', lambda: writer.capture(0, layers[2], 'chunk', fields(), None), 'out-of-order')
        reject('wrong_chunk_recurrent_phase_rejected', lambda: writer.capture(0, layers[1], 'recurrent', fields(), None), 'phase')
        reject('incomplete_32x24_roster_cannot_finalize', writer.finish, 'Incomplete')
        invalid = fields(); invalid[0]['data'] = b'\x00'
        reject('dtype_shape_byte_mismatch_rejected_before_write', lambda: writer.capture(0, layers[1], 'chunk', invalid, None), 'byte length')
        invalid = fields(); invalid[0]['data'] = bytearray(invalid[0]['data'])
        reject('mutable_raw_buffer_rejected', lambda: writer.capture(0, layers[1], 'chunk', invalid, None), 'immutable')
        reject('global_cap_guard_rejects_pending_over8GiB', lambda: writer.guard(8 * 1024**3), 'cap')
        extra = root / 'execution/other_unit/retained.bin'; extra.parent.mkdir(); extra.write_bytes(b'x' * 13)
        ok('global_guard_includes_other_unit_raw', writer.guard()['existing_run_raw_bytes'] == len(raw) + 13)
        W.os.statvfs = lambda _p: types.SimpleNamespace(f_bavail=10 * 1024**3, f_frsize=1)
        reject('disk_reserve_guard_rejects_before_write', lambda: writer.guard(1), 'reserve')
        W.os.statvfs = lambda _p: types.SimpleNamespace(f_bavail=2**40, f_frsize=1)
        for ordinal in range(1, 32 * 24):
            token, layer = ordinal // 24, layers[ordinal % 24]
            writer.capture(token, layer, 'chunk' if token == 0 else 'recurrent', fields(), 'parent_state_hash' if token else None)
        manifest = writer.finish()
        ok('complete32x24_actual_raw_and_index_roster_finalizes', manifest['calls'] == 768 and len(list((target / 'intermediates').glob('*.bin'))) == 768 and len(list((target / 'intermediates').glob('*.json'))) == 768)
        ok('progress_only_emitted_after_actual_capture_commit', len(progress) == 768 and all(x[0] == 'SUCCESS_CAPTURE_COMMITTED' for x in progress))
        reject('extra_769th_call_rejected', lambda: writer.capture(32, layers[0], 'recurrent', fields(), 'x'), 'Extra')
        nonclobber = target / 'intermediates/TOKEN_000_LAYER_00.bin'; before = nonclobber.read_bytes()
        again = W.SuccessActivationWriter(root, target, 'NUMERIC_FRESH0', layers, 32, 8 * 1024**3, 10 * 1024**3, {})
        reject('existing_raw_never_overwritten', lambda: again.capture(0, layers[0], 'chunk', fields(), None), 'already exists')
        ok('existing_raw_bytes_unchanged_after_rejection', nonclobber.read_bytes() == before)
    finally:
        W.os.statvfs = real_statvfs
    allocator_dir = root / 'execution/OLD_FP_R0'; allocator_dir.mkdir()
    cuda = FakeCuda(); clock = Clock(); observer = W.AllocatorObservation(cuda, allocator_dir, 'OLD_FP_R0', clock=clock)
    original = cuda.reset_peak_memory_stats
    observer.install_reset_observer()
    saved_original = observer.original_reset
    device_arg = object(); answer = cuda.reset_peak_memory_stats(device_arg, device='cuda:0')
    ok('reset_observer_forwards_exact_args_kwargs_once', cuda.calls == [('reset', (device_arg,), {'device': 'cuda:0'})])
    ok('reset_observer_preserves_original_return_identity', answer is cuda.marker)
    ok('reset_observer_retains_pre_reset_segment_peak', observer.reset_events[0]['before']['peak_allocated_bytes'] == 30 and observer.reset_events[0]['before']['peak_reserved_bytes'] == 40)
    cuda.pa = 50; cuda.pr = 70
    calls = []; result_object = object()
    def loader(): calls.append('original_loader'); return result_object
    value = observer.loader(loader, 'SYNTHETIC_LOADER')
    ok('model_loader_called_exactly_once_return_preserved', calls == ['original_loader'] and value is result_object)
    ok('model_load_sync_boundaries_separate', len([x for x in cuda.calls if x[0] == 'sync']) == 2 and json.loads((allocator_dir / 'MODEL_LOAD_TIMING.json').read_text())['loader_call_wall_seconds'] > 0)
    cuda.pa = 80; cuda.pr = 90
    final = observer.finish()
    ok('whole_allocator_peak_combines_reset_epochs_and_final_stats', final['peak_allocated_bytes'] == 80 and final['peak_reserved_bytes'] == 90 and len(final['reset_events']) == 2)
    observer.restore_reset_observer()
    ok('original_reset_restored_exact_saved_callable_identity', cuda.reset_peak_memory_stats is saved_original)
    ok('reset_instrumentation_records_caller_file_outside_tensor_math', observer.reset_events[0]['before']['caller_filename'] == __file__)
    reject('second_model_loader_invocation_rejected', lambda: observer.loader(loader, 'SYNTHETIC_LOADER'), 'second')
    failure_dir = root / 'execution/FAILED_FAKE'; failure_dir.mkdir(); failing = W.AllocatorObservation(FakeCuda(), failure_dir, 'FAILED_FAKE', clock=Clock())
    try:
        failing.install_reset_observer()
        original_failure = failing.original_reset
        try: failing.loader(lambda: W.check(False, 'synthetic loader failure'), 'FAILED_LOADER')
        finally: failing.restore_reset_observer()
    except RuntimeError:
        pass
    ok('failure_finally_restores_reset_without_fake_success_receipt', failing.cuda.reset_peak_memory_stats is original_failure and not (failure_dir / 'MODEL_LOAD_TIMING.json').exists())

# Remove only the enumerated observation additions and task metadata adaptation.
# Exact AST equality covers all old model, seed, environment, math, thresholds,
# cache traversal, finite checks, raw trace and historical benchmark invocation.
base_raw = BASE.read_bytes(); new_raw = SOURCE.read_bytes()
assert hashlib.sha256(base_raw).hexdigest() == '8b14e35fdeac501978b0e3f948eb888eacd0e1dce37487afed83181b10bb396e'
base_tree = ast.parse(base_raw)
new_tree = ast.parse(new_raw)
base_doc = base_tree.body[0].value.value


class RevertObservations(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        if node.name in {'SuccessActivationWriter', 'AllocatorObservation'}: return None
        return self.generic_visit(node)
    def visit_FunctionDef(self, node):
        if node.name == 'progress': return None
        return self.generic_visit(node)
    def visit_Constant(self, node):
        if node.value == 'QWEN_GDN_HISTORICAL_RUNTIME_GPU_RECOVERY_V1': return ast.copy_location(ast.Constant('QWEN_GDN_HISTORICAL_RUNTIME_RESTORATION_V1'), node)
        return node
    def visit_Assign(self, node):
        names = {x.id for x in node.targets if isinstance(x, ast.Name)}
        if names & {'allocator', 'success_writer', 'capture_values'}: return None
        return self.generic_visit(node)
    def visit_Expr(self, node):
        v = node.value
        if isinstance(v, ast.Call):
            if isinstance(v.func, ast.Name) and v.func.id == 'progress': return None
            if isinstance(v.func, ast.Attribute) and isinstance(v.func.value, ast.Name) and v.func.value.id in {'allocator', 'success_writer'}: return None
        return self.generic_visit(node)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'allocator' and node.func.attr == 'loader':
            fn = node.args[0]
            return self.visit(fn.body if isinstance(fn, ast.Lambda) else ast.Call(func=fn, args=[], keywords=[]))
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'j' and node.func.attr == 'update':
            node.keywords = [x for x in node.keywords if x.arg != 'observation_amendment']
        return node
    def visit_Try(self, node):
        if len(node.finalbody) == 1 and isinstance(node.finalbody[0], ast.If) and ast.unparse(node.finalbody[0].test) == 'allocator is not None': node.finalbody = []
        return self.generic_visit(node)


new_tree.body[0].value.value = base_doc
reverted = ast.fix_missing_locations(RevertObservations().visit(copy.deepcopy(new_tree)))
old_dump = ast.dump(base_tree, include_attributes=False); reverted_dump = ast.dump(reverted, include_attributes=False)
ast_same = old_dump == reverted_dump
if not ast_same:
    (RUN / 'tests/AST_REVERSION_DIFF.txt').write_text('\n'.join(difflib.unified_diff(ast.unparse(base_tree).splitlines(), ast.unparse(reverted).splitlines())) + '\n')
ok('entire_original_worker_AST_identical_after_enumerated_observation_reversal', ast_same)
ok('no_top_level_Torch_Driver_or_NVML_import', not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(x in ast.unparse(n) for x in ['torch', 'ctypes', 'pynvml']) for n in ast.parse(new_raw).body))
main = next(n for n in ast.parse(new_raw).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
ok('reset_restore_in_outer_finally', any(isinstance(n, ast.Try) and n.finalbody and 'restore_reset_observer' in ast.unparse(n.finalbody[0]) for n in main.body))
matched = RUN.parent / 'execution_payload/candidate/benchmark_replay/source/matched_worker.py'
ok('historical_matched_worker08b_remains_unchanged', hashlib.sha256(matched.read_bytes()).hexdigest() == '08b62839e88fa6bb49f10388a19e152e04beb0ab36b347ffcfdf2d9259e2ff8c')
patch = '\n'.join(difflib.unified_diff(base_raw.decode().splitlines(), new_raw.decode().splitlines(), fromfile='parent8b14/restoration_worker.py', tofile='new_observation/drafts/worker.py')) + '\n'
(RUN / 'tests/worker_observation_amendment.patch').write_text(patch)
proof = dict(status='PASS_EXACT_ORIGINAL_WORKER_AST_AFTER_ENUMERATED_OBSERVATION_REVERSAL', base_worker_sha256=hashlib.sha256(base_raw).hexdigest(), new_worker_sha256=hashlib.sha256(new_raw).hexdigest(), original_AST_sha256=hashlib.sha256(old_dump.encode()).hexdigest(), reverted_new_AST_sha256=hashlib.sha256(reverted_dump.encode()).hexdigest(), reversal_rules=['Remove two new stdlib writer/allocator classes and progress function/calls', 'Remove success capture/resource guard/finalize statements and their bindings', 'Unwrap loader-only timing calls to exact original callable expressions', 'Remove reset observer install/finalize/finally restore', 'Remove result observation label; restore prior TASK metadata/docstring'], math_seed_environment_model_cache_FP64_and_benchmark_AST_unchanged=True, numerical_runtime_not_executed=True)
(RUN / 'tests/WORKER_OBSERVATION_AST_PROOF.json').write_text(json.dumps(proof, indent=2) + '\n')
result = dict(status='PASS_CPU_SUCCESS_RAW_AND_ALLOCATOR_OBSERVATION_TESTS', checks=checks, checks_passed=len(checks), GPU_queries=0, GPU_dispatches=0, Driver_calls=0, NVML_calls=0, Torch_imports=0, network_calls=0, worker_sha256=hashlib.sha256(new_raw).hexdigest(), observed_fake_capture_calls=768, real_model_forwards=0, original_AST_reversal_same=ast_same, scope='Real temporary small serialized bytes with fake CUDA counters; no tensor/model runtime, scientific numerical or GPU qualification claimed.')
(RUN / 'tests/WORKER_OBSERVATION_CPU_TEST_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, sort_keys=True))
