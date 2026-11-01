"""Observational execution provenance. Does not rewrite source or returned code.

Coverage: CPython exec audit boundaries, Python function-entry profile boundaries
in this process and Python threads created after installation, SourceFileLoader
get_code, and ExtensionFileLoader creation. Interpreter/bootstrap execution before
installation and implicit native shared-library/GPU-machine-code execution are
NOT claimed; the latter remain under the frozen runtime/input identity inventory.
"""
import __future__
import ast
import builtins
import hashlib
import importlib.machinery
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types


class CaptureError(RuntimeError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def closure(path):
    """No source-path fallback: only retained blobs and independent event ledger."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    events = manifest['events']
    ledger = manifest['observed_boundaries']
    ids = [e['event_id'] for e in events]
    if ids != ledger or len(set(ids)) != len(ids):
        raise CaptureError('MISSING_EXECUTION_CAPTURE')
    if manifest['failure'] or manifest['unclassified_count']:
        raise CaptureError('CAPTURE_RECORDED_FAILURE')
    closed = 0
    for e in events:
        if not e['executed'] or e['capture_stage'] != 'before_execution':
            raise CaptureError('INVALID_EXECUTION_EVENT')
        blob = (path.parent / e['preserved_artifact_path']).resolve()
        if not blob.is_relative_to(path.parent.resolve()):
            raise CaptureError('BLOB_OUTSIDE_RETENTION_ROOT')
        raw = blob.read_bytes()  # Missing files fail; never skip/reconstruct.
        if len(raw) != e['byte_size']:
            raise CaptureError('BLOB_LENGTH_MISMATCH')
        if not digest(raw) == e['sha256_capture'] == e['sha256_pre_execution']:
            raise CaptureError('BLOB_HASH_MISMATCH')
        if not e['git_classification'] or not e['source_classification']:
            raise CaptureError('UNCLASSIFIED_EXECUTED_DEPENDENCY')
        closed += 1
    if manifest['executed_dependency_count'] != len(events):
        raise CaptureError('EXECUTED_COUNT_MISMATCH')
    return dict(EXECUTED_DEPENDENCY_COUNT=len(events), CLOSED_DEPENDENCY_COUNT=closed,
                MISSING_DEPENDENCY_COUNT=0, HASH_MISMATCH_COUNT=0,
                UNCLASSIFIED_EXECUTED_DEPENDENCY=0, gate='PASS')


class Recorder:
    def __init__(self, root, historical=None, source_root=None, frozen_root=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'blobs').mkdir()
        self.historical = historical or {}
        self.source_root = str(source_root or '')
        self.frozen_root = str(frozen_root or '')
        self.events = []
        self.ledger = []
        self.compiles = {}
        self.code_bindings = {}  # Strong references: no object-id reuse.
        self.called = {}
        self.pending_load = {}
        self.local = threading.local()
        self.lock = threading.RLock()
        self.active = False
        self.failure = None
        self.guard_profile = None
        self.class_cache = {}
        self.original_compile = builtins.compile
        self.original_get_code = importlib.machinery.SourceFileLoader.get_code
        self.original_extension = importlib.machinery.ExtensionFileLoader.create_module
        self.original_setprofile = sys.setprofile
        self.original_threadprofile = threading.setprofile
        self.original_profile = sys.getprofile()
        self.process = dict(pid=os.getpid(), start_time_unix=time.time(),
                            host=os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME'),
                            python_executable=sys.executable)
        self.future_flags = 0
        for name in __future__.all_feature_names:
            self.future_flags |= getattr(__future__, name).compiler_flag

    def classification(self, path):
        if path in self.historical:
            h = self.historical[path]
            return h['classification'], 'frozen_historical_asset', h
        if path.startswith('<'):
            return 'not_in_git', 'interpreter_frozen' if path.startswith('<frozen ') else 'runtime_generated', None
        if path.startswith('/tmp/') or path.startswith('/var/tmp/'):
            return 'not_in_git', 'runtime_generated', None
        # No assertion that installed/inherited files equal a Git HEAD.
        if self.frozen_root and path.startswith(self.frozen_root + '/'):
            return 'external_inherited_frozen_inventory', 'external', None
        if self.source_root and path.startswith(self.source_root + '/'):
            if path not in self.class_cache:
                rel = str(Path(path).relative_to(self.source_root))
                env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
                tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', rel],
                    cwd=self.source_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                if tracked.returncode:
                    self.class_cache[path] = ('untracked', 'git_untracked', None)
                else:
                    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', rel],
                        cwd=self.source_root, env=env)
                    self.class_cache[path] = ('tracked-modified' if diff else 'tracked-clean', 'git_tracked', None)
            return self.class_cache[path]
        return 'not_asserted_git_external_runtime_file', 'external', None

    def preserve(self, raw):
        h = digest(raw)
        p = self.root / 'blobs' / (h + '.bin')
        if p.exists():
            if p.read_bytes() != raw:
                raise CaptureError('EXISTING_BLOB_CORRUPTED')
        else:
            with p.open('xb') as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            p.chmod(0o444)
        return h, str(p.relative_to(self.root))

    def capture_file(self, path):
        path = os.path.abspath(path)
        raw = Path(path).read_bytes()
        h, blob = self.preserve(raw)
        return dict(original_absolute_path=path, raw=raw, sha256_capture=h,
                    preserved_artifact_path=blob, byte_size=len(raw), representation='raw_file_bytes')

    def capture_memory(self, raw, filename, representation='actual_compile_input_bytes'):
        if isinstance(raw, str):
            raw = raw.encode('utf-8')
            representation = 'actual_compile_input_str_utf8'
        if not isinstance(raw, bytes):
            raise CaptureError('NO_RAW_BYTES_FOR_COMPILE_INPUT')
        h, blob = self.preserve(raw)
        return dict(original_absolute_path=None, logical_path=str(filename), raw=raw,
                    sha256_capture=h, preserved_artifact_path=blob, byte_size=len(raw), representation=representation)

    @staticmethod
    def codes(code):
        yield code
        for item in code.co_consts:
            if isinstance(item, types.CodeType):
                yield from Recorder.codes(item)

    def bind_tree(self, code, capture):
        for child in self.codes(code):
            self.code_bindings[child] = capture

    def matches(self, code, raw):
        flags = code.co_flags & self.future_flags
        # Compile solely to compare identity; never execute comparison code.
        # Audit/profile are reentrancy-suppressed. Returned runtime code unchanged.
        for mode in ('exec', 'eval', 'single'):
            try:
                compare = self.original_compile(raw, code.co_filename, mode,
                    flags=flags, dont_inherit=True, optimize=sys.flags.optimize)
            except (SyntaxError, ValueError, TypeError):
                continue
            children = list(self.codes(compare))
            if code in children:
                return compare
        return None

    def binding(self, code):
        if code in self.code_bindings:
            return self.code_bindings[code]
        name = code.co_filename
        for capture in reversed(self.compiles.get(name, [])):
            compare = self.matches(code, capture['raw'])
            if compare is not None:
                self.bind_tree(compare, capture)
                return capture
        if os.path.isabs(name):
            capture = self.capture_file(name)  # Absent original path fails closed.
            compare = self.matches(code, capture['raw'])
            if compare is None:
                raise CaptureError('EXECUTED_CODE_NOT_EQUAL_CAPTURED_SOURCE: ' + name)
            self.bind_tree(compare, capture)
            return capture
        if name.startswith('<frozen '):
            capture = self.capture_memory(marshal.dumps(code), name, 'actual_interpreter_frozen_code_object')
            self.code_bindings[code] = capture
            return capture
        raise CaptureError('UNBOUND_EXECUTED_CODE: ' + name)

    def event(self, capture, kind, name, module_name, code=None):
        event_id = len(self.ledger)
        self.ledger.append(event_id)  # Independent observation precedes entry.
        if capture['original_absolute_path']:
            pre = Path(capture['original_absolute_path']).read_bytes()
        elif capture['representation'] == 'actual_interpreter_frozen_code_object':
            pre = marshal.dumps(code)
        else:
            pre = capture['raw']  # Immutable actual compile input, code equality checked.
        h = digest(pre)
        if h != capture['sha256_capture']:
            raise CaptureError('CAPTURE_PREEXEC_MISMATCH: ' + name)
        git, cls, historic = self.classification(capture['original_absolute_path'] or name)
        e = {k: v for k, v in capture.items() if k not in ('raw', 'logical_path')}
        e.update(event_id=event_id, logical_name=name, module_name=module_name,
                 sha256_pre_execution=h, capture_stage='before_execution', executed=True,
                 execution_kind=kind, python_executable=sys.executable, process_identity=self.process,
                 thread_id=threading.get_ident(), git_classification=git, source_classification=cls,
                 observed_execution_count=1)
        if historic:
            if h != historic['sha256']:
                raise CaptureError('HISTORICAL_CAPTURE_IDENTITY_MISMATCH')
            e['historical_asset'] = historic
        if code:
            e['code_object_sha256'] = digest(marshal.dumps(code))
        self.events.append(e)
        return e

    def audit(self, name, args):
        if not self.active or getattr(self.local, 'busy', False):
            return
        if name not in ('compile', 'exec'):
            return
        with self.lock:
            self.local.busy = True
            try:
                if name == 'compile':
                    raw, filename = args
                    # AST transforms are not guessed/reconstructed as original bytes.
                    if isinstance(raw, ast.AST):
                        raise CaptureError('AST_COMPILE_RAW_BYTE_BOUNDARY_UNAVAILABLE')
                    capture = self.capture_memory(raw, filename)
                    self.compiles.setdefault(str(filename), []).append(capture)
                else:
                    code = args[0]
                    capture = self.binding(code)
                    self.event(capture, 'exec_audit', code.co_filename, None, code)
            except BaseException as exc:
                self.failure = repr(exc)
                raise
            finally:
                self.local.busy = False

    def profile(self, frame, event, arg):
        if not self.active or event != 'call' or getattr(self.local, 'busy', False):
            return
        if self.guard_profile:
            self.guard_profile(frame, event, arg)
        code = frame.f_code
        # Provenance implementation itself is the explicitly hashed trusted
        # observer, not a scientific dependency. Avoid observer self-recursion.
        if code.co_filename == __file__:
            return
        with self.lock:
            self.local.busy = True
            try:
                if code in self.called:
                    self.called[code]['observed_execution_count'] += 1
                else:
                    capture = self.binding(code)
                    self.called[code] = self.event(capture, 'python_function_entry',
                        code.co_filename + ':' + getattr(code, 'co_qualname', code.co_name), frame.f_globals.get('__name__'), code)
            except BaseException as exc:
                self.failure = repr(exc)
                raise
            finally:
                self.local.busy = False

    def install(self):
        rec = self
        def get_code(loader, fullname):
            rec.local.busy = True
            try:
                capture = rec.capture_file(loader.path)
                result = rec.original_get_code(loader, fullname)
                if rec.matches(result, capture['raw']) is None:
                    raise CaptureError('LOADER_CODE_SOURCE_MISMATCH: ' + loader.path)
                rec.bind_tree(result, capture)
                # Audit exec independently rechecks original bytes just before exec.
                return result
            except BaseException as exc:
                rec.failure = repr(exc)
                raise
            finally:
                rec.local.busy = False
        def extension(loader, spec):
            with rec.lock:
                rec.local.busy = True
                try:
                    capture = rec.capture_file(loader.path)
                    rec.event(capture, 'extension_loader_create', loader.path, spec.name)
                    # Native binary unchanged; no source injection/kernel changes.
                    rec.local.busy = False  # Observe Python execution during native init.
                    return rec.original_extension(loader, spec)
                except BaseException as exc:
                    rec.failure = repr(exc)
                    raise
                finally:
                    rec.local.busy = False
        def setprofile(requested):
            if requested not in (None, rec.profile):
                rec.guard_profile = requested
            rec.original_setprofile(rec.profile)
        self.active = True
        sys.addaudithook(self.audit)
        importlib.machinery.SourceFileLoader.get_code = get_code
        importlib.machinery.ExtensionFileLoader.create_module = extension
        sys.setprofile = setprofile
        self.original_threadprofile(self.profile)
        self.original_setprofile(self.profile)

    def finish(self):
        self.active = False
        self.original_setprofile(self.original_profile)
        self.original_threadprofile(None)
        sys.setprofile = self.original_setprofile
        importlib.machinery.SourceFileLoader.get_code = self.original_get_code
        importlib.machinery.ExtensionFileLoader.create_module = self.original_extension
        out = dict(schema='EXECUTION_BYTE_CAPTURE_V1', events=self.events,
            observed_boundaries=self.ledger, executed_dependency_count=len(self.ledger),
            unclassified_count=0, failure=self.failure, process=self.process,
            coverage=__doc__, scientific_measurements=0,
            observer=dict(path=__file__, sha256=digest(Path(__file__).read_bytes())))
        write_json(self.root / 'manifest.json', out)
        return closure(self.root / 'manifest.json')
