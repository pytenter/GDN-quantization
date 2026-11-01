"""Exact generated-code event provenance; virtual filenames are metadata only.

Compilation objects are associated with their observed bytes, not filename keys.
Every execution is separately receipted, including repeated executions of one
code object. CAPTURE is engineering-only; VERIFY consumes an exact frozen sequence.
"""
from pathlib import Path
import ast, builtins, __future__, hashlib, importlib.machinery, json, os, sys, threading, time, types

def digest(data): return hashlib.sha256(data).hexdigest()
def filehash(path): return digest(Path(path).read_bytes())
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')

class GeneratedGuard:
    def __init__(self, static, output, mode='CAPTURE', manifest=None, packages=None):
        self.static = {str(Path(p).resolve()): h for p, h in static.items()}
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=False)
        assert mode in ('CAPTURE', 'VERIFY'); self.mode = mode
        self.manifest = manifest or {}; self.packages = packages or {}
        self.expected = self.manifest.get('GENERATED_EXECUTION_FINGERPRINT_SEQUENCE', [])
        self.allowed = self.manifest.get('modules', [])
        self.events = []; self.execution_events = []; self.writers = {}
        self.sources = {}  # Real filesystem paths ONLY: origin audit, never executable identity.
        self.associated = {}; self.pending = {}; self.contexts = {}
        self.static_ast_events = []
        self.active = False; self.inside = False; self.closed = False; self.audit_owner = None
        self.implementation = str(Path(__file__).resolve())
        self._compile = builtins.compile
        self.future_mask = 0
        for name in __future__.all_feature_names:
            self.future_mask |= getattr(__future__, name).compiler_flag
    def static_identity(self, path):
        if str(path).startswith('<'): return False
        real = str(Path(path).resolve())
        if real not in self.static: return False
        if not Path(path).is_file() or filehash(path) != self.static[real]:
            raise RuntimeError('STATIC_FROZEN_SOURCE_MISMATCH: ' + str(path))
        return True
    def stack(self):
        rows = []; frame = sys._getframe(1)
        while frame:
            name = frame.f_code.co_filename
            if name != self.implementation:
                rows.append(dict(module=frame.f_globals.get('__name__'), function=getattr(frame.f_code, 'co_qualname', frame.f_code.co_name), path=name, line=frame.f_lineno))
            frame = frame.f_back
        return rows
    def caller_frame(self):
        frame = sys._getframe(1)
        while frame and frame.f_code.co_filename == self.implementation: frame = frame.f_back
        return frame
    def producer(self, stack):
        for row in stack:
            path = row['path']; module = row['module'] or ''
            if module.startswith(('importlib', '_frozen_importlib')) or path.startswith('<'): continue
            if not self.static_identity(path): raise RuntimeError('UNKNOWN_GENERATED_PRODUCER: ' + path)
            root = module.split('.')[0]; version = self.packages.get(root)
            if version is None and '/site-packages/' not in path.replace('\\', '/'):
                version = 'Python ' + sys.version.split()[0] + '; exact frozen source SHA256'
            if version is None: raise RuntimeError('UNKNOWN_FROZEN_PRODUCER_PACKAGE_VERSION: ' + module)
            return dict(producer_python_module=module, producer_qualified_function=row['function'], producer_source_path=path, producer_source_sha256=filehash(path), producer_package=root, producer_package_version=version, producer_callsite_line=row['line'])
        raise RuntimeError('UNATTRIBUTABLE_GENERATED_PRODUCER')
    @staticmethod
    def identity(record):
        producer = record['producer']
        return dict(source_sha256=record['source_sha256'], source_size=record['source_size'],
                    producer={k: producer[k] for k in ['producer_python_module', 'producer_qualified_function', 'producer_source_sha256', 'producer_package', 'producer_package_version', 'producer_callsite_line']},
                    compile_mode=record['compile_mode'], compile_flags=record['compile_flags'])
    @classmethod
    def fingerprint(cls, record):
        return digest(json.dumps(cls.identity(record), sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    def capture(self, source, origin, producer, stack, compile_info, kind='DYNAMIC_COMPILE'):
        if not isinstance(source, (str, bytes, bytearray)):
            raise RuntimeError('UNCAPTURABLE_GENERATED_SOURCE: ' + origin)
        data = source.encode('utf-8') if isinstance(source, str) else bytes(source)
        n = len(self.events)
        record = dict(event_local_id='compile_%06d' % n, event_sequence_index=n, origin_label=origin,
                      origin_label_role='ORIGIN_LABEL_ONLY', source_sha256=digest(data), source_size=len(data),
                      producer=producer, producer_module=producer['producer_python_module'],
                      producer_function=producer['producer_qualified_function'], producer_source_sha256=producer['producer_source_sha256'],
                      producer_callsite=producer['producer_callsite_line'], compile_mode=compile_info['mode'], compile_flags=compile_info['flags'],
                      generation_call_stack=stack, event_kind=kind, first_seen_timestamp=time.time(),
                      source_representation='utf8_exact_input_str' if isinstance(source, str) else 'exact_input_bytes')
        record['fingerprint'] = self.fingerprint(record)
        copy = self.output / ('%06d__%s.py' % (n, record['source_sha256']))
        with copy.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        record['canonical_captured_copy_path'] = str(copy)
        if self.mode == 'VERIFY' and self.identity(record) not in self.allowed:
            dump(self.output / ('%06d__REJECTED.json' % n), record)
            raise RuntimeError('UNKNOWN_OR_CHANGED_GENERATED_EXECUTABLE_BEFORE_EXECUTION: ' + origin)
        self.events.append(record); dump(self.output / ('%06d__CAPTURE.json' % n), record)
        return record
    def module_source(self, path, name):
        if self.static_identity(path): return 'STATIC_FROZEN_SOURCE'
        real = str(Path(path).resolve()); writer = self.writers.get(real)
        if writer is None: raise RuntimeError('UNKNOWN_STATIC_EXECUTABLE_NOT_OBSERVED_GENERATED: ' + str(path))
        record = self.capture(Path(path).read_bytes(), str(path), writer['producer'], writer['stack'],
                              {'mode': 'SOURCE_LOADER_INPUT', 'flags': None}, 'GENERATED_MODULE_SOURCE_BEFORE_LOADER')
        record['logical_module_name'] = name
        self.sources[real] = record
        return 'GENERATED_EXECUTABLE_FROM_FROZEN_PRODUCER'
    def bind(self, code, record):
        if not isinstance(code, types.CodeType): return
        old = self.associated.get(id(code))
        if old is not None and old[0] is not code: raise RuntimeError('CODE_OBJECT_REFERENCE_COLLISION')
        self.associated[id(code)] = (code, record)
    def audit(self, event, args):
        if not self.active or self.closed or event not in ('open', 'compile', 'exec'): return
        if self.inside:
            if self.audit_owner != threading.get_ident(): raise RuntimeError('CONCURRENT_GENERATED_INGRESS_NOT_OBSERVABLE')
            return
        self.inside = True; self.audit_owner = threading.get_ident()
        try:
            tid = threading.get_ident()
            if event == 'open':
                raw, mode, flags = args
                if not isinstance(raw, (str, bytes, os.PathLike)): return
                path = os.fsdecode(raw)
                if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) or not path.endswith('.py'): return
                real = str(Path(path).resolve())
                if real in self.static: raise RuntimeError('FROZEN_SOURCE_WRITE_FORBIDDEN: ' + path)
                if Path(real).is_relative_to(self.output): return
                stack = self.stack(); self.writers[real] = dict(producer=self.producer(stack), stack=stack)
            elif event == 'compile':
                source, origin = args
                context = self.contexts.get(tid, [])
                current = context[-1] if context else None
                if origin is None:
                    # CPython reports (AST_object, None) even when compile was
                    # explicitly given an unchanged frozen source filename.
                    # Recover metadata only from that synchronous observed call.
                    # Never manufacture/normalize source bytes or change filename.
                    if not (current and current['input_is_ast'] and isinstance(source,ast.AST)):
                        raise RuntimeError('UNASSOCIATED_AST_COMPILE_AUDIT')
                    requested=current['requested_filename']
                    if not self.static_identity(requested):
                        raise RuntimeError('UNCAPTURABLE_NONSTATIC_AST_BEFORE_EXECUTION')
                    producer=self.producer(self.stack())
                    record=dict(event_sequence_index=len(self.static_ast_events),
                        audit_origin_label=None,requested_compile_filename=os.fsdecode(requested),
                        source_sha256=self.static[str(Path(requested).resolve())],
                        source_identity_class='EXISTING_STATIC_FROZEN_AST_EXTRACTION',
                        producer=producer,compile_mode=current['info']['mode'],compile_flags=current['info']['flags'],
                        generated_source_rewritten=False,AST_normalized=False,compile_filename_changed=False)
                    self.static_ast_events.append(record)
                    dump(self.output/('static_ast_%06d__ASSOCIATION.json'%record['event_sequence_index']),record)
                    return
                origin = os.fsdecode(origin)
                real = None if origin.startswith('<') else str(Path(origin).resolve())
                if real in self.static:
                    self.static_identity(origin)
                    if source is None: return  # Unchanged frozen AST extraction semantics, not generated ingress.
                    data = source.encode('utf8') if isinstance(source, str) else bytes(source)
                    if digest(data) == self.static[real]: return
                if source is None: raise RuntimeError('UNCAPTURABLE_DYNAMIC_AST_BEFORE_EXECUTION: ' + origin)
                stack = self.stack()
                if real is not None:
                    writer = self.writers.get(real)
                    if writer is None and real not in self.static: raise RuntimeError('UNKNOWN_STATIC_SOURCE_COMPILE: ' + origin)
                    producer = writer['producer'] if writer else self.producer(stack)
                    stack = writer['stack'] if writer else stack
                else: producer = self.producer(stack)
                info = current['info'] if current else {'mode': 'NOT_AVAILABLE_IN_PYTHON_AUDIT', 'flags': None}
                record = self.capture(source, origin, producer, stack, info)
                if real is not None: self.sources[real] = record
                if current: current['records'].append(record)
                else:
                    # Implicit eval/exec compilation is synchronous. Ambiguous outstanding
                    # code is rejected, never guessed from a shared virtual filename.
                    if self.pending.get(tid): raise RuntimeError('AMBIGUOUS_IMPLICIT_GENERATED_COMPILE_ASSOCIATION')
                    self.pending[tid] = (record, self.caller_frame())
            else:
                code = args[0]; association = self.associated.get(id(code))
                record = association[1] if association and association[0] is code else None
                if record is None:
                    pending = self.pending.get(tid)
                    if pending:
                        candidate, frame = pending
                        if frame is not self.caller_frame() or code.co_filename != candidate['origin_label']:
                            raise RuntimeError('GENERATED_EXECUTION_ASSOCIATION_MISMATCH_BEFORE_BODY')
                        self.pending.pop(tid); self.bind(code, candidate); record = candidate
                    elif self.static_identity(code.co_filename): return
                    else: raise RuntimeError('UNCAPTURED_CODE_OBJECT_BEFORE_EXECUTION: ' + code.co_filename)
                index = len(self.execution_events); fingerprint = record['fingerprint']
                if self.mode == 'VERIFY' and (index >= len(self.expected) or self.expected[index] != fingerprint):
                    dump(self.output / ('exec_%06d__REJECTED.json' % index), dict(index=index, fingerprint=fingerprint, origin_label=code.co_filename, expected=self.expected[index] if index < len(self.expected) else None))
                    raise RuntimeError('GENERATED_EXECUTION_SEQUENCE_MISMATCH_BEFORE_BODY')
                execution = dict(event_local_id='exec_%06d' % index, event_sequence_index=index,
                                 compilation_event_local_id=record['event_local_id'], fingerprint=fingerprint,
                                 origin_label=code.co_filename, identity=self.identity(record), preexec_verified=True)
                self.execution_events.append(execution)
                dump(self.output / ('exec_%06d__EXECUTION.json' % index), execution)
        finally: self.inside = False; self.audit_owner = None
    def install(self):
        guard = self
        def observed_compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1, *, _feature_version=-1):
            if not guard.active or guard.inside or guard.closed:
                return guard._compile(source, filename, mode, flags, dont_inherit, optimize, _feature_version=_feature_version)
            caller = sys._getframe(1)
            inherited = 0 if dont_inherit else caller.f_code.co_flags & guard.future_mask
            effective = flags | inherited
            context = dict(info={'mode': mode, 'flags': dict(requested_flags=flags, inherited_future_flags=inherited, effective_flags=effective, dont_inherit=dont_inherit, optimize=optimize, feature_version=_feature_version)}, records=[],input_is_ast=isinstance(source,ast.AST),requested_filename=filename)
            tid = threading.get_ident(); guard.contexts.setdefault(tid, []).append(context)
            try:
                code = guard._compile(source, filename, mode, effective, True, optimize, _feature_version=_feature_version)
                if len(context['records']) > 1: raise RuntimeError('AMBIGUOUS_EXPLICIT_GENERATED_COMPILE_ASSOCIATION')
                if context['records']: guard.bind(code, context['records'][0])
                return code
            finally: guard.contexts[tid].pop()
        source_get_code = importlib.machinery.SourceFileLoader.get_code
        def observed_get_code(loader, fullname):
            code = source_get_code(loader, fullname)
            if guard.active and not guard.closed:
                real = str(Path(loader.path).resolve())
                if real not in guard.static and real in guard.sources:
                    if filehash(loader.path) != guard.sources[real]['source_sha256']:
                        raise RuntimeError('GENERATED_SOURCE_CHANGED_BEFORE_EXECUTION')
                    if id(code) not in guard.associated: guard.bind(code, guard.sources[real])
            return code
        builtins.compile = observed_compile
        importlib.machinery.SourceFileLoader.get_code = observed_get_code
        sys.addaudithook(self.audit); self.active = True
    def finish(self, enforce_complete=True):
        self.closed = True
        sequence = [r['fingerprint'] for r in self.execution_events]
        if self.mode == 'VERIFY' and enforce_complete and sequence != self.expected:
            raise RuntimeError('GENERATED_EXECUTION_SEQUENCE_INCOMPLETE')
        multiplicities = {f: sequence.count(f) for f in sorted(set(sequence))}
        receipt = dict(mode=self.mode, modules=[self.identity(r) for r in self.events], generation_count=len(self.events),
                       events=self.events, execution_events=self.execution_events,
                       GENERATED_EXECUTION_FINGERPRINT_SEQUENCE=sequence, multiplicities=multiplicities,
                       static_frozen_AST_associations=self.static_ast_events,
                       preexec_capture=True, formal_measurements=0)
        dump(self.output / 'GENERATED_CODE_RECEIPT.json', receipt)
        with (self.output / 'SHA256SUMS').open('x') as f:
            f.write(''.join(r['source_sha256'] + '  ' + Path(r['canonical_captured_copy_path']).name + '\n' for r in self.events))
        return receipt

def deterministic_receipts(receipts):
    assert len(receipts) >= 3
    for key, gate in [('GENERATED_EXECUTION_FINGERPRINT_SEQUENCE', 'GENERATED_EVENT_SEQUENCE_DETERMINISM_GATE'),
                      ('multiplicities', 'GENERATED_EVENT_MULTIPLICITY_DETERMINISM_GATE'),
                      ('modules', 'GENERATED_MODULE_BYTE_DETERMINISM_GATE')]:
        if any(r[key] != receipts[0][key] for r in receipts[1:]): raise RuntimeError(gate + '_FAIL')
    return dict(GENERATED_EVENT_SEQUENCE_DETERMINISM_GATE='PASS', GENERATED_MODULE_BYTE_DETERMINISM_GATE='PASS',
                GENERATED_PRODUCER_CLOSURE_GATE='PASS', modules=receipts[0]['modules'],
                GENERATED_EXECUTION_FINGERPRINT_SEQUENCE=receipts[0]['GENERATED_EXECUTION_FINGERPRINT_SEQUENCE'],
                multiplicities=receipts[0]['multiplicities'])
