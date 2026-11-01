"""Synthetic isolated-process identity and unchanged-semantics qualification."""
from pathlib import Path
import builtins, hashlib, importlib.util, json, subprocess, sys, tempfile
from generated_code_guard_attempt07 import GeneratedGuard, deterministic_receipts, filehash

CASES = ['T_IDENTITY_%02d' % i for i in range(1, 9)] + [
    'EXPLICIT_COMPILE_RETURN_OBJECT_ASSOCIATION', 'REEXECUTION_MULTIPLICITY',
    'UNKNOWN_CODE_OBJECT_DENIED', 'UNKNOWN_PRODUCER_DENIED',
    'UNKNOWN_STATIC_SOURCE_DENIED', 'STATIC_SOURCE_IDENTITY',
    'AMBIENT_METADATA_NONBLOCKING', 'SOURCE_MUTATION_DENIED',
    'COMPILE_FUTURE_FLAGS_SEMANTICS', 'IMPORT_RESULT_EXCEPTION_SEMANTICS',
    'REAL_NAMEDTUPLE_VIRTUAL_FILENAME_REGRESSION']

def exercise(case, root, mode, manifest_path=None):
    root = Path(root); root.mkdir(); sentinel = root / 'BODY_SIDE_EFFECT'
    static = {str(Path(__file__).resolve()): filehash(__file__)}
    if case == 'REAL_NAMEDTUPLE_VIRTUAL_FILENAME_REGRESSION':
        import collections
        static[str(Path(collections.__file__).resolve())] = filehash(collections.__file__)
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    guard = GeneratedGuard(static, root / 'capture', mode=mode, manifest=manifest)
    namespace = {'SENTINEL': sentinel, '__builtins__': builtins}
    unobserved = compile('VALUE=9', '<unobserved>', 'exec')
    def generated(source, origin='<string>'):
        code = compile(source, origin, 'exec')
        return code
    def other_generated(source):
        return compile(source, '<string>', 'exec')
    def imported(path, name):
        guard.module_source(path, name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    def denied(fn):
        try: fn()
        except (AssertionError, RuntimeError): return
        raise AssertionError('FAIL_CLOSED_GUARD_DID_NOT_FIRE')
    guard.install()
    if case in ('T_IDENTITY_01', 'T_IDENTITY_07'):
        for source in ('1+2', '3+4'):
            assert eval(source) in (3, 7)
        assert len(guard.execution_events) == 2
        assert len({r['fingerprint'] for r in guard.execution_events}) == 2
        assert [Path(r['canonical_captured_copy_path']).read_bytes() for r in guard.events] == [b'1+2', b'3+4']
        assert all(r['origin_label'] == '<string>' for r in guard.events)
        assert not guard.sources, 'VIRTUAL_FILENAME_MUST_NOT_BE_DICTIONARY_KEY'
    elif case in ('T_IDENTITY_02', 'REEXECUTION_MULTIPLICITY'):
        if case == 'T_IDENTITY_02':
            for _ in range(2): exec(generated('VALUE=7'), namespace)
        else:
            code = generated('VALUE=7')
            for _ in range(2): exec(code, namespace)
        assert len(guard.execution_events) == 2
        assert guard.execution_events[0]['fingerprint'] == guard.execution_events[1]['fingerprint']
    elif case == 'T_IDENTITY_03':
        exec(generated('VALUE=7'), namespace); exec(other_generated('VALUE=7'), namespace)
        assert guard.execution_events[0]['fingerprint'] != guard.execution_events[1]['fingerprint']
    elif case in ('T_IDENTITY_04', 'T_IDENTITY_08'):
        source = "SENTINEL.write_text('EXECUTED')" + ('; VALUE=8' if mode == 'VERIFY' else '; VALUE=7')
        if mode == 'VERIFY':
            denied(lambda: exec(generated(source), namespace)); assert not sentinel.exists()
        else: exec(generated(source), namespace); assert sentinel.exists()
    elif case == 'T_IDENTITY_05':
        codes = [generated('VALUE=7'), generated('VALUE=8')]
        if mode == 'VERIFY':
            denied(lambda: exec(codes[1], namespace)); assert 'VALUE' not in namespace
        else:
            for code in codes: exec(code, namespace)
    elif case == 'T_IDENTITY_06':
        # Paths vary across capture/verify roots; neither path nor module name
        # participates in executable fingerprint. Exact source and writer do.
        p = root / 'different_temporary_path.py'; p.write_bytes(b'VALUE=7\n')
        assert imported(p, 'fixture').VALUE == 7
    elif case == 'EXPLICIT_COMPILE_RETURN_OBJECT_ASSOCIATION':
        a = generated('VALUE=1'); b = generated('VALUE=2')
        exec(a, namespace); assert namespace['VALUE'] == 1
        exec(b, namespace); assert namespace['VALUE'] == 2
    elif case == 'UNKNOWN_CODE_OBJECT_DENIED': denied(lambda: exec(unobserved, namespace))
    elif case == 'UNKNOWN_PRODUCER_DENIED':
        stack = guard.stack(); guard.static[str(Path(__file__).resolve())] = '0' * 64
        denied(lambda: guard.producer(stack))
    elif case == 'UNKNOWN_STATIC_SOURCE_DENIED':
        guard.active = False; p = root / 'unknown.py'; p.write_bytes(b'VALUE=7'); guard.active = True
        denied(lambda: imported(p, 'unknown'))
    elif case == 'STATIC_SOURCE_IDENTITY':
        guard.active = False; p = root / 'static.py'; p.write_bytes(b'VALUE=7'); guard.static[str(p.resolve())] = filehash(p); guard.active = True
        assert imported(p, 'ordinary').VALUE == 7 and not guard.events
    elif case == 'AMBIENT_METADATA_NONBLOCKING':
        p = root / 'locale_metadata'; p.write_text('ambient'); assert p.read_text() == 'ambient' and not guard.events
    elif case == 'SOURCE_MUTATION_DENIED':
        guard.active = False; p = root / 'static.py'; p.write_bytes(b'VALUE=7'); guard.static[str(p.resolve())] = filehash(p); p.write_bytes(b'VALUE=8'); guard.active = True
        denied(lambda: imported(p, 'mutated'))
    elif case == 'COMPILE_FUTURE_FLAGS_SEMANTICS':
        # Native and interposed compile inherit the same caller future flags.
        src = "from __future__ import annotations\ndef make():\n return compile('x: Missing = 1', '<string>', 'exec')\n"
        producer = generated(src); g = {}; exec(producer, g)
        code = g['make'](); ns = {}; exec(code, ns); assert ns['__annotations__']['x'] == 'Missing'
    elif case == 'IMPORT_RESULT_EXCEPTION_SEMANTICS':
        p = root / 'semantics.py'; p.write_bytes(b'VALUE=sum([1,2,3]); PACKAGE=__package__; NAME=__name__\n')
        m = imported(p, 'semantics'); assert (m.VALUE, m.PACKAGE, m.NAME) == (6, '', 'semantics')
        p = root / 'exception.py'; p.write_bytes(b"raise ValueError('same exception')\n")
        try: imported(p, 'exception')
        except ValueError as e: assert str(e) == 'same exception'
        else: raise AssertionError('EXCEPTION_SEMANTICS_CHANGED')
    elif case == 'REAL_NAMEDTUPLE_VIRTUAL_FILENAME_REGRESSION':
        import collections
        a = collections.namedtuple('DecimalTuple', 'sign digits exponent')
        b = collections.namedtuple('LinearRegression', 'slope intercept')
        assert a(1, (2,), 3).sign == 1 and b(4, 5).slope == 4
        assert len(guard.execution_events) == 2
        assert len({r['fingerprint'] for r in guard.execution_events}) == 2
    else: raise AssertionError(case)
    guard.finish(enforce_complete=not (mode == 'VERIFY' and case in ('T_IDENTITY_04', 'T_IDENTITY_05', 'T_IDENTITY_08')))
    print(case, mode, 'PASS', flush=True)

def main():
    if len(sys.argv) > 1:
        exercise(*sys.argv[1:]); return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for case in CASES:
            capture = root / (case + '_capture')
            subprocess.run([sys.executable, '-B', __file__, case, str(capture), 'CAPTURE'], check=True)
            receipt = capture / 'capture' / 'GENERATED_CODE_RECEIPT.json'
            if case in ('T_IDENTITY_04', 'T_IDENTITY_05', 'T_IDENTITY_06', 'T_IDENTITY_08'):
                subprocess.run([sys.executable, '-B', __file__, case, str(root / (case + '_verify')), 'VERIFY', str(receipt)], check=True)
            if case == 'T_IDENTITY_05':
                r = json.loads(receipt.read_text()); changed = json.loads(receipt.read_text())
                changed['GENERATED_EXECUTION_FINGERPRINT_SEQUENCE'].reverse()
                try: deterministic_receipts([r, r, changed])
                except RuntimeError: pass
                else: raise AssertionError('ORDER_NOT_ENFORCED')
    print('ATTEMPT07_SYNTHETIC_TESTS_ALL_PASS', len(CASES), flush=True)
if __name__ == '__main__': main()
