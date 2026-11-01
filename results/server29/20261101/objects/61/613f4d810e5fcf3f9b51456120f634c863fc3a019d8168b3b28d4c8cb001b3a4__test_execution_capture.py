"""CPU-only synthetic provenance tests, never loads a scientific model."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from execution_capture import Recorder, CaptureError, closure, write_json


def child(root):
    root = Path(root)
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / 'any_generated_dependency.py'
        module.write_bytes(b'VALUE = 7\ndef get_value():\n    return VALUE\n')
        rec = Recorder(root / 'retained')
        rec.install()
        spec = importlib.util.spec_from_file_location('synthetic_temp', module)
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        assert obj.get_value() == 7
        rec.finish()
        write_json(root / 'receipt.json', dict(original_path=str(module)))


def expected_fail(fn):
    try:
        fn()
    except (CaptureError, FileNotFoundError):
        return
    raise AssertionError('EXPECTED_FAIL_CLOSED')


def suite(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in ('A', 'C', 'D', 'E', 'F1', 'F2'):
        root = out / name
        root.mkdir()
        subprocess.run([sys.executable, __file__, 'child', str(root)], check=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    a = out / 'A'
    assert not Path(json.loads((a / 'receipt.json').read_text())['original_path']).exists()
    closure(a / 'retained/manifest.json')
    results['ENGINEERING_TEST_A'] = 'PASS'
    # Mutation occurs after loader capture, immediately before audit exec.
    b = out / 'B'
    b.mkdir()
    p = b / 'synthetic.py'
    p.write_bytes(b'VALUE = 7\n')
    rec = Recorder(b / 'retained')
    rec.install()
    spec = importlib.util.spec_from_file_location('mutant', p)
    code = spec.loader.get_code('mutant')
    p.write_bytes(b'VALUE = 8\n')
    executed = {}
    expected_fail(lambda: exec(code, executed))
    assert 'VALUE' not in executed
    expected_fail(rec.finish)
    results['ENGINEERING_TEST_B'] = 'PASS_EXPECTED_FAIL_BEFORE_EXECUTION'
    for name in ('C', 'D'):
        root = out / name / 'retained'
        m = json.loads((root / 'manifest.json').read_text())
        blob = root / m['events'][0]['preserved_artifact_path']
        blob.chmod(0o644)
        if name == 'C':
            raw = blob.read_bytes()
            blob.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        else:
            blob.unlink()  # Only explicitly scoped synthetic test artifact.
        expected_fail(lambda: closure(root / 'manifest.json'))
        results['ENGINEERING_TEST_' + name] = 'PASS_EXPECTED_CLOSURE_FAIL'
    e = out / 'E/retained/manifest.json'
    m = json.loads(e.read_text())
    # Actual execution exists in independent boundary ledger, metadata absent.
    assert m['events'] and m['observed_boundaries']
    m['events'].pop()
    write_json(e.with_name('missing_event.json'), m)
    expected_fail(lambda: closure(e.with_name('missing_event.json')))
    results['ENGINEERING_TEST_E'] = 'PASS_EXPECTED_MISSING_CAPTURE_FAIL'
    identity = []
    for name in ('F1', 'F2'):
        root = out / name / 'retained'
        m = json.loads((root / 'manifest.json').read_text())
        events = [x for x in m['events'] if x['original_absolute_path'] and
                  x['original_absolute_path'].endswith('any_generated_dependency.py')]
        assert events
        identity.append([(x['sha256_capture'], x['byte_size'], x['representation'],
                          x['source_classification']) for x in events])
    assert identity[0] == identity[1]
    results['ENGINEERING_TEST_F'] = 'PASS_SAME_BYTES_SAME_CONTENT_IDENTITY'
    write_json(out / 'results.json', dict(**results, scientific_model_loaded=False,
                                        formal_measurements=0, synthetic_inputs_only=True))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'child':
        child(sys.argv[2])
    elif sys.argv[1] == 'suite':
        suite(sys.argv[2])
    else:
        raise RuntimeError('UNKNOWN_TEST_MODE')
