"""Synthetic fixture closure from exact *already frozen* runtime identities.

This never edits/extends the scientific or runtime manifest. Loaded module paths
are inventory candidates, not a directory whitelist. Every file must match an
existing frozen per-file digest or an explicitly named engineering input.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')

def collect(engineering_inputs, runtime_manifest):
    manifest = json.loads(Path(runtime_manifest).read_text())
    frozen = {str(Path(row['path']).resolve()): row['sha256'] for row in manifest['environment_files']}
    engineering = {str(Path(path).resolve()): sha(path) for path in engineering_inputs}
    forbidden = ('torch', 'fla', 'transformers', 'numpy', 'scipy', 'sglang')
    assert not any(name == p or name.startswith(p + '.') for name in sys.modules for p in forbidden), 'SCIENTIFIC_PACKAGE_LOADED_IN_SYNTHETIC_FIXTURE'
    rows = {}
    for name, module in list(sys.modules.items()):
        raw = getattr(module, '__file__', None)
        if not raw: continue
        path = Path(raw)
        if path.suffix == '.pyc': path = Path(importlib.util.source_from_cache(str(path)))
        real = str(path.resolve())
        if not path.is_file():
            raise RuntimeError('FIXTURE_LOADED_SOURCE_UNAVAILABLE: ' + real)
        expected = engineering.get(real, frozen.get(real))
        if expected is None: raise RuntimeError('FIXTURE_DEPENDENCY_NOT_IN_FROZEN_INPUTS: ' + real)
        actual = sha(path)
        if actual != expected: raise RuntimeError('FIXTURE_FROZEN_SOURCE_IDENTITY_MISMATCH: ' + real)
        row = rows.setdefault(real, dict(path=real, sha256=actual, modules=[], classification='ENGINEERING_INPUT' if real in engineering else 'EXISTING_FROZEN_RUNTIME_DEPENDENCY'))
        row['modules'].append(name)
    for real, actual in engineering.items():
        rows.setdefault(real, dict(path=real, sha256=actual, modules=[], classification='ENGINEERING_INPUT'))
    return dict(files=sorted(rows.values(), key=lambda r:r['path']), frozen_runtime_manifest_path=str(runtime_manifest),
                frozen_runtime_manifest_sha256=sha(runtime_manifest), SCIENTIFIC_RUNTIME_MANIFEST_ADDITIONS=0,
                SCIENTIFIC_PROTOCOL_CHANGED='NO', policy='exact loaded fixture dependency closure; no prefix/path-specific admission')

def static_map(manifest):
    return {r['path']:r['sha256'] for r in manifest['files'] if r['path'].endswith('.py')}

class ExecutedClosure:
    def __init__(self, static, guard): self.static=static; self.guard=guard; self.observed={}
    def profile(self, frame, event, arg):
        if event != 'call': return
        path=frame.f_code.co_filename
        if path.startswith('<'): return
        real=str(Path(path).resolve())
        self.observed.setdefault(real,set()).add(getattr(frame.f_code,'co_qualname',frame.f_code.co_name))
    def receipt(self):
        rows=[]
        for path, functions in sorted(self.observed.items()):
            if path in self.static:
                actual=sha(path); assert actual==self.static[path], ('FIXTURE_EXECUTED_SOURCE_CHANGED',path)
                kind='EXACT_FROZEN_FIXTURE_DEPENDENCY'
            elif path in self.guard.sources:
                actual=self.guard.sources[path]['source_sha256'];kind='PREEXEC_CAPTURED_SYNTHETIC_GENERATED_SOURCE'
            elif path in self.guard.static:
                # The unchanged STATIC_SOURCE_IDENTITY fixture deliberately
                # declares exact synthetic bytes before its import; this does
                # not extend any production runtime/scientific manifest.
                actual=sha(path);assert actual==self.guard.static[path]
                kind='EXPLICITLY_DECLARED_SYNTHETIC_STATIC_FIXTURE'
            else: raise RuntimeError('FIXTURE_EXECUTED_DEPENDENCY_NOT_CLOSED: '+path)
            rows.append(dict(path=path,sha256=actual,functions=sorted(functions),classification=kind))
        return dict(files=rows,FIXTURE_EXECUTED_DEPENDENCY_CLOSURE_GATE='PASS')


