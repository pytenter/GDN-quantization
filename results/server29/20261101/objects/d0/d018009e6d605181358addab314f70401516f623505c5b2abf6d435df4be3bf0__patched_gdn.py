"""Compile the unmodified production function body except exactly one cumsum line.

The hub dispatch decorator is not applied to this experiment-local binding: the
reference capture established that the original production Torch fallback is active.
Shared installation files and all other production operations stay unchanged.
"""
import ast, difflib, hashlib, json
from pathlib import Path
from deterministic_cumsum import deterministic_cumsum_fixed_order

ROOT=Path(__file__).resolve().parents[1]

def build(qmod, emit=False):
    source_path=Path(qmod.__file__)
    identity=json.loads((ROOT/'provenance/original_source_identity.json').read_text())
    text=source_path.read_text()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==identity['sha256']
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='torch_chunk_gated_delta_rule')
    original=ast.get_source_segment(text,node)
    old='g = g.cumsum(dim=-1)'; new='g = deterministic_cumsum_fixed_order(g, dim=-1)'
    assert original.count(old)==1
    patched=original.replace(old,new)
    # Source-to-source single replacement; no reconstruction of the numerical body.
    ns=dict(vars(qmod)); ns['deterministic_cumsum_fixed_order']=deterministic_cumsum_fixed_order
    exec(compile(patched,str(ROOT/'runtime_patch/patched_production_chunk.py'),'exec'),ns)
    fn=ns['torch_chunk_gated_delta_rule']
    if emit:
        def once(p,data):
            p=ROOT/p; p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('x') as f: f.write(data)
        once('runtime_patch/original_production_chunk.py',original+'\n')
        once('runtime_patch/patched_production_chunk.py',patched+'\n')
        diff=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile=str(source_path)+':'+str(node.lineno),tofile='experiment-local/patched_production_chunk.py'))
        once('provenance/patch_diff.txt',diff)
        result=dict(original_source_path=str(source_path),original_source_sha256=identity['sha256'],
            function_line_range=[node.lineno,node.end_lineno],single_replaced_line=identity['blocked_line'],
            patched_function_sha256=hashlib.sha256((patched+'\n').encode()).hexdigest(),
            patch_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
            runtime_patch_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'runtime_patch').glob('*.py')},
            changed_scientific_operations=['one inclusive prefix sum association schedule'],
            routing='same observed Torch fallback body compiled locally, hub decorator excluded from local binding',
            shared_installation_changed=False)
        once('provenance/patched_source_identity.json',json.dumps(result,indent=2,sort_keys=True)+'\n')
    return fn

def install(qmod):
    qmod.torch_chunk_gated_delta_rule=build(qmod)
