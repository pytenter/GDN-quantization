"""Compile only named, unchanged AST nodes. Never import model/training runners."""
import ast
from dataclasses import dataclass
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def extract(filename, names, namespace=None):
    path = ROOT/'reference_sources'/filename
    tree = ast.parse(path.read_text())
    nodes = []
    for n in tree.body:
        name = getattr(n, 'name', None)
        assigned = [t.id for t in n.targets if isinstance(t, ast.Name)] if isinstance(n, ast.Assign) else []
        if name in names or any(a in names for a in assigned): nodes.append(n)
    found = {getattr(n, 'name', None) for n in nodes}
    found |= {t.id for n in nodes if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    if not set(names) <= found: raise RuntimeError(f'Missing reference nodes: {set(names)-found}')
    ns = {'torch':torch, 'dataclass':dataclass, 'math':math}
    if namespace: ns.update(namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

CANON = extract('cayley_rotation.py', {'EPS','QuantizationResult','canonical_symmetric_int8_qdq','qwen_c128_qdq'})
FORMAL = extract('run_qwen_c5_c6_formal.py', {'EPS','GDN_LAYERS','get_state','quantize_c128','hadamard'})
STEP = extract('stepquant_quantization.py', {'QuantizationPlan','_fp16_scale','fit_state'})
