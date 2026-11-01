"""One bounded control adaptation of the exact completed C6R training bodies.

No Torch/model import at module import. Only I/O, capture, seeding and memory
helpers are new. Mathematical functions are extracted unchanged from pinned
historical sources. New-machine full-model execution remains NOT_TESTED.
"""
import ast
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from authoritative_training_ast import snapshot_path
from training_state_io import GDN_LAYERS, file_sha256, save_legal_boundary_checkpoint, export_final_rotation

IMPLEMENTATION = 'IMPLEMENTED_CONTROL_ADAPTATION_FULL_MODEL_NOT_TESTED'
TM_SHA = '1ad98ebba548c43679fceef644c152baa1485537109f01da72f1ba4a54d26552'
Q_SHA = '0e613d971124b8cf7acc236d8d10ab13e5bf8d4eea6c9b81d38db0b5ac6f0873'
SOURCE_SHA = '05b4227e7700bcb483667b64ddc274ab119d9f869c7139fbd59292b2936174e8'
CORE_NAMES = {'GDN_LAYERS', 'EPS', 'TARGETS_PER_DOCUMENT', 'SEQUENCE_LENGTH_MAX', 'ORTHOGONALITY_THRESHOLD',
    'corpus_rows', 'sampled_positions', 'hadamard', 'recover_state', 'relative_state_mse', 'freeze_model',
    'detach_tree', 'detach_cache', 'enable_differentiable_cache', 'RecurrentExperimentPatch', 'tokenize',
    'forward_teacher', 'forward_student', 'collect_teacher_targets', 'install_teacher_target'}
ENGINE_NAMES = {'token_hash', 'unified_teacher', 'tensors', 'cached', 'train_document', 'evaluate'}


def pinned_snapshot(root, digest, filename):
    p = Path(root) / 'provenance/snapshots' / (digest + '_' + filename)
    if file_sha256(p) != digest:
        raise ValueError('Historical source identity mismatch: ' + filename)
    return p


def _module(path, name, tree=None, namespace=None):
    spec = importlib.util.spec_from_file_location(name, str(path))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    if namespace:
        result.__dict__.update(namespace)
    if tree is None:
        spec.loader.exec_module(result)
    else:
        exec(compile(tree, str(path), 'exec'), result.__dict__)
    return result


def _exact(path, names, namespace):
    nodes = []
    found = set()
    for n in ast.parse(Path(path).read_text()).body:
        assigned = {t.id for t in n.targets if isinstance(t, ast.Name)} if isinstance(n, ast.Assign) else set()
        if getattr(n, 'name', None) in names or assigned & names:
            nodes.append(n); found |= assigned; found.add(getattr(n, 'name', None))
    if names - found:
        raise ValueError('Missing exact AST nodes: ' + str(names - found))
    ns = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return SimpleNamespace(**ns)


def adapter_dependencies(root):
    # Evaluate only path constants, without importing the Torch adapter.
    tree = ast.parse((Path(root) / 'source/codec_training_adapter.py').read_text())
    names = {'RUNREC', 'CAYLEY_PATH', 'BASE_PATH', 'ARCH_PATH', 'CODEC_PATH', 'V1', 'FROZEN'}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = dict(Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<path inventory only>', 'exec'), ns)
    return ns


def verify_staging(root, config, stage, hash_model=True):
    """CPU-only identity check of explicit future paths; no path guessed."""
    root = Path(root)
    if stage.get('preregistration_sha256') != file_sha256(root / 'PAIRED_TRAINING_PREREGISTRATION.json'):
        raise ValueError('Staging/config SHA mismatch')
    dep = adapter_dependencies(root)
    paths = stage['source_paths']
    if set(paths) != set(dep['FROZEN']):
        raise ValueError('Staged codec dependency inventory differs')
    rows = []
    for old, digest in dep['FROZEN'].items():
        p = Path(paths[old])
        if file_sha256(p) != digest:
            raise ValueError('Staged source drift: ' + old)
        rows.append(dict(path=str(p), sha256=digest))
    env = json.loads((root / 'ENVIRONMENT_PLAN.json').read_text())
    if hash_model:
        for item in env['model_transfer_inventory']:
            p = Path(stage['model_root']) / Path(item['path']).name
            if p.stat().st_size != item['bytes'] or file_sha256(p) != item['sha256']:
                raise ValueError('Staged model/tokenizer drift: ' + str(p))
    historical = json.loads(pinned_snapshot(root, SOURCE_SHA, 'source_manifest.json').read_text())
    transformer_rows = [r for r in historical['canonical_source'] if '/transformers-qwen35/' in r['path']]
    if len(transformer_rows) != 2952:
        raise ValueError('Original full Transformers inventory changed')
    for item in transformer_rows:
        relative = item['path'].split('/transformers-qwen35/', 1)[1]
        p = Path(stage['transformers_root']) / relative
        if p.stat().st_size != item['bytes'] or file_sha256(p) != item['sha256']:
            raise ValueError('Staged complete Transformers source drift: ' + relative)
    # Future data is never inferred from a partial preparation directory.
    if not config['frozen'] or config['data']['status'] != 'PASS':
        raise ValueError('Paired data/preregistration is not frozen and PASS')
    corpus = stage['paired_corpus']
    p = Path(corpus['path'])
    if p.stat().st_size != corpus['bytes'] or file_sha256(p) != corpus['sha256']:
        raise ValueError('Staged paired corpus mismatch')
    for relative, digest in stage['control_hashes'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_sha256(root / relative) != digest:
            raise ValueError('Staged new control/source mismatch: ' + relative)
    required = {'source/a800_entry.py', 'source/paired_training_backend.py', 'source/training_state_io.py',
                'source/codec_training_adapter.py', 'source/authoritative_training_ast.py'}
    if not required <= set(stage['control_hashes']):
        raise ValueError('Incomplete staged new execution-source binding')
    return dict(status='STAGED_IDENTITIES_PASS_NOT_MODEL_EXECUTED', dependencies=rows,
                full_Transformers_files=2952, model_files_verified=len(env['model_transfer_inventory']) if hash_model else 0)


def build_components(root, source_paths):
    """CPU-importable exact AST closure. Does not construct or execute a model."""
    import torch
    root = Path(root)
    dep = adapter_dependencies(root)
    if set(source_paths) != set(dep['FROZEN']):
        raise ValueError('Explicit complete source mapping required')
    for old, digest in dep['FROZEN'].items():
        if file_sha256(source_paths[old]) != digest:
            raise ValueError('Relocated dependency hash mismatch: ' + old)
    adapter = _module(root / 'source/codec_training_adapter.py', 'paired_readiness_codec_adapter')
    for name in ('RUNREC', 'CAYLEY_PATH', 'BASE_PATH', 'ARCH_PATH', 'CODEC_PATH'):
        setattr(adapter, name, Path(source_paths[str(dep[name])]))
    references = Path(source_paths[str(dep['V1'] / 'src/references.py')])
    adapter.V1 = references.parents[1]
    adapter.FROZEN = {str(Path(source_paths[old])): digest for old, digest in dep['FROZEN'].items()}
    ordinary_loader = adapter.load_file
    def relocated_loader(path, name):
        if Path(path) != adapter.CODEC_PATH:
            return ordinary_loader(path, name)
        # Only historical absolute I/O root assignments are relocated. Every
        # codec function/class/encode AST body is retained without transformation.
        tree = ast.parse(Path(path).read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id in {'ROOT', 'PARENT', 'V1'} for t in n.targets))]
        previous = sys.modules.get('references')
        if previous is not None and Path(previous.__file__).resolve() != references.resolve():
            raise ValueError('Different references module already imported')
        sys.path.insert(0, str(references.parent))
        return _module(path, name, tree, dict(ROOT=Path(path).parents[1], PARENT=adapter.V1.parent, V1=adapter.V1))
    adapter.load_file = relocated_loader
    components = adapter.QualifiedComponents()
    core = _exact(snapshot_path(root, 'core'), CORE_NAMES,
        dict(torch=torch, CAYLEY=components.cayley, BASE=components.base, Path=Path, json=json, random=random))
    return SimpleNamespace(torch=torch, adapter=adapter, components=components, core=core)


class HiddenCapture:
    def __init__(self, model):
        self.active = False
        self.value = None
        self.handle = model.lm_head.register_forward_pre_hook(self._capture)
    def _capture(self, module, args):
        if self.active:
            self.value = args[0]
    def clear(self):
        self.value = None
    def close(self):
        self.handle.remove()
        self.clear()


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2); f.write('\n')


def bind_engine(root, bundle, corpus, output):
    """Close original F/M helpers with explicit controls; keep six engine bodies."""
    torch, core = bundle.torch, bundle.core
    output = Path(output)
    # The corpus is explicit frozen I/O, not a change to tokenizer/capture math.
    core.corpus_rows.__globals__['CORPUS'] = Path(corpus)
    def clean(patch=None):
        if patch is not None:
            patch.clear_capture()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def memory_snapshot(device, label):
        return dict(label=label, allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            NVML=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True))
    def fresh_bank(device):
        bank = bundle.components.cayley.PerLayerCayleyRotations(GDN_LAYERS).to(device)
        if sum(p.numel() for p in bank.parameters()) != 195072 or any(bool((p != 0).any()) for p in bank.parameters()):
            raise ValueError('Original fresh Cayley bank identity differs')
        return bank
    helpers = SimpleNamespace(clean=clean, HiddenCapture=HiddenCapture)
    controls = SimpleNamespace(fresh_bank=fresh_bank, memory_snapshot=memory_snapshot,
                               EXPLODING_GRAD_NORM=1e6, MEMORY_CREEP_BYTES=256*1024**2)
    engine = _exact(snapshot_path(root, 'engine'), ENGINE_NAMES,
        dict(torch=torch, core=core, F=controls, M=helpers, ROOT=output, S=file_sha256,
             runtime=SimpleNamespace(tensor_sha256=tensor_sha), time=time, math=math,
             json=json, PRIMARY={'C6R': 'functional'}, METRICS=('state','functional','final_hidden')))
    return SimpleNamespace(engine=engine, F=controls, M=helpers, tensor_sha=tensor_sha)


@contextmanager
def codec_scope(bundle, arm):
    if arm not in ('A', 'D'):
        raise ValueError('Only paired A/D')
    # Both exact namespaces refer to the same Cayley module, verified explicitly.
    if bundle.core.RecurrentExperimentPatch._wrap.__globals__['CAYLEY'] is not bundle.components.cayley:
        raise ValueError('Codec injection misses actual recurrent patch namespace')
    with bundle.components.arm(arm):
        yield


def offline_closure_audit(root, bundle, bound):
    import builtins
    import dis
    import types
    missing = []
    def inspect_code(code, namespace, label):
        for op in dis.get_instructions(code):
            if op.opname == 'LOAD_GLOBAL' and op.argval not in namespace and not hasattr(builtins, op.argval):
                missing.append(label + ':' + op.argval)
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                inspect_code(value, namespace, label + '/' + value.co_name)
    for module, names in ((bundle.core, CORE_NAMES), (bound.engine, ENGINE_NAMES)):
        for name in names:
            obj = getattr(module, name)
            if isinstance(obj, type):
                for method in vars(obj).values():
                    if hasattr(method, '__code__'):
                        inspect_code(method.__code__, method.__globals__, name)
            elif hasattr(obj, '__code__'):
                inspect_code(obj.__code__, obj.__globals__, name)
    if missing:
        raise ValueError('Unclosed original AST globals: ' + str(sorted(set(missing))))
    bank = bound.F.fresh_bank('cpu')
    original = bundle.components.cayley.qwen_c128_ste
    with codec_scope(bundle, 'A'):
        if bundle.components.cayley.qwen_c128_ste is not original:
            raise ValueError('A replaced original C128 STE')
    with codec_scope(bundle, 'D'):
        if bundle.components.cayley.qwen_c128_ste.__self__ is not bundle.components:
            raise ValueError('D codec injection identity mismatch')
    if bundle.components.cayley.qwen_c128_ste is not original:
        raise ValueError('Codec scope did not restore')
    return dict(status='PASS_CPU_IMPORT_AND_DEPENDENCY_CLOSURE_ONLY', implementation=IMPLEMENTATION,
        exact_core_nodes=len(CORE_NAMES), exact_engine_functions=len(ENGINE_NAMES),
        missing_globals=[], original_C128_preserved=True, D_actual_core_injection_and_restore=True,
        fresh_zero_theta_parameters=sum(p.numel() for p in bank.parameters()),
        model_loaded=False, CUDA_initialized=bundle.torch.cuda.is_initialized(),
        full_model_forward_backward='NOT_TESTED')
