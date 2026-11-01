"""Pure-stdlib audit/extraction of original mathematical/training bodies.

Extracted code is not a runnable replacement for missing historical F/M helpers.
No original module is imported and no model/Torch/CUDA is loaded by this audit.
"""
import ast
import hashlib
from pathlib import Path

SNAPSHOTS = {
    'engine': ('0f85e40100070c051a01d05125341c1863b5c9808247803a725b1b218f54412d', 'a800_matched.py'),
    'core': ('48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e', 'run_recurrent_dense.py'),
    'cayley': ('62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067', 'cayley_rotation.py'),
}
SELECT = {
    'engine': {'token_hash', 'unified_teacher', 'tensors', 'cached', 'train_document', 'evaluate'},
    'core': {'recover_state', 'relative_state_mse', 'freeze_model', 'detach_tree', 'detach_cache',
             'enable_differentiable_cache', 'RecurrentExperimentPatch', 'sampled_positions', 'hadamard',
             'tokenize', 'forward_teacher', 'forward_student', 'collect_teacher_targets', 'install_teacher_target'},
    'cayley': {'CayleyDenseRotation', 'PerLayerCayleyRotations', 'relative_mse', 'qwen_c128_ste',
               '_axis_to_last', 'QuantizationResult', 'canonical_symmetric_int8_qdq', 'qwen_c128_qdq', 'ste_from_exact'},
}


def snapshot_path(root, family):
    sha, filename = SNAPSHOTS[family]
    p = Path(root) / 'provenance/snapshots' / (sha + '_' + filename)
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        raise ValueError('Authoritative source drift: ' + family)
    return p


def selected_nodes(root, family):
    p = snapshot_path(root, family)
    tree = ast.parse(p.read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', None) in SELECT[family]]
    found = {n.name for n in nodes}
    missing = SELECT[family] - found
    if missing:
        raise ValueError('Missing authoritative AST bodies: ' + str(sorted(missing)))
    return p, nodes


def audit(root):
    rows = []
    for family in SNAPSHOTS:
        p, nodes = selected_nodes(root, family)
        module = ast.Module(body=nodes, type_ignores=[])
        compile(module, str(p), 'exec')  # Compile only. No Torch/model execution.
        rows.append(dict(family=family, source_path=str(p.relative_to(root)),
            source_sha256=SNAPSHOTS[family][0],
            nodes=[dict(name=n.name, line=n.lineno,
                        ast_sha256=hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()) for n in nodes]))
    return dict(status='EXACT_ORIGINAL_AST_RECOVERED_NOT_FULL_MODEL_RUNNER', sources=rows,
        copied_math_changes=0, compiled_without_execution=True,
        unresolved_control_helpers=['M.HiddenCapture', 'M.clean', 'F.memory_snapshot',
            'F.fresh_bank / initialization verification', 'full model load/materializer/integrity wiring',
            'new paired teacher cache and source/data identity binding'],
        full_model_runner_implementation='BLOCKED_UNCLOSED_CONTROL_DEPENDENCIES')
