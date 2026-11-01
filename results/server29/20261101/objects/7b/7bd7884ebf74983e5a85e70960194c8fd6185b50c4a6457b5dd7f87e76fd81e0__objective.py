"""Reuse historical L7 helpers without importing or executing training phases."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

SOURCE_SHA256 = "d1c433598875ea84c80c6f6d4fb5fe48088d5ca0bf44ccff9d950a77560fd9ae"
HELPERS = {"local_output", "relative_mse", "relative_state_loss", "exact_qdq"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_helpers(torch, path):
    path = Path(path)
    if digest(path) != SOURCE_SHA256:
        raise RuntimeError("ORIGINAL_L7_SOURCE_IDENTITY_FAIL")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    if {n.name for n in nodes} != HELPERS:
        raise RuntimeError("ORIGINAL_L7_HELPER_COVERAGE_FAIL")
    eps = next(n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "EPS" for t in n.targets))
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[eps] + nodes, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["EPS"] == 1e-12
    return namespace


def combined_scores(torch, helpers, student_outputs, teacher_outputs,
                    student_state, teacher_state, layers):
    """Exact loss definitions on matched full-model deployment tensors.

    Outputs are the actual post-o_proj tensors, before current-token QDQ.
    States are post-current-QDQ native-coordinate recurrent cache values.
    This does not replay historical FP-conditioned local training drivers.
    """
    if set(student_outputs) != set(layers) or set(teacher_outputs) != set(layers):
        raise RuntimeError("POST_O_PROJ_LAYER_COVERAGE_FAIL")
    functional = torch.zeros((), device=student_state.device)
    state = torch.zeros_like(functional)
    for i, layer in enumerate(layers):
        a, b = student_outputs[layer], teacher_outputs[layer]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError("POST_O_PROJ_TENSOR_VALIDITY_FAIL")
        functional = functional + helpers["relative_mse"](a, b) / len(layers)
        state = state + helpers["relative_state_loss"](
            student_state[i:i+1], teacher_state[i:i+1]) / len(layers)
    total = functional + 0.1 * state
    if not torch.isfinite(torch.stack([functional, state, total])).all():
        raise RuntimeError("ORIGINAL_L7_NONFINITE_LOSS")
    return functional, state, total
