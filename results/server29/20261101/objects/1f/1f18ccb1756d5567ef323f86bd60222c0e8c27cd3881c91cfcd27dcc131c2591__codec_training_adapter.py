"""Frozen numerical forward; legacy C128 and a detached-scale identity STE for Dual-M.

Only CPU/local qualification is authorized by this module. No model loader,
optimizer loop, recurrence implementation, launch, or GPU admission is here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import importlib.util
import math
import os
import sys
import torch

RUNREC = Path('/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py')
CAYLEY_PATH = RUNREC.parents[1] / 'shared/rotation/cayley_rotation.py'
BASE_PATH = Path('/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py')
ARCH_PATH = Path('/data/zypan/transformers-qwen35/src/transformers/models/qwen3_5/modeling_qwen3_5.py')
CODEC_PATH = Path('/data/zypan/analyses/qwen_gdn_c6r_dual_m_interaction_reuse_5doc_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_C6R_DUAL_M_INTERACTION_REUSE_5DOC_V1/src/codec.py')
V1 = Path('/data/zypan/analyses/qwen_hadamard_dual_axis_real_state_runtime_gate_v1/experiments/qwen_gdn/quantizer/QWEN_GDN_HADAMARD_DUAL_AXIS_REAL_STATE_RUNTIME_GATE_V1/inherited_cpu_v1')
FROZEN = {
    str(RUNREC): '48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e',
    str(CAYLEY_PATH): '62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067',
    str(BASE_PATH): '9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd',
    str(ARCH_PATH): '90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a',
    str(CODEC_PATH): 'ece859b901145c6d9d646b8befd01916037c9117ef4a80f7b801f27f79f8c23f',
    str(V1/'src/dual_axis_codec.py'): 'cd494824c242b2ce967dc470b50160aff25cf1776e1d45344d547a30fb6253e6',
    str(V1/'src/references.py'): 'e9b611e0f961dce27a21549018d25a00eede13a83c2a69ca8ca81984a1268dbf',
    str(V1/'reference_sources/cayley_rotation.py'): '62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067',
    str(V1/'reference_sources/run_qwen_c5_c6_formal.py'): '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69',
    str(V1/'reference_sources/stepquant_quantization.py'): '0c39fe6da2b5ad68e2a081471b1ec22465a9c9f914c367ed984a78ad3717ee69',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024**2), b''):
            h.update(part)
    return h.hexdigest()

def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def extract(path, names, namespace, remove_decorators=False):
    selected = []
    identities = {}
    for node in ast.parse(Path(path).read_text()).body:
        assigned = {x.id for x in node.targets if isinstance(x, ast.Name)} if isinstance(node, ast.Assign) else set()
        if getattr(node, 'name', None) in names or assigned.intersection(names):
            key = getattr(node, 'name', None) or ','.join(sorted(assigned))
            identities[key] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            if remove_decorators and isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                node.decorator_list = []
            selected.append(node)
    present = {getattr(n, 'name', None) for n in selected}
    present |= {x.id for n in selected if isinstance(n, ast.Assign) for x in n.targets if isinstance(x, ast.Name)}
    if not set(names) <= present:
        raise RuntimeError('Missing original AST: '+str(set(names)-present))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace), identities

class ExactForwardIdentity(torch.autograd.Function):
    """No saved tensors: true detached hard value forward, identity surrogate backward."""
    @staticmethod
    def forward(ctx, source, hard):
        if source.shape != hard.shape or source.dtype != torch.float32 or hard.dtype != torch.float32:
            raise RuntimeError('FP32 state/codec shape mismatch')
        if hard.requires_grad:
            raise RuntimeError('Hard codec path must be detached')
        return hard

    @staticmethod
    def backward(ctx, upstream):
        return upstream, None

@dataclass
class DualMTrainingResult:
    dequant: torch.Tensor
    scale: torch.Tensor
    codes: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor

class QualifiedComponents:
    def __init__(self):
        self.source_hashes = {path: sha(path) for path in FROZEN}
        if self.source_hashes != FROZEN:
            raise RuntimeError('Frozen dependency mismatch')
        self.cayley = load_file(CAYLEY_PATH, 'readiness_authoritative_cayley')
        self.legacy_c128 = self.cayley.qwen_c128_ste
        self.codec = load_file(CODEC_PATH, 'readiness_frozen_dualm_codec')
        self.base, b = extract(BASE_PATH, {'normalize_qk', 'hadamard', 'get_state'}, dict(torch=torch, math=math))
        self.core, c = extract(RUNREC, {'GDN_LAYERS', 'EPS', 'recover_state', 'relative_state_mse', 'RecurrentExperimentPatch'}, dict(torch=torch, CAYLEY=self.cayley, BASE=self.base))
        self.arch, a = extract(ARCH_PATH, {'l2norm', 'torch_recurrent_gated_delta_rule', 'Qwen3_5RMSNormGated'},
                               dict(torch=torch, nn=torch.nn, ACT2FN={'silu': torch.nn.functional.silu}), remove_decorators=True)
        self.AST_provenance = dict(base=b, core=c, architecture=a,
            decorator_scope='Only optional dispatch decorators omitted for an explicit original Torch CPU operator unit; original function/class bodies unchanged')

    def scoped_core(self, layer_ids):
        """Original helper bodies with an explicitly local representative roster."""
        ids = tuple(layer_ids)
        if not ids or not set(ids) <= set(self.core.GDN_LAYERS):
            raise ValueError('Invalid local GDN layer roster')
        module, _ = extract(RUNREC, {'EPS', 'recover_state', 'relative_state_mse', 'RecurrentExperimentPatch'},
                            dict(torch=torch, CAYLEY=self.cayley, BASE=self.base, GDN_LAYERS=ids))
        return module

    def c128(self, value):
        # The authoritative old expression and metadata are unchanged.
        return self.legacy_c128(value)

    def dual_m(self, value):
        encoded = self.codec.encode(value, 'DUAL_M_INT8')
        hard = encoded.decode()
        dequant = ExactForwardIdentity.apply(value.float(), hard)
        return DualMTrainingResult(dequant, encoded.columns, encoded.codes, encoded.rows, encoded.columns)

    @contextmanager
    def arm(self, arm):
        if arm not in ('A', 'D'):
            raise ValueError('Only paired A/D conditions')
        previous = self.cayley.qwen_c128_ste
        self.cayley.qwen_c128_ste = self.legacy_c128 if arm == 'A' else self.dual_m
        try:
            yield
        finally:
            self.cayley.qwen_c128_ste = previous

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def local_diagnostic_loss(components, bank, states, norms, projections, arm, tokens=64):
    """Representative local operator qualification, never a full-model training trajectory.

    Original recurrent, patch, state/functional helpers and norm forward are used.
    Inputs q/k/v/g/beta/gate are synthetic diagnostics; states may be actual
    captured fixtures. A local roster controls only coverage and equal weighting.
    ``norms``/``projections`` can contain separately bound real frozen weights for
    a later GPU micro-test; this function does not load models or launch anything.
    """
    ids = tuple(sorted(states))
    core = components.scoped_core(ids)
    h = components.base.hadamard(128, dtype=torch.float32, device=next(bank.parameters()).device)
    patch = core.RecurrentExperimentPatch(None, bank, h)
    recurrence = components.arch.torch_recurrent_gated_delta_rule
    wrapped = patch._wrap(recurrence)
    teacher = {i: states[i].detach().clone() for i in ids}
    student = {}
    for i in ids:
        delta = bank.layer(i).matrix()
        student[i] = torch.einsum('ij,bhjv->bhiv', delta.transpose(0,1),
                                 torch.einsum('ij,bhjv->bhiv', h.transpose(0,1), teacher[i]))
    previous = {}
    next_checks = 0
    diagnostics = hashlib.sha256()
    with components.arm(arm):
        for t in range(tokens):
            patch.capture = t == tokens-1
            for i in ids:
                state = teacher[i]
                b, heads, _, vdim = state.shape
                shape = (b, 1, heads, 128)
                device = state.device
                q, k, v = [torch.randn(shape, device=device, dtype=torch.float32).to(torch.bfloat16) for _ in range(3)]
                gate = torch.randn(shape, device=device, dtype=torch.float32).to(torch.bfloat16)
                g = -torch.rand((b,1,heads), device=device, dtype=torch.float32)*.01
                beta = torch.sigmoid(torch.randn((b,1,heads), device=device, dtype=torch.float32).to(torch.bfloat16))
                if t == tokens-1:
                    for value in (q,k,v,gate,g,beta):
                        diagnostics.update(bytes.fromhex(tensor_hash(value)))
                patch.current_layer = i
                patch.mode = 'teacher'
                with torch.no_grad():
                    output, teacher[i] = wrapped(q,k,v,g=g,beta=beta,initial_state=teacher[i],output_final_state=True,use_qk_l2norm_in_kernel=True)
                    native = projections[i](norms[i](output,gate).reshape(b,1,-1))
                    patch._out_hook(i)(None,None,native)
                patch.mode = 'student'
                if i in previous:
                    if student[i].data_ptr() != previous[i][0] or tensor_hash(student[i]) != previous[i][1]:
                        raise RuntimeError('Next diagnostic token did not consume hard QDQ state')
                    next_checks += 1
                output, student[i] = wrapped(q,k,v,g=g,beta=beta,initial_state=student[i],output_final_state=True,use_qk_l2norm_in_kernel=True)
                previous[i] = student[i].data_ptr(), tensor_hash(student[i])
                transformed = projections[i](norms[i](output,gate).reshape(b,1,-1))
                patch._out_hook(i)(None,None,transformed)
    state_loss, functional, _, loss = patch.captured_losses()
    if not torch.equal(loss, functional+.1*state_loss):
        raise RuntimeError('Original captured loss definition changed')
    return loss, dict(state_loss=state_loss, functional_loss=functional,
        QDQ_calls=patch.qdq_audit['calls'], QDQ_layers=sorted(patch.qdq_audit['layers']),
        next_QDQ_consumption_checks=next_checks, diagnostic_input_sha256=diagnostics.hexdigest(),
        teacher_target_sha256={str(i):tensor_hash(patch.teacher_out[i]) for i in ids},
        tokens_per_optimizer_microstep=tokens, layer_roster=list(ids),
        state_hashes={str(i):tensor_hash(student[i]) for i in ids},
        scope='LOCAL_OPERATOR_ONLY; original helper/recurrence bodies; diagnostic q/k/v/gate, not a full9B teacher/student trajectory')
