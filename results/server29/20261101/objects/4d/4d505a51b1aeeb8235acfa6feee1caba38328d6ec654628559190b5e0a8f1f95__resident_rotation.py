"""Instance-local, resident-coordinate Qwen3.5 GDN adapter.

The state is created by the unchanged Torch GDN helper in H/UH/UHP coordinates
and never leaves those coordinates.  Only the 16 unique Query/Key heads are
rotated.  The 32 recurrent heads, Value, gates, Beta and helper Query scaling
retain their original semantics.  No function in this module rotates State.

The forward edit is deliberately narrow: delete the original two Q/K head
repeats, and resolve the two original helper names to instance-local wrappers.
The original Conv, projection, gates, readout, cache update and decorators are
compiled from the SHA-pinned production source without other edits.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE_SHA256 = "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a"
CODEC_SOURCE_SHA256 = "110a3e928ffe9d81fa2db6fcfb0fecdfd39bd478dde7e41dc9d9209807c731de"
GDN_LAYERS = tuple(i for i in range(32) if i % 4 != 3)
PBR128 = tuple(int(f"{i:07b}"[::-1], 2) for i in range(128))
RECURRENT_TO_UNIQUE = tuple(i // 2 for i in range(32))
_CODEC_MODULE = None


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pbr_indices(device=None) -> torch.Tensor:
    return torch.tensor(PBR128, dtype=torch.long, device=device)


def hadamard128(*, dtype=torch.float32, device=None) -> torch.Tensor:
    """The original Sylvester H128 construction and normalization graph."""
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(128)


def normalize_qk_legacy(value: torch.Tensor) -> torch.Tensor:
    """Preserve original BF16 multiply/reduce/rsqrt/multiply, before FP32 cast."""
    return value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1e-6)


def prepare_unique_qk(query: torch.Tensor, key: torch.Tensor,
                      matrix: torch.Tensor, *, normalize: bool = True):
    """Native [B,T,16,128] row Q/K -> resident [B,T,32,128] FP32 Q/K.

    Matrix columns define resident coordinates, i.e. row vectors multiply R.
    For S3 matrix = A[..., PBR128].  Thus P is executed once, as part of this
    multiply, and is never applied inside the QDQ codec.  Stacking Q/K yields
    one batched matmul with 16 matrix batches, rather than 32 repeated matrices.
    Q scaling remains inside the original GDN helper, exactly once.
    """
    if query.shape != key.shape or query.ndim != 4 or tuple(query.shape[-2:]) != (16, 128):
        raise ValueError("Q/K must share [B,T,16,128] before the original head repeat")
    if query.dtype != key.dtype or query.device != key.device:
        raise ValueError("Q/K input dtype/device differ")
    if matrix.shape != (16, 128, 128) or matrix.dtype != torch.float32 or matrix.device != query.device:
        raise ValueError("Expected colocated FP32 [16,128,128] resident matrix")
    # Every row's reduction graph is the original graph; stacking does not mix
    # rows, heads, tokens, or Query with Key during normalization.
    both = torch.stack((query, key), dim=0)
    if normalize:
        both = normalize_qk_legacy(both)
    batch, length = query.shape[:2]
    with torch.autocast(device_type=query.device.type, enabled=False):
        packed = both.float().permute(3, 0, 1, 2, 4).reshape(16, 2 * batch * length, 128)
        rotated = torch.bmm(packed, matrix)
        rotated = rotated.reshape(16, 2, batch, length, 128).permute(1, 2, 3, 0, 4)
        return (rotated[0].repeat_interleave(2, dim=2),
                rotated[1].repeat_interleave(2, dim=2))


prepare_qk = prepare_unique_qk


class ResidentMatrices:
    """Immutable deployment assets, shared by C1 and S3, with no fitting."""

    def __init__(self, matrices: dict[int, torch.Tensor], *, asset_metadata=None):
        if not matrices:
            raise ValueError("No deployment matrices")
        self.matrices = {}
        for layer, matrix in matrices.items():
            if matrix.shape != (16, 128, 128) or matrix.dtype != torch.float32:
                raise ValueError(f"Invalid unique-head A32 dimensions/dtype at layer {layer}")
            if not bool(torch.isfinite(matrix).all().item()):
                raise FloatingPointError("Nonfinite deployment matrix")
            self.matrices[int(layer)] = matrix.detach().contiguous()
        devices = {str(value.device) for value in self.matrices.values()}
        if len(devices) != 1:
            raise ValueError("Deployment matrices must share one device")
        self.device = next(iter(self.matrices.values())).device
        self.asset_metadata = dict(asset_metadata or {})
        self._h = hadamard128(device=self.device).expand(16, -1, -1).contiguous()
        self._p = pbr_indices(self.device)
        self._s3 = {layer: value.index_select(-1, self._p).contiguous()
                    for layer, value in self.matrices.items()}

    @classmethod
    def from_root(cls, root: str | Path, *, device="cpu"):
        import numpy as np

        root = Path(root).resolve()
        path = root / "CANONICAL_UH_MATRICES_AND_SHA.json"
        manifest = json.loads(path.read_text())
        if (manifest.get("status") != "PASS_SEALED_SHARED_CANONICAL_UH"
                or manifest.get("refit_from_AIME") is not False
                or manifest.get("C1_S3_same_files") is not True
                or manifest.get("recurrent_mapping") != list(RECURRENT_TO_UNIQUE)):
            raise RuntimeError("Canonical shared frozen matrix manifest identity differs")
        rows = manifest["layers"]
        if sorted(int(row["layer"]) for row in rows) != list(GDN_LAYERS):
            raise RuntimeError("Expected all 24 original GDN layers exactly once")
        matrices, checked = {}, []
        for row in rows:
            layer = int(row["layer"])
            asset = (root / row["path"]).resolve()
            if not asset.is_relative_to(root):
                raise RuntimeError("Matrix asset escapes independent execution root")
            if asset.stat().st_size != row["bytes"] or file_sha256(asset) != row["sha256"]:
                raise RuntimeError(f"Frozen U/A asset bytes changed at layer {layer}")
            with np.load(asset, allow_pickle=False) as arrays:
                a, u = arrays["A_fp32_unique"].copy(), arrays["U_fp64_unique"].copy()
                mapping = arrays["recurrent_to_unique"].copy()
            if (a.dtype != np.float32 or u.dtype != np.float64
                    or a.shape != (16, 128, 128) or u.shape != a.shape
                    or not np.array_equal(mapping, np.array(RECURRENT_TO_UNIQUE))
                    or not np.isfinite(a).all() or not np.isfinite(u).all()):
                raise RuntimeError(f"Frozen matrix shape, dtype, mapping or finite identity at layer {layer}")
            if len(row["head_matrices"]) != 16:
                raise RuntimeError("Expected exactly 16 frozen unique-head SHA records")
            for head, head_record in enumerate(row["head_matrices"]):
                if (head_record["unique_query_head"] != head
                        or head_record["recurrent_heads"] != [2 * head, 2 * head + 1]
                        or hashlib.sha256(a[head].tobytes()).hexdigest() != head_record["A32_sha256"]
                        or hashlib.sha256(u[head].tobytes()).hexdigest() != head_record["U64_sha256"]):
                    raise RuntimeError(f"Per-head frozen U/A identity differs: {layer}/{head}")
            matrices[layer] = torch.from_numpy(a).to(device=device)
            checked.append({"layer": layer, "sha256": row["sha256"], "bytes": row["bytes"]})
        return cls(matrices, asset_metadata={"manifest_sha256": file_sha256(path),
                   "checked_frozen_assets": checked, "eigen_refits": 0, "new_query_tokens": 0})

    def matrix(self, layer: int, condition: str) -> torch.Tensor:
        if int(layer) not in self.matrices:
            raise KeyError(f"Missing canonical layer {layer}")
        if condition == "H":
            return self._h
        if condition == "C1":
            return self.matrices[int(layer)]
        if condition == "S3":
            return self._s3[int(layer)]
        raise ValueError("Resident conditions are H, C1 and S3 only")


def _load_codec():
    global _CODEC_MODULE
    if _CODEC_MODULE is not None:
        return _CODEC_MODULE
    path = ROOT / "vendor/group_codecs.py"
    if file_sha256(path) != CODEC_SOURCE_SHA256:
        raise RuntimeError("Original Sym32 codec source SHA changed")
    name = "_qwen_resident_frozen_sym32_codec"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("Frozen codec import identity collision")
        _CODEC_MODULE = module
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    _CODEC_MODULE = module
    return module


def symmetric32_qdq(state: torch.Tensor, *, defer_finite=False):
    """Unchanged B1 arithmetic; optionally defer host finite verdict to token end.

    The deferred path removes only the repeated per-layer .item() barriers.
    It returns a GPU Boolean covering the original pre-state, scale and levels
    checks plus decoded State; the model forward post-hook must consume it.
    Codes/scales/signed zero and arithmetic order match the frozen B1 codec.
    No full-precision continuation State is retained by this function.
    """
    codec = _load_codec()
    if not defer_finite:
        encoded = codec.encode(state, "B1")
        decoded = codec.decode(encoded)
        state.copy_(decoded)
        return encoded, None
    if state.ndim != 4 or tuple(state.shape[-2:]) != (128, 128) or state.dtype != torch.float32:
        raise ValueError("Sym32 persistent carrier must be FP32 [B,H,128,128]")
    grouped = state.detach().reshape(state.shape[0], state.shape[1], 4, 32, 128)
    peak = grouped.abs().amax(dim=-2)
    safe = peak.clamp_min(codec.EPS)
    scale = safe / 127.0
    levels = grouped / scale.unsqueeze(-2)
    rounded = torch.round(levels)
    clipped = rounded.clamp(-127, 127)
    codes = clipped.to(torch.int8)
    negative_zero = (clipped == 0) & torch.signbit(clipped)
    encoded = codec.Encoded("B1", tuple(state.shape), codes, scale, None, negative_zero)
    decoded = codes.float() * scale.unsqueeze(-2)
    decoded = torch.where(negative_zero, torch.full_like(decoded, -0.0), decoded).reshape(state.shape)
    finite = (torch.isfinite(state).all() & torch.isfinite(scale).all()
              & (scale > 0).all() & torch.isfinite(levels).all() & torch.isfinite(decoded).all())
    state.copy_(decoded)
    return encoded, finite


def load_torch_helpers(source_path: str | Path | None = None, namespace=None):
    """Load exact original Torch fallback bodies, never optional fused dispatch."""
    source_path = Path(source_path or ROOT / "vendor/modeling_qwen3_5.py")
    if file_sha256(source_path) != MODEL_SOURCE_SHA256:
        raise RuntimeError("Original Qwen modeling source SHA changed")
    wanted = {"l2norm", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"}
    nodes = [node for node in ast.parse(source_path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("Original Torch helper definitions incomplete")
    for node in nodes:
        node.decorator_list = []
    scope = dict(namespace or {}, torch=torch, F=F, math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source_path) + "::resident_original_torch_helpers", "exec"), scope)
    return {name: scope[name] for name in wanted}


def _forward_tree(source_path):
    tree = ast.parse(Path(source_path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Qwen3_5GatedDeltaNet")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    removed = []
    for node in list(forward.body):
        if not isinstance(node, ast.If):
            continue
        if len(node.body) != 2 or not all(isinstance(part, ast.Assign) for part in node.body):
            continue
        names = [part.targets[0].id if isinstance(part.targets[0], ast.Name) else None for part in node.body]
        if names != ["query", "key"]:
            continue
        expected = ast.parse("if self.num_v_heads // self.num_k_heads > 1:\n"
                             " query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)\n"
                             " key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)\n").body[0]
        if ast.dump(node, include_attributes=False) != ast.dump(expected, include_attributes=False):
            raise RuntimeError("Original repeat_interleave block changed")
        forward.body.remove(node)
        removed.append(node)
    if len(removed) != 1:
        raise RuntimeError("Expected exactly the original two Q/K repeats")
    if sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule")
           for node in ast.walk(forward)) != 2:
        raise RuntimeError("Original GDN helper dispatch structure changed")
    return forward


class ResidentRotationPatch:
    """Context manager binding only this model's 24 GDN module instances.

    observer(event, token_index, layer, payload) may synchronously persist exact
    values/bytes. Payload tensors are live references: an observer must copy
    evidence before returning if it needs to retain it. In particular, no
    pre-QDQ State is kept by the adapter. audit_next_consumption keeps only
    encoded INT8/scales/signed-zero CPU evidence, never a high-precision shadow.
    """

    def __init__(self, model, matrices: ResidentMatrices, condition: str, *, qdq=True,
                 observer: Callable | None = None, audit_next_consumption=False,
                 finite_policy="token", source_path=None, layers=GDN_LAYERS):
        if condition not in ("H", "C1", "S3"):
            raise ValueError("Unknown resident rotation condition")
        if finite_policy not in ("token", "immediate"):
            raise ValueError("Finite verdict must be token or immediate")
        self.model, self.matrices, self.condition = model, matrices, condition
        self.qdq, self.observer = bool(qdq), observer
        self.audit_next_consumption, self.finite_policy = bool(audit_next_consumption), finite_policy
        self.layers = tuple(int(layer) for layer in layers)
        self.source_path = Path(source_path or ROOT / "vendor/modeling_qwen3_5.py")
        self.codec = _load_codec()
        self.helpers = load_torch_helpers(self.source_path)
        self.saved, self.handles, self.previous_encoded, self.seen = [], [], {}, set()
        self.finite_flags, self.token_layers = [], set()
        self.token_index, self.active = 0, False
        self.counters = dict(qk_batched_matmul_calls=0, unique_qk_vectors_rotated=0,
                             original_gdn_helper_calls=0, qdq_calls=0,
                             next_consumption_full_value_byte_checks=0,
                             state_rotation_matmul_calls=0, full_token_finite_checks=0,
                             chunk_prefill_calls=0, recurrent_decode_calls=0)

    def _observe(self, event, layer, **payload):
        if self.observer is not None:
            self.observer(event, self.token_index, int(layer), payload)

    def _helper(self, layer, operator):
        original = self.helpers["torch_" + operator + "_gated_delta_rule"]

        def call(query, key, value, *args, **kwargs):
            if args:
                raise RuntimeError("Pinned production forward passes GDN gate/Beta/cache by keyword")
            if query.shape[1] != 1:
                raise RuntimeError("Resident persistent protocol requires q_length=1 from Prompt token0")
            if layer in self.token_layers:
                raise RuntimeError("Repeated GDN layer within the same model forward")
            self.token_layers.add(layer)
            if kwargs.pop("use_qk_l2norm_in_kernel", None) is not True:
                raise RuntimeError("Original forward must request Q/K normalization exactly once")
            initial = kwargs.get("initial_state")
            if layer not in self.seen and initial is not None:
                raise RuntimeError("Resident coordinates must start from a fresh empty recurrent cache")
            if layer in self.seen and initial is None:
                raise RuntimeError("Recurrent cache reset requires a fresh adapter/request")
            if initial is not None:
                if initial.dtype != torch.float32 or tuple(initial.shape[-3:]) != (32, 128, 128):
                    raise RuntimeError("Persistent resident cache carrier shape/dtype changed")
                if self.audit_next_consumption and self.qdq:
                    if layer not in self.previous_encoded:
                        raise RuntimeError("Missing previous encoded resident writeback evidence")
                    self.codec.require_consumed(initial, self.previous_encoded.pop(layer))
                    self.counters["next_consumption_full_value_byte_checks"] += 1
                self._observe("NEXT_CONSUMPTION", layer, actual=initial)
            with torch.profiler.record_function("resident.QK"):
                query, key = prepare_unique_qk(query, key, self.matrices.matrix(layer, self.condition))
            self.counters["qk_batched_matmul_calls"] += 1
            self.counters["unique_qk_vectors_rotated"] += 32 * query.shape[0] * query.shape[1]
            self._observe("PREPARED_QK", layer, query=query, key=key, value=value,
                          g=kwargs.get("g"), beta=kwargs.get("beta"), initial_state=initial)
            with torch.profiler.record_function("resident.GDN"):
                core, state = original(query, key, value, **kwargs, use_qk_l2norm_in_kernel=False)
            self.counters["original_gdn_helper_calls"] += 1
            self.counters["recurrent_decode_calls" if operator == "recurrent" else "chunk_prefill_calls"] += 1
            if state is None:
                raise RuntimeError("Resident protocol requires output_final_state=True/use_cache=True")
            if state.dtype != torch.float32 or tuple(state.shape[-3:]) != (32, 128, 128):
                raise RuntimeError("Original helper did not return FP32 resident State")
            self._observe("PRE_QDQ", layer, state=state)
            if self.qdq:
                with torch.profiler.record_function("resident.QDQ"):
                    encoded, finite = symmetric32_qdq(state, defer_finite=self.finite_policy == "token")
                self.counters["qdq_calls"] += 1
                if finite is not None:
                    self.finite_flags.append(finite)
                if self.audit_next_consumption:
                    expected = encoded.cpu()
                    self.codec.require_consumed(state, expected)
                    self.previous_encoded[layer] = expected
                self._observe("POST_QDQ", layer, codes=encoded.codes, scales=encoded.scales,
                              negative_zero_flags=encoded.negative_zero_flags, writeback=state)
            else:
                self.finite_flags.append(torch.isfinite(state).all())
            self.finite_flags.append(torch.isfinite(core).all())
            self.seen.add(layer)
            return core.to(value.dtype), state

        return call

    def _begin_token(self, _module, args, kwargs):
        if self.token_layers or self.finite_flags:
            raise RuntimeError("Previous model forward did not reach its finite/cache verdict")
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        embeds = kwargs.get("inputs_embeds")
        source = input_ids if input_ids is not None else embeds
        if source is None or source.shape[1] != 1:
            raise RuntimeError("Resident model input must contain exactly one Prompt/Decode token")

    def _finish_token(self, _module, _args, output):
        if self.token_layers != set(self.layers):
            raise RuntimeError("Model did not execute every resident GDN layer exactly once")
        logits = getattr(output, "logits", None)
        if logits is not None:
            self.finite_flags.append(torch.isfinite(logits).all())
        if not self.finite_flags or not bool(torch.stack(self.finite_flags).all().item()):
            raise FloatingPointError("Nonfinite State/QDQ/Core/Logits in resident token forward")
        self.counters["full_token_finite_checks"] += 1
        self.token_index += 1
        self.token_layers.clear()
        self.finite_flags.clear()

    def __enter__(self):
        if self.active or getattr(self.model, "_resident_rotation_owner", None) is not None:
            raise RuntimeError("Nested/reused resident adapter is forbidden")
        if file_sha256(self.source_path) != MODEL_SOURCE_SHA256:
            raise RuntimeError("Pinned forward source changed")
        if self.matrices.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("Deployment requires TF32 disabled for the FP32 rotation contract")
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        try:
            for index in self.layers:
                module = getattr(layers[index], "linear_attn", None)
                if module is None or type(module).__name__ != "Qwen3_5GatedDeltaNet":
                    raise RuntimeError(f"Expected actual GDN module at layer {index}")
                if (module.num_k_heads, module.num_v_heads, module.head_k_dim, module.head_v_dim) != (16, 32, 128, 128):
                    raise RuntimeError("Actual Qwen unique/recurrent head configuration differs")
                if "forward" in module.__dict__ or module._forward_pre_hooks or module._forward_hooks:
                    raise RuntimeError("Existing instance forward/hooks may apply old H/U/P transforms")
                qmod = importlib.import_module(type(module).__module__)
                if file_sha256(qmod.__file__) != MODEL_SOURCE_SHA256:
                    raise RuntimeError("Live loaded Qwen module file identity differs")
                # Reject an active old global H/rotation wrapper instead of
                # silently bypassing it and presenting ambiguous provenance.
                for helper_name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
                    if getattr(getattr(qmod, helper_name), "__name__", None) != helper_name:
                        raise RuntimeError("Live helper name indicates an active global adapter")
                scope = dict(qmod.__dict__)
                scope["torch_recurrent_gated_delta_rule"] = self._helper(index, "recurrent")
                scope["torch_chunk_gated_delta_rule"] = self._helper(index, "chunk")
                node = _forward_tree(self.source_path)
                exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                             str(self.source_path) + "::resident_instance_forward", "exec"), scope)
                self.saved.append((module, "forward" in module.__dict__, module.forward))
                module.forward = types.MethodType(scope["forward"], module)
            self.handles.append(self.model.register_forward_pre_hook(self._begin_token, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._finish_token))
            self.model._resident_rotation_owner = self
            self.active = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, had_instance_forward, original in reversed(self.saved):
            if had_instance_forward:
                module.forward = original
            elif "forward" in module.__dict__:
                del module.forward
        self.saved.clear()
        if getattr(self.model, "_resident_rotation_owner", None) is self:
            del self.model._resident_rotation_owner
        self.previous_encoded.clear()
        self.active = False

    def audit(self):
        return dict(self.counters, condition=self.condition,
                    coordinate={"H": "H128", "C1": "Uq@H128", "S3": "Uq@H128@PBR128"}[self.condition],
                    qdq=self.qdq, codec="Frozen B1 Symmetric INT8 Key-axis Group32",
                    scale_dtype="FP32", state_dtype="FP32", packed_INT8_persistent_storage=False,
                    qk_normalize="original incoming BF16 arithmetic once before FP32 rotation",
                    q_scale="original helper 1/sqrt(128) exactly once",
                    qk_matmul_batches=16, shared_head_repeat=2,
                    state_rotation_matmul_calls=0, P_inside_QDQ=False,
                    high_precision_continuation_shadow=False,
                    original_torch_helper_body_sha256=MODEL_SOURCE_SHA256,
                    original_codec_source_sha256=CODEC_SOURCE_SHA256,
                    finite_policy=self.finite_policy, processed_tokens=self.token_index,
                    assets=self.matrices.asset_metadata)
