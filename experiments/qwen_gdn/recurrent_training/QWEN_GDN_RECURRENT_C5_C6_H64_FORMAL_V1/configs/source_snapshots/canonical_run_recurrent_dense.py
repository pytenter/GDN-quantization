#!/usr/bin/env python3
"""Recurrent-aware dense/Cayley rotation training for Qwen3.5-9B GDN.

This experiment keeps the audited C3/C4 objectives and changes only the
source of recurrent state: the student consumes its own post-QDQ state on the
next token.  Teacher trajectories provide detached targets only.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


TASK = "QWEN_RECURRENT_DENSE_C5_C6_V1"
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
BASE_REPO = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
BASE_RUNNER = BASE_REPO / "experiments/aime26/run_qwen_aime26_formal.py"
CAYLEY_SOURCE = REPO / "experiments/shared/rotation/cayley_rotation.py"
OLD_TRAINER = REPO / "experiments/qwen_gdn/rotation/hadamard_init_dense_orthogonal_oracle_v1/run_qwen_dense_oracle.py"
OLD_AUDIT = REPO / "experiments/QWEN_DENSE_TRAINING_CHECKPOINT_PROVENANCE_AUDIT_V1"
OLD_E2E = REPO / "experiments/QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1"
CORPUS = REPO / "CALIBRATION_RAW_TEXTS.jsonl"
OLD_TRACE_ROOT = REPO / "results/rotation/qwen_dense_orthogonal_oracle_v1/traces"
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
EPS = 1.0e-12
TRAIN_SEED = 0
TARGET_EXPOSURES = 1000
TARGETS_PER_DOCUMENT = 8
OPTIMIZER_UPDATES = TARGET_EXPOSURES // TARGETS_PER_DOCUMENT
VALIDATION_INTERVAL_EXPOSURES = 50
SEQUENCE_LENGTH_MAX = 1024
HORIZON_CANDIDATES = (512, 256, 128, 64, 32)
ORTHOGONALITY_THRESHOLD = 1.0e-4


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(BASE_RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(BASE_RUNNER.parent))
BASE = import_file(BASE_RUNNER, "qwen_recurrent_dense_base")
CAYLEY = import_file(CAYLEY_SOURCE, "qwen_recurrent_dense_cayley")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()


def corpus_rows(split: str) -> list[dict]:
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row["split"] == split.upper()]
    if not selected:
        raise RuntimeError(f"no corpus rows for {split}")
    return selected


def sampled_positions(row: dict, token_count: int) -> list[int]:
    if token_count <= 136:
        raise RuntimeError(f"{row['document_id']} has only {token_count} tokens")
    seed = int(row["raw_text_sha256"][:16], 16) ^ 20260921
    return sorted(random.Random(seed).sample(list(range(128, token_count)), TARGETS_PER_DOCUMENT))


def hadamard(device: torch.device) -> torch.Tensor:
    return BASE.hadamard(128, dtype=torch.float32, device=device)


def recover_state(rotated_state: torch.Tensor, delta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    corrected = torch.einsum("ij,bhjv->bhiv", delta, rotated_state.float())
    return torch.einsum("ij,bhjv->bhiv", h, corrected)


def relative_state_mse(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = (value.float() - reference.float()).square().sum(dim=(-2, -1))
    denominator = reference.float().square().sum(dim=(-2, -1)).clamp_min(EPS)
    return (numerator / denominator).mean()


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != 0:
        raise RuntimeError(f"MODEL_FREEZE_GATE_FAIL trainable={count}")


def detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    return value


def detach_cache(cache) -> None:
    if cache is None:
        return
    names = ("recurrent_states", "conv_states", "keys", "values", "key_cache", "value_cache")
    for name in names:
        if hasattr(cache, name):
            try:
                setattr(cache, name, detach_tree(getattr(cache, name)))
            except (AttributeError, TypeError):
                current = getattr(cache, name)
                if isinstance(current, list):
                    for index, item in enumerate(current):
                        current[index] = detach_tree(item)
    for layer in getattr(cache, "layers", []):
        for name in names:
            if hasattr(layer, name):
                try:
                    setattr(layer, name, detach_tree(getattr(layer, name)))
                except (AttributeError, TypeError):
                    pass
        # record_past=True selects the differentiable concat path for the
        # depthwise-convolution cache.  Retain only the formal kernel window at
        # graph boundaries so numerical state continues without unbounded
        # history growth.
        conv_states = getattr(layer, "conv_states", None)
        kernels = getattr(layer, "conv_kernel_size", None)
        if isinstance(conv_states, dict) and isinstance(kernels, dict):
            for state_idx, value in list(conv_states.items()):
                if torch.is_tensor(value) and state_idx in kernels:
                    conv_states[state_idx] = value.detach()[..., -int(kernels[state_idx]) :]
        if hasattr(layer, "record_past"):
            layer.record_past = True


def enable_differentiable_cache(cache) -> None:
    """Select assignment/concat cache updates instead of static copy_ writes."""
    if cache is None:
        return
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "record_past"):
            layer.record_past = True


def cache_description(cache) -> dict:
    result = {"type": type(cache).__name__, "module": type(cache).__module__}
    try:
        result["top_level_attributes"] = sorted(name for name in vars(cache) if not name.startswith("_"))
    except TypeError:
        result["top_level_attributes"] = "NOT_AVAILABLE"
    layers = getattr(cache, "layers", None)
    if layers:
        result["layer_count"] = len(layers)
        result["layer0_type"] = type(layers[0]).__name__
        result["layer0_attributes"] = sorted(name for name in vars(layers[0]) if not name.startswith("_"))
    return result


class RecurrentExperimentPatch:
    """Audited H@Delta path plus differentiable post-update C128 writeback."""

    def __init__(self, model, bank, h):
        self.model = model
        self.bank = bank
        self.h = h
        self.mode = "idle"
        self.capture = False
        self.current_layer = None
        self.teacher_prev: dict[int, torch.Tensor] = {}
        self.teacher_out: dict[int, torch.Tensor] = {}
        self.student_prev: dict[int, torch.Tensor] = {}
        self.student_post: dict[int, torch.Tensor] = {}
        self.state_losses: dict[int, torch.Tensor] = {}
        self.functional_losses: dict[int, torch.Tensor] = {}
        self.qdq_audit = {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}
        self.saved = []
        self.handles = []
        self.saved_cache_method = None

    def clear_capture(self) -> None:
        self.teacher_prev.clear()
        self.teacher_out.clear()
        self.student_prev.clear()
        self.student_post.clear()
        self.state_losses.clear()
        self.functional_losses.clear()

    def _wrap(self, function):
        def wrapped(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            layer = int(self.current_layer)
            initial = kwargs.get("initial_state")
            if self.mode == "teacher":
                core, state = function(query, key, value, *args, use_qk_l2norm_in_kernel=use_norm, **kwargs)
                if self.capture and layer in GDN_LAYERS and initial is not None:
                    self.teacher_prev[layer] = initial.detach().float().clone()
                return core, state
            if self.mode != "student":
                return function(query, key, value, *args, use_qk_l2norm_in_kernel=use_norm, **kwargs)
            if layer not in GDN_LAYERS:
                raise RuntimeError(f"student dense patch invoked on unmapped layer {layer}")
            normalized_q = BASE.normalize_qk(query) if use_norm else query
            normalized_k = BASE.normalize_qk(key) if use_norm else key
            delta = self.bank.layer(layer).matrix()
            q_rot = normalized_q.float().matmul(self.h).matmul(delta)
            k_rot = normalized_k.float().matmul(self.h).matmul(delta)
            core, updated = function(
                q_rot, k_rot, value, *args, use_qk_l2norm_in_kernel=False, **kwargs
            )
            quant = CAYLEY.qwen_c128_ste(updated.float())
            if not bool(torch.isfinite(quant.dequant).all().detach().cpu()):
                raise FloatingPointError(f"nonfinite student QDQ at layer {layer}")
            self.qdq_audit["calls"] += 1
            self.qdq_audit["layers"].add(layer)
            self.qdq_audit["scale_shapes"][str(layer)] = list(quant.scale.shape)
            self.qdq_audit["code_min"] = min(self.qdq_audit["code_min"], int(quant.codes.min().detach().cpu()))
            self.qdq_audit["code_max"] = max(self.qdq_audit["code_max"], int(quant.codes.max().detach().cpu()))
            if self.capture:
                if initial is None or layer not in self.teacher_prev:
                    raise RuntimeError(f"missing recurrent previous state at captured layer {layer}")
                self.student_prev[layer] = initial
                self.student_post[layer] = quant.dequant
                recovered = recover_state(initial, delta, self.h)
                self.state_losses[layer] = relative_state_mse(recovered, self.teacher_prev[layer])
            return core.to(value.dtype), quant.dequant
        return wrapped

    def _out_hook(self, layer: int):
        def hook(_module, _arguments, output):
            if not self.capture or layer not in GDN_LAYERS:
                return
            if self.mode == "teacher":
                self.teacher_out[layer] = output.detach().float().clone()
            elif self.mode == "student":
                if layer not in self.teacher_out:
                    raise RuntimeError(f"missing teacher out-proj target at layer {layer}")
                self.functional_losses[layer] = CAYLEY.relative_mse(output.float(), self.teacher_out[layer])
        return hook

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        import transformers.cache_utils as cache_utils

        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is None:
                continue
            self.handles.append(module.register_forward_pre_hook(lambda _m, _a, i=index: setattr(self, "current_layer", i)))
            if index in GDN_LAYERS:
                self.handles.append(module.out_proj.register_forward_hook(self._out_hook(index)))
        for name in ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"):
            if hasattr(qmod, name):
                original = getattr(qmod, name)
                self.saved.append((qmod, name, original))
                setattr(qmod, name, self._wrap(original))
        cache_class = cache_utils.LinearAttentionLayer
        original_cache_update = cache_class.update_recurrent_state

        def differentiable_recurrent_update(layer, recurrent_states, state_idx=0, **kwargs):
            if not getattr(layer, "record_past", False):
                return original_cache_update(layer, recurrent_states, state_idx, **kwargs)
            if not layer.is_recurrent_states_initialized[state_idx]:
                layer.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            layer.recurrent_states[state_idx] = recurrent_states
            return recurrent_states

        self.saved_cache_method = (cache_class, original_cache_update)
        cache_class.update_recurrent_state = differentiable_recurrent_update
        return self

    def __exit__(self, *_):
        for module, name, original in self.saved:
            setattr(module, name, original)
        for handle in self.handles:
            handle.remove()
        if self.saved_cache_method is not None:
            cache_class, original = self.saved_cache_method
            cache_class.update_recurrent_state = original

    def captured_losses(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        missing_state = sorted(set(GDN_LAYERS) - set(self.state_losses))
        missing_func = sorted(set(GDN_LAYERS) - set(self.functional_losses))
        if missing_state or missing_func:
            raise RuntimeError(f"capture incomplete state={missing_state} functional={missing_func}")
        state = sum(self.state_losses.values()) / len(GDN_LAYERS)
        functional = sum(self.functional_losses.values()) / len(GDN_LAYERS)
        return state, functional, state, functional + 0.1 * state


def load_model_bank():
    os.environ.setdefault("GDN_DATA_ROOT", "/data/zypan")
    model, tokenizer = BASE.load_model_and_tokenizer()
    freeze_model(model)
    device = model.get_input_embeddings().weight.device
    bank = CAYLEY.PerLayerCayleyRotations(GDN_LAYERS).to(device)
    return model, tokenizer, bank, device


def tokenize(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:SEQUENCE_LENGTH_MAX]
    return ids, sampled_positions(row, len(ids))


def forward_teacher(model, patch, token_ids, cache, capture: bool):
    patch.mode = "teacher"
    patch.capture = capture
    device = model.get_input_embeddings().weight.device
    token = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
    return output.past_key_values


def forward_student(model, patch, token_id: int, cache, capture: bool):
    patch.mode = "student"
    patch.capture = capture
    device = model.get_input_embeddings().weight.device
    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
    output = model(input_ids=token, past_key_values=cache, use_cache=True)
    result = output.past_key_values
    enable_differentiable_cache(result)
    return result


def collect_teacher_targets(model, patch, ids: list[int], positions: list[int]) -> dict[int, dict[str, dict[int, torch.Tensor]]]:
    """Reproduce the old trace collector's chunk-to-position teacher path."""
    targets = {}
    cache = None
    current = 0
    for position in positions:
        if current < position:
            cache = forward_teacher(model, patch, ids[current:position], cache, False)
        patch.clear_capture()
        cache = forward_teacher(model, patch, [ids[position]], cache, True)
        missing_state = sorted(set(GDN_LAYERS) - set(patch.teacher_prev))
        missing_out = sorted(set(GDN_LAYERS) - set(patch.teacher_out))
        if missing_state or missing_out:
            raise RuntimeError(f"teacher capture incomplete state={missing_state} out={missing_out}")
        targets[position] = {
            "prev": {layer: value.detach().cpu() for layer, value in patch.teacher_prev.items()},
            "out": {layer: value.detach().cpu() for layer, value in patch.teacher_out.items()},
        }
        current = position + 1
    del cache
    patch.clear_capture()
    return targets


def install_teacher_target(patch, target, device: torch.device) -> None:
    patch.clear_capture()
    patch.teacher_prev.update({layer: value.to(device) for layer, value in target["prev"].items()})
    patch.teacher_out.update({layer: value.to(device) for layer, value in target["out"].items()})


def prepare() -> None:
    for name in ("configs", "manifests", "analysis", "checkpoints/c5", "checkpoints/c6", "outputs/C5", "outputs/C6", "logs", "reports", "hashes"):
        (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)
    e2e_summary = OLD_E2E / "analysis/summary.csv"
    e2e_config = OLD_E2E / "configs/frozen_eval_config.json"
    eval_samples = OLD_E2E / "manifests/eval_samples.jsonl"
    data_audit = load_json(OLD_AUDIT / "manifests/data_manifest.json")
    objective_audit = load_json(OLD_AUDIT / "analysis/exact_objective_audit.json")
    training_runs = load_json(OLD_AUDIT / "manifests/training_run_manifest.json")
    selection = load_json(OLD_AUDIT / "analysis/checkpoint_selection_audit.json")
    prereg = {
        "experiment_name": TASK,
        "created_before_training": True,
        "hypothesis": "recurrent_int8_exposure_may_improve_learned_rotation_transfer",
        "fixed_baseline": {"C2_correct": 13, "C2_total": 20},
        "old_dense": {"C3_correct": 12, "C4_correct": 8, "training_mode": "FP_STATE_RESET_EACH_TOKEN"},
        "new_conditions": ["C5_RECURRENT_DENSE_STATE", "C6_RECURRENT_DENSE_FUNCTIONAL"],
        "training_data": {"path": str(CORPUS), "sha256": sha256(CORPUS), "excludes_aime26": True},
        "rotation": {
            "formula": "A[upper]=theta; A[lower]=-theta; DeltaR=(I-A)(I+A)^-1; row-vector R_final=H128@DeltaR",
            "initialization": "theta=0, DeltaR=I, R_final=H128",
            "warm_start": False,
            "layers": list(GDN_LAYERS),
            "head_shared": True,
        },
        "formal_eval": {
            "question_ids": [f"aime26_{i:02d}" for i in range(11, 31)],
            "generation_seed": 1,
            "max_new_tokens": 81920,
            "scorer": "Frozen V4",
        },
        "primary_comparisons": ["C5_vs_C2", "C6_vs_C2"],
        "primary_multiplicity": "Holm correction",
        "secondary_comparisons": ["C5_vs_C3", "C6_vs_C4", "C6_vs_C5"],
        "optimizer": {
            "C5": {"name": "Adam", "lr": 0.003, "weight_decay": 0.0},
            "C6": {"name": "Adam", "lr": 0.001, "weight_decay": 0.0},
            "training_seed": 0,
            "gradient_clip_global_norm": 1.0,
        },
        "budget": {
            "target_exposures": TARGET_EXPOSURES,
            "targets_per_document": TARGETS_PER_DOCUMENT,
            "optimizer_updates": OPTIMIZER_UPDATES,
            "validation_interval_target_exposures": VALIDATION_INTERVAL_EXPOSURES,
            "sequence_length_max": SEQUENCE_LENGTH_MAX,
            "numerical_state_recurrence": "continuous within each document rollout",
            "gradient_horizon": "largest passing candidate selected by resource feasibility only",
            "candidate_gradient_horizons": list(HORIZON_CANDIDATES),
        },
        "checkpoint_selection": {
            "C5": selection["conditions"]["C3"]["selection_rule"],
            "C6": selection["conditions"]["C4"]["selection_rule"],
            "dataset": "fixed WikiText-2 raw VALIDATION panel",
            "AIME_used": False,
        },
        "source_commit": git_commit(),
        "source_hashes": {
            "old_trainer": sha256(OLD_TRAINER),
            "cayley": sha256(CAYLEY_SOURCE),
            "formal_runner": sha256(BASE_RUNNER),
        },
    }
    atomic_json(EXPERIMENT / "preregistration.json", prereg)
    atomic_json(EXPERIMENT / "protocol_amendment.json", {
        "status": "PRE_REGISTERED",
        "changes_from_old_local_training": [
            "microbatch is one continuous document rollout",
            "eight frozen sampled targets are gradient-accumulated before one optimizer update",
            "autograd is truncated at a resource-selected interval while numerical state remains continuous",
        ],
        "unchanged": ["optimizer", "learning_rate", "training_seed", "target exposure count", "objective formula", "data splits", "checkpoint-selection metric"],
        "reason": "real recurrent state writeback requires keeping rotation coordinates fixed throughout a numerical trajectory",
        "selection_rule": "RESOURCE_FEASIBILITY_ONLY",
    })
    atomic_json(EXPERIMENT / "configs/inherited_c3_training_config.json", {"source": training_runs["conditions"]["C3"], "objective": objective_audit["DENSE_STATE_EXACT_OBJECTIVE"]})
    atomic_json(EXPERIMENT / "configs/inherited_c4_training_config.json", {"source": training_runs["conditions"]["C4"], "objective": objective_audit["DENSE_FUNCTIONAL_EXACT_OBJECTIVE"]})
    for condition, objective, lr in (("c5", "DENSE_STATE", 0.003), ("c6", "DENSE_FUNCTIONAL", 0.001)):
        atomic_json(EXPERIMENT / f"configs/recurrent_training_config_{condition}.json", {
            "condition": condition.upper(), "objective": objective, "lr": lr, "optimizer": "Adam", "weight_decay": 0.0,
            "seed": 0, "target_exposures": TARGET_EXPOSURES, "targets_per_document": 8,
            "optimizer_updates": OPTIMIZER_UPDATES, "sequence_length_max": SEQUENCE_LENGTH_MAX,
            "numerical_recurrence": "CONTINUOUS_INT8_STUDENT", "gradient_horizon": "FROM_HORIZON_GATE",
            "checkpoint_selection": "MIN_FIXED_VALIDATION_PRIMARY", "AIME_used": False,
        })
    shutil.copy2(e2e_config, EXPERIMENT / "configs/frozen_eval_config.json")
    shutil.copy2(eval_samples, EXPERIMENT / "manifests/eval_samples.jsonl")
    atomic_json(EXPERIMENT / "manifests/training_data_manifest.json", {**data_audit, "TRAINING_DATA_AIME26_OVERLAP": "NO"})
    atomic_json(EXPERIMENT / "manifests/validation_data_manifest.json", {
        "split": "VALIDATION", "documents": data_audit["split_counts"]["VALIDATION"],
        "targets_per_document": 8, "source_manifest": data_audit["manifests"]["VALIDATION"], "AIME26_OVERLAP": "NO",
    })
    atomic_json(EXPERIMENT / "manifests/baseline_reuse_manifest.json", {
        "source_experiment": str(OLD_E2E), "summary_path": str(e2e_summary), "summary_sha256": sha256(e2e_summary),
        "conditions": {"C0": [17, 20], "C1": [4, 20], "C2": [13, 20], "C3": [12, 20], "C4": [8, 20]},
        "regenerate": False,
    })
    atomic_json(EXPERIMENT / "analysis/old_vs_new_training_semantics.json", {
        "C3": "FP_STATE_RESET_EACH_TOKEN", "C4": "FP_STATE_RESET_EACH_TOKEN",
        "C5": "REAL_RECURRENT_INT8_STATE_WRITEBACK", "C6": "REAL_RECURRENT_INT8_STATE_WRITEBACK",
        "C3_EXACT_OBJECTIVE": objective_audit["DENSE_STATE_EXACT_OBJECTIVE"],
        "C5_EXACT_OBJECTIVE": objective_audit["DENSE_STATE_EXACT_OBJECTIVE"],
        "C4_EXACT_OBJECTIVE": objective_audit["DENSE_FUNCTIONAL_EXACT_OBJECTIVE"],
        "C6_EXACT_OBJECTIVE": objective_audit["DENSE_FUNCTIONAL_EXACT_OBJECTIVE"],
        "ONLY_PRIMARY_CHANGE": "FP_RESET -> REAL_RECURRENT_WRITEBACK",
    })
    print(json.dumps({"status": "PREPARED", "experiment": str(EXPERIMENT)}, indent=2), flush=True)


def qdq_static_gate() -> dict:
    torch.manual_seed(20260926)
    value = torch.randn(1, 32, 128, 128, dtype=torch.float32)
    exact = CAYLEY.qwen_c128_qdq(value)
    source = value.float()
    scale = source.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
    codes = torch.round(source / scale).clamp(-127, 127)
    formal = codes * scale
    maximum = float((formal - exact.dequant).abs().max())
    return {
        "TRAIN_VS_FORMAL_QDQ_MATCH": "EXACT_MATCH" if maximum == 0.0 else "FAIL",
        "max_abs": maximum, "scale_shape": list(scale.shape), "axis": -2,
        "qmin": int(codes.min()), "qmax": int(codes.max()), "zero_point": 0,
        "rounding": "torch.round ties-to-even", "floor": EPS,
    }


def smoke() -> None:
    model, tokenizer, bank, device = load_model_bank()
    row = corpus_rows("TRAIN")[0]
    ids, _positions = tokenize(tokenizer, row)
    ids = ids[:8]
    teacher_cache = None
    student_cache = None
    provenance = []
    gradient_results = {}
    h = hadamard(device)
    with RecurrentExperimentPatch(model, bank, h) as patch:
        for index, token_id in enumerate(ids):
            capture = index >= 1
            if capture:
                patch.clear_capture()
            teacher_cache = forward_teacher(model, patch, [token_id], teacher_cache, capture)
            student_cache = forward_student(model, patch, token_id, student_cache, capture)
            if capture:
                state, functional, c5_loss, c6_loss = patch.captured_losses()
                layer = GDN_LAYERS[0]
                teacher_hash = tensor_sha256(patch.teacher_prev[layer])
                student_prev_hash = tensor_sha256(patch.student_prev[layer])
                post_hash = tensor_sha256(patch.student_post[layer])
                previous_post = provenance[-1]["student_post_qdq_state_hash"] if provenance else None
                provenance.append({
                    "token_index": index,
                    "layer": layer,
                    "teacher_prev_state_hash": teacher_hash,
                    "student_prev_state_hash": student_prev_hash,
                    "student_post_qdq_state_hash": post_hash,
                    "previous_student_post_qdq_state_hash": previous_post,
                    "consumed_matches_previous_post": previous_post is None or student_prev_hash == previous_post,
                    "student_differs_from_teacher": student_prev_hash != teacher_hash,
                    "state_loss": float(state.detach().cpu()),
                    "functional_loss": float(functional.detach().cpu()),
                })
                if index == 1:
                    for name, loss in (("C5", c5_loss), ("C6", c6_loss)):
                        bank.zero_grad(set_to_none=True)
                        loss.backward(retain_graph=True)
                        grads = [parameter.grad for parameter in bank.parameters()]
                        finite = all(grad is not None and bool(torch.isfinite(grad).all()) for grad in grads)
                        norm = math.sqrt(sum(float(grad.detach().float().square().sum().cpu()) for grad in grads if grad is not None))
                        gradient_results[name] = {
                            f"{name}_GRADIENT_GATE": "PASS" if finite and norm > 0 else "FAIL",
                            "loss_finite": bool(torch.isfinite(loss).detach().cpu()),
                            "rotation_gradient_finite": finite,
                            "rotation_gradient_nonzero": norm > 0,
                            "rotation_grad_norm": norm,
                            "rotation_parameters_with_grad": sum(grad is not None for grad in grads),
                            "base_model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "hard_qdq_forward_active": True,
                        }
                    bank.zero_grad(set_to_none=True)
            detach_cache(teacher_cache)
        qdq = qdq_static_gate()
        writeback = all(row["consumed_matches_previous_post"] for row in provenance[1:])
        diverged = any(row["student_differs_from_teacher"] for row in provenance)
        recurrent = {
            "REAL_RECURRENT_WRITEBACK_GATE": "PASS" if writeback else "FAIL",
            "RECURRENT_STATE_PROVENANCE_GATE": "PASS" if writeback and diverged else "FAIL",
            "numerical_state_recurrence": "continuous",
            "autograd_graph": "smoke-only; horizon selected separately",
            "records": provenance,
        }
        atomic_json(EXPERIMENT / "analysis/recurrent_state_provenance.json", recurrent)
        atomic_json(EXPERIMENT / "analysis/gradient_gate_c5.json", gradient_results["C5"])
        atomic_json(EXPERIMENT / "analysis/gradient_gate_c6.json", gradient_results["C6"])
        atomic_json(EXPERIMENT / "analysis/train_vs_formal_quantization.json", qdq)
        result = {
            **{k: recurrent[k] for k in ("REAL_RECURRENT_WRITEBACK_GATE", "RECURRENT_STATE_PROVENANCE_GATE")},
            "TRAIN_VS_FORMAL_QDQ_MATCH": qdq["TRAIN_VS_FORMAL_QDQ_MATCH"],
            "C5_GRADIENT_GATE": gradient_results["C5"]["C5_GRADIENT_GATE"],
            "C6_GRADIENT_GATE": gradient_results["C6"]["C6_GRADIENT_GATE"],
            "cache": cache_description(student_cache),
            "qdq_calls": patch.qdq_audit["calls"],
            "qdq_layers": sorted(patch.qdq_audit["layers"]),
        }
        atomic_json(EXPERIMENT / "analysis/pretraining_smoke_summary.json", result)
        print(json.dumps(result, indent=2), flush=True)
        if any(result[key] not in ("PASS", "EXACT_MATCH", "NUMERICALLY_EQUIVALENT") for key in (
            "REAL_RECURRENT_WRITEBACK_GATE", "RECURRENT_STATE_PROVENANCE_GATE", "TRAIN_VS_FORMAL_QDQ_MATCH", "C5_GRADIENT_GATE", "C6_GRADIENT_GATE"
        )):
            raise RuntimeError("PRETRAINING_CORE_SEMANTICS_GATE_FAIL")


def horizon_feasibility() -> None:
    model, tokenizer, bank, device = load_model_bank()
    row = corpus_rows("TRAIN")[0]
    ids, _positions = tokenize(tokenizer, row)
    h = hadamard(device)
    results = []
    selected = None
    with RecurrentExperimentPatch(model, bank, h) as patch:
        for horizon in HORIZON_CANDIDATES:
            if selected is not None:
                results.append({"horizon": horizon, "status": "SKIPPED_SMALLER_AFTER_PASS"})
                continue
            teacher_cache = None
            student_cache = None
            bank.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.time()
            try:
                if horizon > len(ids):
                    raise RuntimeError(f"sequence only has {len(ids)} tokens")
                if horizon > 1:
                    teacher_cache = forward_teacher(model, patch, ids[: horizon - 1], None, False)
                patch.clear_capture()
                teacher_cache = forward_teacher(model, patch, [ids[horizon - 1]], teacher_cache, True)
                for index in range(horizon):
                    student_cache = forward_student(model, patch, ids[index], student_cache, index == horizon - 1)
                _state, _functional, _c5, c6 = patch.captured_losses()
                if not bool(torch.isfinite(c6).detach().cpu()):
                    raise FloatingPointError("nonfinite C6 feasibility loss")
                c6.backward()
                grads = [p.grad for p in bank.parameters() if p.grad is not None]
                grad_norm = math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads))
                finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
                if not finite or grad_norm <= 0:
                    raise FloatingPointError(f"invalid gradients finite={finite} norm={grad_norm}")
                selected = horizon
                results.append({
                    "horizon": horizon, "status": "PASS", "gradient_finite": finite,
                    "gradient_nonzero": grad_norm > 0, "gradient_norm": grad_norm,
                    "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    "runtime_seconds": time.time() - started,
                })
            except torch.cuda.OutOfMemoryError:
                results.append({
                    "horizon": horizon, "status": "OOM", "error": traceback.format_exc(),
                    "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    "runtime_seconds": time.time() - started,
                })
            except Exception:
                results.append({
                    "horizon": horizon, "status": "FAIL", "error": traceback.format_exc(),
                    "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    "runtime_seconds": time.time() - started,
                })
            finally:
                bank.zero_grad(set_to_none=True)
                del teacher_cache, student_cache
                gc.collect()
                torch.cuda.empty_cache()
    payload = {
        "selection_rule": "RESOURCE_FEASIBILITY_ONLY",
        "candidate_order": list(HORIZON_CANDIDATES),
        "numerical_state_recurrence": "continuous up to 1024 document tokens",
        "gradient_horizon": selected,
        "detach_interval": selected,
        "results": results,
        "status": "PASS" if selected is not None else "FAIL",
    }
    atomic_json(EXPERIMENT / "analysis/horizon_feasibility.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if selected is None:
        raise RuntimeError("NO_FEASIBLE_GRADIENT_HORIZON")


def boundary_smoke() -> None:
    """Verify backward/detach/reuse across two selected BPTT windows."""
    horizon = int(load_json(EXPERIMENT / "analysis/horizon_feasibility.json")["gradient_horizon"])
    model, tokenizer, bank, device = load_model_bank()
    ids, _ = tokenize(tokenizer, corpus_rows("TRAIN")[0])
    ids = ids[: 2 * horizon]
    positions = [horizon - 1, horizon, 2 * horizon - 1]
    h = hadamard(device)
    records = []
    with RecurrentExperimentPatch(model, bank, h) as patch:
        targets = collect_teacher_targets(model, patch, ids, positions)
        cache = None
        segment_losses = []
        post_before_boundary = None
        for index, token_id in enumerate(ids):
            capture = index in set(positions)
            if capture:
                install_teacher_target(patch, targets[index], device)
            cache = forward_student(model, patch, token_id, cache, capture)
            if capture:
                _state, _functional, _c5, c6 = patch.captured_losses()
                segment_losses.append(c6)
                layer = GDN_LAYERS[0]
                current = {
                    "token_index": index,
                    "student_prev_hash": tensor_sha256(patch.student_prev[layer]),
                    "student_post_hash": tensor_sha256(patch.student_post[layer]),
                    "loss": float(c6.detach().cpu()),
                }
                records.append(current)
                if index == horizon - 1:
                    post_before_boundary = current["student_post_hash"]
            if (index + 1) % horizon == 0:
                loss = sum(segment_losses)
                loss.backward()
                segment_losses.clear()
                detach_cache(cache)
        grads = [p.grad for p in bank.parameters() if p.grad is not None]
        finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        norm = math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads))
        boundary_match = records[1]["student_prev_hash"] == post_before_boundary
        result = {
            "BPTT_BOUNDARY_GATE": "PASS" if finite and norm > 0 and boundary_match else "FAIL",
            "gradient_horizon": horizon,
            "numerical_state_continues_across_detach": boundary_match,
            "gradient_finite": finite,
            "gradient_nonzero": norm > 0,
            "gradient_norm": norm,
            "records": records,
        }
        atomic_json(EXPERIMENT / "analysis/bptt_boundary_gate.json", result)
        print(json.dumps(result, indent=2), flush=True)
        if result["BPTT_BOUNDARY_GATE"] != "PASS":
            raise RuntimeError("BPTT_BOUNDARY_GATE_FAIL")


def evaluate_validation(model, tokenizer, bank, patch, rows: list[dict], horizon: int, objective: str) -> dict:
    totals = {"primary": 0.0, "state": 0.0, "functional": 0.0, "combined": 0.0, "samples": 0}
    for row in rows:
        ids, positions = tokenize(tokenizer, row)
        targets = collect_teacher_targets(model, patch, ids, positions)
        student_cache = None
        current = 0
        position_set = set(positions)
        for index, token_id in enumerate(ids):
            capture = index in position_set
            if capture:
                install_teacher_target(patch, targets[index], model.get_input_embeddings().weight.device)
            with torch.no_grad():
                student_cache = forward_student(model, patch, token_id, student_cache, capture)
            if capture:
                state, functional, c5, c6 = patch.captured_losses()
                primary = state if objective == "DENSE_STATE" else functional
                combined = c5 if objective == "DENSE_STATE" else c6
                totals["primary"] += float(primary.cpu())
                totals["state"] += float(state.cpu())
                totals["functional"] += float(functional.cpu())
                totals["combined"] += float(combined.cpu())
                totals["samples"] += 1
                del state, functional, c5, c6, primary, combined
            current += 1
            if current % horizon == 0:
                detach_cache(student_cache)
        del targets, student_cache
        patch.clear_capture()
        gc.collect()
        torch.cuda.empty_cache()
    samples = totals.pop("samples")
    return {key: value / samples for key, value in totals.items()} | {"samples": samples, "documents": len(rows)}


def train(condition: str) -> None:
    condition = condition.upper()
    if condition not in ("C5", "C6"):
        raise ValueError("condition must be C5 or C6")
    objective = "DENSE_STATE" if condition == "C5" else "DENSE_FUNCTIONAL"
    lr = 0.003 if condition == "C5" else 0.001
    horizon_gate = load_json(EXPERIMENT / "analysis/horizon_feasibility.json")
    horizon = int(horizon_gate["gradient_horizon"])
    smoke_gate = load_json(EXPERIMENT / "analysis/pretraining_smoke_summary.json")
    required = ("REAL_RECURRENT_WRITEBACK_GATE", "RECURRENT_STATE_PROVENANCE_GATE", "C5_GRADIENT_GATE", "C6_GRADIENT_GATE")
    if any(smoke_gate[key] != "PASS" for key in required) or smoke_gate["TRAIN_VS_FORMAL_QDQ_MATCH"] not in ("PASS", "EXACT_MATCH", "NUMERICALLY_EQUIVALENT"):
        raise RuntimeError("PRETRAINING_GATE_NOT_PASS")
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    model, tokenizer, bank, device = load_model_bank()
    optimizer = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=0.0)
    train_rows = corpus_rows("TRAIN")
    validation_rows = corpus_rows("VALIDATION")
    schedule = [train_rows[random.randrange(len(train_rows))] for _ in range(OPTIMIZER_UPDATES)]
    outdir = EXPERIMENT / "checkpoints" / condition.lower()
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / f"best_{condition.lower()}_seed0.pt"
    history = []
    best = float("inf")
    best_exposure = 0
    next_validation = VALIDATION_INTERVAL_EXPOSURES
    started = time.time()
    h = hadamard(device)
    with RecurrentExperimentPatch(model, bank, h) as patch:
        for update, row in enumerate(schedule, 1):
            ids, positions = tokenize(tokenizer, row)
            position_set = set(positions)
            targets = collect_teacher_targets(model, patch, ids, positions)
            student_cache = None
            optimizer.zero_grad(set_to_none=True)
            doc_primary = doc_state = doc_functional = doc_combined = 0.0
            segment_losses = []
            captured = 0
            for index, token_id in enumerate(ids):
                capture = index in position_set
                if capture:
                    install_teacher_target(patch, targets[index], device)
                student_cache = forward_student(model, patch, token_id, student_cache, capture)
                if capture:
                    state, functional, c5, c6 = patch.captured_losses()
                    primary = state if objective == "DENSE_STATE" else functional
                    combined = c5 if objective == "DENSE_STATE" else c6
                    segment_losses.append(combined / TARGETS_PER_DOCUMENT)
                    doc_primary += float(primary.detach().cpu()) / TARGETS_PER_DOCUMENT
                    doc_state += float(state.detach().cpu()) / TARGETS_PER_DOCUMENT
                    doc_functional += float(functional.detach().cpu()) / TARGETS_PER_DOCUMENT
                    doc_combined += float(combined.detach().cpu()) / TARGETS_PER_DOCUMENT
                    captured += 1
                    del state, functional, c5, c6, primary, combined
                boundary = (index + 1) % horizon == 0 or index + 1 == len(ids)
                if boundary:
                    if segment_losses:
                        segment_loss = sum(segment_losses)
                        if not bool(torch.isfinite(segment_loss).detach().cpu()):
                            raise FloatingPointError(f"nonfinite loss update={update} token={index}")
                        segment_loss.backward()
                        segment_losses.clear()
                        patch.clear_capture()
                        del segment_loss
                    detach_cache(student_cache)
            if captured != TARGETS_PER_DOCUMENT:
                raise RuntimeError(f"captured {captured}, expected {TARGETS_PER_DOCUMENT}")
            grads = [p.grad for p in bank.parameters()]
            if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError(f"invalid gradient at update {update}")
            grad_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            optimizer.step()
            # The completed rollout and its loss graph must be released before
            # loading the fixed validation panel; keeping either cache alive
            # leaves too little headroom on a 24GB card for the next rollout.
            del targets, student_cache
            patch.clear_capture()
            gc.collect()
            torch.cuda.empty_cache()
            exposure = update * TARGETS_PER_DOCUMENT
            row_log = {
                "optimizer_update": update, "target_exposures": exposure, "document_id": row["document_id"],
                "train_combined": doc_combined, "train_primary": doc_primary, "train_state": doc_state,
                "train_functional": doc_functional, "gradient_norm": float(grad_norm.detach().cpu()),
            }
            if exposure >= next_validation or exposure == TARGET_EXPOSURES:
                with torch.no_grad():
                    validation = evaluate_validation(model, tokenizer, bank, patch, validation_rows, horizon, objective)
                ortho = bank.runtime_gate(threshold=ORTHOGONALITY_THRESHOLD)
                if ortho["status"] != "PASS":
                    raise RuntimeError(f"ORTHOGONALITY_GATE_FAIL update={update}")
                row_log["validation"] = validation
                row_log["orthogonality"] = {key: ortho[key] for key in ("status", "threshold", "max_abs_rt_r_minus_i", "all_finite", "all_proper")}
                if validation["primary"] < best:
                    best = validation["primary"]
                    best_exposure = exposure
                    torch.save({
                        "task": TASK, "model": "Qwen3.5-9B/GDN", "condition": condition,
                        "objective": objective, "training_mode": "REAL_RECURRENT_INT8_STATE_WRITEBACK",
                        "seed": TRAIN_SEED, "lr": lr, "target_exposures": exposure,
                        "optimizer_updates": update, "gradient_horizon": horizon,
                        "numerical_rollout": "continuous document up to 1024 tokens",
                        "validation": validation, "bank": bank.state_dict(), "layer_ids": GDN_LAYERS,
                    }, checkpoint)
                while next_validation <= exposure:
                    next_validation += VALIDATION_INTERVAL_EXPOSURES
            history.append(row_log)
            print(json.dumps(row_log, sort_keys=True), flush=True)
            patch.clear_capture()
            gc.collect()
            torch.cuda.empty_cache()
    summary = {
        "task": TASK, "condition": condition, "objective": objective, "training_mode": "REAL_RECURRENT_INT8_STATE_WRITEBACK",
        "seed": TRAIN_SEED, "lr": lr, "optimizer": "Adam", "weight_decay": 0.0,
        "target_exposures_requested": TARGET_EXPOSURES, "optimizer_updates_requested": OPTIMIZER_UPDATES,
        "best_target_exposure": best_exposure, "best_validation_primary": best,
        "gradient_horizon": horizon, "numerical_state_recurrence": "continuous within each document rollout",
        "history": history, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "runtime_seconds": time.time() - started, "model_weights_changed": False, "AIME26_used": False,
        f"{condition}_TRAIN_COMPLETE": "PASS", "status": "PASS",
    }
    atomic_json(outdir / "training_summary.json", summary)
    atomic_json(EXPERIMENT / f"analysis/training_curves_{condition.lower()}.json", {"condition": condition, "history": history})
    print(json.dumps({key: summary[key] for key in ("condition", "best_target_exposure", "best_validation_primary", "checkpoint_sha256", "runtime_seconds", "status")}, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "smoke", "horizon", "boundary", "train"), required=True)
    parser.add_argument("--condition", choices=("C5", "C6"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "smoke":
        smoke()
    elif args.phase == "horizon":
        horizon_feasibility()
    elif args.phase == "boundary":
        boundary_smoke()
    elif args.phase == "train":
        if not args.condition:
            raise ValueError("--condition is required for training")
        train(args.condition)


if __name__ == "__main__":
    main()
