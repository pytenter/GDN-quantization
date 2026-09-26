#!/usr/bin/env python3
"""Prospective dual-RTX3090 recurrent C5/C6 experiment, V2.

The first phases preregister the topology and audit complete-block partitions.
No frozen V1, memory-audit, or C0-C4 artifact is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


TASK = "QWEN_RECURRENT_DENSE_C5_C6_DUAL_GPU_HORIZON_V2"
EXPERIMENT = Path(__file__).resolve().parent
REPO = EXPERIMENT.parents[1]
V1 = REPO / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1"
MEMORY_AUDIT = REPO / "experiments/QWEN_C5_RECURRENT_MEMORY_LIFETIME_AUDIT_V1"
OLD_E2E = REPO / "experiments/QWEN_AIME26_LAST20_SEED1_DENSE_E2E_V1"
V1_SCRIPT = V1 / "run_recurrent_dense.py"
HORIZON_CANDIDATES = (128, 64, 32)
STABILITY_UPDATES = 30
MIN_FREE_FRACTION = 0.05
GPU_IDS = (0, 1)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1_IMPL = import_file(V1_SCRIPT, "qwen_recurrent_v1_read_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_inventory():
    devices = []
    for index in GPU_IDS:
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        devices.append({
            "index": index,
            "name": props.name,
            "capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor],
            "total_bytes": total,
            "free_bytes_at_prepare": free,
            "pci_bus_id": None,
        })
    pair_pass = (
        len(devices) == 2
        and devices[0]["name"] == devices[1]["name"] == "NVIDIA GeForce RTX 3090"
        and devices[0]["capability"] == devices[1]["capability"]
        and all(row["free_bytes_at_prepare"] / row["total_bytes"] >= 0.90 for row in devices)
    )
    return {"devices": devices, "GPU_PAIR_GATE": "PASS" if pair_pass else "FAIL"}


def prepare():
    if EXPERIMENT.exists() and (EXPERIMENT / "preregistration.json").exists():
        raise RuntimeError("V2 already preregistered; refusing to overwrite")
    for relative in (
        "configs", "analysis", "checkpoints/C5", "checkpoints/C6",
        "outputs/C5", "outputs/C6", "logs", "reports", "hashes",
    ):
        (EXPERIMENT / relative).mkdir(parents=True, exist_ok=True)
    audit = read_json(MEMORY_AUDIT / "analysis/final_classification.json")
    if audit["FINAL_VERDICT"] != "NO_LEAK_H32_TOO_LARGE":
        raise RuntimeError("single-GPU memory audit verdict does not authorize this amendment")
    gpu = gpu_inventory()
    if gpu["GPU_PAIR_GATE"] != "PASS":
        raise RuntimeError(f"GPU_PAIR_GATE_FAIL {gpu}")
    old = read_json(V1 / "preregistration.json")
    amendment = {
        "task": TASK,
        "version": 2,
        "created_before_formal_training_results": True,
        "reason": [
            "single-GPU H32 sustained feasibility failed",
            "memory audit proved NO_LEAK_H32_TOO_LARGE",
            "two same-architecture RTX3090 GPUs are available",
            "dual-GPU topology may support a larger gradient horizon",
        ],
        "training_topology_change": {
            "from": {"gpu_count": 1},
            "to": {"gpu_count": 2, "topology": "contiguous_layer_model_parallel", "one_process": True},
        },
        "horizon_policy": {
            "parity_reference_horizon": 32,
            "formal_candidates": list(HORIZON_CANDIDATES),
            "selection_rule": "largest_joint_sustained_feasible_horizon",
            "selection_depends_on_task_accuracy": False,
            "conditions_share_final_horizon": True,
            "short_screen_updates_per_condition_split": 2,
            "sustained_updates_per_condition": STABILITY_UPDATES,
            "minimum_peak_free_fraction_per_gpu": MIN_FREE_FRACTION,
            "validation_boundary_required": True,
        },
        "unchanged": [
            "recurrent numerical writeback semantics", "quantizer", "QDQ timing",
            "training data", "validation data", "optimizer", "learning rate",
            "training seed", "loss definitions", "rotation parameterization",
            "checkpoint-selection rule", "formal AIME protocol",
        ],
        "forbidden_parallelism": [
            "DDP", "DataParallel", "FSDP", "ZeRO", "tensor parallel",
            "pipeline microbatch overlap", "RPC pipeline", "CPU offload",
            "activation checkpointing",
        ],
        "old_memory_audit_sha256": sha256(MEMORY_AUDIT / "analysis/final_classification.json"),
        "old_v1_preregistration_sha256": sha256(V1 / "preregistration.json"),
    }
    atomic_json(EXPERIMENT / "protocol_amendment_dual_gpu_horizon_v2.json", amendment)
    prereg = {
        "task": TASK,
        "created_before_formal_training_results": True,
        "protocol_amendment_sha256": sha256(EXPERIMENT / "protocol_amendment_dual_gpu_horizon_v2.json"),
        "source_v1_training_script": str(V1_SCRIPT),
        "source_v1_training_script_sha256": sha256(V1_SCRIPT),
        "source_memory_audit": str(MEMORY_AUDIT),
        "source_memory_audit_verdict": audit["FINAL_VERDICT"],
        "source_v1_preregistration": old,
        "same_dataset_seed_objective_quantizer_and_rotation": True,
        "formal_baselines": {"C0": [17, 20], "C1": [4, 20], "C2": [13, 20], "C3": [12, 20], "C4": [8, 20]},
        "formal_eval": {"question_ids": [f"aime26_{i:02d}" for i in range(11, 31)], "seed": 1, "max_new_tokens": 81920, "scorer": "Frozen V4"},
        "GPU_PAIR_GATE": gpu["GPU_PAIR_GATE"],
        "gpu_inventory": gpu,
        "formal_training": "NOT_STARTED",
        "formal_AIME": "NOT_STARTED",
    }
    atomic_json(EXPERIMENT / "preregistration.json", prereg)
    for condition in ("c5", "c6"):
        source = V1 / f"configs/recurrent_training_config_{condition}.json"
        shutil.copy2(source, EXPERIMENT / f"configs/original_{condition}_config.json")
    shutil.copy2(V1 / "configs/frozen_eval_config.json", EXPERIMENT / "configs/frozen_eval_config.json")
    atomic_json(EXPERIMENT / "configs/horizon_candidates.json", {
        "ordered_candidates": list(HORIZON_CANDIDATES),
        "selection": "largest_joint_sustained_feasible_horizon",
        "minimum_free_fraction": MIN_FREE_FRACTION,
        "objective_pair": ["C5", "C6"],
    })
    print(json.dumps({"status": "PREPARED", "GPU_PAIR_GATE": gpu["GPU_PAIR_GATE"], "experiment": str(EXPERIMENT)}, indent=2), flush=True)


def model_inventory():
    if not (EXPERIMENT / "preregistration.json").exists():
        raise RuntimeError("run prepare before inventory")
    model, tokenizer = V1_IMPL.BASE.load_model_and_tokenizer()
    layers = model.model.layers
    if len(layers) != 32:
        raise RuntimeError(f"expected 32 complete blocks, got {len(layers)}")
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    tie = embedding is head or embedding.data_ptr() == head.data_ptr()
    config_tie = bool(getattr(model.config, "tie_word_embeddings", False))
    shared_parameter_names: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        shared_parameter_names.setdefault(parameter.data_ptr(), []).append(name)
    actual_shared = [names for names in shared_parameter_names.values() if len(names) > 1]
    tying_pass = not tie and not config_tie and not actual_shared
    blocks = []
    for layer_id, layer in enumerate(layers):
        gdn = hasattr(layer, "linear_attn") and layer_id in V1_IMPL.GDN_LAYERS
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in layer.parameters())
        buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in layer.buffers())
        blocks.append({
            "module_name": f"model.layers.{layer_id}",
            "block_id": layer_id,
            "gdn_layer_id": layer_id if gdn else None,
            "block_type": "GDN" if gdn else "FULL_ATTENTION",
            "parameter_bytes": parameter_bytes,
            "buffer_bytes": buffer_bytes,
            "rotation_parameter_bytes": 8128 * 4 if gdn else 0,
            "recurrent_state_shape": [1, 32, 128, 128] if gdn else None,
            "device_compatible_split_boundary_after_block": True,
        })
    all_gdn = [row["block_id"] for row in blocks if row["gdn_layer_id"] is not None]
    if tuple(all_gdn) != V1_IMPL.GDN_LAYERS:
        raise RuntimeError("GDN layer mapping mismatch")
    candidate_gdn_counts = (10, 11, 12, 13, 14)
    candidates = []
    for target_count in candidate_gdn_counts:
        same_gdn_boundaries = []
        for split in range(1, len(blocks)):
            if sum(row["gdn_layer_id"] is not None for row in blocks[:split]) != target_count:
                continue
            gpu0_bytes = embedding.numel() * embedding.element_size() + sum(
                row["parameter_bytes"] + row["buffer_bytes"] for row in blocks[:split]
            )
            gpu1_bytes = head.numel() * head.element_size() + sum(
                row["parameter_bytes"] + row["buffer_bytes"] for row in blocks[split:]
            )
            same_gdn_boundaries.append((abs(gpu0_bytes - gpu1_bytes), split, gpu0_bytes, gpu1_bytes))
        _imbalance, split, gpu0_bytes, gpu1_bytes = min(same_gdn_boundaries)
        candidates.append({
            "split_block": split,
            "gpu0_block_ids": list(range(split)),
            "gpu1_block_ids": list(range(split, len(blocks))),
            "gpu0_gdn_count": target_count,
            "gpu1_gdn_count": len(all_gdn) - target_count,
            "gpu0_static_model_bytes": gpu0_bytes,
            "gpu1_static_model_bytes": gpu1_bytes,
            "within_gdn_count_static_balance_rule": True,
        })
    inventory = {
        "task": TASK,
        "model": type(model).__name__,
        "module_path": "model.layers",
        "block_count": len(blocks),
        "gdn_count": len(all_gdn),
        "embedding_module": "model.embed_tokens",
        "lm_head_module": "lm_head",
        "final_norm_module": "model.norm",
        "embedding_bytes": embedding.numel() * embedding.element_size(),
        "lm_head_bytes": head.numel() * head.element_size(),
        "blocks": blocks,
        "candidate_splits": candidates,
        "WEIGHT_TYING_GATE": "PASS" if tying_pass else "FAIL",
        "weight_tying_evidence": {
            "config_tie_word_embeddings": config_tie,
            "embedding_head_same_object": embedding is head,
            "embedding_head_same_storage": embedding.data_ptr() == head.data_ptr(),
            "shared_parameter_names": actual_shared,
        },
        "MODEL_PARTITION_GATE": "PASS" if len(blocks) == 32 and len(all_gdn) == 24 and tying_pass else "FAIL",
    }
    atomic_json(EXPERIMENT / "analysis/model_device_partition_inventory.json", inventory)
    atomic_json(EXPERIMENT / "configs/model_partition_inventory.json", inventory)
    atomic_json(EXPERIMENT / "configs/dual_gpu_topology.json", {
        "architecture": "one-process complete-block contiguous model parallel",
        "gpu_ids": list(GPU_IDS),
        "candidate_splits": candidates,
        "rotations_colocated_with_gdn_layers": True,
        "optimizer_state_colocated_with_rotation_parameters": True,
        "selected_split": None,
        "selection_rule": "minimax worst peak fraction across C5/C6 and both GPUs",
    })
    report = (
        f"# V2 model topology inventory\n\n"
        f"The actual model is `{type(model).__name__}` with {len(blocks)} complete decoder blocks and {len(all_gdn)} GDN blocks. "
        f"Embedding and LM head have distinct storage (`WEIGHT_TYING_GATE={inventory['WEIGHT_TYING_GATE']}`). "
        "All candidates split only between complete decoder blocks. The rotations will be colocated with their GDN block.\n\n"
        "| Split after block | GPU0 GDN | GPU1 GDN | GPU0 static model GiB | GPU1 static model GiB |\n"
        "|---:|---:|---:|---:|---:|\n"
        + "".join(
            f"| {row['split_block'] - 1} | {row['gpu0_gdn_count']} | {row['gpu1_gdn_count']} | {row['gpu0_static_model_bytes'] / 2**30:.3f} | {row['gpu1_static_model_bytes'] / 2**30:.3f} |\n"
            for row in candidates
        )
    )
    (EXPERIMENT / "reports/topology_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"MODEL_PARTITION_GATE": inventory["MODEL_PARTITION_GATE"], "WEIGHT_TYING_GATE": inventory["WEIGHT_TYING_GATE"], "candidates": candidates}, indent=2), flush=True)


def explicit_device_map(split_block: int) -> dict[str, int]:
    if not 1 <= split_block <= 31:
        raise ValueError("split must be between complete decoder blocks")
    mapping = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    for block_id in range(32):
        mapping[f"model.layers.{block_id}"] = 0 if block_id < split_block else 1
    return mapping


def load_dual_model_bank(split_block: int):
    for source_path in (V1_IMPL.BASE.STATE_QUANT_SRC, V1_IMPL.BASE.TRANSFORMERS_SRC):
        path = str(source_path)
        if path not in sys.path:
            sys.path.insert(0, path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(V1_IMPL.BASE.MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=explicit_device_map(split_block),
        trust_remote_code=True,
        local_files_only=True,
    )
    V1_IMPL.freeze_model(model)
    bank = V1_IMPL.CAYLEY.PerLayerCayleyRotations(V1_IMPL.GDN_LAYERS)
    for layer_id in V1_IMPL.GDN_LAYERS:
        device = next(model.model.layers[layer_id].parameters()).device
        bank.layer(layer_id).to(device)
    layer_devices = {}
    for block_id, block in enumerate(model.model.layers):
        devices = {str(parameter.device) for parameter in block.parameters()}
        if len(devices) != 1:
            raise RuntimeError(f"block {block_id} spans devices: {devices}")
        expected = f"cuda:{0 if block_id < split_block else 1}"
        if devices != {expected}:
            raise RuntimeError(f"block {block_id} on {devices}, expected {expected}")
        layer_devices[block_id] = expected
    for layer_id in V1_IMPL.GDN_LAYERS:
        rotation_device = str(bank.layer(layer_id).theta.device)
        if rotation_device != layer_devices[layer_id]:
            raise RuntimeError(f"rotation {layer_id} on {rotation_device}, layer on {layer_devices[layer_id]}")
    if str(model.get_input_embeddings().weight.device) != "cuda:0":
        raise RuntimeError("embedding must be on GPU0")
    if str(model.get_output_embeddings().weight.device) != "cuda:1":
        raise RuntimeError("LM head must be on GPU1")
    return model, tokenizer, bank, layer_devices


class DualRecurrentPatch(V1_IMPL.RecurrentExperimentPatch):
    def __init__(self, model, bank):
        self.h_by_device = {
            "cuda:0": V1_IMPL.hadamard(torch.device("cuda:0")),
            "cuda:1": V1_IMPL.hadamard(torch.device("cuda:1")),
        }
        super().__init__(model, bank, self.h_by_device["cuda:0"])

    def __enter__(self):
        super().__enter__()
        for block_id in V1_IMPL.GDN_LAYERS:
            layer = self.model.model.layers[block_id].linear_attn
            device = str(next(layer.parameters()).device)
            self.handles.append(layer.register_forward_pre_hook(
                lambda _module, _args, selected=device: setattr(self, "h", self.h_by_device[selected])
            ))
        return self

    def captured_losses(self):
        missing_state = sorted(set(V1_IMPL.GDN_LAYERS) - set(self.state_losses))
        missing_func = sorted(set(V1_IMPL.GDN_LAYERS) - set(self.functional_losses))
        if missing_state or missing_func:
            raise RuntimeError(f"capture incomplete state={missing_state} functional={missing_func}")
        destination = self.h_by_device["cuda:0"].device
        state = sum(value.to(destination) for value in self.state_losses.values()) / len(V1_IMPL.GDN_LAYERS)
        functional = sum(value.to(destination) for value in self.functional_losses.values()) / len(V1_IMPL.GDN_LAYERS)
        return state, functional, state, functional + 0.1 * state


def install_dual_teacher_target(patch: DualRecurrentPatch, target, layer_devices: dict[int, str]):
    patch.clear_capture()
    patch.teacher_prev.update({layer: value.to(layer_devices[layer]) for layer, value in target["prev"].items()})
    patch.teacher_out.update({layer: value.to(layer_devices[layer]) for layer, value in target["out"].items()})


def dual_load_smoke(split_block: int):
    inventory = read_json(EXPERIMENT / "analysis/model_device_partition_inventory.json")
    allowed = {row["split_block"] for row in inventory["candidate_splits"]}
    if split_block not in allowed:
        raise RuntimeError(f"split {split_block} not preregistered among {sorted(allowed)}")
    torch.manual_seed(V1_IMPL.TRAIN_SEED)
    model, tokenizer, bank, layer_devices = load_dual_model_bank(split_block)
    ids, _ = V1_IMPL.tokenize(tokenizer, V1_IMPL.corpus_rows("TRAIN")[0])
    cache = None
    with DualRecurrentPatch(model, bank) as patch:
        for token_id in ids[:3]:
            cache = V1_IMPL.forward_student(model, patch, token_id, cache, False)
    state_devices = {}
    for layer_id in V1_IMPL.GDN_LAYERS:
        state = cache.layers[layer_id].recurrent_states[0]
        state_devices[str(layer_id)] = str(state.device)
        if str(state.device) != layer_devices[layer_id]:
            raise RuntimeError(f"state {layer_id} on {state.device}, expected {layer_devices[layer_id]}")
    result = {
        "split_block": split_block,
        "layer_devices": layer_devices,
        "recurrent_state_devices": state_devices,
        "rotation_devices": {str(layer_id): str(bank.layer(layer_id).theta.device) for layer_id in V1_IMPL.GDN_LAYERS},
        "output_cache_type": type(cache).__name__,
        "DUAL_LOAD_SMOKE_GATE": "PASS",
    }
    atomic_json(EXPERIMENT / f"analysis/dual_load_smoke_split{split_block}.json", result)
    print(json.dumps({"split_block": split_block, "DUAL_LOAD_SMOKE_GATE": "PASS"}, indent=2), flush=True)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def one_step_trace(mode: str, condition: str, split_block: int | None, repeat: int):
    """One fixed 192-token, eight-target H32 non-AIME optimizer update."""
    mode = mode.lower()
    condition = condition.upper()
    if mode not in ("single", "dual") or condition not in ("C5", "C6"):
        raise ValueError("one-step trace requires single/dual and C5/C6")
    if mode == "dual":
        if split_block is None:
            raise ValueError("dual mode requires --split-block")
        allowed = {row["split_block"] for row in read_json(EXPERIMENT / "analysis/model_device_partition_inventory.json")["candidate_splits"]}
        if split_block not in allowed:
            raise RuntimeError("unregistered split")
    torch.manual_seed(V1_IMPL.TRAIN_SEED)
    if mode == "single":
        model, tokenizer, bank, device = V1_IMPL.load_model_bank()
        layer_devices = {layer_id: str(device) for layer_id in V1_IMPL.GDN_LAYERS}
        patch_factory = lambda: V1_IMPL.RecurrentExperimentPatch(model, bank, V1_IMPL.hadamard(device))
    else:
        model, tokenizer, bank, layer_devices = load_dual_model_bank(split_block)
        device = model.get_input_embeddings().weight.device
        patch_factory = lambda: DualRecurrentPatch(model, bank)
    lr = 0.003 if condition == "C5" else 0.001
    optimizer = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=0.0)
    row = V1_IMPL.corpus_rows("TRAIN")[0]
    ids = tokenizer(row["raw_text"], add_special_tokens=False).input_ids[:192]
    positions = V1_IMPL.sampled_positions(row, len(ids))
    selected_tokens = sorted(set((31, 32, *positions)))
    trace_set = set(selected_tokens)
    position_set = set(positions)
    records: dict[str, dict[str, Any]] = {}
    loss_records = []
    boundary_records = []
    cache = None
    segment_losses = []
    current_token = -1
    original_qdq = V1_IMPL.CAYLEY.qwen_c128_ste
    with patch_factory() as patch:
        targets = V1_IMPL.collect_teacher_targets(model, patch, ids, positions)

        def traced_qdq(value):
            quant = original_qdq(value)
            if current_token in trace_set:
                layer_id = int(patch.current_layer)
                key = f"{current_token}:{layer_id}"
                records.setdefault(key, {}).update({
                    "token_index": current_token,
                    "layer_id": layer_id,
                    "device": str(value.device),
                    "pre_qdq_state_sha256": tensor_hash(value),
                    "scale_sha256": tensor_hash(quant.scale),
                    "qcodes_sha256": tensor_hash(quant.codes),
                    "post_qdq_state_sha256": tensor_hash(quant.dequant),
                    "pre_qdq_shape": list(value.shape),
                    "scale_shape": list(quant.scale.shape),
                    "qcodes_shape": list(quant.codes.shape),
                    "post_qdq_shape": list(quant.dequant.shape),
                })
            return quant

        V1_IMPL.CAYLEY.qwen_c128_ste = traced_qdq
        try:
            optimizer.zero_grad(set_to_none=True)
            for index, token_id in enumerate(ids):
                current_token = index
                capture = index in position_set
                if capture:
                    if mode == "single":
                        V1_IMPL.install_teacher_target(patch, targets[index], device)
                    else:
                        install_dual_teacher_target(patch, targets[index], layer_devices)
                if cache is not None and index in trace_set:
                    for layer_id in V1_IMPL.GDN_LAYERS:
                        key = f"{index}:{layer_id}"
                        state = V1_IMPL.BASE.get_state(cache, layer_id)
                        records.setdefault(key, {})["consumed_prev_state_sha256"] = tensor_hash(state)
                cache = V1_IMPL.forward_student(model, patch, token_id, cache, capture)
                if capture:
                    state, functional, c5, c6 = patch.captured_losses()
                    loss = c5 if condition == "C5" else c6
                    segment_losses.append(loss / V1_IMPL.TARGETS_PER_DOCUMENT)
                    loss_records.append({
                        "token_index": index,
                        "state": float(state.detach().cpu()),
                        "functional": float(functional.detach().cpu()),
                        "c5": float(c5.detach().cpu()),
                        "c6": float(c6.detach().cpu()),
                    })
                    del state, functional, c5, c6, loss
                if (index + 1) % 32 == 0 or index + 1 == len(ids):
                    if segment_losses:
                        segment_loss = sum(segment_losses)
                        segment_loss.backward()
                        segment_losses.clear()
                        patch.clear_capture()
                        del segment_loss
                    if index == 31:
                        before = {str(layer_id): tensor_hash(V1_IMPL.BASE.get_state(cache, layer_id)) for layer_id in V1_IMPL.GDN_LAYERS}
                    V1_IMPL.detach_cache(cache)
                    if index == 31:
                        after = {str(layer_id): tensor_hash(V1_IMPL.BASE.get_state(cache, layer_id)) for layer_id in V1_IMPL.GDN_LAYERS}
                        boundary_records.append({"token_index": index, "before_detach": before, "after_detach": after, "value_preserved": before == after})
            if len(loss_records) != V1_IMPL.TARGETS_PER_DOCUMENT:
                raise RuntimeError(f"expected eight targets, got {len(loss_records)}")
            raw_gradient = {str(layer_id): tensor_hash(bank.layer(layer_id).theta.grad) for layer_id in V1_IMPL.GDN_LAYERS}
            raw_gradient_finite = all(bool(torch.isfinite(bank.layer(layer_id).theta.grad).all()) for layer_id in V1_IMPL.GDN_LAYERS)
            if not raw_gradient_finite:
                raise FloatingPointError("nonfinite rotation gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            clipped_gradient = {str(layer_id): tensor_hash(bank.layer(layer_id).theta.grad) for layer_id in V1_IMPL.GDN_LAYERS}
            optimizer.step()
            post_theta = {str(layer_id): tensor_hash(bank.layer(layer_id).theta) for layer_id in V1_IMPL.GDN_LAYERS}
            post_rotation = {str(layer_id): tensor_hash(bank.layer(layer_id).matrix()) for layer_id in V1_IMPL.GDN_LAYERS}
            optimizer_state_devices = {
                str(layer_id): sorted({str(value.device) for value in optimizer.state[bank.layer(layer_id).theta].values() if torch.is_tensor(value)})
                for layer_id in V1_IMPL.GDN_LAYERS
            }
        finally:
            V1_IMPL.CAYLEY.qwen_c128_ste = original_qdq
    for index in selected_tokens:
        for layer_id in V1_IMPL.GDN_LAYERS:
            key = f"{index}:{layer_id}"
            if key not in records or "pre_qdq_state_sha256" not in records[key]:
                raise RuntimeError(f"missing trace {key}")
    writeback = all(
        records[f"31:{layer_id}"]["post_qdq_state_sha256"] == records[f"32:{layer_id}"]["consumed_prev_state_sha256"]
        for layer_id in V1_IMPL.GDN_LAYERS
    )
    boundary = all(record["value_preserved"] for record in boundary_records)
    optimizer_colocated = all(
        all(device_name == layer_devices[layer_id] for device_name in optimizer_state_devices[str(layer_id)] if device_name.startswith("cuda"))
        for layer_id in V1_IMPL.GDN_LAYERS
    )
    result = {
        "task": TASK,
        "mode": mode,
        "condition": condition,
        "split_block": split_block,
        "repeat": repeat,
        "source_document_id": row["document_id"],
        "input_token_ids": ids,
        "target_positions": positions,
        "trace_token_indices": selected_tokens,
        "records": records,
        "loss_components": loss_records,
        "total_primary_loss": sum(record["state" if condition == "C5" else "functional"] for record in loss_records) / 8,
        "total_training_loss": sum(record["c5" if condition == "C5" else "c6"] for record in loss_records) / 8,
        "gradient_norm_after_clip": float(grad_norm.detach().cpu()),
        "raw_rotation_gradient_sha256": raw_gradient,
        "clipped_rotation_gradient_sha256": clipped_gradient,
        "post_optimizer_theta_sha256": post_theta,
        "post_optimizer_rotation_sha256": post_rotation,
        "optimizer_state_devices": optimizer_state_devices,
        "OPTIMIZER_COLOCATION_GATE": "PASS" if optimizer_colocated else "FAIL",
        "REAL_RECURRENT_WRITEBACK_GATE": "PASS" if writeback else "FAIL",
        "BPTT_BOUNDARY_GATE": "PASS" if boundary and writeback else "FAIL",
        "peak_allocated_bytes": {str(index): torch.cuda.max_memory_allocated(index) for index in (GPU_IDS if mode == "dual" else (0,))},
    }
    file_name = f"one_step_{condition.lower()}_{mode}" + (f"_split{split_block}" if mode == "dual" else f"_repeat{repeat}") + ".json"
    atomic_json(EXPERIMENT / "analysis" / file_name, result)
    print(json.dumps({
        "file": file_name,
        "mode": mode,
        "condition": condition,
        "loss": result["total_training_loss"],
        "gradient_norm": result["gradient_norm_after_clip"],
        "REAL_RECURRENT_WRITEBACK_GATE": result["REAL_RECURRENT_WRITEBACK_GATE"],
        "BPTT_BOUNDARY_GATE": result["BPTT_BOUNDARY_GATE"],
        "OPTIMIZER_COLOCATION_GATE": result["OPTIMIZER_COLOCATION_GATE"],
    }, indent=2), flush=True)
    if any(result[key] != "PASS" for key in ("REAL_RECURRENT_WRITEBACK_GATE", "BPTT_BOUNDARY_GATE", "OPTIMIZER_COLOCATION_GATE")):
        raise RuntimeError("one-step semantic gate failed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "inventory", "dual-load-smoke", "one-step"), required=True)
    parser.add_argument("--split-block", type=int)
    parser.add_argument("--mode", choices=("single", "dual"))
    parser.add_argument("--condition", choices=("C5", "C6"))
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "inventory":
        model_inventory()
    elif args.phase == "dual-load-smoke":
        if args.split_block is None:
            raise ValueError("--split-block required")
        dual_load_smoke(args.split_block)
    elif args.phase == "one-step":
        if args.mode is None or args.condition is None:
            raise ValueError("--mode and --condition required")
        one_step_trace(args.mode, args.condition, args.split_block, args.repeat)


if __name__ == "__main__":
    main()
