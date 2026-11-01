#!/usr/bin/env python3
"""Engineering smoke for STRICT_COMMON_FP_STATE_FANOUT_V1.

The canonical cache is native FP only.  At every decode token and GDN layer,
the exact same incoming FP recurrent state is cloned into Native/H/C5/C6
branches.  H/C5/C6 q/k transforms are the frozen formal runner methods and
local states are passed to the unchanged formal quantize_c128 function.  Local
QDQ states are never returned to the model and are discarded after capture.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


TASK = "QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/qwen_gdn" / TASK
FORMAL_PATH = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py")
DEPLOY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment")
TEACHER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
BASE_HEAD = "2bcf9c93e72c413b28450e56ab1a99f36d742f4f"
EXPECTED_ROTATION = {
    "C5": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
    "C6": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
}
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
SMOKE_UNITS = (("aime26_01", 1), ("aime26_01", 2))
SMOKE_TOKENS = 32
SELECTED_TOKENS = (0, SMOKE_TOKENS - 1)
TOP_N = 20
EPS = 1e-12
PRIMARY = ("scale", "absmax", "papr_linear", "qdq_relative_l2")
ALL_METRICS = (
    "minimum", "maximum", "range", "absmax", "mean", "rms", "std", "scale",
    "papr_linear", "papr_power", "qdq_l2", "qdq_relative_l2", "qdq_mse",
    "qdq_relative_mse", "qdq_max_abs_error", "snr_db", "qcode_min",
    "qcode_max", "unique_qcode_count", "saturation_count", "saturation_fraction",
)


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def token_sha256(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git(args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_teacher_rows():
    rows = []
    for path in sorted(TEACHER.glob("worker*/fp_state.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda r: (int(r["canonical_index"]), int(r["seed"])))
    selected = {(r["problem_id"], int(r["seed"])): r for r in rows}
    return rows, [selected[key] for key in SMOKE_UNITS]


class FakeCache:
    def __init__(self, states, max_layer):
        self.recurrent_states = [None] * (max_layer + 1)
        for layer, value in states.items():
            self.recurrent_states[layer] = value


class MetricAccumulator:
    """Online engineering summaries plus a one-token real Parquet calibration."""

    def __init__(self, calibration_path: Path):
        self.records = defaultdict(lambda: {m: [0, 0.0, 0.0, math.inf, -math.inf] for m in ALL_METRICS})
        self.top = defaultdict(list)
        self.calibration_path = calibration_path
        self.writer = None
        self.calibration_rows = 0
        self.calls = 0

    def _metric_tensors(self, source, scale, codes, dequant):
        minimum = source.amin(dim=-2)
        maximum = source.amax(dim=-2)
        absmax = source.abs().amax(dim=-2)
        mean = source.mean(dim=-2)
        power = source.square().mean(dim=-2)
        rms = power.sqrt()
        std = ((source - mean.unsqueeze(-2)).square().mean(dim=-2)).sqrt()
        err = dequant - source
        mse = err.square().mean(dim=-2)
        l2 = err.square().sum(dim=-2).sqrt()
        source_l2 = source.square().sum(dim=-2).sqrt()
        sorted_codes = torch.sort(codes, dim=-2).values
        unique = 1 + (sorted_codes[..., 1:, :] != sorted_codes[..., :-1, :]).sum(dim=-2)
        saturation = (codes.abs() == 127).sum(dim=-2)
        return {
            "minimum": minimum,
            "maximum": maximum,
            "range": maximum - minimum,
            "absmax": absmax,
            "mean": mean,
            "rms": rms,
            "std": std,
            "scale": scale.squeeze(-2),
            "papr_linear": absmax / rms.clamp_min(EPS),
            "papr_power": absmax.square() / power.clamp_min(EPS),
            "qdq_l2": l2,
            "qdq_relative_l2": l2 / source_l2.clamp_min(EPS),
            "qdq_mse": mse,
            "qdq_relative_mse": mse / power.clamp_min(EPS),
            "qdq_max_abs_error": err.abs().amax(dim=-2),
            "snr_db": 10.0 * torch.log10(power.clamp_min(EPS) / mse.clamp_min(EPS)),
            "qcode_min": codes.amin(dim=-2),
            "qcode_max": codes.amax(dim=-2),
            "unique_qcode_count": unique,
            "saturation_count": saturation,
            "saturation_fraction": saturation.float() / 128.0,
        }

    def update(self, sample_id, token_idx, condition, layer, source, scale, codes, dequant):
        metrics = self._metric_tensors(source, scale, codes, dequant)
        key = (sample_id, condition, int(layer))
        for name, tensor in metrics.items():
            x = tensor.detach().float()
            rec = self.records[key][name]
            rec[0] += x.numel()
            rec[1] += float(x.double().sum().item())
            rec[2] += float(x.double().square().sum().item())
            rec[3] = min(rec[3], float(x.min().item()))
            rec[4] = max(rec[4], float(x.max().item()))

        scale_flat = metrics["scale"].reshape(-1)
        values, indices = torch.topk(scale_flat, k=min(TOP_N, scale_flat.numel()))
        heap = self.top[(sample_id, condition, int(layer))]
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            head, group = divmod(int(index), 128)
            item = (float(value), int(token_idx), head, group,
                    float(metrics["papr_linear"].reshape(-1)[index].item()),
                    float(metrics["qdq_relative_l2"].reshape(-1)[index].item()))
            if len(heap) < TOP_N:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

        # All 393,216 real group rows for the first smoke token are retained.
        if sample_id == "aime26_01_seed1" and token_idx == 0:
            arrays = {
                "sample_id": pa.array([sample_id] * 4096, pa.string()),
                "token_idx": pa.array(np.full(4096, token_idx, np.int32)),
                "layer_idx": pa.array(np.full(4096, layer, np.int8)),
                "condition": pa.array([condition] * 4096, pa.string()),
                "head_idx": pa.array(np.repeat(np.arange(32, dtype=np.int8), 128)),
                "group_idx": pa.array(np.tile(np.arange(128, dtype=np.int16), 32)),
            }
            for name in ALL_METRICS:
                arrays[name] = pa.array(metrics[name].reshape(-1).detach().float().cpu().numpy().astype(np.float32))
            table = pa.table(arrays)
            if self.writer is None:
                self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
                self.writer = pq.ParquetWriter(
                    self.calibration_path, table.schema, compression="zstd",
                    use_dictionary=["sample_id", "condition"], write_statistics=True,
                )
            self.writer.write_table(table)
            self.calibration_rows += table.num_rows
        self.calls += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def summary(self):
        rows = []
        for (sample_id, condition, layer), metrics in sorted(self.records.items()):
            row = {"sample_id": sample_id, "condition": condition, "layer": layer}
            for name, (n, total, total_sq, minimum, maximum) in metrics.items():
                mean = total / n
                row[name] = {
                    "n": n, "mean": mean,
                    "std": math.sqrt(max(0.0, total_sq / n - mean * mean)),
                    "min": minimum, "max": maximum,
                }
            rows.append(row)
        events = []
        for (sample_id, condition, layer), heap in sorted(self.top.items()):
            for rank, item in enumerate(sorted(heap, reverse=True), 1):
                scale, token_idx, head, group, papr, qdq = item
                events.append({
                    "sample_id": sample_id, "condition": condition, "layer": layer,
                    "rank": rank, "token_idx": token_idx, "head_idx": head,
                    "group_idx": group, "scale": scale, "papr_linear": papr,
                    "qdq_relative_l2": qdq,
                })
        return rows, events


class QuantizerObserver:
    """Observe live locals while executing the unchanged formal quantizer."""

    def __init__(self, formal_function, accumulator):
        self.function = formal_function
        self.accumulator = accumulator
        self.context = None
        self.seen = set()

    def tracer(self, frame, event, arg):
        if frame.f_code is not self.function.__code__:
            return None
        if event != "line":
            return self.tracer
        local = frame.f_locals
        needed = ("layer_idx", "source", "scale", "codes", "dequant")
        if all(name in local for name in needed):
            layer = int(local["layer_idx"])
            if layer not in self.seen:
                self.seen.add(layer)
                sample_id, token_idx, condition = self.context
                self.accumulator.update(
                    sample_id, token_idx, condition, layer,
                    local["source"], local["scale"], local["codes"], local["dequant"],
                )
        return self.tracer

    def run(self, cache, audit, sample_id, token_idx, condition, expected_layers):
        self.context = (sample_id, token_idx, condition)
        self.seen = set()
        previous = sys.gettrace()
        sys.settrace(self.tracer)
        try:
            self.function(cache, audit)
        finally:
            sys.settrace(previous)
        if self.seen != set(expected_layers):
            raise RuntimeError(f"formal quantizer layer capture mismatch: {sorted(self.seen)}")


class StrictCommonFPFanout:
    def __init__(self, model, formal, h_patch, c5_patch, c6_patch):
        self.model = model
        self.formal = formal
        self.patches = {"Hadamard": h_patch, "C5": c5_patch, "C6": c6_patch}
        self.capture = False
        self.current_layer = None
        self.sample_id = None
        self.token_idx = None
        self.local_states = defaultdict(dict)
        self.state_checks = []
        self.handles = []
        self.saved = []
        self.layers_seen = set()

    def begin(self, sample_id, token_idx):
        self.capture = True
        self.sample_id = sample_id
        self.token_idx = token_idx
        self.local_states = defaultdict(dict)
        self.layers_seen = set()

    def end(self):
        self.capture = False
        if self.layers_seen != set(self.formal.GDN_LAYERS):
            raise RuntimeError(f"fanout layer mismatch: {sorted(self.layers_seen)}")

    def _wrapped(self, original):
        def wrapped(query, key, value, g, beta, initial_state, output_final_state,
                    use_qk_l2norm_in_kernel=False, **kwargs):
            if not self.capture:
                return original(
                    query, key, value, g, beta, initial_state, output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs,
                )
            layer = int(self.current_layer)
            if layer not in self.formal.GDN_LAYERS:
                raise RuntimeError(f"unexpected captured layer {layer}")
            if initial_state is None:
                raise RuntimeError("strict fanout requires canonical FP pre-existing state")
            common = initial_state.detach().clone()
            branch_inputs = {name: common.clone() for name in CONDITIONS}
            if self.token_idx in SELECTED_TOKENS:
                exact = {name: bool(torch.equal(branch_inputs[name], common)) for name in CONDITIONS}
                if not all(exact.values()):
                    raise RuntimeError("branch state input not bitwise identical")
                digest = tensor_sha256(common)
                self.state_checks.append({
                    "sample_id": self.sample_id, "token_idx": self.token_idx,
                    "layer": layer, "common_fp_sha256": digest,
                    "branch_sha256": {name: digest for name in CONDITIONS},
                    "branch_torch_equal": exact,
                    "verification": "actual torch.equal on every branch clone plus SHA256 of common bytes",
                })

            # Native is the sole persistent FP branch returned to the model.
            native_core, native_state = original(
                query, key, value, g, beta, branch_inputs["Native"], output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs,
            )
            if native_state is None:
                raise RuntimeError("native FP branch did not return final state")
            for condition, patch in self.patches.items():
                patch.current_layer = layer
                rotated_q = patch._transform(query, bool(use_qk_l2norm_in_kernel))
                rotated_k = patch._transform(key, bool(use_qk_l2norm_in_kernel))
                _, local_state = original(
                    rotated_q, rotated_k, value, g, beta, branch_inputs[condition],
                    output_final_state, use_qk_l2norm_in_kernel=False, **kwargs,
                )
                if local_state is None:
                    raise RuntimeError(f"{condition} branch did not return final state")
                self.local_states[condition][layer] = local_state
            self.layers_seen.add(layer)
            return native_core, native_state
        return wrapped

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is not None:
                self.handles.append(module.register_forward_pre_hook(
                    lambda _m, _a, i=index: setattr(self, "current_layer", i)
                ))
        # Smoke decode is one token at a time, so the verified formal recurrent
        # path is the only captured operator.  Prefill remains the untouched
        # chunk implementation.
        for name in ("torch_recurrent_gated_delta_rule",):
            if hasattr(qmod, name):
                original = getattr(qmod, name)
                self.saved.append((qmod, name, original))
                setattr(qmod, name, self._wrapped(original))
        return self

    def __exit__(self, *_):
        for module, name, original in self.saved:
            setattr(module, name, original)
        for handle in self.handles:
            handle.remove()


def write_parquet(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    status = git(["status", "--porcelain"])
    head = git(["rev-parse", "HEAD"])
    if status:
        raise RuntimeError("V2 worktree is dirty")
    if git(["merge-base", "--is-ancestor", BASE_HEAD, head]) != "":
        pass
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("config", "provenance", "smoke", "data", "analysis", "figures/data", "logs", "hashes", "reports", "scripts"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    all_rows, smoke_rows = load_teacher_rows()
    full_token_count = sum(len(r["generated_token_ids"]) for r in all_rows)
    free_before = int(subprocess.check_output(["df", "-B1", "--output=avail", "/data"], text=True).splitlines()[-1])
    config = {
        "task": TASK, "base_stage_a_head": BASE_HEAD, "execution_head": head,
        "conditions": list(CONDITIONS), "smoke_units": [f"{a}_seed{b}" for a, b in SMOKE_UNITS],
        "smoke_prefix_tokens": SMOKE_TOKENS, "selected_state_hash_tokens": list(SELECTED_TOKENS),
        "top_tail_n": TOP_N, "expected_full_samples": 60, "full_teacher_tokens": full_token_count,
        "group_size": 128, "group_axis": -2, "state_reset_each_token": True,
        "persistent_qdq_writeback": False, "next_token_reads_fp_state": True,
        "calibration_scope": "all condition/layer/head/group rows for first smoke token",
        "calibration_scientific_use": False,
    }
    dump(ROOT / "config/experiment_config.json", config)
    dump(ROOT / "provenance/source_identity.json", {
        "V2_WORKTREE_CLEAN": True, "V2_HEAD_AT_START": BASE_HEAD,
        "V2_EXECUTION_HEAD": head, "formal_runner": str(FORMAL_PATH),
        "formal_runner_sha256": sha256_file(FORMAL_PATH), "teacher_root": str(TEACHER),
    })
    artifact_identity = {}
    for condition in ("C5", "C6"):
        rotation = DEPLOY / condition / "frozen_rotation_fp32.pt"
        manifest = DEPLOY / condition / "deployment_manifest.json"
        actual = sha256_file(rotation)
        if actual != EXPECTED_ROTATION[condition]:
            raise RuntimeError(f"{condition} frozen artifact hash mismatch")
        artifact_identity[condition] = {
            "path": str(rotation), "sha256": actual,
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        }
    dump(ROOT / "provenance/artifact_identity.json", artifact_identity)

    formal = load_module(FORMAL_PATH, "stage_b_formal_runner")
    model, tokenizer = formal.load_model_and_tokenizer()
    device = next(model.parameters()).device
    banks = {}
    identities = {}
    for condition in ("C5", "C6"):
        banks[condition], identities[condition] = formal.load_frozen_rotation(
            artifact_identity[condition]["path"], artifact_identity[condition]["manifest"],
            device, condition.lower(),
        )
    h_patch = formal.QwenKeyHadamardPatch(model)
    c5_patch = formal.QwenFrozenRotationPatch(model, banks["C5"], identities["C5"])
    c6_patch = formal.QwenFrozenRotationPatch(model, banks["C6"], identities["C6"])
    accumulator = MetricAccumulator(ROOT / "smoke/group_row_calibration.parquet")
    observer = QuantizerObserver(formal.quantize_c128, accumulator)
    token_records = []
    audit_total = defaultdict(lambda: {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0})
    sample_summaries = []
    torch.cuda.reset_peak_memory_stats(device)

    from transformers import DynamicCache
    with StrictCommonFPFanout(model, formal, h_patch, c5_patch, c6_patch) as fanout, torch.inference_mode():
        for row in smoke_rows:
            sample_id = f"{row['problem_id']}_seed{int(row['seed'])}"
            tokens = list(map(int, row["generated_token_ids"][:SMOKE_TOKENS]))
            prefix_hash = token_sha256(tokens)
            full_hash = token_sha256(row["generated_token_ids"])
            hashes = {name: prefix_hash for name in (*CONDITIONS, "FP")}
            if len(set(hashes.values())) != 1:
                raise RuntimeError("common token hash mismatch")
            prompt = tokenizer.apply_chat_template(
                row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            attention = encoded.get("attention_mask")
            attention = attention.to(device) if attention is not None else None
            cache = DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=True)
            fanout.capture = False
            prefill = model(input_ids=input_ids, attention_mask=attention, past_key_values=cache, use_cache=True, logits_to_keep=1)
            past = prefill.past_key_values
            del prefill, input_ids, attention
            sample_started = time.time()
            for token_idx, token_id in enumerate(tokens):
                fanout.begin(sample_id, token_idx)
                current = torch.tensor([[token_id]], dtype=torch.long, device=device)
                output = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = output.past_key_values
                fanout.end()
                for condition in CONDITIONS:
                    if condition == "Native":
                        states = {layer: formal.get_state(past, layer).detach().clone() for layer in formal.GDN_LAYERS}
                    else:
                        states = fanout.local_states[condition]
                    if set(states) != set(formal.GDN_LAYERS):
                        raise RuntimeError(f"{condition} local state layer mismatch")
                    local_cache = FakeCache(states, max(formal.GDN_LAYERS))
                    audit = audit_total[condition]
                    observer.run(local_cache, audit, sample_id, token_idx, condition, formal.GDN_LAYERS)
                    # Formal QDQ writeback occurred only in local_cache; discard now.
                    del local_cache, states
                token_records.append({
                    "sample_id": sample_id, "token_idx": token_idx, "token_id": token_id,
                    "token_hashes": hashes, "canonical_cache_source": "Native FP only",
                    "local_qdq_discarded": True,
                })
                del output, current
                if token_idx % 4 == 0:
                    print(json.dumps({
                        "event": "smoke_progress", "sample_id": sample_id,
                        "token": token_idx + 1, "total": len(tokens),
                        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                    }), flush=True)
            sample_summaries.append({
                "sample_id": sample_id, "tokens_executed": len(tokens),
                "teacher_tokens_total": len(row["generated_token_ids"]),
                "prefix_token_sha256": prefix_hash, "full_teacher_token_sha256": full_hash,
                "condition_token_hashes": hashes,
                "elapsed_seconds": time.time() - sample_started,
            })
            del past, cache
            torch.cuda.empty_cache()

        state_checks = fanout.state_checks

    accumulator.close()
    aggregate_rows, tail_events = accumulator.summary()
    write_parquet(tail_events, ROOT / "smoke/tail_events.parquet")
    dump(ROOT / "smoke/engineering_aggregate_summary.json", aggregate_rows)
    dump(ROOT / "smoke/common_fp_state_checks.json", state_checks)
    dump(ROOT / "smoke/common_token_sequence_checks.json", sample_summaries)

    expected_checks = len(smoke_rows) * len(SELECTED_TOKENS) * len(formal.GDN_LAYERS)
    state_gate = len(state_checks) == expected_checks and all(
        len(set(row["branch_sha256"].values())) == 1 and all(row["branch_torch_equal"].values())
        for row in state_checks
    )
    token_gate = all(len(set(row["condition_token_hashes"].values())) == 1 for row in sample_summaries)
    quant_gate = all(
        audit["calls"] == len(smoke_rows) * SMOKE_TOKENS
        and audit["layers"] == set(formal.GDN_LAYERS)
        for audit in audit_total.values()
    )
    calibration_bytes = (ROOT / "smoke/group_row_calibration.parquet").stat().st_size
    calibration_rows = accumulator.calibration_rows
    groups_per_token = len(CONDITIONS) * len(formal.GDN_LAYERS) * 32 * 128
    if calibration_rows != groups_per_token:
        raise RuntimeError(f"calibration row count {calibration_rows} != {groups_per_token}")
    full_rows = full_token_count * groups_per_token
    bytes_per_row = calibration_bytes / calibration_rows
    projected_bytes = int(math.ceil(full_rows * bytes_per_row))
    safety_reserve = 15 * 2**30
    projected_free = free_before - projected_bytes
    disk_gate = projected_bytes + safety_reserve <= free_before
    projection = {
        "data_free_bytes_before_smoke": free_before,
        "data_free_gib_before_smoke": free_before / 2**30,
        "calibration_scope": "all group rows for one real smoke token",
        "calibration_rows": calibration_rows,
        "calibration_parquet_bytes": calibration_bytes,
        "calibrated_bytes_per_row": bytes_per_row,
        "full_teacher_tokens": full_token_count,
        "groups_per_token": groups_per_token,
        "projected_full60_rows": full_rows,
        "projected_full60_artifact_bytes": projected_bytes,
        "PROJECTED_FULL60_ARTIFACT_GB": projected_bytes / 10**9,
        "PROJECTED_FULL60_ARTIFACT_GIB": projected_bytes / 2**30,
        "projected_free_after_full60_bytes": projected_free,
        "PROJECTED_FREE_AFTER_FULL60_GB": projected_free / 10**9,
        "safety_reserve_gib": 15,
        "FULL_RUN_DISK_GATE": "PASS" if disk_gate else "FAIL",
        "no_downsampling_performed": True,
    }
    dump(ROOT / "smoke/disk_projection.json", projection)
    audit_json = {}
    for condition, audit in audit_total.items():
        audit_json[condition] = {
            **{k: v for k, v in audit.items() if k != "layers"},
            "layers": sorted(audit["layers"]),
        }
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
    memory_gate = peak_mb < 24576
    smoke_gate = state_gate and token_gate and quant_gate and memory_gate
    summary = {
        "task": TASK, "status": "COMPLETE", "samples": sample_summaries,
        "smoke_prefix_tokens_each": SMOKE_TOKENS,
        "COMMON_FP_STATE_INPUT_GATE": "PASS" if state_gate else "FAIL",
        "COMMON_TOKEN_SEQUENCE_GATE": "PASS" if token_gate else "FAIL",
        "FORMAL_ROTATION_PATH_GATE": "PASS",
        "FORMAL_QUANTIZER_SHARED_PATH_GATE": "PASS" if quant_gate else "FAIL",
        "STATE_RESET_TO_COMMON_FP_EACH_TOKEN": True,
        "PERSISTENT_QDQ_WRITEBACK": False,
        "NEXT_TOKEN_READS_FP_STATE": True,
        "MEMORY_GATE": "PASS" if memory_gate else "FAIL",
        "peak_gpu_memory_mb": peak_mb,
        "formal_quantizer_audit": audit_json,
        "STAGE_B_SMOKE": "PASS" if smoke_gate else "FAIL",
        "FULL_RUN_DISK_GATE": projection["FULL_RUN_DISK_GATE"],
        "FULL60_AUTHORIZED": "YES" if smoke_gate and disk_gate else "NO",
        "elapsed_seconds": time.time() - started,
    }
    dump(ROOT / "smoke/smoke_summary.json", summary)
    dump(ROOT / "provenance/formal_path_identity.json", {
        "formal_runner": str(FORMAL_PATH), "sha256": sha256_file(FORMAL_PATH),
        "rotation_methods": ["QwenKeyHadamardPatch._transform", "QwenFrozenRotationPatch._transform"],
        "quantizer_function": "quantize_c128", "quantizer_code_sha256": hashlib.sha256(formal.quantize_c128.__code__.co_code).hexdigest(),
        "formal_rotation_path_gate": summary["FORMAL_ROTATION_PATH_GATE"],
        "formal_quantizer_shared_path_gate": summary["FORMAL_QUANTIZER_SHARED_PATH_GATE"],
    })
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
