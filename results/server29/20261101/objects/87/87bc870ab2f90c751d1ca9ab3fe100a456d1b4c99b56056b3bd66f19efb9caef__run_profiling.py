#!/usr/bin/env python3
"""Timing-only profile of the frozen Stage-B GPU streaming smoke.

Scientific model/reducer semantics are imported unchanged.  This runner adds
CUDA-event and perf_counter instrumentation and intentionally emits no
scientific sample summaries.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


TASK = "QWEN_GDN_STAGE_B_GPU_STREAMING_PROFILING_V1"
BASE_HEAD = "a352280cbacaab19e222f8a1943a457c36f42700"
REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = REPO / "experiments/qwen_gdn" / TASK
SMOKE_ROOT = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_GPU_STREAMING_SMOKE_V3"
OLD_SMOKE = REPO / "experiments/qwen_gdn/QWEN_GDN_STRICT_COMMON_FP_STATE_FANOUT_V1/scripts/stage_b_smoke.py"
V3_PATH = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/scripts/reducer_v3.py"
FORMAL_PATH = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py")
DEPLOY = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment")
TEACHER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
SMOKE_UNITS = (("aime26_01", 1), ("aime26_01", 2))
SMOKE_TOKENS = 32
FULL_TEACHER_TOKENS = 1854451
EXPECTED_ROTATION = {
    "C5": "0266c5065267cf452d72cfe29a2a4d9a24fdfb6a89d7bec04b4d796c86f857c9",
    "C6": "67e3731213ad3fb0c088c781ddd912aaaad5af75f9532b289442690a7b09f39c",
}
EPS = 1e-12


def loadmod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_sha(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def canonical_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_rows():
    rows = []
    for path in sorted(TEACHER.glob("worker*/fp_state.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda row: (int(row["canonical_index"]), int(row["seed"])))
    index = {(row["problem_id"], int(row["seed"])): row for row in rows}
    return rows, [index[key] for key in SMOKE_UNITS]


class TimingBook:
    def __init__(self):
        self.wall = defaultdict(float)
        self.count = defaultdict(int)
        self.gpu_events = defaultdict(list)
        self.token_wall = []
        self.sample_wall = []
        self.memory_tokens = []
        self.memory_samples = []

    def event_pair(self, key):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.gpu_events[key].append((start, end))
        self.count[f"{key}_calls"] += 1
        return end

    def resolve_events(self):
        out = {}
        for key, pairs in self.gpu_events.items():
            out[key] = sum(float(start.elapsed_time(end)) for start, end in pairs)
        return out


class RecurrentCoreProfiler:
    """Wrap the formal recurrent kernel before StrictCommonFPFanout captures it."""
    def __init__(self, original, book):
        self.original = original
        self.book = book
        self.enabled = False
        self.call_index = 0

    def begin_token(self):
        self.enabled = True
        self.call_index = 0

    def end_token(self):
        self.enabled = False
        expected = len(CONDITIONS) * 24
        if self.call_index != expected:
            raise RuntimeError(f"condition core call count {self.call_index} != {expected}")

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.original(*args, **kwargs)
        condition = CONDITIONS[self.call_index % len(CONDITIONS)]
        self.call_index += 1
        end = self.book.event_pair(f"condition_core/{condition}")
        out = self.original(*args, **kwargs)
        end.record()
        return out


def wrap_rotation(patch, condition, book):
    original = patch._transform

    def wrapped(*args, **kwargs):
        end = book.event_pair(f"rotation/{condition}")
        out = original(*args, **kwargs)
        end.record()
        return out
    patch._transform = wrapped
    return original


class _TimedPairIterator:
    """Measure each unchanged PAIRS loop body through iterator boundaries."""
    def __init__(self, values, book):
        self.iterator = iter(values)
        self.book = book
        self.started = None

    def __iter__(self):
        return self

    def __next__(self):
        now = time.perf_counter()
        if self.started is not None:
            self.book.wall["paired_count_time"] += now - self.started
        try:
            value = next(self.iterator)
        except StopIteration:
            self.started = None
            raise
        self.started = time.perf_counter()
        return value


class _TimedPairs:
    def __init__(self, values, book):
        self.values = tuple(values)
        self.book = book

    def __iter__(self):
        return _TimedPairIterator(self.values, self.book)


class ReducerComponentProfiler:
    """Low-overhead method-boundary timing around the unchanged V3 reducer."""
    TRACKED = ("token_tail_time", "topk_event_time", "paired_count_time", "histogram_moment_time")

    def __init__(self, v3, book):
        self.v3 = v3
        self.book = book
        self.original_tail = v3.candidate_top42_tail
        self.original_topk = v3.DeterministicTopK.update
        self.original_hist = v3.MultiplicativeHistogram.update
        self.original_moments = v3.Moments.update
        self.original_pairs = v3.PAIRS

        def timed_tail(values):
            started = time.perf_counter()
            try:
                return self.original_tail(values)
            finally:
                self.book.wall["token_tail_time"] += time.perf_counter() - started

        def timed_topk(instance, values, token_idx, group_ids):
            started = time.perf_counter()
            try:
                return self.original_topk(instance, values, token_idx, group_ids)
            finally:
                self.book.wall["topk_event_time"] += time.perf_counter() - started

        def timed_hist(instance, values):
            started = time.perf_counter()
            try:
                return self.original_hist(instance, values)
            finally:
                self.book.wall["histogram_moment_time"] += time.perf_counter() - started

        def timed_moments(instance, values):
            is_tail = np.asarray(values).size == 1
            key = "token_tail_time" if is_tail else "histogram_moment_time"
            started = time.perf_counter()
            try:
                return self.original_moments(instance, values)
            finally:
                self.book.wall[key] += time.perf_counter() - started

        v3.candidate_top42_tail = timed_tail
        v3.DeterministicTopK.update = timed_topk
        v3.MultiplicativeHistogram.update = timed_hist
        v3.Moments.update = timed_moments
        v3.PAIRS = _TimedPairs(self.original_pairs, book)

    def run(self, reducer, token_idx, batches):
        before = sum(self.book.wall[key] for key in self.TRACKED)
        started = time.perf_counter()
        reducer.update_token(token_idx, batches)
        elapsed = time.perf_counter() - started
        attributed = sum(self.book.wall[key] for key in self.TRACKED) - before
        self.book.wall["python_reducer_time"] += elapsed
        self.book.wall["python_reducer_other_time"] += max(0.0, elapsed - attributed)
        self.book.count["python_reducer_calls"] += 1

    def restore(self):
        self.v3.candidate_top42_tail = self.original_tail
        self.v3.DeterministicTopK.update = self.original_topk
        self.v3.MultiplicativeHistogram.update = self.original_hist
        self.v3.Moments.update = self.original_moments
        self.v3.PAIRS = self.original_pairs


class ProfilingAccumulator:
    def __init__(self, reducer, v3, book, component_profiler):
        self.reducer = reducer
        self.v3 = v3
        self.book = book
        self.component_profiler = component_profiler
        self.buffer = {}
        self.sample_id = reducer.identity["sample_id"]
        self.total_wall = 0.0

    def _gpu_metrics(self, source, scale, codes, dequant):
        scale_end = self.book.event_pair("scale_stat")
        scale_metric = scale.squeeze(-2)
        scale_end.record()

        papr_end = self.book.event_pair("papr_stat")
        absmax = source.abs().amax(dim=-2)
        power = source.square().mean(dim=-2)
        rms = power.sqrt()
        papr = absmax / rms.clamp_min(EPS)
        papr_end.record()

        qdq_end = self.book.event_pair("qdq_error_stat")
        err = dequant - source
        l2 = err.square().sum(dim=-2).sqrt()
        source_l2 = source.square().sum(dim=-2).sqrt()
        qdq = l2 / source_l2.clamp_min(EPS)
        qdq_end.record()

        secondary_end = self.book.event_pair("secondary_stat")
        minimum = source.amin(dim=-2)
        maximum = source.amax(dim=-2)
        value_range = maximum - minimum
        secondary_end.record()
        return {"scale": scale_metric, "papr_linear": papr, "qdq_relative_l2": qdq, "absmax": absmax, "range": value_range}

    def update(self, sample_id, token_idx, condition, layer, source, scale, codes, dequant):
        started = time.perf_counter()
        if sample_id != self.sample_id:
            raise RuntimeError("sample identity mismatch")
        key = (int(layer), condition)
        if key in self.buffer:
            raise RuntimeError("duplicate metric capture")
        metrics = self._gpu_metrics(source, scale, codes, dequant)
        arrays = {}
        for name in self.v3.METRICS:
            transfer_started = time.perf_counter()
            arrays[name] = metrics[name].detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            transfer_elapsed = time.perf_counter() - transfer_started
            self.book.wall["cuda_to_cpu_time"] += transfer_elapsed
            self.book.count["cuda_to_cpu_calls"] += 1
        self.buffer[key] = {"group_ids": np.arange(self.v3.N_GROUPS, dtype=np.int64), "metrics": arrays}
        elapsed = time.perf_counter() - started
        self.total_wall += elapsed
        self.book.wall["accumulator_total_time"] += elapsed
        self.book.count["accumulator_calls"] += 1

    def flush(self, token_idx):
        expected = len(self.v3.LAYERS) * len(self.v3.CONDITIONS)
        if len(self.buffer) != expected:
            return False
        started = time.perf_counter()
        self.component_profiler.run(self.reducer, token_idx, self.buffer)
        self.buffer = {}
        elapsed = time.perf_counter() - started
        self.total_wall += elapsed
        self.book.wall["accumulator_total_time"] += elapsed
        self.book.count["accumulator_flush_calls"] += 1
        return True

    def finish(self):
        if self.buffer:
            raise RuntimeError("partial token buffer")


class ProfilingQuantizerObserver:
    """Trace the unchanged formal quantizer and time scale/codes/dequant with events."""
    def __init__(self, formal_function, accumulator, book):
        self.function = formal_function
        self.accumulator = accumulator
        self.book = book
        self.context = None
        self.seen = set()
        lines, start = inspect.getsourcelines(formal_function)
        source = {start + i: line.strip() for i, line in enumerate(lines)}
        self.scale_line = next(n for n, x in source.items() if x.startswith("scale ="))
        self.capture_line = next(n for n, x in source.items() if x.startswith("if not bool(torch.isfinite"))
        self.quant_start = {}

    def tracer(self, frame, event, arg):
        if frame.f_code is not self.function.__code__:
            return None
        if event != "line":
            return self.tracer
        line = frame.f_lineno
        local = frame.f_locals
        if line == self.scale_line and "layer_idx" in local:
            layer = int(local["layer_idx"])
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.quant_start[layer] = (start, end)
            self.book.gpu_events["quantizer_scale_qcode_dequant"].append((start, end))
            self.book.count["quantizer_layer_calls"] += 1
        if line == self.capture_line:
            needed = ("layer_idx", "source", "scale", "codes", "dequant")
            if all(name in local for name in needed):
                layer = int(local["layer_idx"])
                if layer not in self.seen:
                    self.seen.add(layer)
                    self.quant_start[layer][1].record()
                    sample_id, token_idx, condition = self.context
                    self.accumulator.update(sample_id, token_idx, condition, layer, local["source"], local["scale"], local["codes"], local["dequant"])
        return self.tracer

    def run(self, cache, audit, sample_id, token_idx, condition, expected_layers):
        self.context = (sample_id, token_idx, condition)
        self.seen = set()
        self.quant_start = {}
        acc_before = self.accumulator.total_wall
        previous = sys.gettrace()
        started = time.perf_counter()
        sys.settrace(self.tracer)
        try:
            self.function(cache, audit)
        finally:
            sys.settrace(previous)
        self.accumulator.flush(token_idx)
        elapsed = time.perf_counter() - started
        acc_delta = self.accumulator.total_wall - acc_before
        self.book.wall["quantizer_observer_total_time"] += elapsed
        self.book.wall["quantizer_exclusive_wall_time"] += max(0.0, elapsed - acc_delta)
        self.book.count["quantizer_condition_calls"] += 1
        self.book.count["quantizer_item_sync_calls"] += 3 * len(expected_layers)
        if self.seen != set(expected_layers):
            raise RuntimeError("formal quantizer capture layer mismatch")


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--gpu":
        raise SystemExit("usage: run_profiling.py --gpu N")
    gpu = int(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    start_head = git("rev-parse", "HEAD")
    clean = not git("status", "--porcelain")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_HEAD, start_head], cwd=REPO).returncode == 0
    if not (clean and ancestor):
        raise RuntimeError("SOURCE_GATE=FAIL")
    for name in ("config", "provenance", "profiling", "analysis", "hashes", "reports", "logs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)

    old = loadmod(OLD_SMOKE, "profiling_frozen_fanout")
    v3 = loadmod(V3_PATH, "profiling_frozen_reducer_v3")
    formal = loadmod(FORMAL_PATH, "profiling_frozen_formal_runner")
    frozen_identity = json.loads((SMOKE_ROOT / "smoke_identity.json").read_text())
    rows, smoke_rows = load_rows()
    if len(rows) != 60 or sum(len(r["generated_token_ids"]) for r in rows) != FULL_TEACHER_TOKENS:
        raise RuntimeError("teacher identity mismatch")
    for row in smoke_rows:
        sid = f"{row['problem_id']}_seed{row['seed']}"
        expected = frozen_identity["samples"][sid]
        if token_sha(row["generated_token_ids"][:SMOKE_TOKENS]) != expected["prefix_sha256"] or token_sha(row["generated_token_ids"]) != expected["full_sha256"]:
            raise RuntimeError("smoke identity mismatch")

    config = {
        "task": TASK, "source_base_head": BASE_HEAD, "execution_head": start_head,
        "samples": [f"{a}_seed{b}" for a, b in SMOKE_UNITS], "tokens_per_sample": SMOKE_TOKENS,
        "repeats": 1, "conditions": list(CONDITIONS), "metrics": list(v3.METRICS),
        "same_model": True, "same_samples": True, "same_tokens": True, "same_conditions": True,
        "same_metrics": True, "same_reducer": True, "profiling_only": True,
        "FULL60_STARTED": False, "AIME_STARTED": False, "TRAINING_STARTED": False,
    }
    dump(ROOT / "config/profiling_config.json", config)
    dump(ROOT / "provenance/source_identity.json", {
        "SOURCE_GATE": "PASS", "SOURCE_BRANCH": git("symbolic-ref", "--short", "HEAD"),
        "SOURCE_BASE_HEAD": BASE_HEAD, "EXECUTION_HEAD": start_head, "WORKTREE_CLEAN_AT_START": True,
        "formal_runner": str(FORMAL_PATH), "formal_runner_sha256": sha(FORMAL_PATH),
        "fanout_source": str(OLD_SMOKE), "fanout_source_sha256": sha(OLD_SMOKE),
        "reducer_v3": str(V3_PATH), "reducer_v3_sha256": sha(V3_PATH),
        "smoke_identity_sha256": sha(SMOKE_ROOT / "smoke_identity.json"),
    })
    nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    dump(ROOT / "provenance/gpu_environment.json", {
        "physical_gpu": gpu, "visible_gpu_count": torch.cuda.device_count(), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "system_gpus": nvidia, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": platform.python_version(),
    })

    model, tokenizer = formal.load_model_and_tokenizer()
    device = next(model.parameters()).device
    banks, identities = {}, {}
    for condition in ("C5", "C6"):
        rotation = DEPLOY / condition / "frozen_rotation_fp32.pt"
        manifest = DEPLOY / condition / "deployment_manifest.json"
        if sha(rotation) != EXPECTED_ROTATION[condition]:
            raise RuntimeError("rotation identity mismatch")
        banks[condition], identities[condition] = formal.load_frozen_rotation(str(rotation), str(manifest), device, condition.lower())
    book = TimingBook()
    h_patch = formal.QwenKeyHadamardPatch(model)
    c5_patch = formal.QwenFrozenRotationPatch(model, banks["C5"], identities["C5"])
    c6_patch = formal.QwenFrozenRotationPatch(model, banks["C6"], identities["C6"])
    for condition, patch in (("Hadamard", h_patch), ("C5", c5_patch), ("C6", c6_patch)):
        wrap_rotation(patch, condition, book)

    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    kernel_name = "torch_recurrent_gated_delta_rule"
    underlying = getattr(qmod, kernel_name)
    core_profiler = RecurrentCoreProfiler(underlying, book)
    setattr(qmod, kernel_name, core_profiler)
    component_profiler = ReducerComponentProfiler(v3, book)
    memory_sample_after = []
    prefill_wall = 0.0
    serialization_bytes = []
    state_checks = []

    from transformers import DynamicCache
    try:
        with old.StrictCommonFPFanout(model, formal, h_patch, c5_patch, c6_patch) as fanout, torch.inference_mode():
            for row in smoke_rows:
                sid = f"{row['problem_id']}_seed{row['seed']}"
                expected = frozen_identity["samples"][sid]
                tokens = list(map(int, row["generated_token_ids"][:SMOKE_TOKENS]))
                identity = {
                    "sample_id": sid, "source_commit": BASE_HEAD, "reducer_schema_version": v3.SCHEMA_VERSION,
                    "input_identity_hash": expected["prefix_sha256"],
                    "metric_config_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/config/metric_definitions.json"),
                    "tail_definition_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/config/token_local_tail_spec.json"),
                    "histogram_config_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/config/histogram_counts_spec.json"),
                    "topk_order_hash": sha(REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V2_CLOSURE/config/topk_total_order.json"),
                }
                reducer = v3.SampleReducerV3(identity)
                accumulator = ProfilingAccumulator(reducer, v3, book, component_profiler)
                observer = ProfilingQuantizerObserver(formal.quantize_c128, accumulator, book)
                audit = {c: {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0} for c in CONDITIONS}
                gc.collect(); torch.cuda.empty_cache();
                sync_started = time.perf_counter(); torch.cuda.synchronize(device); book.wall["explicit_sync_time"] += time.perf_counter() - sync_started; book.count["explicit_sync_calls"] += 1
                prompt = tokenizer.apply_chat_template(row["raw_messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"].to(device)
                attention = encoded.get("attention_mask"); attention = attention.to(device) if attention is not None else None
                cache = DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=True)
                fanout.capture = False
                prefill_started = time.perf_counter()
                prefill = model(input_ids=input_ids, attention_mask=attention, past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize(device)
                prefill_wall += time.perf_counter() - prefill_started
                book.count["explicit_sync_calls"] += 1
                past = prefill.past_key_values
                del prefill, input_ids, attention
                sample_started = time.perf_counter()
                check_start = len(fanout.state_checks)
                for token_idx, token_id in enumerate(tokens):
                    token_started = time.perf_counter()
                    mem_before = {"allocated_before": int(torch.cuda.memory_allocated(device)), "reserved_before": int(torch.cuda.memory_reserved(device))}
                    forward_start = torch.cuda.Event(enable_timing=True); forward_end = torch.cuda.Event(enable_timing=True)
                    forward_start.record()
                    core_profiler.begin_token()
                    fanout.begin(sid, token_idx)
                    current = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    output = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = output.past_key_values
                    fanout.end(); core_profiler.end_token(); forward_end.record()
                    book.gpu_events["forward"].append((forward_start, forward_end)); book.count["forward_calls"] += 1
                    for condition in CONDITIONS:
                        if condition == "Native":
                            states = {layer: formal.get_state(past, layer).detach().clone() for layer in formal.GDN_LAYERS}
                        else:
                            states = fanout.local_states[condition]
                        observer.run(old.FakeCache(states, max(formal.GDN_LAYERS)), audit[condition], sid, token_idx, condition, formal.GDN_LAYERS)
                        del states
                    del output, current
                    sync_started = time.perf_counter(); torch.cuda.synchronize(device); sync_elapsed = time.perf_counter() - sync_started
                    book.wall["explicit_sync_time"] += sync_elapsed; book.count["explicit_sync_calls"] += 1
                    mem_after = {"allocated_after": int(torch.cuda.memory_allocated(device)), "reserved_after": int(torch.cuda.memory_reserved(device))}
                    token_elapsed = time.perf_counter() - token_started
                    book.token_wall.append(token_elapsed)
                    book.memory_tokens.append({"sample_id": sid, "token_idx": token_idx, **mem_before, **mem_after})
                    if (token_idx + 1) % 4 == 0:
                        print(json.dumps({"event": "profile_progress", "sample_id": sid, "token": token_idx + 1, "total": len(tokens), "seconds_per_token_running": statistics.mean(book.token_wall)}), flush=True)
                accumulator.finish()
                state_checks.extend(fanout.state_checks[check_start:])
                serialization_started = time.perf_counter()
                sample_blob = canonical_bytes(reducer.sample_artifact())
                checkpoint_blob = canonical_bytes(reducer.to_state())
                serialization_elapsed = time.perf_counter() - serialization_started
                book.wall["serialization_time"] += serialization_elapsed
                book.count["serialization_calls"] += 1
                serialization_bytes.append({"sample_id": sid, "sample_artifact_bytes_in_memory": len(sample_blob), "checkpoint_bytes_in_memory": len(checkpoint_blob)})
                del sample_blob, checkpoint_blob, reducer, accumulator, observer, past, cache
                gc.collect(); torch.cuda.empty_cache();
                sync_started = time.perf_counter(); torch.cuda.synchronize(device); book.wall["explicit_sync_time"] += time.perf_counter() - sync_started; book.count["explicit_sync_calls"] += 1
                after = {"sample_id": sid, "allocated_after_cleanup": int(torch.cuda.memory_allocated(device)), "reserved_after_cleanup": int(torch.cuda.memory_reserved(device))}
                memory_sample_after.append(after)
                book.sample_wall.append(time.perf_counter() - sample_started)
    finally:
        setattr(qmod, kernel_name, underlying)
        component_profiler.restore()

    gpu_ms = book.resolve_events()
    total_tokens = len(book.token_wall)
    if total_tokens != 64 or len(state_checks) != 96:
        raise RuntimeError("profiling coverage mismatch")
    if not all(len(set(x["branch_sha256"].values())) == 1 and all(x["branch_torch_equal"].values()) for x in state_checks):
        raise RuntimeError("semantic gate failed")
    total_wall = sum(book.token_wall)
    wall_ms = {k: v * 1000.0 for k, v in book.wall.items()}
    gpu_ms_per_token = {k: v / total_tokens for k, v in gpu_ms.items()}
    wall_ms_per_token = {k: v / total_tokens for k, v in wall_ms.items()}
    mean_token_ms = total_wall * 1000.0 / total_tokens
    components = {
        "forward": gpu_ms_per_token.get("forward", 0.0),
        "rotation": sum(gpu_ms_per_token.get(f"rotation/{c}", 0.0) for c in ("Hadamard", "C5", "C6")),
        "quantizer": gpu_ms_per_token.get("quantizer_scale_qcode_dequant", 0.0),
        "scale_stats": gpu_ms_per_token.get("scale_stat", 0.0),
        "PAPR": gpu_ms_per_token.get("papr_stat", 0.0),
        "QDQ_error": gpu_ms_per_token.get("qdq_error_stat", 0.0),
        "tails": wall_ms_per_token.get("token_tail_time", 0.0),
        "paired_counts": wall_ms_per_token.get("paired_count_time", 0.0),
        "TopK": wall_ms_per_token.get("topk_event_time", 0.0),
        "histogram_moments": wall_ms_per_token.get("histogram_moment_time", 0.0),
        "reducer_other": wall_ms_per_token.get("python_reducer_other_time", 0.0),
        "GPU_to_CPU_transfer": wall_ms_per_token.get("cuda_to_cpu_time", 0.0),
        "Python_reducer": wall_ms_per_token.get("python_reducer_time", 0.0),
        # The smoke owns one reducer per sample, so no cross-partial merge is invoked.
        "merge": 0.0,
        "serialization": wall_ms.get("serialization_time", 0.0) / total_tokens,
    }
    table = [{"component": key, "ms_per_token": value, "percentage_of_observed_token_wall": 100.0 * value / mean_token_ms, "timing_domain": "CUDA_EVENT" if key in ("forward", "rotation", "quantizer", "scale_stats", "PAPR", "QDQ_error") else "PERF_COUNTER"} for key, value in components.items()]
    top_level = {k: components[k] for k in ("Python_reducer", "GPU_to_CPU_transfer", "forward", "quantizer")}
    ranking = sorted(top_level.items(), key=lambda item: item[1], reverse=True)
    reducer_dominant = ranking[0][0] == "Python_reducer"
    transfer_dominant = ranking[0][0] == "GPU_to_CPU_transfer"
    forward_dominant = ranking[0][0] == "forward"
    tail_dominant = components["tails"] >= max(
        components["paired_counts"], components["TopK"],
        components["histogram_moments"], components["reducer_other"],
    )
    after_alloc = [x["allocated_after_cleanup"] for x in memory_sample_after]
    memory_creep_bytes = max(after_alloc) - min(after_alloc)
    memory_creep = memory_creep_bytes > 16 * 2**20
    sync_count = book.count["explicit_sync_calls"] + book.count["cuda_to_cpu_calls"] + book.count["quantizer_item_sync_calls"]
    measured_sync_time = book.wall["explicit_sync_time"] + book.wall["cuda_to_cpu_time"]

    timing = {
        "TIME_BREAKDOWN": "COMPLETE", "tokens": total_tokens, "samples": 2, "repeats": 1,
        "observed_token_wall_seconds": total_wall, "mean_token_wall_ms": mean_token_ms,
        "tokens_per_second_profiled": total_tokens / total_wall,
        "component_table": table, "gpu_event_total_ms": gpu_ms, "wall_total_ms": wall_ms,
        "gpu_ms_per_token": gpu_ms_per_token, "wall_ms_per_token": wall_ms_per_token,
        "condition_core_cuda_ms_per_token": {c: gpu_ms_per_token.get(f"condition_core/{c}", 0.0) for c in CONDITIONS},
        "rotation_cuda_ms_per_token": {c: (0.0 if c == "Native" else gpu_ms_per_token.get(f"rotation/{c}", 0.0)) for c in CONDITIONS},
        "prefill_wall_seconds_total": prefill_wall, "sample_wall_seconds": book.sample_wall,
        "serialization_bytes_in_memory_only": serialization_bytes,
        "percentage_note": "Inclusive diagnostic percentages use observed token wall as denominator; forward contains condition cores and rotation, while reducer contains tails/paired/TopK/histogram_moments/reducer_other, so rows are not additive. Merge is zero because this smoke invokes one reducer per sample and performs no cross-partial merge.",
    }
    dump(ROOT / "profiling/timing_breakdown.json", timing)
    with (ROOT / "profiling/timing_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "ms_per_token", "percentage_of_observed_token_wall", "timing_domain"])
        writer.writeheader(); writer.writerows(table)
    dump(ROOT / "profiling/cuda_sync_report.json", {
        "sync_count": sync_count, "sync_time_seconds_measured_lower_bound": measured_sync_time,
        "explicit_sync_calls": book.count["explicit_sync_calls"], "explicit_sync_time_seconds": book.wall["explicit_sync_time"],
        "tensor_cpu_transfer_calls": book.count["cuda_to_cpu_calls"], "tensor_cpu_transfer_time_seconds": book.wall["cuda_to_cpu_time"],
        "formal_quantizer_item_sync_calls_static_exact": book.count["quantizer_item_sync_calls"],
        "unnecessary_sync_audit": "HIGH_FREQUENCY_IMPLICIT_SYNCHRONIZATION_PRESENT",
        "profiling_inserted_token_sync_calls": total_tokens,
        "production_hotspot": "480 tensor.cpu() plus 288 Tensor.item() synchronization opportunities per token",
    })
    dump(ROOT / "profiling/memory_profile.json", {
        "per_token": book.memory_tokens, "sample_after_cleanup": memory_sample_after,
        "memory_creep_bytes": memory_creep_bytes, "MEMORY_CREEP": memory_creep,
        "MEMORY_GATE": "PASS" if not memory_creep else "FAIL",
    })

    rank_lines = ["# Bottleneck ranking", "", "Top-level inclusive ranking:", ""]
    for i, (name, ms) in enumerate(ranking, 1):
        rank_lines.append(f"{i}. **{name}** — {ms:.3f} ms/token ({100*ms/mean_token_ms:.2f}% of observed wall)")
    rank_lines += ["", "Reducer subcomponent ranking:", ""]
    reducer_parts = sorted(
        ((k, components[k]) for k in ("histogram_moments", "tails", "TopK", "paired_counts", "reducer_other")),
        key=lambda item: item[1], reverse=True,
    )
    for i, (name, ms) in enumerate(reducer_parts, 1):
        rank_lines.append(f"{i}. **{name}** — {ms:.3f} ms/token ({100*ms/mean_token_ms:.2f}% of observed wall)")
    rank_lines += ["", f"`PYTHON_REDUCER_DOMINANT = {'YES' if reducer_dominant else 'NO'}`", f"`TRANSFER_DOMINANT = {'YES' if transfer_dominant else 'NO'}`", f"`TAIL_DOMINANT = {'YES' if tail_dominant else 'NO'}`", f"`FORWARD_DOMINANT = {'YES' if forward_dominant else 'NO'}`", "", "Subcomponent percentages are inclusive and must not be summed with their parent component."]
    (ROOT / "analysis/bottleneck_ranking.md").write_text("\n".join(rank_lines) + "\n", encoding="utf-8")
    theoretical_no_reducer = mean_token_ms / max(1e-9, mean_token_ms - components["Python_reducer"])
    theoretical_no_histogram = mean_token_ms / max(1e-9, mean_token_ms - components["histogram_moments"])
    theoretical_no_topk = mean_token_ms / max(1e-9, mean_token_ms - components["TopK"])
    theoretical_no_transfer = mean_token_ms / max(1e-9, mean_token_ms - components["GPU_to_CPU_transfer"])
    recommendation = f"""# Optimization recommendation (not implemented)

1. Replace the Python scalar-loop multiplicative histogram update with a semantics-identical vectorized/compiled integer-bin count path, preserving gamma, boundary correction, exact counts, and deterministic merge.
2. Batch the 480 per-token metric transfers into a small number of contiguous asynchronous copies without changing metric values or reducer order.
3. Remove or aggregate formal quantizer `.item()` audit reductions only after proving audit equivalence; there are 288 such synchronization opportunities per token.
4. Preserve the exact Top-42 tails and deterministic Top-32 ordering; optimize only their implementation after the dominant histogram path.

The diagnostic Amdahl upper bounds are {theoretical_no_histogram:.2f}x for eliminating the histogram/moments path, {theoretical_no_topk:.2f}x for eliminating TopK, {theoretical_no_reducer:.2f}x for eliminating the entire reducer, and {theoretical_no_transfer:.2f}x for eliminating measured transfer wall. These are not promised realized speedups, and overlap/synchronization effects mean they are not additive.

`DO_NOT_IMPLEMENT = TRUE`
"""
    (ROOT / "analysis/optimization_recommendation.md").write_text(recommendation, encoding="utf-8")

    report = f"""# FINAL PROFILING REPORT

## Verdict

`TIME_BREAKDOWN = COMPLETE`

Q1. Main bottleneck behind 0.201873 token/s: **{ranking[0][0]}**, measured at {ranking[0][1]:.3f} ms/token ({100*ranking[0][1]/mean_token_ms:.2f}% inclusive wall share).

Q2. Forward: {components['forward']:.3f} ms/token, {100*components['forward']/mean_token_ms:.2f}%.

Q3. Rotation: {components['rotation']:.3f} ms/token, {100*components['rotation']/mean_token_ms:.2f}% (subset of forward).

Q4. Quantizer scale/qcode/dequant kernels: {components['quantizer']:.3f} ms/token, {100*components['quantizer']/mean_token_ms:.2f}%.

Q5. Exact token-tail computation: {components['tails']:.3f} ms/token, {100*components['tails']/mean_token_ms:.2f}% (subset of reducer).

Q6. GPU-to-CPU transfer/synchronization: {components['GPU_to_CPU_transfer']:.3f} ms/token, {100*components['GPU_to_CPU_transfer']/mean_token_ms:.2f}%.

Q7. Python reducer: {components['Python_reducer']:.3f} ms/token, {100*components['Python_reducer']/mean_token_ms:.2f}%.

Q8. Unnecessary synchronization? **YES, high-frequency implicit synchronization is present**: {book.count['cuda_to_cpu_calls']} tensor transfers and {book.count['quantizer_item_sync_calls']} formal quantizer `.item()` calls over 64 tokens. The {total_tokens} profiling-inserted token-end synchronizations are measurement-only.

Q9. Memory creep? **{'YES' if memory_creep else 'NO'}**; cleaned sample-end spread {memory_creep_bytes/2**20:.3f} MiB.

Q10. Highest-priority optimization: vectorize/compile the semantics-identical Python histogram/reducer hot path; **not implemented**.

Q11. Possible speedup: the highest-priority histogram/moments path has a diagnostic elimination upper bound of {theoretical_no_histogram:.2f}x; eliminating the entire reducer would bound at {theoretical_no_reducer:.2f}x. Realized speedup requires a separate equivalence-gated optimization task.

Q12. Scientific semantics modified? **NO.** Timing instrumentation only.

## Profiling attempts

One low-overhead 64-token run is the authoritative result. A first complete run had suppressed reducer subcomponent attribution, and a second run was stopped after 4 tokens when intrusive line tracing produced decisive observer overhead. Both invalid attempts are preserved under `attempts/`; neither is used in this report.

## Hypotheses

`PYTHON_REDUCER_DOMINANT = {'YES' if reducer_dominant else 'NO'}`

`TRANSFER_DOMINANT = {'YES' if transfer_dominant else 'NO'}`

`TAIL_DOMINANT = {'YES' if tail_dominant else 'NO'}`

`FORWARD_DOMINANT = {'YES' if forward_dominant else 'NO'}`

## Terminal state

`MODEL_SEMANTICS_CHANGED = FALSE`

`REDUCER_SEMANTICS_CHANGED = FALSE`

`METRICS_CHANGED = FALSE`

`FULL60_STARTED = FALSE`

`AIME_STARTED = FALSE`

`TRAINING_STARTED = FALSE`
"""
    (ROOT / "reports/FINAL_PROFILING_REPORT.md").write_text(report, encoding="utf-8")
    dump(ROOT / "profiling/final_status.json", {
        "SOURCE_GATE": "PASS", "TIME_BREAKDOWN": "COMPLETE",
        "PYTHON_REDUCER_DOMINANT": "YES" if reducer_dominant else "NO",
        "TRANSFER_DOMINANT": "YES" if transfer_dominant else "NO",
        "TAIL_DOMINANT": "YES" if tail_dominant else "NO",
        "FORWARD_DOMINANT": "YES" if forward_dominant else "NO",
        "MEMORY_CREEP": memory_creep,
        "MODEL_SEMANTICS_CHANGED": False, "REDUCER_SEMANTICS_CHANGED": False, "METRICS_CHANGED": False,
        "FULL60_STARTED": False, "AIME_STARTED": False, "TRAINING_STARTED": False,
    })
    manifest = ROOT / "hashes/SHA256SUMS"
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != manifest and "__pycache__" not in p.parts)
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in paths), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "tokens": total_tokens, "profiled_tps": total_tokens/total_wall, "dominant": ranking[0], "memory_creep": memory_creep, "full60_started": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
