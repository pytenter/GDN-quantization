#!/usr/bin/env python3
"""One manifest-bound, qualification-only worker. No launcher or formal path.

All imports capable of creating a CUDA context occur after the independent
stdlib-only admission guard. Merely importing this script, --list-stages and
--manifest-template do not import Torch, load a model, or dispatch a process.
An implementation here is not evidence that its GPU qualification has run.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


TASK = "QWEN_GDN_OSCAR_RESIDENT_ROTATION_EFFICIENT_QUALIFICATION_V1"
UUIDS = {1: "GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3",
         2: "GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31",
         3: "GPU-30b07aba-a728-7649-4899-e09d33f8bc75"}
STAGES = {
    "codec_math": "CPU legacy-emulation vs GPU B1 fields; optimized token-finite QDQ vs original GPU B1 fields",
    "fp128": "128 fixed nonAIME tokens, Native/H/C1/S3 without QDQ; finite and structural checks; FP errors diagnostic only",
    "qdq32": "32 fixed inputs, all State/codec/writeback/next-consumption checks",
    "qdq128": "128 fixed inputs, all State/codec/writeback/next-consumption checks",
    "qdq512": "512 fixed inputs, fresh process R0/R1 full actual Tensor byte packets and real next-consumption",
    "free64": "Original unchanged nonAIME prompt, Sampler/RNG/EOS, max64; optional manifest-bound fresh pair",
    "context82293": "Live model plus actual short QDQ and synthetic maximum-context native SDPA pressure, not81K trajectory",
    "perf_short": "SameGPU/fixed-input short128-window TPS, genuine QDQ; original sampler included",
    "perf_late": "SameGPU/fixed-input late128-window after3968 warmup TPS, genuine QDQ; original sampler included",
    "profile_short": "Separate short bounded CUDA profiler+QK/GDN/QDQ events; never throughput evidence",
    "profile_late": "Separate late bounded CUDA profiler+QK/GDN/QDQ events; never throughput evidence",
}
CONDITIONS = ("H", "C1", "S3")
PERFORMANCE_CONDITIONS = ("LEGACY_H", *CONDITIONS)
LONG_FIXTURE = "calibration/doc1_tokens_32768.json"
LONG_FIXTURE_SHA256 = "18faa0ca311362181c5c3340d19d72ac39422d1ee07846e516bf34074a51703d"
# Input indices are zero based.  These strict actual-byte checks are outside
# both the first128 prompt timing and the measured TPS/profiler windows.
LATE_NEXT_CONSUMPTION_INPUTS = (129, 511, 1023, 2047, 3071, 3967)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def objsha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def progress(target, phase, **values):
    path = Path(target) / "PROGRESS.json"
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temp.open("wb") as f:
        f.write(canonical(dict(phase=phase, monotonic=time.monotonic(), **values)) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def manifest_template():
    """A reviewable template, never a dispatch or executable permission."""
    units = [dict(id="GPU1_CODEC", gpu=1, kind="codec_math", condition=None),
             dict(id="GPU1_FP128", gpu=1, kind="fp128", condition=None, tokens=128)]
    for gpu in UUIDS:
        for condition in ("C1", "S3"):
            for tokens in (32, 128):
                units.append(dict(id=f"GPU{gpu}_{condition}_{tokens}", gpu=gpu,
                                  kind=f"qdq{tokens}", condition=condition, tokens=tokens))
            ref = f"GPU{gpu}_{condition}_512_R0"
            for repeat in (0, 1):
                units.append(dict(id=f"GPU{gpu}_{condition}_512_R{repeat}", gpu=gpu,
                                  kind="qdq512", condition=condition, tokens=512,
                                  repeat=repeat, reference_unit=ref))
            free_ref = f"GPU{gpu}_{condition}_FREE64_R0"
            for repeat in (0, 1):
                units.append(dict(id=f"GPU{gpu}_{condition}_FREE64_R{repeat}", gpu=gpu,
                                  kind="free64", condition=condition, tokens=64,
                                  repeat=repeat, reference_unit=free_ref))
            units.append(dict(id=f"GPU{gpu}_{condition}_CONTEXT", gpu=gpu,
                              kind="context82293", condition=condition, tokens=16))
    for condition in PERFORMANCE_CONDITIONS:
        for kind, warmup, measured in (("perf_short", 128, 128), ("perf_late", 3968, 128),
                                       ("profile_short", 128, 8), ("profile_late", 3968, 8)):
            if condition == "LEGACY_H" and kind.startswith("profile_"):
                continue
            units.append(dict(id=f"GPU1_{condition}_{kind.upper()}", gpu=1,
                              kind=kind, condition=condition, tokens=warmup + measured,
                              performance_window=dict(warmup=warmup, measured=measured),
                              fixture=LONG_FIXTURE, fixture_sha256=LONG_FIXTURE_SHA256,
                              frozen_fixture_tokens=32768, prefix_only_no_cycle=True))
    for u in units:
        u.update(gpu_uuid=UUIDS[u["gpu"]], status="NOT_RUN")
    return dict(task_id=TASK, status="TEMPLATE_NOT_ADMITTED_NO_DISPATCH",
                mandatory_documentation="Bind fixture SHA, current source freeze, exact unit scopes and resource guards before admission.",
                units=units, formal_dispatch_implemented=False)


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def save_tensor_diagnostic(target, basename, fields, **scope):
    rows, chunks, offset = [], [], 0
    for name, tensor in fields:
        raw = tensor_bytes(tensor)
        rows.append(dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape),
                         offset=offset, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        chunks.append(raw)
        offset += len(raw)
        require(offset <= 1024 ** 3, "First-difference diagnostic exceeds1GiB bound")
    path = target / (basename + ".bin")
    with path.open("xb") as handle:
        for raw in chunks:
            handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_new(target / (basename + ".json"), dict(tensors=rows, bytes=offset,
                sha256=sha(path), **scope))


def metric(actual, expected):
    a, e = actual.double(), expected.double()
    diff = a - e
    return dict(max_abs=float(diff.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(diff) /
                                  torch.linalg.vector_norm(e).clamp_min(1e-30)),
                reference_max=float(e.abs().max()), finite=bool(torch.isfinite(a).all()))


def walk_tensors(value, name="cache", seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        yield name, value
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from walk_tensors(value[key], name + "/" + str(key), seen)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_tensors(child, name + "/" + str(index), seen)
    elif type(value).__module__.startswith("transformers") and hasattr(value, "__dict__"):
        for key in sorted(vars(value)):
            yield from walk_tensors(vars(value)[key], name + "/" + key, seen)


def packet(out, cache, patch):
    fields = [("logits", out.logits)] + list(walk_tensors(cache))
    for layer, encoded in sorted(patch.previous_encoded.items()):
        for name in ("codes", "scales", "negative_zero_flags"):
            value = getattr(encoded, name)
            if value is not None:
                fields.append((f"encoded/{layer}/{name}", value))
    chunks, rows, dedup, offset = [], [], {}, 0
    for name, value in fields:
        raw = tensor_bytes(value)
        digest = hashlib.sha256(raw).hexdigest()
        prior = dedup.get((digest, len(raw)))
        if prior is not None and prior[1] == raw:
            where = prior[0]
        else:
            where = offset
            chunks.append(raw)
            dedup[(digest, len(raw))] = (where, raw)
            offset += len(raw)
        rows.append(dict(name=name, dtype=str(value.dtype), shape=list(value.shape),
                         offset=where, bytes=len(raw), sha256=digest))
    return b"".join(chunks), rows


class FreshReference:
    """One globally owned reference pair; failed comparison data are retained."""
    def __init__(self, root, target, unit):
        import fcntl
        self.root, self.target, self.unit = root, target, unit
        self.ref = unit["reference_unit"]
        self.repeat = unit["repeat"]
        require(self.repeat in (0, 1), "Only fresh R0/R1 are allowed")
        # This authorized continuation owns an independent RAM namespace.
        self.ram = Path("/dev/shm") / root.resolve().name
        self.ram.mkdir(parents=True, exist_ok=True)
        self.lock = (self.ram / "PAIR.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = self.ram / "PAIR_OWNER.json"
        if self.repeat == 0:
            require(not owner.exists(), "Another R0/R1 pair is pending; no overlapping references")
            free = os.statvfs(self.ram)
            require(free.f_bavail * free.f_frsize >= 100 * 1024 ** 3,
                    "Fresh R0 requires80GiB reservation plus20GiB margin")
            atomic_new(owner, dict(reference_unit=self.ref, source_freeze_sha256=sha(root / "SOURCE_FREEZE.json")))
        else:
            require(read(owner) == dict(reference_unit=self.ref, source_freeze_sha256=sha(root / "SOURCE_FREEZE.json")),
                    "Fresh reference pair/source identity mismatch")
        self.path = self.ram / self.ref
        if self.repeat == 0:
            self.path.mkdir(exist_ok=False)
        else:
            old = read(root / "qualification" / self.ref / "TRACE.json")
            require(old["status"] == "REFERENCE_COLLECTED", "Reference trace is not a complete R0")
            self.expected = old["tokens"]
        self.rows, self.total = [], 0

    def step(self, index, out, cache, patch):
        raw, tensors = packet(out, cache, patch)
        row = dict(token=index, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), tensors=tensors)
        path = self.path / f"{index:05d}.bin"
        if self.repeat == 0:
            free = os.statvfs(self.ram)
            require(free.f_bavail * free.f_frsize >= len(raw) + 20 * 1024 ** 3,
                    "Fresh reference physical margin below20GiB")
            self.total += len(raw)
            require(self.total <= 80 * 1024 ** 3, "Fresh reference exceeds80GiB")
            with path.open("xb") as handle:
                handle.write(raw)
        else:
            require(index < len(self.expected), "Fresh R1 longer than R0")
            expected = path.read_bytes()
            if self.expected[index] != row or expected != raw:
                for name, data in (("EXPECTED_FIRST_DIFFERENCE.bin", expected),
                                   ("ACTUAL_FIRST_DIFFERENCE.bin", raw)):
                    with (self.target / name).open("xb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                atomic_new(self.target / "FULL_FIRST_DIFFERENCE.json",
                           dict(token=index, expected=self.expected[index], actual=row))
                raise RuntimeError("Fresh process full actual Tensor values/bytes differ; retained without retry")
        self.rows.append(row)

    def finish(self, expected_count):
        require(len(self.rows) == expected_count, "Incomplete fresh reference trace")
        if self.repeat:
            require(len(self.expected) == expected_count, "Fresh R1 shorter than R0")
        atomic_new(self.target / "TRACE.json",
                   dict(status="PASS_FULL_ACTUAL_TENSOR_BYTES" if self.repeat else "REFERENCE_COLLECTED",
                        tokens=self.rows, reference_unit=self.ref, RAM_path=str(self.path),
                        source_freeze_sha256=sha(self.root / "SOURCE_FREEZE.json"),
                        continuation_state=False))

    def cleanup_after_durable_pass(self):
        if self.repeat:
            require(read(self.target / "RESULT.json")["status"] == "PASS", "Cleanup requires durable unit PASS")
            for index, row in enumerate(self.expected):
                path = self.path / f"{index:05d}.bin"
                require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                        "Reference changed before exact pair cleanup")
            require({p.name for p in self.path.iterdir()} == {f"{i:05d}.bin" for i in range(len(self.expected))},
                    "Unknown file in reference pair; retain")
            for i in range(len(self.expected)):
                (self.path / f"{i:05d}.bin").unlink()
            self.path.rmdir()
            (self.ram / "PAIR_OWNER.json").unlink()
            atomic_new(self.target / "REFERENCE_RELEASED.json", dict(status="PASS", paired_actual_bytes_PASS=True,
                        exact_engineering_reference_only=str(self.path), no_continuation=True))
        self.lock.close()


class Runtime:
    def __init__(self, root, target, gpu, condition=None):
        self.root, self.target, self.gpu = root, target, gpu
        torch.set_num_threads(24)
        torch.set_num_interop_threads(24)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        self.legacy = importlib.import_module("canonical_h128")
        self.legacy.setup_seed(1)
        progress(target, "LOAD_MODEL")
        start = time.monotonic()
        self.model, self.tokenizer = self.legacy.load_model_and_tokenizer()
        self.environment = dict(torch=torch.__version__, CUDA=torch.version.cuda,
                                model_load_seconds=time.monotonic() - start,
                                CVD=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                device_name=torch.cuda.get_device_name(0),
                                capability=list(torch.cuda.get_device_capability(0)),
                                TF32_matmul=torch.backends.cuda.matmul.allow_tf32)
        require(torch.__version__ == "2.5.1+cu121" and torch.version.cuda == "12.1", "Torch/CUDA runtime differs")
        require(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 6), "Single3090 runtime differs")
        require(all(p.dtype == torch.bfloat16 and p.device.type == "cuda" and p.device.index == 0
                    for p in self.model.parameters()), "BF16 singleGPU/no-offload model identity differs")
        self.qmod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        require(sha(self.qmod.__file__) == resident.MODEL_SOURCE_SHA256, "Live Qwen architecture differs")
        config = self.model.config.text_config if hasattr(self.model.config, "text_config") else self.model.config
        require(config._attn_implementation == "sdpa" and config.num_attention_heads == 16
                and config.num_key_value_heads == 4 and config.head_dim == 256
                and config.max_position_embeddings >= 82293, "Full attention/context protocol differs")
        # A legacy baseline must not preload unused C1/S3 deployment matrices.
        self.assets = (None if condition == "LEGACY_H" else
                       resident.ResidentMatrices.from_root(root, device="cuda:0"))
        self.vocab = self.model.get_output_embeddings().weight.shape[0]
        require(self.vocab == 248320, "Vocabulary differs")
        self.last_cache = None
        self.last_out = None
        self.last_input_cache = None
        self.last_attempt_token = None
        progress(target, "MODEL_LOADED")

    def patch(self, condition, qdq, observer=None, audit=True):
        return resident.ResidentRotationPatch(self.model, self.assets, condition, qdq=qdq,
                    observer=observer, audit_next_consumption=audit and qdq, finite_policy="token")

    def forward(self, token, cache):
        self.last_attempt_token, self.last_input_cache = int(token), cache
        expected = 1
        if cache is not None:
            lengths = [layer.keys.shape[-2] for layer in cache.layers
                       if isinstance(getattr(layer, "keys", None), torch.Tensor)]
            require(lengths and len(set(lengths)) == 1, "Attention cache lengths differ before token")
            expected = lengths[0] + 1
        out = self.model(input_ids=torch.tensor([[int(token)]], device="cuda:0"),
                         past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        lengths = [layer.keys.shape[-2] for layer in cache.layers
                   if isinstance(getattr(layer, "keys", None), torch.Tensor)]
        require(lengths and all(x == expected for x in lengths), "Attention cache failed exact1 advancement")
        # Resident token-end hook already checks State/QDQ/Core/Logits once.
        # Keep the original standalone finite verdict for Native/legacy paths.
        if getattr(self.model, "_resident_rotation_owner", None) is None:
            require(bool(torch.isfinite(out.logits).all()), "Nonfinite logits")
        self.last_cache = cache
        self.last_out = out
        return out, cache

    def ids(self, unit, count):
        relative = unit.get("fixture", "calibration/doc_05_tokens.json")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "NonAIME fixture escapes task root")
        require(unit.get("fixture_sha256") == sha(path), "Unit must bind exact frozen nonAIME fixture SHA")
        payload = read(path)
        ids = payload if isinstance(payload, list) else payload.get("tokens", payload.get("input_ids"))
        require(isinstance(ids, list) and len(ids) >= count and all(type(x) is int for x in ids), "Fixture too short or malformed")
        return ids[:count]

    def sampler(self):
        return importlib.import_module("protocol_binding").sampler(self.root, torch)


def assert_audit(audit, tokens, qdq, full_next=True):
    layers = len(resident.GDN_LAYERS)
    require(audit["processed_tokens"] == tokens, "Resident token count differs")
    for key in ("qk_batched_matmul_calls", "original_gdn_helper_calls", "full_token_finite_checks"):
        expected = tokens if key == "full_token_finite_checks" else tokens * layers
        require(audit[key] == expected, "Missing or duplicate resident operation:" + key)
    require(audit["unique_qk_vectors_rotated"] == tokens * layers * 32,
            "Q/K rotations must cover16 uniqueheads each, before repeat")
    require(audit["state_rotation_matmul_calls"] == 0 and not audit["P_inside_QDQ"], "State conversion or duplicatePBR")
    require(audit["qdq_calls"] == (tokens * layers if qdq else 0), "Actual QDQ count differs")
    if qdq and full_next:
        require(audit["next_consumption_full_value_byte_checks"] == (tokens - 1) * layers,
                "Missing full actual resident next-consumption checks")
    require(audit["chunk_prefill_calls"] == layers and
            audit["recurrent_decode_calls"] == (tokens - 1) * layers, "Prefill/decode helper path differs")


def teacher(runtime, unit, reference=None):
    count = unit["tokens"]
    require(count == {"qdq32": 32, "qdq128": 128, "qdq512": 512}[unit["kind"]], "Unexpected qualification length")
    ids = runtime.ids(unit, count)
    counters = {"PRE_QDQ": 0, "POST_QDQ": 0, "NEXT_CONSUMPTION": 0, "PREPARED_QK": 0}
    evidence, prior_writebacks = [], {}

    def observe(event, token, layer, payload):
        counters[event] += 1
        if event == "NEXT_CONSUMPTION":
            actual = tensor_bytes(payload["actual"])
            require(hashlib.sha256(actual).hexdigest() == prior_writebacks[layer],
                    "Actual resident next-consumption bytes differ from last writeback receipt")
        if event == "POST_QDQ":
            require(payload["codes"].dtype == torch.int8 and payload["scales"].dtype == torch.float32,
                    "Actual persistent codec dtypes differ")
            require(payload["writeback"].dtype == torch.float32, "Actual resident writeback dtype differs")
            encoded_rows = []
            for field in ("codes", "scales", "negative_zero_flags", "writeback"):
                value = payload[field]
                raw = tensor_bytes(value)
                digest = hashlib.sha256(raw).hexdigest()
                encoded_rows.append(dict(field=field, dtype=str(value.dtype), shape=list(value.shape),
                                         bytes=len(raw), sha256=digest))
                if field == "writeback":
                    prior_writebacks[layer] = digest
            evidence.append(dict(token=token, layer=layer, fields=encoded_rows,
                                 actual_QDQ_consumption_verified_by_values_and_bytes=True))
    cache = None
    with torch.inference_mode(), runtime.patch(unit["condition"], True, observe) as patch:
        for index, token in enumerate(ids):
            out, cache = runtime.forward(token, cache)
            if reference:
                reference.step(index, out, cache, patch)
            if (index + 1) % 16 == 0:
                progress(runtime.target, "ACTUAL_PERSISTENT_QDQ_TOKEN", token=index + 1, total=count)
        audit = patch.audit()
        assert_audit(audit, count, True)
    if reference:
        reference.finish(count)
    atomic_new(runtime.target / "QDQ_ENCODING_AND_WRITEBACK_EVIDENCE.json",
               dict(status="PASS_ACTUAL_FIELDS_WRITEBACK_AND_FULL_NEXT_CONSUMPTION", tokens=count,
                    layer_tokens=count * 24, next_layer_tokens=(count - 1) * 24,
                    rows=evidence, rawfields_observed_before_return=True))
    require(counters == {"PRE_QDQ": count * 24, "POST_QDQ": count * 24,
                         "NEXT_CONSUMPTION": (count - 1) * 24, "PREPARED_QK": count * 24}, "Observer coverage differs")
    return dict(status="PASS_ACTUAL_PERSISTENT_QDQ", tokens=count, condition=unit["condition"],
                audit=audit, observer_counts=counters, prompt_from_token0=True,
                resident_carrier="FP32 decoded INT8 State; not packedINT8 storage")


def fp128(runtime, unit):
    ids = runtime.ids(unit, 128)
    native_logits, native_states = [], {}
    results = {}
    original_helpers = {name: getattr(runtime.qmod, name) for name in
                        ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule")}
    for condition in ("NATIVE", "H", "C1", "S3"):
        cache, diagnostics = None, []
        if condition == "NATIVE":
            raw = resident.load_torch_helpers()
            for name in original_helpers:
                setattr(runtime.qmod, name, raw[name])
            context = contextlib.nullcontext()
        else:
            context = runtime.patch(condition, False, audit=False)
        try:
            with torch.inference_mode(), context as patch:
                for index, token in enumerate(ids):
                    out, cache = runtime.forward(token, cache)
                    current = out.logits.detach().cpu()
                    if condition == "NATIVE":
                        native_logits.append(current)
                    else:
                        diagnostics.append(dict(token=index, logits_vs_native=metric(current, native_logits[index])))
                    if index in (0, 31, 127):
                        for layer in resident.GDN_LAYERS:
                            state = runtime.legacy.get_state(cache, layer).detach().cpu()
                            require(state.dtype == torch.float32 and bool(torch.isfinite(state).all()), "FP State invalid")
                            if condition == "NATIVE":
                                native_states[(index, layer)] = state
                            else:
                                matrix = runtime.assets.matrix(layer, condition).detach().cpu().double()
                                matrix = matrix.repeat_interleave(2, dim=0)
                                back = torch.matmul(matrix, state.double())
                                diagnostics.append(dict(token=index, layer=layer,
                                                        state_backprojection_vs_native=metric(back, native_states[(index, layer)])))
                    if (index + 1) % 16 == 0:
                        progress(runtime.target, "FP128_FIXED_INPUT", condition=condition, token=index + 1)
                if condition != "NATIVE":
                    audit = patch.audit()
                    assert_audit(audit, 128, False)
                else:
                    audit = dict(native_original_Torch_fallback=True, QDQ=False)
                results[condition] = dict(tokens=128, finite=True, audit=audit, numerical_diagnostics=diagnostics)
        finally:
            if condition == "NATIVE":
                for name, original in original_helpers.items():
                    setattr(runtime.qmod, name, original)
        del cache, out
        runtime.last_cache = None
        gc.collect()
        torch.cuda.empty_cache()
    return dict(status="PASS_STRUCTURAL_AND_FINITE_FP128_DIAGNOSTICS_RECORDED", results=results,
                maxabs_or_relative_error_used_as_hard_gate=False,
                old_4xH_and_2e_6_gates_used=False,
                no_claim_of_bitwise_equality_for_different_floating_graphs=True,
                independent_FP64_math_qualification_required_separately=True)


def codec_math(runtime, unit):
    codec = resident._load_codec()
    generator = torch.Generator(device="cpu").manual_seed(271828)
    fixtures = [("zero", torch.zeros(1, 32, 128, 128)),
                ("negative_zero", torch.full((1, 32, 128, 128), -0.0)),
                ("constant", torch.ones(1, 32, 128, 128))]
    for amplitude in (1e-20, 1e-12, 1e-5, 1., 1e5):
        fixtures.append((str(amplitude), torch.randn((1, 32, 128, 128), generator=generator) * amplitude))
    rows = []
    for name, cpu in fixtures:
        expected = codec.encode(cpu, "B1", cpu_semantics="CUDA_LEGACY_EMULATED")
        source = cpu.to("cuda:0")
        original = codec.encode(source, "B1")
        deployed = source.clone()
        actual, finite = resident.symmetric32_qdq(deployed, defer_finite=True)
        require(bool(finite.item()), "Deferred token finite verdict failed")
        fields = []
        for field in ("codes", "scales", "negative_zero_flags"):
            e, o, a = getattr(expected, field), getattr(original, field), getattr(actual, field)
            if not (tensor_bytes(e) == tensor_bytes(o) == tensor_bytes(a)):
                save_tensor_diagnostic(runtime.target, "CODEC_FIRST_DIFFERENCE",
                    [("expected_CPU_legacy/" + field, e), ("original_GPU/" + field, o),
                     ("resident_GPU/" + field, a), ("source", source)], fixture=name,
                    field=field, scope="Actual original/deployed codec fields; no GPU retry")
                raise RuntimeError("GPU/CPU codec actual field bytes differ:" + name + "/" + field)
            fields.append(dict(field=field, bytes=len(tensor_bytes(a)), sha256=hashlib.sha256(tensor_bytes(a)).hexdigest()))
        require(tensor_bytes(deployed) == tensor_bytes(codec.decode(original)), "Optimized QDQ actual writeback differs")
        rows.append(dict(fixture=name, fields=fields))
    for bad in (float("nan"), float("inf"), -float("inf")):
        source = torch.zeros((1, 32, 128, 128), device="cuda:0")
        source[0, 0, 0, 0] = bad
        _, finite = resident.symmetric32_qdq(source, defer_finite=True)
        require(not bool(finite.item()), "Deferred finite check missed actual nonfinite State")
    return dict(status="PASS_ACTUAL_GPU_CODEC_FIELDS_AND_DEFERRED_FINITE", fixtures=rows,
                original_B1_source_sha256=resident.CODEC_SOURCE_SHA256,
                canonical_scale_floor="clamp_peak_1e-12 then scalar_div127", signed_zero_preserved=True)


def free64(runtime, unit, reference=None):
    fixture = read(runtime.root / "protocol/FREE_CANARY_INPUT.json")
    sampler, ast_records = runtime.sampler()
    sampler.setup_seed(1)
    messages, rendered = sampler.render_input(runtime.tokenizer, fixture["problem"])
    ids = rendered[0].tolist()
    require(ids == fixture["input_ids"], "Actual original nonAIME rendered prompt differs")
    generator = torch.Generator(device="cuda:0").manual_seed(1)
    initial_rng = hashlib.sha256(tensor_bytes(generator.get_state())).hexdigest()
    mask = torch.zeros((1, runtime.vocab), device="cuda:0", dtype=torch.bool)
    warpers, stops = sampler.build_warpers(), sampler.stop_ids(runtime.tokenizer)
    tokens, survivors, cache, forwards = [], [], None, 0
    with torch.inference_mode(), runtime.patch(unit["condition"], True) as patch:
        def step(token):
            nonlocal cache, forwards
            out, cache = runtime.forward(token, cache)
            if reference:
                reference.step(forwards, out, cache, patch)
            forwards += 1
            return out
        for token in ids:
            out = step(token)
        while len(tokens) < 64:
            current, count = sampler.sample_next(warpers, out.logits[:, -1, :], mask, generator)
            mask.scatter_(1, current, True)
            token = int(current.item())
            tokens.append(token)
            survivors.append(count)
            progress(runtime.target, "FREE_ORIGINAL_SAMPLER_TOKEN", generated_tokens=len(tokens), input_forwards=forwards)
            if token in stops or len(tokens) == 64:
                break
            out = step(token)
        audit = patch.audit()
        assert_audit(audit, forwards, True)
    require(forwards == len(ids) + len(tokens) - 1, "Prompt/last-output QDQ timing differs")
    if reference:
        reference.finish(forwards)
    result = dict(status="PASS_ORIGINAL_FREE64_SAMPLER", generated_token_ids=tokens,
                  generated_token_count=len(tokens), input_ids=ids, input_forwards=forwards,
                  finish_reason="eos" if tokens[-1] in stops else "length",
                  stop_token_ids=sorted(stops), initial_rng_sha256=initial_rng,
                  final_rng_sha256=hashlib.sha256(tensor_bytes(generator.get_state())).hexdigest(),
                  original_sampler_ast=ast_records, survivors_min=min(survivors), survivors_max=max(survivors),
                  ties_over20_allowed=True, audit=audit, messages_sha256=objsha(messages),
                  response=runtime.tokenizer.decode(tokens, skip_special_tokens=True))
    if reference and reference.repeat:
        old = read(runtime.root / "qualification" / reference.ref / "RESULT.json")["result"]
        for key in ("generated_token_ids", "input_ids", "input_forwards", "finish_reason",
                    "initial_rng_sha256", "final_rng_sha256"):
            require(result[key] == old[key], "Original sampler/RNG fresh repeat mismatch:" + key)
    return result


def context_pressure(runtime, unit):
    count, total = 16, 82293
    ids = runtime.ids(unit, count)
    cache, held, overlap, expanded = None, [], [], []
    actual_encoded = {}

    def retain_formal_encoded(event, token, layer, payload):
        if event == "POST_QDQ":
            # Mirror the formal sparse-observer's bounded *encoded* retention:
            # one actual GPU INT8/scales/signed-zero set per layer. Never keep
            # an unquantized State, reconstructed reference, or CPU shadow.
            actual_encoded[layer] = tuple(payload[name].detach() for name in
                                          ("codes", "scales", "negative_zero_flags"))

    with torch.inference_mode(), runtime.patch(unit["condition"], True,
            observer=retain_formal_encoded, audit=False) as patch:
        for token in ids:
            out, cache = runtime.forward(token, cache)
        audit = patch.audit()
        assert_audit(audit, count, True, full_next=False)
        require(set(actual_encoded) == set(resident.GDN_LAYERS),
                "Formal-equivalent actual encoded retention lacks a GDN layer")
        for codes, scales, negative_zero in actual_encoded.values():
            require(codes.dtype == torch.int8 and tuple(codes.shape) == (1, 32, 4, 32, 128)
                    and scales.dtype == torch.float32 and tuple(scales.shape) == (1, 32, 4, 128)
                    and negative_zero.dtype == torch.bool and negative_zero.shape == codes.shape
                    and all(value.device.type == "cuda" and value.device.index == 0
                            for value in (codes, scales, negative_zero)),
                    "Formal-equivalent actual encoded dtype/shape/device differs")
        encoded_bytes = sum(value.numel() * value.element_size()
                            for values in actual_encoded.values() for value in values)
        require(encoded_bytes == 26738688, "Actual bounded formal encoded retention is not25.5MiB")
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(8):
                held.extend(torch.zeros((1, 4, total, 256), device="cuda:0", dtype=torch.bfloat16) for _ in range(2))
            overlap = [torch.zeros((1, 4, total + 1, 256), device="cuda:0", dtype=torch.bfloat16) for _ in range(2)]
            expanded = [torch.zeros((1, 16, total + 1, 256), device="cuda:0", dtype=torch.bfloat16) for _ in range(2)]
            query = torch.zeros((1, 16, 1, 256), device="cuda:0", dtype=torch.bfloat16)
            attended = torch.nn.functional.scaled_dot_product_attention(query, *expanded, dropout_p=0., is_causal=False)
            require(bool(torch.isfinite(attended).all()), "Nonfinite maxcontext nativeSDPA")
            torch.cuda.synchronize()
            free, _ = torch.cuda.mem_get_info()
            require(free >= 1024 ** 3, "Maxcontext VRAM remaining below1GiB")
            return dict(status="PASS_SYNTHETIC_LONG_CONTEXT_RESOURCE_PRESSURE", context=total,
                        actual_QDQ_tokens=count, audit=audit, free_VRAM_bytes=free,
                        actual_formal_encoded_retention_bytes=encoded_bytes,
                        actual_formal_encoded_retention_layers=len(actual_encoded),
                        encoded_retention_scope="actual latest GPU codes/scales/negative-zero only; no FP State shadow",
                        synthetic_KV_bytes=sum(t.numel() * t.element_size() for t in held),
                        overlap_expanded_bytes=sum(t.numel() * t.element_size() for t in overlap + expanded),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), numerical81K_proof=False,
                        full81K_GDN_trajectory=False)
        finally:
            held.clear()
            overlap.clear()
            expanded.clear()
            actual_encoded.clear()


class ComponentEvents:
    """Instrumentation only in a separately labelled profiler qualification."""
    def __init__(self, patch):
        self.patch, self.rows, self.restore = patch, [], []
        self.active = False

    def wrap(self, function, label):
        def call(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            value = function(*args, **kwargs)
            end.record()
            self.rows.append((label, start, end))
            return value
        return call

    def __enter__(self):
        for name, label in (("prepare_unique_qk", "resident.QK"), ("symmetric32_qdq", "resident.QDQ")):
            original = getattr(resident, name)
            self.restore.append((resident, name, original))
            setattr(resident, name, self.wrap(original, label))
        for name, function in list(self.patch.helpers.items()):
            if name.startswith("torch_"):
                self.patch.helpers[name] = self.wrap(function, "resident.GDN")
        return self

    def __exit__(self, *_):
        for module, name, original in reversed(self.restore):
            setattr(module, name, original)

    def summary(self):
        torch.cuda.synchronize()
        labels = sorted({row[0] for row in self.rows})
        return {name: dict(calls=sum(row[0] == name for row in self.rows),
                           total_CUDA_ms=sum(start.elapsed_time(end) for label, start, end in self.rows if label == name))
                for name in labels}


def legacy_cached_h_patch(runtime):
    """The old P0 helper wrapper/post-forward B1 graph, private worker only.

    This is a nonAIME performance control, not regeneration of any old sample.
    The canonical sources are copied to this new task and frozen independently.
    """
    base = importlib.import_module("persistent_adapter")

    class CachedH(runtime.legacy.QwenKeyHadamardPatch):
        def __init__(self, model):
            super().__init__(model)
            self.cached_h = runtime.legacy.hadamard(128, dtype=torch.float32, device="cuda:0")

        def _transform(self, value, use_norm):
            value = runtime.legacy.normalize_qk(value) if use_norm else value
            return value.float() @ self.cached_h

    patch = base.PersistentQDQ(runtime.model, "B1", resident.GDN_LAYERS, observe=False)
    patch.h = CachedH(runtime.model)
    return patch


def performance(runtime, unit):
    profile = unit["kind"].startswith("profile_")
    legacy = unit["condition"] == "LEGACY_H"
    require(not (legacy and profile), "NOT_IMPLEMENTED_BLOCKING: legacy H component profiler")
    window = unit["performance_window"]
    warmup, measured = window["warmup"], window["measured"]
    require(warmup == (3968 if unit["kind"].endswith("late") else 128), "Unexpected warmup window")
    require(measured == (8 if profile else 128), "Unexpected measurement window")
    ids = runtime.ids(unit, warmup + measured)
    sampler, _ = runtime.sampler()
    sampler.setup_seed(1)
    generator = torch.Generator(device="cuda:0").manual_seed(1)
    mask = torch.zeros((1, runtime.vocab), dtype=torch.bool, device="cuda:0")
    warpers, cache = sampler.build_warpers(), None
    sparse_inputs = (set(LATE_NEXT_CONSUMPTION_INPUTS)
                     if not legacy and unit["kind"].endswith("late") else set())
    require(not sparse_inputs or max(sparse_inputs) < warmup,
            "Sparse next-consumption checks must precede the measured window")
    capture_inputs = {index - 1 for index in sparse_inputs}
    sparse_rows, expected_writebacks = [], {}

    def observe_warmup(event, token, layer, payload):
        if event == "POST_QDQ" and token in capture_inputs:
            # Preserve exact *actual encoded output* from the preceding input;
            # never re-encode/reconstruct a continuation State as a reference.
            encoded = patch_context.codec.Encoded("B1", tuple(payload["writeback"].shape),
                payload["codes"].detach().cpu().clone(),
                payload["scales"].detach().cpu().clone(), None,
                payload["negative_zero_flags"].detach().cpu().clone())
            patch_context.previous_encoded[layer] = encoded
            raw = tensor_bytes(payload["writeback"])
            expected_writebacks[(token + 1, layer)] = hashlib.sha256(raw).hexdigest()
        elif event == "NEXT_CONSUMPTION" and token in sparse_inputs:
            raw = tensor_bytes(payload["actual"])
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == expected_writebacks[(token, layer)],
                    "Actual long warmup input differs from preceding resident writeback")
            sparse_rows.append(dict(input_index=token, layer=layer, bytes=len(raw),
                actual_state_sha256=digest, actual_values_and_bytes_compared_by_original_codec=True))

    patch_context = (legacy_cached_h_patch(runtime) if legacy else
                     runtime.patch(unit["condition"], True, audit=False))
    # Bind helper timing wrappers before ResidentRotationPatch.__enter__ closes
    # over its original helper functions. Recording stays disabled in warmup.
    event_context = ComponentEvents(patch_context) if profile else contextlib.nullcontext()
    with torch.inference_mode(), event_context as events, patch_context as patch:
        def step(index):
            nonlocal cache
            if sparse_inputs and index < warmup:
                # No observers, CPU tensor transfers, or extra consumption
                # synchronizations are present in either measured window.
                patch.audit_next_consumption = index in sparse_inputs
                patch.observer = observe_warmup if index in (capture_inputs | sparse_inputs) else None
                if index not in sparse_inputs:
                    patch.previous_encoded.clear()
            out, cache = runtime.forward(ids[index], cache)
            if legacy:
                # The historical graph applies QDQ after the full model step.
                # Resident candidates already QDQ before native cache writeback.
                patch.after_token(cache)
            current, _ = sampler.sample_next(warpers, out.logits[:, -1, :], mask, generator)
            mask.scatter_(1, current, True)
        torch.cuda.synchronize()
        prompt_start = time.perf_counter()
        prompt_window = None
        for index in range(warmup):
            step(index)
            if index == 127:
                torch.cuda.synchronize()
                prompt_window = dict(tokens=128, seconds=time.perf_counter() - prompt_start)
                prompt_window["TPS"] = 128 / prompt_window["seconds"]
            if (index + 1) % 32 == 0:
                progress(runtime.target, "PERFORMANCE_WARMUP", token=index + 1, total=warmup)
        torch.cuda.synchronize()
        if sparse_inputs:
            patch.audit_next_consumption = False
            patch.observer = None
            patch.previous_encoded.clear()
            require(len(sparse_rows) == len(sparse_inputs) * 24,
                    "Sparse long next-consumption evidence is incomplete")
            require(patch.counters["next_consumption_full_value_byte_checks"] == len(sparse_inputs) * 24,
                    "Sparse long actual helper-consumption count differs")
        torch.cuda.reset_peak_memory_stats()
        if profile:
            events.active = True
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=False, with_stack=False) as profiler:
                for index in range(warmup, warmup + measured):
                    step(index)
                    profiler.step()
            events.active = False
            timing = events.summary()
            require(set(timing) == {"resident.QK", "resident.GDN", "resident.QDQ"}
                    and all(row["calls"] == measured * 24 for row in timing.values()),
                    "Actual CUDA-event component timing coverage differs")
            profiler.export_chrome_trace(str(runtime.target / "CUDA_PROFILE.json"))
            # Chrome category 'kernel' separates real launches from CUDA copies,
            # allocation events and CPU launch calls. Do not count all CUDA events.
            trace = read(runtime.target / "CUDA_PROFILE.json")
            kernel_rows = [dict(name=event["name"], duration_us=event["dur"])
                           for event in trace["traceEvents"]
                           if event.get("cat") == "kernel" and event.get("ph") == "X"]
            require(kernel_rows, "Profiler did not contain actual CUDA kernel activities")
            labels = {event.key: dict(calls=event.count, CPU_total_us=event.cpu_time_total,
                                     CUDA_total_us=getattr(event, "device_time_total", 0))
                      for event in profiler.key_averages() if event.key in ("resident.QK", "resident.GDN", "resident.QDQ")}
            require(set(labels) == {"resident.QK", "resident.GDN", "resident.QDQ"}, "Missing actual profiler component labels")
            require(all(x["calls"] == measured * 24 for x in labels.values()),
                    "Actual profiler QK/GDN/QDQ layer coverage differs")
            matrix_ops = [dict(operator=event.key, calls=event.count, input_shapes=event.input_shapes)
                          for event in profiler.key_averages(group_by_input_shape=True)
                          if event.key in ("aten::bmm", "aten::matmul")]
            qk_calls = sum(row["calls"] for row in matrix_ops if row["operator"] == "aten::bmm"
                           and row["input_shapes"][:2] == [[16, 2, 128], [16, 128, 128]])
            require(qk_calls == measured * 24, "Actual QK dense rotation must be16 matrix batches perlayer")
            forbidden = [row for row in matrix_ops if len(row["input_shapes"]) >= 2
                         and len(row["input_shapes"][0]) >= 2 and len(row["input_shapes"][1]) >= 2
                         and row["input_shapes"][0][-2:] == [128, 128]
                         and row["input_shapes"][1][-2:] == [128, 128]]
            require(not forbidden, "Forbidden whole128x128 State dense matrix rotation in measured path")
            result = dict(status="PASS_SEPARATE_ACTUAL_CUDA_PROFILE", component_CUDA_events=timing,
                          component_profiler_ranges=labels, actual_CUDA_kernel_count=len(kernel_rows),
                          actual_CUDA_kernels_per_measured_token=len(kernel_rows) / measured,
                          CUDA_kernel_events=kernel_rows, matrix_operator_shapes=matrix_ops,
                          QK16head_bmm_calls=qk_calls, wholeState_dense_rotations_observed=0,
                          TPS=None, profiled_TPS_is_not_throughput=True)
        else:
            begin = time.perf_counter()
            for index in range(warmup, warmup + measured):
                step(index)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            result = dict(status="PASS_ACTUAL_UNPROFILED_FIXED_INPUT_WINDOW", seconds=seconds,
                          TPS=measured / seconds, ms_per_token=seconds * 1000 / measured,
                          all60_hit81920_conditional_generation_GPUh=60 * 81920 / (measured / seconds) / 3600,
                          perUUID20samples_hit81920_conditional_hours=20 * 81920 / (measured / seconds) / 3600,
                          not_free_generation_or81K_speed_guarantee=True)
        audit = patch.audit()
        if legacy:
            total = warmup + measured
            require(audit["tokens"] == total and audit["qdq_layer_updates"] == total * 24
                    and patch.h.calls == total * 24 and patch.h.layers == set(resident.GDN_LAYERS),
                    "Legacy cached-H helper/QDQ coverage differs")
            audit.update(legacy_control=True, QK_rotation_after32head_repeat=True,
                         QDQ_after_full_model_forward=True, H_cached=True,
                         static_H_bytes=patch.h.cached_h.numel() * patch.h.cached_h.element_size())
        else:
            assert_audit(audit, warmup + measured, True, full_next=False)
            require(audit["next_consumption_full_value_byte_checks"] == len(sparse_inputs) * 24,
                    "Unexpected measured-window full-byte observer or missing warmup evidence")
    result.update(condition=unit["condition"], GPU_UUID=UUIDS[runtime.gpu], warmup_tokens=warmup,
                  measured_tokens=measured, actual_original_sampler_invoked=True,
                  fixture_sha256=unit["fixture_sha256"], QDQ_and_finite_checks_included=True,
                  observer_fullbytes_disabled_in_measurement=True, audit=audit,
                  initial_prompt128_window=prompt_window,
                  legacy_H_control=legacy,
                  prompt_definition="q_length1 from token0; warmup includes first-token chunk path",
                  decode_definition="measured recurrent fixed-input forwards plus original sampler; outputs not fed back",
                  continuous_QDQ_input_forwards=warmup + measured,
                  long_sparse_next_consumption=dict(input_indices=sorted(sparse_inputs),
                      checked_layer_inputs=len(sparse_rows), rows=sparse_rows,
                      checks_outside_measured_and_prompt128_windows=True,
                      reference_is_actual_previous_encoded_fields_not_reencoding=True,
                      no_claim_of_all4096_next_inputs_checked=True, numerical81K_proof=False),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--unit")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--list-stages", action="store_true")
    ap.add_argument("--manifest-template", action="store_true")
    args = ap.parse_args(argv)
    if args.list_stages:
        print(json.dumps(dict(task_id=TASK, stages={k: dict(implementation="IMPLEMENTED_REQUIRES_GPU_REVIEW",
                     execution="NOT_RUN", scope=v) for k, v in STAGES.items()},
                     formal_dispatch_implemented=False, auto_launcher_implemented=False), ensure_ascii=False, indent=2))
        return 0
    if args.manifest_template:
        print(json.dumps(manifest_template(), ensure_ascii=False, indent=2))
        return 0
    root = args.root.resolve()
    require(args.unit and args.gpu in UUIDS, "A single manifest unit and authorizedGPU1/2/3 are required")
    permit = read(root / "GPU_PERMIT.json")
    require(permit.get("task_id") == TASK and permit.get("ALLOW_GPU") == "QUALIFICATION_ONLY",
            "GPU not admitted: requires explicit frozen QUALIFICATION_ONLY permit before any CUDA import")
    manifest = read(root / "QUALIFICATION_JOB_MANIFEST.json")
    matches = [u for u in manifest["units"] if u["id"] == args.unit]
    require(len(matches) == 1, "Unit identity not unique in frozen qualification manifest")
    unit = matches[0]
    require(unit["gpu"] == args.gpu and unit["gpu_uuid"] == UUIDS[args.gpu], "Exact unitUUID differs")
    require(unit["kind"] in STAGES and unit.get("condition") in (None, *PERFORMANCE_CONDITIONS), "Unsupported unqualified unit")
    require(unit.get("condition") != "LEGACY_H" or unit["kind"].startswith("perf_"),
            "Legacy H is only a bounded performance control")
    sys.path.insert(0, str(root / "src"))
    guards = importlib.import_module("qualification_guards")
    admission = guards.admit_single_unit(root, args.unit, args.gpu, os.getpid())
    # No Torch import or CUDA call exists above this admission boundary.
    target = Path(admission["target"])
    require(target == root / "qualification" / args.unit and admission["unit"] == unit,
            "Guard returned a different admitted unit")
    target.mkdir(parents=True, exist_ok=True)
    require(not (target / "STARTED.json").exists(), "Unit already STARTED; no worker retry/reset")
    atomic_new(target / "STARTED.json", dict(pid=os.getpid(), identity=admission["identity"],
                monotonic=time.monotonic(), admission=admission["admission"],
                source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"), permit_sha256=sha(root / "GPU_PERMIT.json")))
    reference = None
    runtime = None
    try:
        if unit.get("reference_unit"):
            reference = FreshReference(root, target, unit)
        sys.path.insert(0, str(root / "vendor"))
        global torch, resident
        import torch
        import resident_rotation as resident
        runtime = Runtime(root, target, args.gpu, unit.get("condition"))
        guards.check_after_model(root, args.gpu, os.getpid())
        if unit["kind"] == "codec_math":
            result = codec_math(runtime, unit)
        elif unit["kind"] == "fp128":
            result = fp128(runtime, unit)
        elif unit["kind"].startswith("qdq"):
            result = teacher(runtime, unit, reference)
        elif unit["kind"] == "free64":
            result = free64(runtime, unit, reference)
        elif unit["kind"] == "context82293":
            result = context_pressure(runtime, unit)
        elif unit["kind"].startswith(("perf_", "profile_")):
            result = performance(runtime, unit)
        else:
            raise RuntimeError("NOT_IMPLEMENTED_BLOCKING: " + unit["kind"])
        guards.check_after_model(root, args.gpu, os.getpid())
        atomic_new(target / "RESULT.json", dict(status="PASS", task_id=TASK, unit=unit,
                    result=result, environment=runtime.environment, pid=os.getpid(),
                    monotonic=time.monotonic(), source_freeze_sha256=sha(root / "SOURCE_FREEZE.json"),
                    permit_sha256=sha(root / "GPU_PERMIT.json"), formal_permission=False))
        if reference:
            reference.cleanup_after_durable_pass()
        progress(target, "UNIT_PASS")
        return 0
    except BaseException as exc:
        if runtime is not None:
            try:
                cache = runtime.last_cache if runtime.last_cache is not None else runtime.last_input_cache
                fields = list(walk_tensors(cache)) if cache is not None else []
                if runtime.last_out is not None:
                    fields.append(("last_completed_forward_logits", runtime.last_out.logits))
                if fields:
                    save_tensor_diagnostic(target, "ACTUAL_CACHE_AT_FAILURE", fields,
                        scope="Actual possibly partially advanced cache at failure; logits are last COMPLETED forward, not fabricated current output",
                        last_attempt_token=runtime.last_attempt_token, continuation_allowed=False)
            except BaseException as diagnostic_error:
                atomic_new(target / "DIAGNOSTIC_FAILURE.json", dict(error=repr(diagnostic_error)))
        atomic_new(target / "FIRST_FAILURE.json", dict(status="QUALIFICATION_FAIL_CLOSED_STOP",
                    unit=unit, error=repr(exc), traceback=traceback.format_exc(), pid=os.getpid(),
                    monotonic=time.monotonic(), no_retry=True,
                    RAM_reference_retained=bool(reference), formal_permission=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
