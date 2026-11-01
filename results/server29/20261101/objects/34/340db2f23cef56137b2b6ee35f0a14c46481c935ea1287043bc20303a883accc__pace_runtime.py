"""Scoped PACE integration over the pinned native Qwen model and sampler.

Importing this module does not import torch, load weights, or touch CUDA.  The
scientific conventions without official-code verification are recorded in the
paper audit and method specification, not presented as official implementation.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

TASK = "QWEN_GDN_PACE48_AIME26_SEED1_60_V1"
GDN_LAYERS = tuple(i for i in range(32) if i % 4 != 3)
NATIVE_SHA256 = "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a"
CONDITIONS = {"R2": 8, "R3": 1}
MARKER_MAGIC = 48256.0


def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _progress(callback, phase, **values):
    if callback is not None:
        callback(phase, **values)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_vendor(root):
    root = Path(root)
    manifest = json.loads((root / "operations/MINIMAL_PINNED_RUNTIME_VENDOR.json").read_text())
    for row in manifest["files"]:
        path = root / row["path"]
        _require(path.is_file() and path.stat().st_size == row["bytes"]
                 and file_sha(path) == row["sha256"], "Pinned vendor asset differs: " + str(path))
    return file_sha(root / "operations/MINIMAL_PINNED_RUNTIME_VENDOR.json")


def _layers(model):
    return model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers


def _condition(value):
    aliases = {"R2_PACE48_DP8": "R2", "R3_PACE48_DP1": "R3",
               "R2_PACE48_dP8": "R2", "R3_PACE48_dP1": "R3"}
    value = aliases.get(value, value)
    _require(value in CONDITIONS, "Only R2 and R3 are admitted")
    return value


def load_head_mask(path, layer_ids=GDN_LAYERS):
    path = Path(path)
    obj = json.loads(path.read_text())
    _require(obj.get("selection_decode_period") == 8, "Mask must be selected once with dP8")
    _require(obj.get("head_count") == 32 and obj.get("int8_per_layer") == 8,
             "Mask must allocate exactly eight of32 INT8 heads per layer")
    rows = obj["layers"]
    _require(set(rows) == {str(i) for i in layer_ids}, "Head-mask layer roster differs")
    masks = {}
    for layer in layer_ids:
        row = rows[str(layer)]
        high = row["int8_heads"]
        _require(len(high) == 8 and len(set(high)) == 8 and all(type(h) is int and 0 <= h < 32 for h in high),
                 "Invalid INT8 head allocation")
        low = sorted(set(range(32)) - set(high))
        _require(row.get("int4_heads", low) == low, "INT4 head roster differs")
        masks[layer] = [h in high for h in range(32)]
    return masks, file_sha(path), obj


class NativeLayerTracker:
    """Only label actual linear-attention invocations; never rotate q/k."""
    def __init__(self, model, layer_ids):
        self.model, self.layer_ids = model, tuple(layer_ids)
        self.current_layer = None
        self.handles = []

    def __enter__(self):
        layers = _layers(self.model)
        for index in self.layer_ids:
            module = getattr(layers[index], "linear_attn", None)
            _require(module is not None, "Expected linear attention at layer " + str(index))
            self.handles.append(module.register_forward_pre_hook(
                lambda _m, _args, i=index: setattr(self, "current_layer", i)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _bind(kind, args, kwargs):
    names = (["g", "beta", "initial_state", "output_final_state"] if kind == "recurrent"
             else ["g", "beta", "chunk_size", "initial_state", "output_final_state"])
    _require(len(args) <= len(names), "Unexpected native positional signature")
    values = dict(kwargs)
    for name, value in zip(names, args):
        _require(name not in values, "Duplicate native argument: " + name)
        values[name] = value
    return values


class PACEAdapter:
    """Own packed state; the native recurrent cache contains only a tiny marker."""
    def __init__(self, model, native_module, masks, decode_period, prompt_length,
                 *, buffer_factory=None, observe=True):
        if buffer_factory is None:
            from pace_buffer import PACEBuffer
            buffer_factory = PACEBuffer
        self.model, self.native = model, native_module
        self.layers = tuple(sorted(masks))
        self.tracker = NativeLayerTracker(model, self.layers)
        self.buffers = {}
        self.buffer_factory, self.masks = buffer_factory, masks
        self.decode_period, self.prompt_length = decode_period, prompt_length
        self.observe = observe
        self.saved = []
        self.marker_checks = 0
        self.helper_calls = {layer: 0 for layer in self.layers}
        self.native_first_chunk_compat_calls = {layer: 0 for layer in self.layers}
        self.native_first_chunk_state_checked = {layer: 0 for layer in self.layers}
        self.native_first_chunk_max_state_abs_error = 0.0

    def __enter__(self):
        _require(not getattr(self.native, "_pace48_active", False), "Nested PACE helper patch")
        self.native._pace48_active = True
        try:
            self.tracker.__enter__()
            for kind, name in (("recurrent", "torch_recurrent_gated_delta_rule"),
                               ("chunk", "torch_chunk_gated_delta_rule")):
                original = getattr(self.native, name)
                self.saved.append((name, original))
                setattr(self.native, name, self._wrap(kind, original))
        except BaseException:
            self.__exit__()
            raise
        return self

    def _wrap(self, kind, original):
        def call(query, key, value, *args, **kwargs):
            import torch
            options = _bind(kind, args, kwargs)
            layer = self.tracker.current_layer
            _require(layer in self.helper_calls, "Native helper called without an admitted layer")
            _require(query.ndim == 4 and query.shape[1] == 1
                     and query.shape == key.shape, "PACE frozen runtime requires one-token q/k")
            _require(query.dtype == key.dtype == value.dtype == torch.bfloat16,
                     "Native q/k/value BF16 contract differs")
            _require(query.shape[2] == value.shape[2] == 32,
                     "Native q/k repeat to32 heads must occur before the helper")
            _require(options.get("use_qk_l2norm_in_kernel") is True,
                     "Native core must request q/k normalization exactly once")
            _require(options.get("output_final_state") is True, "PACE requires native cache enabled")
            g, beta = options["g"], options["beta"]
            _require(g.dtype == torch.float32 and beta.dtype == torch.bfloat16,
                     "Native log-decay/beta precision differs")
            if layer not in self.buffers:
                mask = torch.tensor(self.masks[layer], dtype=torch.bool, device=query.device)
                buffer = self.buffer_factory(self.decode_period, mask, prefill_period=256)
                buffer.set_prompt_length(self.prompt_length)
                self.buffers[layer] = buffer
            buffer = self.buffers[layer]
            state = options.get("initial_state")
            if buffer.tokens:
                _require(isinstance(state, torch.Tensor) and tuple(state.shape) == (2,)
                         and state.dtype == torch.float32 and state.device == query.device,
                         "Native cache retained an invalid or dense recurrent carrier")
                if self.observe and (buffer.tokens < 128 or buffer.tokens % 2048 == 0):
                    expected = torch.tensor([MARKER_MAGIC, float(buffer.tokens)],
                                            dtype=torch.float32, device=query.device)
                    _require(torch.equal(state, expected), "Native recurrent cache marker changed")
                    self.marker_checks += 1
            else:
                _require(state is None, "Initial PACE call must start from an empty native cache")
                _require(kind == "chunk", "The actual first empty-cache helper must be native chunk")
            _require(kind != "chunk" or buffer.tokens == 0,
                     "Native chunk helper is admitted only for the initial empty-cache token")
            native_first_out = None
            if kind == "chunk" and buffer.tokens == 0:
                # Native empty-cache T1 uses a padded chunk matmul readout.
                # Its BF16 rounding can differ from sum(updated_state*q).
                # Keep that one original readout, while the production buffer
                # still consumes the token and follows its unchanged periods.
                native_first_out, native_first_state = original(query, key, value, *args, **kwargs)
                normalized_key = self.native.l2norm(key[:, 0], dim=-1, eps=1e-6).float()
                first_delta = value[:, 0].float() * beta[:, 0].float().unsqueeze(-1)
                first_state = normalized_key.unsqueeze(-1) * first_delta.unsqueeze(-2)
                torch.testing.assert_close(native_first_state, first_state, atol=1e-5, rtol=1e-5)
                error = float((native_first_state - first_state).abs().max().item())
                self.native_first_chunk_max_state_abs_error = max(self.native_first_chunk_max_state_abs_error, error)
                self.native_first_chunk_compat_calls[layer] += 1
                self.native_first_chunk_state_checked[layer] += 1
                # These are transient checks only: never put a native dense
                # state in a cache, buffer, adapter attribute, or packed base.
                del native_first_state, first_state, first_delta, normalized_key
            out = buffer.step(query[:, 0], key[:, 0], value[:, 0], beta[:, 0], g[:, 0],
                              scale=1.0 / math.sqrt(query.shape[-1]))
            if native_first_out is not None:
                out = native_first_out[:, 0]
            _require(tuple(out.shape) == tuple(value[:, 0].shape) and out.dtype == query.dtype,
                     "PACE helper output shape/dtype differs")
            self.helper_calls[layer] += 1
            marker = torch.tensor([MARKER_MAGIC, float(buffer.tokens)],
                                  dtype=torch.float32, device=value.device)
            return out.unsqueeze(1), marker
        return call

    def assert_cache(self, cache, expected_tokens):
        import torch
        _require(set(self.buffers) == set(self.layers), "Missing actual GDN layer calls")
        for layer in self.layers:
            b = self.buffers[layer]
            _require(b.tokens == expected_tokens and self.helper_calls[layer] == expected_tokens,
                     "PACE per-layer token count differs")
            state = cache.layers[layer].recurrent_states[0]
            _require(isinstance(state, torch.Tensor) and tuple(state.shape) == (2,)
                     and state.dtype == torch.float32, "Dense recurrent state persisted in native cache")
            _require(not b.fixture_only and getattr(b, "_fp_state", None) is None,
                     "Unquantized fixture state entered the production adapter")
            for name, tensor in b.resident_tensors().items():
                _require(not (tensor.is_floating_point() and tuple(tensor.shape) == tuple(b._shape)),
                         "Dense floating-point state persisted in buffer: " + name)
        return True

    def audit(self):
        rows = {str(layer): self.buffers[layer].storage_report() for layer in self.layers}
        _require(all(self.native_first_chunk_compat_calls[layer] == 1 and
                     self.native_first_chunk_state_checked[layer] == 1 for layer in self.layers),
                 "Every layer must use one checked native empty-cache first-chunk readout")
        return dict(layers=list(self.layers), helper_calls={str(k): v for k, v in self.helper_calls.items()},
                    native_first_chunk_compat_calls={str(k): v for k, v in self.native_first_chunk_compat_calls.items()},
                    native_first_chunk_state_checked={str(k): v for k, v in self.native_first_chunk_state_checked.items()},
                    native_first_chunk_max_state_abs_error=self.native_first_chunk_max_state_abs_error,
                    native_first_chunk_state_retained=False,
                    native_first_chunk_state_discarded=True,
                    marker_checks=self.marker_checks, native_cache_carrier="two_element_FP32_marker",
                    dense_recurrent_carrier=False, storage_by_layer=rows,
                    decode_period=self.decode_period, prefill_period=256,
                    output_before_state_writeback=True, actual_layer_coverage=len(self.buffers),
                    native_marker_tensor_nbytes=8*len(self.layers),
                    total_state_resident_tensor_nbytes=8*len(self.layers)+sum(
                        row["total_resident_tensor_nbytes"] for row in rows.values()))

    def __exit__(self, *_):
        for name, original in reversed(self.saved):
            setattr(self.native, name, original)
        self.saved.clear()
        self.tracker.__exit__()
        self.native._pace48_active = False


def _load_frozen_model(root):
    import torch
    root = Path(root)
    verify_vendor(root)
    _require(torch.__version__ == "2.5.1+cu121" and torch.version.cuda == "12.1",
             "Frozen Torch/CUDA runtime differs")
    torch.set_num_threads(24)
    torch.set_num_interop_threads(24)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    loader = _module(root / "vendor/canonical_h128.py", "pace48_frozen_loader")
    loader.setup_seed(1)
    model, tokenizer = loader.load_model_and_tokenizer()
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    _require(file_sha(native.__file__) == NATIVE_SHA256, "Actual native architecture differs")
    _require(all(p.dtype == torch.bfloat16 and p.device.type == "cuda" and p.device.index == 0
                 for p in model.parameters()), "Model must be BF16 on single visible CUDA0")
    _require(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (8, 6),
             "Frozen visible singleGPU/capability differs")
    actual = tuple(i for i, layer in enumerate(_layers(model)) if getattr(layer, "linear_attn", None) is not None)
    _require(len(_layers(model)) == 32 and actual == GDN_LAYERS, "Actual32/24 layer mapping differs")
    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    _require(config._attn_implementation == "sdpa" and config.max_position_embeddings >= 82293,
             "Frozen attention/context differs")
    _require(model.get_output_embeddings().weight.shape[0] == 248320, "Actual vocabulary differs")
    return model, tokenizer, native


def load_native_reference(root):
    """Load actual pinned native helpers for tiny qualification; no weights."""
    import torch
    root = Path(root)
    verify_vendor(root)
    loader = _module(root / "vendor/canonical_h128.py", "pace48_reference_loader")
    for path in (str(loader.STATE_QUANT_SRC), str(loader.TRANSFORMERS_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    _require(file_sha(native.__file__) == NATIVE_SHA256, "Actual native reference differs")
    _require(not getattr(native, "_pace48_active", False), "Native reference helper is already patched")
    recurrent = native.torch_recurrent_gated_delta_rule
    chunk = native.torch_chunk_gated_delta_rule

    def reference(q, k, v, beta, g, initial_state=None, *, kind="recurrent",
                  use_qk_l2norm_in_kernel=True, output_final_state=True):
        single = q.ndim == 3
        if single:
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            beta, g = beta.unsqueeze(1), g.unsqueeze(1)
        _require(q.ndim == 4 and kind in ("recurrent", "chunk"), "Invalid native reference request")
        helper = recurrent if kind == "recurrent" else chunk
        out, state = helper(q, k, v, g=g, beta=beta, initial_state=initial_state,
                            output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        return (out[:, 0] if single else out), state
    reference.source_identity = dict(path=str(native.__file__), sha256=NATIVE_SHA256,
                                     actual_installed_native=True, model_weights_loaded=False)
    reference.source_sha256 = NATIVE_SHA256
    reference.native_module = native
    return reference


def run_math_device_gates(root, progress=None):
    import torch
    import traceback
    root = Path(root)
    callback = load_native_reference(root)
    # Match the already frozen full-model loader flags before chunk matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    _progress(progress, "NATIVE_REFERENCE_READY", native_sha256=NATIVE_SHA256)
    tests = _module(root / "tests/test_pace_math.py", "pace48_actual_device_gates")
    result = tests.run_device_gates(root / "validation", device="cuda:0", progress=progress,
                                   native_reference=callback)
    if isinstance(result, dict):
        result["native_reference_identity"] = callback.source_identity
        if result["status"] == "PASS":
            diagnostic = _module(root / "tests/first_chunk_adapter_parity.py", "pace48_first_chunk_actual_device")
            try:
                first_chunk = diagnostic.run_adapter_parity(
                    root, device="cuda:0", native_module=callback.native_module,
                    native_identity=dict(callback.source_identity, kind="ACTUAL_INSTALLED_NATIVE_HELPERS"),
                    progress=progress)
            except BaseException as exc:
                first_chunk = dict(status="FAIL_NATIVE_FIRST_CHUNK_ADAPTER_PARITY", actual_cuda=True,
                                   error=repr(exc), traceback=traceback.format_exc())
            first_chunk["backend_flags"] = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
            path = root / "validation/fp_first_chunk_adapter_parity.json"
            with path.open("x") as stream:
                json.dump(first_chunk, stream, sort_keys=True, indent=2)
                stream.write("\n")
            result["gates"]["B"]["native_first_chunk_subgate"] = dict(
                status=first_chunk["status"], path=str(path), sha256=file_sha(path))
            if first_chunk["status"] != "PASS":
                result["status"] = result["gates"]["B"]["status"] = "FAIL"
                result["GPU_qualification_credit"] = False
            with (root / "validation/fp_buffer_parity.json").open("w") as stream:
                json.dump(result["gates"]["B"], stream, sort_keys=True, indent=2)
                stream.write("\n")
    return result


class PACEEngine:
    def __init__(self, root, condition, *, model, tokenizer, native_module,
                 head_mask_path=None, masks=None, device="cuda:0", sampler_functions=None,
                 progress=None, require_native_pin=True):
        import torch
        self.root, self.condition = Path(root), _condition(condition)
        self.model, self.tokenizer, self.native = model, tokenizer, native_module
        self.device, self.progress = torch.device(device), progress
        self.layers = tuple(i for i, layer in enumerate(_layers(model)) if getattr(layer, "linear_attn", None) is not None)
        self.head_mask_path = Path(head_mask_path or self.root / "calibration/head_mask.json")
        if masks is None:
            masks, self.head_mask_sha, _ = load_head_mask(self.head_mask_path, self.layers)
        else:
            self.head_mask_sha = canonical_sha(masks)
        self.masks = masks
        if require_native_pin:
            _require(file_sha(native_module.__file__) == NATIVE_SHA256, "Actual native architecture differs")
        if sampler_functions is None:
            binding = _module(self.root / "vendor/protocol_binding.py", "pace48_protocol_binding")
            self.f, self.sampler_ast = binding.sampler(self.root, torch)
        else:
            self.f, self.sampler_ast = sampler_functions, []

    @classmethod
    def from_frozen_loader(cls, root, condition, **kwargs):
        _progress(kwargs.get("progress"), "LOAD_MODEL")
        model, tokenizer, native = _load_frozen_model(root)
        engine = cls(root, condition, model=model, tokenizer=tokenizer, native_module=native, **kwargs)
        _progress(kwargs.get("progress"), "MODEL_LOADED")
        return engine

    def generate(self, input_ids, *, problem_id, max_new_tokens=81920):
        import torch
        _require(isinstance(input_ids, list) and input_ids and all(type(x) is int for x in input_ids),
                 "Generation needs explicit frozen input ids")
        _require(type(max_new_tokens) is int and max_new_tokens > 0, "Invalid generation cap")
        self.f.setup_seed(1)
        generator = torch.Generator(device=self.device).manual_seed(1)
        initial_generator_sha = hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
        warpers, stops = self.f.build_warpers(), self.f.stop_ids(self.tokenizer)
        vocab = self.model.get_output_embeddings().weight.shape[0]
        mask = torch.zeros((1, vocab), dtype=torch.bool, device=self.device)
        tokens, survivors, cache = [], [], None
        total, start = 0, time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        adapter = PACEAdapter(self.model, self.native, self.masks, CONDITIONS[self.condition], len(input_ids))

        def step(token):
            nonlocal cache, total
            previous_attention_lengths = ([] if cache is None else
                [layer.keys.shape[-2] for layer in cache.layers
                 if isinstance(getattr(layer, "keys", None), torch.Tensor)])
            out = self.model(input_ids=torch.tensor([[token]], dtype=torch.long, device=self.device),
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            total += 1
            _require(bool(torch.isfinite(out.logits).all()), "Nonfinite model logits")
            adapter.assert_cache(cache, total)
            actual = [layer.keys.shape[-2] for layer in cache.layers
                      if isinstance(getattr(layer, "keys", None), torch.Tensor)]
            if previous_attention_lengths:
                _require(len(set(previous_attention_lengths)) == 1
                         and actual and all(x == previous_attention_lengths[0] + 1 for x in actual),
                         "Native full-attention cache did not advance exactly one token")
            if total % 128 == 0:
                _progress(self.progress, "FORWARD_COMMITTED", input_forwards=total,
                          generated_tokens=len(tokens), problem_id=problem_id, condition=self.condition)
            return out

        with torch.inference_mode(), adapter:
            for token in input_ids:
                out = step(token)
            prefill_seconds = time.monotonic() - start
            while len(tokens) < max_new_tokens:
                current, count = self.f.sample_next(warpers, out.logits[:, -1, :], mask, generator)
                mask.scatter_(1, current, True)
                token = int(current.item())
                tokens.append(token)
                survivors.append(int(count))
                if len(tokens) % 128 == 0 or token in stops or len(tokens) == max_new_tokens:
                    _progress(self.progress, "GENERATED_TOKENS_COMMITTED", input_forwards=total,
                              generated_tokens=len(tokens), problem_id=problem_id, condition=self.condition)
                if token in stops or len(tokens) == max_new_tokens:
                    break
                out = step(token)
            _require(total == len(input_ids) + len(tokens) - 1, "Prompt/decode forward count differs")
            for buffer in adapter.buffers.values():
                buffer.finalize(policy="preserve")
            audit = adapter.audit()
        elapsed = time.monotonic() - start
        gpu_memory = (dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
                      if self.device.type == "cuda" else {})
        result = dict(status="COMPLETE", task_id=TASK, condition=self.condition,
                      configuration="R2_PACE48_DP8" if self.condition == "R2" else "R3_PACE48_DP1",
                      seed=1, problem_id=problem_id, input_ids_sha256=self.f.ids_sha256(input_ids),
                      input_tokens=len(input_ids), generated_token_ids=tokens,
                      generated_token_ids_sha256=canonical_sha(tokens), generated_token_count=len(tokens),
                      response=self.tokenizer.decode(tokens, skip_special_tokens=True),
                      finish_reason="eos" if tokens[-1] in stops else "length",
                      stop_token_ids=sorted(stops), wall_seconds=elapsed, prompt_seconds=prefill_seconds,
                      decode_and_sampling_seconds=elapsed-prefill_seconds, TPS=len(tokens)/elapsed,
                      actual_model_forward_calls=total, sampler_calls=len(tokens), nonfinite=False,
                      runtime_error=None, head_mask_sha256=self.head_mask_sha,
                      quantizer_audit=audit, memory=gpu_memory, **gpu_memory,
                      initial_generator_sha256=initial_generator_sha,
                      final_generator_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
                      sampling_audit=dict(top_k_parameter=20, min_survivors=min(survivors),
                                          max_survivors=max(survivors), original_warper_unchanged=True),
                      protocol_identity={name: file_sha(self.root / "protocol" / name)
                                         for name in ("sampler.py", "common.py", "aime26_scorer_v4.py", "aime26_frozen.jsonl")},
                      actual_native_sha256=NATIVE_SHA256,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      implementation_identity="UNVERIFIED_AGAINST_OFFICIAL_CODE")
        del cache, out, adapter
        gc.collect()
        return result


class CalibrationTap:
    """Measure native FP states; Q4 error observation never changes the carrier."""
    def __init__(self, model, native_module, *, layer_ids=GDN_LAYERS):
        self.model, self.native, self.layers = model, native_module, tuple(layer_ids)
        self.tracker = NativeLayerTracker(model, self.layers)
        self.tokens = {i: 0 for i in self.layers}
        self.decay_sums, self.error_sums = {}, {}
        self.decay_counts = {i: 0 for i in self.layers}
        self.error_counts = {i: 0 for i in self.layers}
        self.saved = []

    def __enter__(self):
        _require(not getattr(self.native, "_pace48_active", False), "Calibration nested inside PACE")
        self.native._pace48_active = True
        try:
            self.tracker.__enter__()
            for kind, name in (("recurrent", "torch_recurrent_gated_delta_rule"),
                               ("chunk", "torch_chunk_gated_delta_rule")):
                original = getattr(self.native, name)
                self.saved.append((name, original))
                setattr(self.native, name, self._wrap(kind, original))
        except BaseException:
            self.__exit__()
            raise
        return self

    def _wrap(self, kind, original):
        def call(q, k, v, *args, **kwargs):
            import torch
            from pace_quantizer import PackedAffineState
            values = _bind(kind, args, kwargs)
            layer = self.tracker.current_layer
            _require(q.shape[1] == 1 and q.shape[2] == 32, "Calibration requires frozen one-token32-head helper")
            out, state = original(q, k, v, *args, **kwargs)
            _require(state is not None and state.dtype == torch.float32,
                     "Calibration must observe original FP32 native state")
            alpha = values["g"].detach().float().exp().double().sum(dim=(0, 1))
            self.decay_sums[layer] = self.decay_sums.get(layer, torch.zeros_like(alpha)) + alpha
            self.decay_counts[layer] += q.shape[0]
            self.tokens[layer] += 1
            if self.tokens[layer] % 256 == 0:
                low = torch.zeros(32, dtype=torch.bool, device=state.device)
                encoded = PackedAffineState.from_tensor(state, low, group_size=32)
                recovered = encoded.dequantize(dtype=torch.float32)
                sse = (state.double() - recovered.double()).square().sum(dim=(0, 2, 3))
                self.error_sums[layer] = self.error_sums.get(layer, torch.zeros_like(sse)) + sse
                self.error_counts[layer] += 1
            return out, state
        return call

    def __exit__(self, *_):
        for name, original in reversed(self.saved):
            setattr(self.native, name, original)
        self.saved.clear()
        self.tracker.__exit__()
        self.native._pace48_active = False


def _write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rank_head_scores(totals, layer_ids=GDN_LAYERS):
    """Aggregate energies/decays first, then rank with fixed selection dP8."""
    rows, masks = [], {}
    for layer in layer_ids:
        total = totals[layer]
        _require(total["error_count"] == 64 and total["decay_count"] == 16384,
                 "Calibration must aggregate64 state samples and16384 decays per head")
        layer_rows = []
        for head in range(32):
            e = total["error"][head] / 64.0
            a = total["decay"][head] / 16384.0
            _require(math.isfinite(e) and e >= 0 and math.isfinite(a) and 0 < a < 1,
                     "Invalid native decay/error aggregate; no denominator fallback")
            denominator = -math.expm1(16.0 * math.log(a))
            score = e / denominator
            _require(math.isfinite(score), "Nonfinite head score")
            layer_rows.append(dict(layer=layer, head=head, e_h=e, a_h=a,
                                   denominator=denominator, E_h=score,
                                   error_samples=64, decay_samples=16384))
        high = sorted(x["head"] for x in sorted(layer_rows, key=lambda x: (-x["E_h"], x["head"]))[:8])
        masks[str(layer)] = dict(int8_heads=high, int4_heads=sorted(set(range(32))-set(high)))
        for row in layer_rows:
            row["selected_bits"] = 8 if row["head"] in high else 4
        rows.extend(layer_rows)
    return rows, masks


def collect_calibration(model, tokenizer, native_module, root, windows, *, manifest_path,
                        progress=None, output_dir=None, layer_ids=GDN_LAYERS,
                        input_verification=None):
    import torch
    root = Path(root)
    output_dir = Path(output_dir or root / "calibration")
    _require(len(windows) == 8 and all(len(x) == 2048 for x in windows),
             "Calibration must contain eight independent2048-token windows")
    _require(len({canonical_sha(x) for x in windows}) == 8, "Duplicate calibration windows")
    _require(all(all(type(t) is int for t in x) for x in windows), "Invalid calibration token ids")
    for name in ("head_scores.csv", "head_mask.json", "calibration_manifest.json"):
        _require(not (output_dir / name).exists(), "Calibration output already exists; no rerun: " + name)
    totals = {i: dict(error=[0.0]*32, decay=[0.0]*32, error_count=0, decay_count=0) for i in layer_ids}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, ids in enumerate(windows):
            cache = None
            with CalibrationTap(model, native_module, layer_ids=layer_ids) as tap:
                for token_index, token in enumerate(ids):
                    out = model(input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                                past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    _require(bool(torch.isfinite(out.logits).all()), "Nonfinite FP calibration logits")
                    if (token_index + 1) % 256 == 0:
                        _progress(progress, "CALIBRATION_BOUNDARY", window=index+1, tokens=token_index+1)
            for layer in layer_ids:
                _require(tap.tokens[layer] == 2048 and tap.error_counts[layer] == 8
                         and tap.decay_counts[layer] == 2048, "Calibration boundary/token coverage differs")
                row = totals[layer]
                for key, tensor in (("error", tap.error_sums[layer]), ("decay", tap.decay_sums[layer])):
                    values = tensor.cpu().tolist()
                    row[key] = [a+b for a, b in zip(row[key], values)]
                row["error_count"] += tap.error_counts[layer]
                row["decay_count"] += tap.decay_counts[layer]
            del cache, out, tap
            gc.collect()
    rows, masks = rank_head_scores(totals, layer_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "head_scores.csv"
    with score_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mask = dict(schema="PACE48_HEAD_MASK_V1", selection_decode_period=8, head_count=32,
                int8_per_layer=8, layers=masks, input_manifest_sha256=file_sha(manifest_path),
                head_scores_sha256=file_sha(score_path), tie_break="ascending_head_index",
                implementation_identity="UNVERIFIED_AGAINST_OFFICIAL_CODE")
    mask_path = output_dir / "head_mask.json"
    _write_new_json(mask_path, mask)
    manifest = dict(status="PASS", actual_cuda=device.type == "cuda", windows=8,
                    tokens_per_window=2048, state_boundaries=list(range(256, 2049, 256)),
                    error_samples_per_head=64, decay_samples_per_head=16384,
                    e_definition="mean single-Q4 SSE of originalFP states; no quantization carry",
                    a_definition="arithmetic mean nativeFP32 exp(g) across all windows/tokens",
                    score_definition="e_h/(1-a_h**16)", selection_decode_period=8,
                    error_reduction="FP64 squared difference and sum per head", decay_reduction="nativeFP32 exp then FP64 sum",
                    input_manifest_sha256=file_sha(manifest_path),
                    window_token_sha256=[canonical_sha(x) for x in windows],
                    head_mask_sha256=file_sha(mask_path), head_scores_sha256=file_sha(score_path),
                    native_sha256=NATIVE_SHA256, AIME_used_for_calibration=False,
                    actual_tokenizer_prefix_verification=input_verification,
                    implementation_identity="UNVERIFIED_AGAINST_OFFICIAL_CODE")
    _write_new_json(output_dir / "calibration_manifest.json", manifest)
    return manifest


def _load_windows(root):
    root = Path(root)
    candidates = [root / "protocol/calibration_windows.json", root / "calibration/INPUTS.json"]
    path = next((p for p in candidates if p.is_file()), None)
    _require(path is not None, "Frozen non-AIME calibration windows missing")
    manifest = json.loads(path.read_text())
    rows = manifest["windows"] if isinstance(manifest, dict) else manifest
    windows = []
    for row in rows:
        if isinstance(row, list):
            ids = row
        elif "input_ids" in row:
            ids = row["input_ids"]
        elif "tokens" in row:
            ids = row["tokens"]
        else:
            p = Path(row.get("token_path", row.get("input_ids_path", "")))
            if not p.is_absolute():
                p = root / p
            ids = json.loads(p.read_text())
        windows.append(ids)
    return windows, path


def run_calibration(root, progress=None):
    windows, path = _load_windows(root)
    _progress(progress, "LOAD_MODEL")
    model, tokenizer, native = _load_frozen_model(root)
    _progress(progress, "MODEL_LOADED")
    try:
        root = Path(root)
        input_manifest = json.loads(path.read_text())
        _require(input_manifest.get("AIME_data_used") is False,
                 "Calibration input manifest must explicitly exclude AIME")
        rows = input_manifest["windows"]
        _require(len({row["document_id"] for row in rows}) == 8,
                 "Calibration must use eight distinct frozen document ids")
        checks = []
        for row, ids in zip(rows, windows):
            source = Path(row["source_text_path"])
            if not source.is_absolute():
                source = root / source
            _require(file_sha(source) == row["source_text_sha256"], "Frozen calibration source text differs")
            actual = tokenizer(source.read_text(), add_special_tokens=False)["input_ids"][:2048]
            _require(actual == ids and canonical_sha(ids) == row["input_ids_sha256"],
                     "Actual model tokenizer differs from frozen calibration prefix")
            checks.append(dict(document_id=row["document_id"], source_text_sha256=file_sha(source),
                               input_ids_sha256=canonical_sha(actual), tokens=2048, status="PASS"))
            _progress(progress, "CALIBRATION_TOKENIZER_PREFIX_VERIFIED", document_id=row["document_id"])
        return collect_calibration(model, tokenizer, native, root, windows,
                                   manifest_path=path, progress=progress, input_verification=checks)
    finally:
        del model, tokenizer, native
        gc.collect()


def run_formal(root, job, progress=None):
    root = Path(root)
    _require(job.get("seed", 1) == 1, "Only seed1 is admitted")
    condition = _condition(job.get("condition", job.get("configuration")))
    data = json.loads((root / "AIME26_DATASET_FREEZE.json").read_text())
    found = [x for x in data["records"] if x["problem_id"] == job["problem_id"]]
    _require(len(found) == 1, "Canonical AIME problem identity missing/duplicated")
    item = found[0]
    engine = PACEEngine.from_frozen_loader(root, condition, progress=progress)
    try:
        _, rendered = engine.f.render_input(engine.tokenizer, item["problem"])
        _require(rendered[0].tolist() == item["input_ids"], "Actual tokenizer/chat template differs from frozen input")
        result = engine.generate(item["input_ids"], problem_id=item["problem_id"], max_new_tokens=81920)
        _require(result["input_ids_sha256"] == item["input_ids_sha256"], "Frozen prompt ids differ")
        result.update(prompt_text_sha256=item["prompt_text_sha256"],
                      problem_sha256=item["problem_sha256"], dataset_freeze_sha256=file_sha(root / "AIME26_DATASET_FREEZE.json"))
        result["source_identity"] = {name: file_sha(root / name) for name in
            ("src/pace_runtime.py", "src/pace_quantizer.py", "src/pace_buffer.py",
             "source_manifest.json", "SOURCE_FREEZE.json", "method_spec.json") if (root / name).is_file()}
        return result
    finally:
        del engine
        gc.collect()


def run_gate(root, stage, unit, progress=None):
    _require(stage == "gate_d", "pace_runtime.run_gate implements full-model gate_d only")
    root = Path(root)
    fixture = json.loads((root / "protocol/non_aime_fixture.json").read_text())
    ids = fixture["tokens"][:int(unit.get("prompt_tokens", 32))]
    cap = int(unit.get("max_new_tokens", 32))
    _require(ids and 1 <= cap <= 128, "Smoke bounds differ")
    reference = load_native_reference(root)
    _progress(progress, "NATIVE_REFERENCE_READY", native_sha256=NATIVE_SHA256)
    mask_checks = _module(root / "tests/actual_mask_gate.py", "pace48_actual_mask_checks")
    actual_mask_gate = mask_checks.run_actual_mask_check(root, device="cuda:0", progress=progress,
                                                        native_reference=reference)
    _require(actual_mask_gate["status"] == "PASS" and actual_mask_gate["actual_cuda"] is True,
             "Actual calibrated-mask device check failed")
    engine = PACEEngine.from_frozen_loader(root, "R2", progress=progress)
    try:
        r2 = engine.generate(ids, problem_id="NON_AIME_SMOKE", max_new_tokens=cap)
        r3_engine = PACEEngine(root, "R3", model=engine.model, tokenizer=engine.tokenizer,
                              native_module=engine.native, progress=progress)
        r3 = r3_engine.generate(ids, problem_id="NON_AIME_SMOKE", max_new_tokens=cap)
        _require(r2["head_mask_sha256"] == r3["head_mask_sha256"], "Smoke head masks differ")
        return dict(status="PASS", actual_cuda=True, configs=dict(R2=r2, R3=r3),
                    actual_mask_gate=actual_mask_gate,
                    numerical_81K_claim=False, scope="bounded_nonAIME_packed_full_model_generation")
    finally:
        del engine
        gc.collect()
