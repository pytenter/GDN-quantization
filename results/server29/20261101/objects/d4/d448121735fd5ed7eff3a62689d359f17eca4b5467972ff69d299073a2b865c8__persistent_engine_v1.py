"""Unqualified Phase B engine: unchanged model-step and genuine R128 QDQ.

No launcher or command-line entry point. Importing this module authorizes no
GPU work and no 32K run. Separate frozen qualification/cost gates are required.
No historical outputs, other experiments' labels, or optimizer are accessed.
"""
import hashlib
import math
import time


def tensor_bytes(torch, value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def percentile(values, fraction):
    assert values and 0 <= fraction <= 1
    values = sorted(values)
    at = fraction * (len(values) - 1)
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] * (hi - at) + values[hi] * (at - lo)


def summarize(values, start, end):
    assert len(values) == end - start and values
    assert all(math.isfinite(x) for x in values)
    return dict(window=[start, end], count=len(values), mean=sum(values) / len(values),
                auc=sum(values), p50=percentile(values, .5), p99=percentile(values, .99),
                min=min(values), max=max(values))


def full_kl(torch, reference, student):
    """Exact frozen direction scorer KL arithmetic; no clipping or subset."""
    ref, other = reference[:, -1, :].float(), student[:, -1, :].float()
    logp, logq = torch.log_softmax(ref, -1), torch.log_softmax(other, -1)
    p = torch.softmax(ref, -1)
    return float(torch.sum(p * (logp - logq), -1).mean().item())


def cache_tensors(torch, cache):
    for layer, entry in enumerate(cache.layers):
        assert torch.is_tensor(entry.keys), ("CACHE_KEY_MISSING", layer)
        yield f"layer{layer}.keys", entry.keys
        if torch.is_tensor(entry.values):
            yield f"layer{layer}.values", entry.values
        else:
            assert isinstance(entry.values, tuple) and len(entry.values) == 3
            for index, value in enumerate(entry.values):
                assert torch.is_tensor(value), ("CONV_CACHE_MISSING", layer, index)
                yield f"layer{layer}.conv{index}", value


def hybrid_inventory(torch, cache, position, layers, model_layer_count):
    """Real native hybrid schema, not an isolated recurrent-state stand-in."""
    assert len(cache.layers) == model_layer_count, "HYBRID_LAYER_COVERAGE_FAIL"
    items, total_bytes = {}, 0
    for name, value in cache_tensors(torch, cache):
        assert bool(torch.isfinite(value).all()), ("NONFINITE_CACHE", name)
        items[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                           bytes=value.numel() * value.element_size())
        total_bytes += items[name]["bytes"]
    for layer, entry in enumerate(cache.layers):
        if layer in layers:
            assert tuple(entry.keys.shape) == (1, 16, 128, 128)
            assert entry.keys.dtype == torch.float32
            assert isinstance(entry.values, tuple) and len(entry.values) == 3
        else:
            assert torch.is_tensor(entry.values)
            assert entry.keys.shape[-2] == entry.values.shape[-2] == position + 1
            assert cache.get_seq_length(layer_idx=layer) == position + 1
    return dict(position=position, layers=items, cache_bytes=total_bytes,
                kda_layers=list(layers), mla_layers=[x for x in range(model_layer_count) if x not in layers])


class PersistentAudit:
    def __init__(self, torch, direction, core, capture, layers, conditions):
        self.torch, self.direction, self.core, self.capture = torch, direction, core, capture
        self.layers, self.conditions = list(layers), list(conditions)
        self.active = None
        self.previous = {c: {} for c in conditions}
        self.writes = {c: 0 for c in conditions}
        self.consumed = {c: 0 for c in conditions}
        self.after_anchors = {c: {2048: 0, 8192: 0} for c in conditions}
        self.proofs = {c: [] for c in conditions}
        self.original_kernel = capture.kernel
        capture.kernel = self.kernel

    def kernel(self, **kwargs):
        condition = self.active
        if condition is not None:
            layer = int(self.capture.probe.current_layer)
            assert layer in self.layers
            initial = kwargs.get("initial_state")
            if layer in self.previous[condition]:
                assert initial is not None and self.torch.equal(initial, self.previous[condition][layer]), \
                    ("POST_QDQ_NOT_CONSUMED", condition, layer)
                self.consumed[condition] += 1
            else:
                assert initial is None or not bool(initial.count_nonzero()), "NONZERO_DOCUMENT_START"
        return self.original_kernel(**kwargs)

    def quantize(self, condition, cache, rotation, position, probe):
        torch = self.torch
        expected, untouched = {}, {}
        if probe:
            # A second evaluation is diagnostic only. It never enters the model.
            for name, value in cache_tensors(torch, cache):
                if name.endswith(".keys") and int(name.split(".")[0][5:]) in self.layers:
                    continue
                untouched[name] = value.detach().clone()
            for index, layer in enumerate(self.layers):
                state = self.direction.cache_state(self.capture, cache, layer)
                rotated = state.float().matmul(rotation[index][None])
                dequant, codes, scale = self.core.qdq_r128(rotated)
                assert codes.dtype == torch.int8 and codes.shape[-1] == 128
                assert int(codes.min()) >= -127 and int(codes.max()) <= 127
                expected[layer] = dequant.matmul(rotation[index][None].transpose(-1, -2)).to(state.dtype)
        metadata = self.direction.quantize_l6_cache(torch, self.core, self.capture, cache, rotation)
        assert sorted(metadata) == sorted(self.layers), "QDQ_LAYER_COVERAGE_FAIL"
        self.writes[condition] += len(self.layers)
        for anchor in (2048, 8192):
            if position >= anchor:
                self.after_anchors[condition][anchor] += len(self.layers)
        self.previous[condition] = {layer: self.direction.cache_state(self.capture, cache, layer).detach().clone()
                                    for layer in self.layers}
        if probe:
            assert all(torch.equal(expected[layer], self.previous[condition][layer]) for layer in self.layers), \
                "ACTUAL_INT8_ROTATE_INVERSE_WRITEBACK_MISMATCH"
            assert all(torch.equal(value, untouched[name]) for name, value in cache_tensors(torch, cache)
                       if name in untouched), "QDQ_CHANGED_MLA_OR_CONV"
            self.proofs[condition].append(dict(position=position, true_int8_r128=True,
                physical_inverse_writeback_bitwise_equal=True, mla_and_conv_untouched=True))
        return metadata

    def integrity(self, condition, endpoint):
        count = len(self.layers)
        assert self.writes[condition] == endpoint * count
        assert self.consumed[condition] == (endpoint - 1) * count
        assert all(self.after_anchors[condition][a] == max(0, endpoint - a) * count for a in (2048, 8192))
        return dict(every_token_layer_qdq_count=self.writes[condition],
                    actual_next_kernel_equal_post_qdq_count=self.consumed[condition],
                    post_anchor_qdq_counts=self.after_anchors[condition],
                    coordinate_and_quantization_proofs=self.proofs[condition], passed=True)

    def close(self):
        self.capture.kernel = self.original_kernel


def state_diagnostic(torch, direction, capture, fp_cache, branch_cache, layers):
    error_squared, fp_squared, branch_squared, numel = 0., 0., 0., 0
    for layer in layers:
        fp = direction.cache_state(capture, fp_cache, layer).float()
        branch = direction.cache_state(capture, branch_cache, layer).float()
        error_squared += float((branch - fp).square().sum())
        fp_squared += float(fp.square().sum())
        branch_squared += float(branch.square().sum())
        numel += fp.numel()
    return dict(accumulated_error_l2=math.sqrt(error_squared), accumulated_error_mse=error_squared / numel,
                fp_state_l2=math.sqrt(fp_squared), student_state_l2=math.sqrt(branch_squared))


def stream(torch, direction, core, capture, tokens, rotations, layers, endpoint,
           primary_start, secondary_start, guard, *, fp_replay=False, progress=None):
    """One fresh native FP cache + two fresh quantized full-model caches.

    Caller must have independent frozen launch authority. No execution here
    infers scientific permission from calibration, imports, or CPU tests.
    """
    assert 0 <= secondary_start <= primary_start < endpoint <= 32768
    assert len(tokens) >= endpoint and len(rotations) == 2
    conditions = sorted(rotations)
    caches, fp_cache = {c: None for c in conditions}, None
    audit = PersistentAudit(torch, direction, core, capture, layers, conditions)
    hashes = {c: hashlib.sha256() for c in ["FP"] + conditions}
    kl = {c: [] for c in conditions}
    diagnostics, inventory, blocks = {c: {} for c in conditions}, {}, []
    scale = {c: dict(min=float("inf"), max=0., sum=0., count=0) for c in conditions}
    probes = {0, 255, 256, 511, 2047, 2048, 8191, 8192, 16383, 24575, endpoint - 1}
    parity_state, parity_tokens = None, 0
    start = block_start = time.perf_counter()
    fp_seconds, student_seconds = 0., {c: 0. for c in conditions}
    try:
        for position in range(endpoint):
            guard()
            audit.active = None
            tick = time.perf_counter()
            fp_cache, fp_logits, records = direction.model_step(torch, capture, fp_cache, tokens[position], position)
            assert bool(torch.isfinite(fp_logits).all()), "NONFINITE_FP_LOGITS"
            hashes["FP"].update(position.to_bytes(8, "little")); hashes["FP"].update(tensor_bytes(torch, fp_logits))
            if fp_replay:
                states = torch.cat([direction.cache_state(capture, fp_cache, layer) for layer in layers])
                if parity_state is None:
                    parity_state = torch.zeros_like(states)
                record = {key: torch.cat([records[layer][key] for layer in layers])
                          for key in ("q", "k", "v", "beta", "log_decay")}
                with torch.inference_mode():
                    parity_state = core.update_pre_q(parity_state, record)
                assert torch.equal(parity_state, states), "NEW_ENGINE_FP_REPLAY_PARITY_FAIL"
                parity_tokens += 1
            fp_seconds += time.perf_counter() - tick
            for condition in conditions:
                tick = time.perf_counter()
                audit.active = condition
                caches[condition], logits, _ = direction.model_step(torch, capture, caches[condition], tokens[position], position)
                assert bool(torch.isfinite(logits).all()), "NONFINITE_STUDENT_LOGITS"
                metadata = audit.quantize(condition, caches[condition], rotations[condition], position, position in probes)
                hashes[condition].update(position.to_bytes(8, "little")); hashes[condition].update(tensor_bytes(torch, logits))
                for item in metadata.values():
                    summary = scale[condition]
                    summary["min"] = min(summary["min"], item["scale_min"])
                    summary["max"] = max(summary["max"], item["scale_max"])
                    summary["sum"] += item["scale_mean"]; summary["count"] += 1
                if position >= secondary_start:
                    value = full_kl(torch, fp_logits, logits)
                    assert math.isfinite(value), "NONFINITE_FULL_VOCAB_KL"
                    kl[condition].append(value)
                student_seconds[condition] += time.perf_counter() - tick
                del logits
            audit.active = None
            if position in probes:
                branch_caches = {"FP": fp_cache, **caches}
                assert len({id(c) for c in branch_caches.values()}) == 3, "SHARED_MUTABLE_CACHE_OBJECT"
                pointers = {key: {value.data_ptr() for _, value in cache_tensors(torch, cache)}
                            for key, cache in branch_caches.items()}
                assert all(not pointers[a].intersection(pointers[b]) for a in pointers for b in pointers if a < b), \
                    "CROSS_BRANCH_CACHE_STORAGE_ALIAS"
                inventory[str(position)] = {key: hybrid_inventory(torch, cache, position, layers, len(capture.model.model.layers))
                                            for key, cache in branch_caches.items()}
                for condition in conditions:
                    diagnostics[condition][str(position)] = state_diagnostic(torch, direction, capture, fp_cache, caches[condition], layers)
            if (position + 1) % 512 == 0 or position + 1 == endpoint:
                now = time.perf_counter()
                blocks.append(dict(end=position + 1, tokens=512 if (position + 1) % 512 == 0 else (position + 1) % 512,
                                   elapsed_seconds=now - block_start))
                block_start = now
                if progress is not None:
                    progress(position + 1, endpoint, now - start)
            del fp_logits
        results = []
        for condition in conditions:
            summary = scale[condition]
            results.append(dict(condition=condition, primary=summarize(kl[condition][primary_start - secondary_start:], primary_start, endpoint),
                secondary=summarize(kl[condition], secondary_start, endpoint), kl_trajectory=kl[condition],
                persistent_integrity=audit.integrity(condition, endpoint), state_diagnostics=diagnostics[condition],
                scale=dict(min=summary["min"], max=summary["max"], mean=summary["sum"] / summary["count"], count=summary["count"]),
                full_logit_sha256=hashes[condition].hexdigest(), all_finite=True))
        return dict(endpoint=endpoint, fp_reference_full_logit_sha256=hashes["FP"].hexdigest(),
            fp_replay_parity_tokens=parity_tokens, hybrid_cache_inventories=inventory,
            model_step_fp_seconds=fp_seconds, model_step_student_seconds=student_seconds,
            blocks=blocks, elapsed_seconds=time.perf_counter() - start, results=results,
            full_model_teacher_forced=True, every_token_qdq_including_after_anchors=True,
            logits_archived=False, same_gpu_live_fp_reference=True)
    finally:
        audit.close()

