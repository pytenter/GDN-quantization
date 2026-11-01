#!/usr/bin/env python3
"""LING KDA long-horizon direction-causal pilot.

This is an independent, fail-closed experiment. It never writes historical
worktrees or canonical outputs.  It uses the frozen L6 value rotation and the
current same-GPU FP reference, forks the complete model cache, replaces one
KDA layer's accumulated error direction, performs real L6 R128 QDQ/writeback,
and keeps every-token QDQ active during teacher-forced continuation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


TASK = "LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1"
OUT = Path("/data/zypan/experiment_runs") / TASK
BRIDGE = Path("/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1")
TRANSPORT = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py")
L6_EXPECTED_SHA = "51a532aad371e9af1d65ed2d2d5d0c41b1369f0635ba2c11adb713283038e964"
LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
TARGET_LAYER = 20
BRANCHES = ["NATURAL", "VISIBLE_POS", "VISIBLE_NEG", "LOW_POS", "LOW_NEG"]
GPU_UUIDS = {
    3: "GPU-759526ed-efb2-e7f6-881c-ab098e0b05ae",
    6: "GPU-de53f243-b271-521c-d875-d96611ed8574",
}
DOC_GPU = {
    "pg19-test-37702": 6,
    "pg19-test-28988": 6,
    "pg19-test-26493": 3,
}
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x) -> str:
    import torch
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"NO_OVERWRITE: {path}")
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"PARTIAL_EXISTS: {tmp}")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(torch, path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"NO_OVERWRITE: {path}")
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"PARTIAL_EXISTS: {tmp}")
    torch.save(value, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def import_file(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def command(args) -> str:
    return subprocess.check_output(args, text=True).strip()


def load_assets(gpu: int):
    transport = import_file(TRANSPORT, f"direction_transport_gpu{gpu}")
    transport.GPUS = [gpu]
    transport.UUIDS = {gpu: GPU_UUIDS[gpu]}
    common, capture_module, core, _, _ = transport.scientific_modules()
    return transport, common, capture_module, core


def gpu_guard(torch, gpu: int) -> dict:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    assert torch.cuda.device_count() == 1
    lines = command([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    row = None
    for line in lines:
        parts = [v.strip() for v in line.split(",")]
        if int(parts[0]) == gpu:
            row = parts
            break
    assert row is not None
    assert row[1] == GPU_UUIDS[gpu] and "3090" in row[2]
    occupants = command([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    assigned = [x for x in occupants.splitlines() if GPU_UUIDS[gpu] in x]
    assert all(int(x.split(",")[1]) == os.getpid() for x in assigned), ("GPU_OCCUPIED", assigned)
    return {
        "host": command(["hostname"]),
        "kernel": platform.release(),
        "physical_gpu": gpu,
        "gpu_uuid": row[1],
        "name": row[2],
        "driver": row[3],
        "initial_memory_used_mib": int(row[4]),
        "memory_total_mib": int(row[5]),
        "initial_utilization_percent": int(row[6]),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "python": sys.version,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selector": os.environ.get("LING_GATED_RMSNORM_FIXED_CONFIG"),
        "occupants": assigned,
    }


def load_manifest_and_tokens(transport, doc: str):
    dm = json.loads((BRIDGE / "document_manifest.json").read_text())
    d = next(x for x in dm["documents"] if x["document_id"] == doc)
    token_path = transport.mapped(d["token_file"]["path"])
    assert sha256(token_path) == d["token_file"]["sha256"]
    ids = json.loads(token_path.read_text())
    assert isinstance(ids, list) and len(ids) >= 32768
    rm = json.loads((BRIDGE / "rotation_manifest.json").read_text())
    cond = next(x for x in rm["bank"]["conditions"] if x["condition_id"] == "L6")
    rotation_path = transport.mapped(cond["artifact"]["path"])
    assert sha256(rotation_path) == cond["artifact"]["sha256"] == L6_EXPECTED_SHA
    return d, [int(v) for v in ids], cond, rotation_path


def model_step(torch, capture, cache, token: int, position: int):
    capture.current = {}
    capture.post = {}
    capture.probe.begin("FP", "native", None)
    try:
        with torch.inference_mode():
            out = capture.model(
                input_ids=torch.tensor([[token]], device=capture.device),
                attention_mask=torch.ones((1, position + 1), device=capture.device, dtype=torch.long),
                past_key_values=cache,
                cache_position=torch.tensor([position], device=capture.device),
                use_cache=True,
            )
    finally:
        capture.probe.end()
    cache = out.past_key_values
    logits = out.logits
    assert sorted(capture.current) == LAYERS
    for layer in LAYERS:
        assert capture.P2.get_state(cache, layer).data_ptr() == capture.post[layer].data_ptr()
    return cache, logits, capture.current


def cache_state(capture, cache, layer: int):
    value = capture.P2.get_state(cache, int(layer))
    if value is None:
        raise RuntimeError(f"missing cache state layer={layer}")
    return value


def quantize_l6_cache(torch, core, capture, cache, rotations):
    summaries = {}
    # The recurrent cache is created by model_step inside inference_mode.  PyTorch
    # therefore requires every subsequent in-place writeback to execute inside
    # inference_mode as well; grad-disabled alone is insufficient.
    with torch.inference_mode():
        for idx, layer in enumerate(LAYERS):
            state = cache_state(capture, cache, layer)
            rotated_pre = state.float().matmul(rotations[idx][None])
            qdq_rot, codes, scale = core.qdq_r128(rotated_pre)
            physical = qdq_rot.matmul(rotations[idx][None].transpose(-1, -2))
            assert bool(torch.isfinite(physical).all())
            state.copy_(physical.to(state.dtype))
            summaries[layer] = {
                "scale_min": float(scale.min().item()),
                "scale_mean": float(scale.mean().item()),
                "scale_max": float(scale.max().item()),
                "code_min": int(codes.min().item()),
                "code_max": int(codes.max().item()),
            }
    return summaries


def iter_named_tensors(value, prefix="root", seen=None):
    import torch
    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return
    seen.add(ident)
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda x: str(x)):
            yield from iter_named_tensors(value[key], f"{prefix}.{key}", seen)
    elif isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            yield from iter_named_tensors(item, f"{prefix}[{idx}]", seen)
    elif hasattr(value, "__dict__"):
        for key in sorted(vars(value)):
            yield from iter_named_tensors(getattr(value, key), f"{prefix}.{key}", seen)


def cache_parity(torch, a, b):
    ta = list(iter_named_tensors(a))
    tb = list(iter_named_tensors(b))
    assert [x[0] for x in ta] == [x[0] for x in tb]
    alias = set(x.data_ptr() for _, x in ta) & set(x.data_ptr() for _, x in tb)
    maximum = 0.0
    exact = True
    for (na, xa), (nb, xb) in zip(ta, tb):
        assert na == nb and xa.shape == xb.shape and xa.dtype == xb.dtype
        exact = exact and bool(torch.equal(xa, xb))
        if xa.numel():
            maximum = max(maximum, float((xa.float() - xb.float()).abs().max().item()))
    return {
        "tensor_count": len(ta),
        "alias_pointer_count": len(alias),
        "all_tensors_bitwise_equal": exact,
        "max_abs_difference": maximum,
    }


def prepared_query(torch, rec):
    q = rec["q"].float()
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (128.0 ** -0.5)
    return q[:, 0]


def oriented(vector):
    idx = int(vector.abs().argmax().item())
    return -vector if float(vector.flatten()[idx]) < 0 else vector


def build_directions(torch, covariance, natural_error):
    visible = torch.zeros_like(natural_error, dtype=torch.float32)
    low = torch.zeros_like(natural_error, dtype=torch.float32)
    eigenvalues = []
    for head in range(natural_error.shape[1]):
        c = covariance[head].double()
        vals, vecs = torch.linalg.eigh(c)
        qlow = oriented(vecs[:, 0].float())
        qhigh = oriented(vecs[:, -1].float())
        _, _, vh = torch.linalg.svd(natural_error[0, head].float(), full_matrices=False)
        v = oriented(vh[0].float())
        weight = torch.linalg.vector_norm(natural_error[0, head].float()).clamp_min(EPS)
        visible[0, head] = weight * torch.outer(qhigh, v)
        low[0, head] = weight * torch.outer(qlow, v)
        eigenvalues.append({
            "head": head,
            "lambda_min": float(vals[0].item()),
            "lambda_max": float(vals[-1].item()),
            "condition_ratio": float(vals[-1].item() / max(vals[0].item(), EPS)),
        })
    visible /= torch.linalg.vector_norm(visible)
    low /= torch.linalg.vector_norm(low)
    return visible, low, eigenvalues


def sensitivity(torch, covariance, error):
    # Sum_h trace(E_h^T C_h E_h), with covariance already averaged over tokens.
    error64 = error.double()
    value = torch.einsum("hkq,bhqv,bhkv->", covariance.double(), error64, error64)
    norm2 = error64.square().sum().clamp_min(EPS)
    return float(value.item()), float((value / norm2).item())


def qdq_candidate(torch, core, fp_block, direction, sign: int, alpha: float, rotation):
    nominal = fp_block.float() + float(sign) * float(alpha) * direction.float()
    rotated = nominal.matmul(rotation[None])
    qdq_rot, codes, scale = core.qdq_r128(rotated)
    physical = qdq_rot.matmul(rotation[None].transpose(-1, -2))
    error = physical - fp_block.float()
    return physical, error, codes, scale


def match_candidate(torch, core, fp_block, direction, sign: int, desired: float, rotation):
    tested = []

    def evaluate(factor):
        physical, error, codes, scale = qdq_candidate(
            torch, core, fp_block, direction, sign, desired * factor, rotation
        )
        norm = float(torch.linalg.vector_norm(error).item())
        tested.append((abs(norm - desired), abs(factor - 1.0), factor, norm, physical, error, codes, scale))
        return norm

    grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    values = [(factor, evaluate(factor)) for factor in grid]
    low, high = None, None
    for factor, norm in values:
        if norm <= desired:
            low = factor
        if high is None and norm >= desired:
            high = factor
    if low is not None and high is not None:
        for _ in range(18):
            mid = (low + high) / 2.0
            norm = evaluate(mid)
            if norm < desired:
                low = mid
            else:
                high = mid
    tested.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, factor, norm, physical, error, codes, scale = tested[0]
    cosine = float((error.flatten() * (float(sign) * direction).flatten()).sum().item() /
                   (torch.linalg.vector_norm(error).item() + EPS))
    return {
        "factor": float(factor),
        "alpha": float(desired * factor),
        "realized_norm": float(norm),
        "relative_norm_error": float(abs(norm - desired) / (desired + EPS)),
        "direction_cosine": cosine,
        "physical": physical,
        "error": error,
        "codes": codes,
        "scale": scale,
        "search_evaluations": len(tested),
        "search_range": [0.25, 2.0],
    }


def kl_metrics(torch, ref_logits, branch_logits):
    ref = ref_logits[:, -1, :].float()
    other = branch_logits[:, -1, :].float()
    logp = torch.log_softmax(ref, dim=-1)
    logq = torch.log_softmax(other, dim=-1)
    p = torch.softmax(ref, dim=-1)
    return {
        "KL": float(torch.sum(p * (logp - logq), dim=-1).mean().item()),
        "top1_match": float((ref.argmax(dim=-1) == other.argmax(dim=-1)).float().mean().item()),
        "logit_rel_l2": float(torch.linalg.vector_norm((ref - other).flatten()).item() /
                              (torch.linalg.vector_norm(ref.flatten()).item() + EPS)),
    }


class LogitStore:
    def __init__(self, torch, root: Path, mode: str, block=128):
        self.torch = torch
        self.root = root
        self.mode = mode
        self.block = block
        self.buffer = []
        self.chunks = []
        self.loaded_idx = None
        self.loaded = None
        if mode == "write":
            if root.exists():
                raise FileExistsError(f"NO_OVERWRITE: {root}")
            root.mkdir(parents=True)

    def append(self, logits):
        self.buffer.append(logits[:, -1, :].detach().to("cpu").contiguous())
        if len(self.buffer) == self.block:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        value = self.torch.cat(self.buffer, dim=0).contiguous()
        path = self.root / f"chunk_{len(self.chunks):04d}.pt"
        atomic_torch_save(self.torch, path, value)
        self.chunks.append({"path": str(path), "rows": int(value.shape[0]), "sha256": sha256(path), "bytes": path.stat().st_size})
        self.buffer.clear()

    def finish(self):
        self.flush()
        manifest = {"dtype": "model_output_native", "block": self.block, "chunks": self.chunks,
                    "rows": sum(x["rows"] for x in self.chunks)}
        atomic_json(self.root / "manifest.json", manifest)
        return manifest

    def get(self, index: int, device):
        chunk_idx = index // self.block
        offset = index % self.block
        if self.loaded_idx != chunk_idx:
            path = self.root / f"chunk_{chunk_idx:04d}.pt"
            self.loaded = self.torch.load(path, map_location="cpu", weights_only=True)
            self.loaded_idx = chunk_idx
        assert self.loaded is not None and offset < self.loaded.shape[0]
        return self.loaded[offset:offset + 1].to(device).unsqueeze(0)


def run_reference(torch, capture, cache, ids, checkpoint, calibration, end, root: Path, target_layer: int):
    store = LogitStore(torch, root / "fp_logits", "write")
    covariance = torch.zeros((16, 128, 128), device="cuda", dtype=torch.float64)
    count = 0
    started = time.time()
    for pos in range(checkpoint, end):
        cache, logits, records = model_step(torch, capture, cache, ids[pos], pos)
        if pos < checkpoint + calibration:
            q = prepared_query(torch, records[target_layer])[0].double()
            covariance += torch.einsum("hk,hl->hkl", q, q)
            count += 1
        else:
            store.append(logits)
        if (pos + 1) % 512 == 0:
            print("REFERENCE_PROGRESS", checkpoint, pos + 1, end, round(time.time() - started, 2), flush=True)
        del logits, records
    manifest = store.finish()
    assert count == calibration and manifest["rows"] == end - checkpoint - calibration
    covariance /= float(count)
    endpoint = {layer: cache_state(capture, cache, layer).detach().clone() for layer in LAYERS}
    return cache, covariance, endpoint, manifest, time.time() - started


def run_branch(torch, core, capture, base_cache, ids, checkpoint, calibration, end, root: Path,
               branch: str, fp_endpoint, rotations, initial_state, ref_store_root: Path):
    cache = copy.deepcopy(base_cache)
    with torch.inference_mode():
        target_state = cache_state(capture, cache, TARGET_LAYER)
        target_state.copy_(initial_state.to(target_state))
    reader = LogitStore(torch, ref_store_root, "read")
    sums = {"KL": 0.0, "top1_match": 0.0, "logit_rel_l2": 0.0}
    count = 0
    qscale_min = float("inf")
    qscale_max = 0.0
    qscale_sum = 0.0
    qscale_n = 0
    started = time.time()
    for pos in range(checkpoint, end):
        cache, logits, _ = model_step(torch, capture, cache, ids[pos], pos)
        qmeta = quantize_l6_cache(torch, core, capture, cache, rotations)
        m = qmeta[TARGET_LAYER]
        qscale_min = min(qscale_min, m["scale_min"])
        qscale_max = max(qscale_max, m["scale_max"])
        qscale_sum += m["scale_mean"]
        qscale_n += 1
        if pos >= checkpoint + calibration:
            ref = reader.get(count, logits.device)
            metrics = kl_metrics(torch, ref, logits)
            for key in sums:
                sums[key] += metrics[key]
            count += 1
            del ref
        if (pos + 1) % 512 == 0:
            print("BRANCH_PROGRESS", branch, checkpoint, pos + 1, end, round(time.time() - started, 2), flush=True)
        del logits
    global_sq = 0.0
    per_layer = {}
    for layer in LAYERS:
        diff = cache_state(capture, cache, layer).float() - fp_endpoint[layer].float()
        norm = float(torch.linalg.vector_norm(diff).item())
        global_sq += norm * norm
        per_layer[str(layer)] = norm
    result = {
        "branch": branch,
        "tokens_total_continuation": end - checkpoint,
        "calibration_tokens_unscored": calibration,
        "evaluation_tokens": count,
        "future_kl_auc": sums["KL"],
        "future_kl_mean": sums["KL"] / max(count, 1),
        "top1_match_mean": sums["top1_match"] / max(count, 1),
        "logit_rel_l2_mean": sums["logit_rel_l2"] / max(count, 1),
        "endpoint_target_error_norm": per_layer[str(TARGET_LAYER)],
        "endpoint_global_kda_error_norm": math.sqrt(global_sq),
        "endpoint_per_layer_error_norm": per_layer,
        "target_qdq_scale_min": qscale_min,
        "target_qdq_scale_mean_over_steps": qscale_sum / max(qscale_n, 1),
        "target_qdq_scale_max": qscale_max,
        "all_finite": all(math.isfinite(v) for v in sums.values()) and math.isfinite(global_sq),
        "elapsed_seconds": time.time() - started,
        "every_token_l6_qdq_writeback": True,
    }
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_parity(torch, core, capture, natural_cache, snapshot_path, ids, checkpoint, rotations, steps=8):
    restored = torch.load(snapshot_path, map_location="cuda", weights_only=False)
    initial = cache_parity(torch, natural_cache, restored)
    assert initial["alias_pointer_count"] == 0 and initial["all_tensors_bitwise_equal"]
    max_logit = 0.0
    logits_exact = True
    started = time.time()
    for pos in range(checkpoint, checkpoint + steps):
        natural_cache, logits_a, _ = model_step(torch, capture, natural_cache, ids[pos], pos)
        restored, logits_b, _ = model_step(torch, capture, restored, ids[pos], pos)
        max_logit = max(max_logit, float((logits_a.float() - logits_b.float()).abs().max().item()))
        logits_exact = logits_exact and bool(torch.equal(logits_a, logits_b))
        quantize_l6_cache(torch, core, capture, natural_cache, rotations)
        quantize_l6_cache(torch, core, capture, restored, rotations)
        del logits_a, logits_b
    final = cache_parity(torch, natural_cache, restored)
    result = {
        "gate": "PASS" if initial["all_tensors_bitwise_equal"] and logits_exact and final["all_tensors_bitwise_equal"] and final["alias_pointer_count"] == 0 else "FAIL",
        "steps": steps,
        "snapshot_roundtrip_initial": initial,
        "suffix_logits_bitwise_equal": logits_exact,
        "suffix_logits_max_abs_difference": max_logit,
        "suffix_full_cache_final": final,
        "elapsed_seconds": time.time() - started,
    }
    del natural_cache, restored
    gc.collect()
    torch.cuda.empty_cache()
    assert result["gate"] == "PASS", result
    return result


def role_config(role: str):
    if role == "micro":
        return dict(checkpoint=8, calibration=4, end=16, parity_steps=2, do_smoke=False, evidence="MECHANICAL_ONLY")
    if role == "development":
        return dict(checkpoint=24576, calibration=512, end=32768, parity_steps=8, do_smoke=True, smoke_end=25600, evidence="EXPLORATORY_DEVELOPMENT")
    if role == "validation":
        return dict(checkpoint=24576, calibration=512, end=32768, parity_steps=8, do_smoke=False, evidence="PROSPECTIVE_VALIDATION")
    raise ValueError(role)


def intervention_bundle(torch, core, fp_block, natural_block, rotation, covariance):
    natural_error = natural_block.float() - fp_block.float()
    desired = float(torch.linalg.vector_norm(natural_error).item())
    assert desired > EPS
    visible, low, eigenvalues = build_directions(torch, covariance, natural_error)
    bundle = {}
    for name, direction, sign in [
        ("VISIBLE_POS", visible, 1), ("VISIBLE_NEG", visible, -1),
        ("LOW_POS", low, 1), ("LOW_NEG", low, -1),
    ]:
        bundle[name] = match_candidate(torch, core, fp_block, direction, sign, desired, rotation)
    natural_raw, natural_per_norm = sensitivity(torch, covariance, natural_error)
    rows = []
    for name, item in bundle.items():
        raw, per_norm = sensitivity(torch, covariance, item["error"])
        item["readout_sensitivity"] = raw
        item["readout_sensitivity_per_norm2"] = per_norm
        item["state_sha256"] = tensor_sha(item["physical"])
        item["error_sha256"] = tensor_sha(item["error"])
        item["scale_summary"] = {
            "min": float(item["scale"].min().item()),
            "mean": float(item["scale"].mean().item()),
            "max": float(item["scale"].max().item()),
            "shape": list(item["scale"].shape),
        }
        rows.append(item)
    visible_s = 0.5 * (bundle["VISIBLE_POS"]["readout_sensitivity_per_norm2"] + bundle["VISIBLE_NEG"]["readout_sensitivity_per_norm2"])
    low_s = 0.5 * (bundle["LOW_POS"]["readout_sensitivity_per_norm2"] + bundle["LOW_NEG"]["readout_sensitivity_per_norm2"])
    realized = [x["realized_norm"] for x in bundle.values()]
    gates = {
        "post_qdq_each_relative_norm_error_le_0p02": all(x["relative_norm_error"] <= 0.02 for x in rows),
        "post_qdq_cross_branch_norm_spread_le_0p02": (max(realized) - min(realized)) / (desired + EPS) <= 0.02,
        "post_qdq_signed_direction_cosine_ge_0p50": all(x["direction_cosine"] >= 0.50 for x in rows),
        "visible_to_low_sensitivity_ratio_ge_4": visible_s / (low_s + EPS) >= 4.0,
    }
    gate = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "gate": gate,
        "target_layer": TARGET_LAYER,
        "target_heads": list(range(16)),
        "desired_natural_error_norm": desired,
        "natural_error_sha256": tensor_sha(natural_error),
        "natural_readout_sensitivity": natural_raw,
        "natural_readout_sensitivity_per_norm2": natural_per_norm,
        "visible_direction_sha256": tensor_sha(visible),
        "low_direction_sha256": tensor_sha(low),
        "pre_qdq_direction_inner_product": float((visible * low).sum().item()),
        "eigenvalues": eigenvalues,
        "visible_mean_sensitivity_per_norm2": visible_s,
        "low_mean_sensitivity_per_norm2": low_s,
        "visible_to_low_sensitivity_ratio": visible_s / (low_s + EPS),
        "mechanical_gates": gates,
        "branches": {},
    }
    for name, item in bundle.items():
        result["branches"][name] = {k: v for k, v in item.items() if k not in {"physical", "error", "codes", "scale"}}
    assert gate == "PASS", result
    states = {"NATURAL": natural_block.detach().clone()}
    states.update({name: item["physical"].detach().clone() for name, item in bundle.items()})
    return result, states


def run_experiment(doc: str, gpu: int, role: str, tag: str | None = None):
    assert DOC_GPU[doc] == gpu
    config = role_config(role)
    suffix = f"__{tag}" if tag else ""
    run_root = OUT / "runs" / f"{doc}__gpu{gpu}__{role}{suffix}"
    if run_root.exists():
        raise FileExistsError(f"NO_OVERWRITE: {run_root}")
    run_root.mkdir(parents=True)
    started = time.time()
    transport, common, capture_module, core = load_assets(gpu)
    torch = core.torch
    torch.set_grad_enabled(False)
    hardware = gpu_guard(torch, gpu)
    document, ids, condition, rotation_path = load_manifest_and_tokens(transport, doc)
    rot_obj = torch.load(rotation_path, map_location="cuda", weights_only=False)
    rotations = torch.stack([rot_obj["rotations"][layer].float() for layer in LAYERS])
    assert rotations.shape == (18, 128, 128)
    checkpoint = config["checkpoint"]
    calibration = config["calibration"]
    end = config["end"]
    start_receipt = {
        "task": TASK, "document": doc, "role": role, "evidence_role": config["evidence"],
        "checkpoint": checkpoint, "calibration": calibration, "endpoint": end,
        "target_layer": TARGET_LAYER, "target_heads": list(range(16)), "condition": "L6",
        "hardware": hardware, "pid": os.getpid(), "time_unix": started,
        "document_identity": document, "rotation_identity": condition["artifact"],
        "source_sha256": sha256(Path(__file__)), "append_only_tag": tag,
    }
    atomic_json(run_root / "STARTED.json", start_receipt)
    capture = capture_module.Capture()
    try:
        fp_cache = None
        natural_cache = None
        prefix_started = time.time()
        last_qmeta = None
        for pos in range(checkpoint):
            fp_cache, fp_logits, _ = model_step(torch, capture, fp_cache, ids[pos], pos)
            natural_cache, natural_logits, _ = model_step(torch, capture, natural_cache, ids[pos], pos)
            last_qmeta = quantize_l6_cache(torch, core, capture, natural_cache, rotations)
            del fp_logits, natural_logits
            if (pos + 1) % 512 == 0:
                print("PREFIX_PROGRESS", doc, pos + 1, checkpoint, round(time.time() - prefix_started, 2), flush=True)
        prefix_seconds = time.time() - prefix_started
        fp_block = cache_state(capture, fp_cache, TARGET_LAYER).detach().clone()
        natural_block = cache_state(capture, natural_cache, TARGET_LAYER).detach().clone()
        snapshot_started = time.time()
        fp_snapshot = run_root / "FP_FULL_CACHE_CHECKPOINT.pt"
        natural_snapshot = run_root / "NATURAL_FULL_CACHE_CHECKPOINT.pt"
        atomic_torch_save(torch, fp_snapshot, fp_cache)
        atomic_torch_save(torch, natural_snapshot, natural_cache)
        snapshot_seconds = time.time() - snapshot_started
        del fp_cache
        gc.collect(); torch.cuda.empty_cache()
        parity = run_parity(torch, core, capture, natural_cache, natural_snapshot, ids, checkpoint, rotations, config["parity_steps"])
        atomic_json(run_root / "FORK_PARITY.json", parity)
        natural_base = torch.load(natural_snapshot, map_location="cuda", weights_only=False)

        def phase(label: str, phase_end: int):
            fp_base = torch.load(fp_snapshot, map_location="cuda", weights_only=False)
            phase_root = run_root / label
            phase_root.mkdir()
            fp_final_cache, covariance, fp_endpoint, ref_manifest, ref_seconds = run_reference(
                torch, capture, fp_base, ids, checkpoint, calibration, phase_end, phase_root, TARGET_LAYER
            )
            del fp_final_cache, fp_base
            gc.collect(); torch.cuda.empty_cache()
            direction_audit, branch_states = intervention_bundle(
                torch, core, fp_block, natural_block, rotations[LAYERS.index(TARGET_LAYER)], covariance
            )
            atomic_json(phase_root / "DIRECTION_AUDIT.json", direction_audit)
            branch_results = []
            for branch in BRANCHES:
                result = run_branch(
                    torch, core, capture, natural_base, ids, checkpoint, calibration, phase_end,
                    phase_root, branch, fp_endpoint, rotations, branch_states[branch], phase_root / "fp_logits",
                )
                initial_error = branch_states[branch].float() - fp_block.float()
                raw, per_norm = sensitivity(torch, covariance, initial_error)
                result.update({
                    "initial_target_error_norm": float(torch.linalg.vector_norm(initial_error).item()),
                    "initial_readout_sensitivity": raw,
                    "initial_readout_sensitivity_per_norm2": per_norm,
                    "initial_target_state_sha256": tensor_sha(branch_states[branch]),
                })
                atomic_json(phase_root / f"BRANCH_{branch}.json", result)
                branch_results.append(result)
            by_name = {x["branch"]: x for x in branch_results}
            contrast = 0.5 * (by_name["VISIBLE_POS"]["future_kl_auc"] + by_name["VISIBLE_NEG"]["future_kl_auc"]) - 0.5 * (by_name["LOW_POS"]["future_kl_auc"] + by_name["LOW_NEG"]["future_kl_auc"])
            summary = {
                "phase": label, "document": doc, "evidence_role": config["evidence"],
                "checkpoint": checkpoint, "calibration_window": [checkpoint, checkpoint + calibration],
                "evaluation_window": [checkpoint + calibration, phase_end],
                "reference_seconds": ref_seconds, "fp_logit_manifest": ref_manifest,
                "direction_gate": direction_audit["gate"], "fork_gate": parity["gate"],
                "branches": branch_results, "primary_contrast_D_auc": contrast,
                "visible_signs_individually": {
                    "positive": by_name["VISIBLE_POS"]["future_kl_auc"],
                    "negative": by_name["VISIBLE_NEG"]["future_kl_auc"],
                },
                "low_signs_individually": {
                    "positive": by_name["LOW_POS"]["future_kl_auc"],
                    "negative": by_name["LOW_NEG"]["future_kl_auc"],
                },
                "all_branch_integrity": all(x["all_finite"] for x in branch_results),
            }
            atomic_json(phase_root / "PHASE_RESULT.json", summary)
            del fp_endpoint, covariance, branch_states
            gc.collect(); torch.cuda.empty_cache()
            return summary

        smoke_result = None
        if config.get("do_smoke"):
            smoke_result = phase("smoke_512", config["smoke_end"])
            assert smoke_result["fork_gate"] == "PASS" and smoke_result["direction_gate"] == "PASS" and smoke_result["all_branch_integrity"]
        full_result = phase("full", end)
        total_seconds = time.time() - started
        receipt = {
            "gate": "PASS", "task": TASK, "document": doc, "role": role,
            "hardware": hardware, "checkpoint": checkpoint, "calibration": calibration, "endpoint": end,
            "prefix_seconds": prefix_seconds, "snapshot_seconds": snapshot_seconds,
            "snapshot_files": [
                {"path": str(fp_snapshot), "sha256": sha256(fp_snapshot), "bytes": fp_snapshot.stat().st_size},
                {"path": str(natural_snapshot), "sha256": sha256(natural_snapshot), "bytes": natural_snapshot.stat().st_size},
            ],
            "fork_parity": parity, "prefix_target_qdq_meta": last_qmeta[TARGET_LAYER] if last_qmeta else None,
            "smoke_result": smoke_result, "full_result": full_result,
            "elapsed_seconds": total_seconds, "gpu_hours": total_seconds / 3600.0,
            "budget_limit_gpu_hours": 16.0,
            "old_canonical_modified": False, "old_fail_modified": False,
            "new_rotation_training": False, "aime": False, "free_generation": False,
        }
        atomic_json(run_root / "COMPLETE.json", receipt)
        print("DIRECTION_CAUSAL_RUN_COMPLETE", doc, role, round(total_seconds, 2), full_result["primary_contrast_D_auc"], flush=True)
    finally:
        capture.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", choices=sorted(DOC_GPU), required=True)
    parser.add_argument("--gpu", type=int, choices=[3, 6], required=True)
    parser.add_argument("--role", choices=["micro", "development", "validation"], required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    run_experiment(args.doc, args.gpu, args.role, args.tag)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # A tiny pre-freeze engineering probe may expose a plumbing defect and
        # is not a scientific run. Preserve it independently; formal roles are
        # fail-closed at the task level.
        is_micro = "--role" in sys.argv and sys.argv[sys.argv.index("--role") + 1] == "micro"
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "untagged"
        stop = OUT / (f"ENGINEERING_MICRO_FAIL__{tag}.json" if is_micro else "STOP.json")
        if not stop.exists():
            atomic_json(stop, {
                "gate": "FAIL", "task": TASK, "stage": "execution", "error": repr(exc),
                "traceback": traceback.format_exc(), "time_unix": time.time(),
                "final_action": "PRESERVE_AND_DEBUG" if is_micro else "FAIL_AND_STOP",
                "no_automatic_retry": not is_micro,
            })
        raise
