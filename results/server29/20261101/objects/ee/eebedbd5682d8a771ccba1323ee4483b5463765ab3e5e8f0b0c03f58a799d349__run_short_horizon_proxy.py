#!/usr/bin/env python3
"""Frozen forward-only evaluator for natural KDA rotation surrogates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path


ROOT = Path("/data/zypan/experiment_runs/LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1")
CONFIG_PATH = ROOT / "frozen_config.json"
BRIDGE = Path("/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1")
DIRECTION_RUNNER = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py")
TRANSPORT = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py")
EPS = 1e-30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return value == value and abs(value) != float("inf")
    return True


def rec_at(records, pos):
    return {key: value[:, pos : pos + 1] for key, value in records.items()}


def query_hat(torch, rec):
    q = rec["q"].float()
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    return q[:, 0] * (128.0 ** -0.5)


def stack_cache(torch, runner, capture, cache, layers):
    return torch.cat([runner.cache_state(capture, cache, int(layer)).float() for layer in layers], dim=0)


def state_metrics(torch, error, qhat, auxiliary_index):
    readout = torch.einsum("lhkv,lhk->lhv", error, qhat)
    aux_error = error[auxiliary_index]
    aux_q = qhat[auxiliary_index]
    aux_readout = torch.einsum("hkv,hk->hv", aux_error, aux_q)
    return {
        "state_mse": float(error.square().mean().item()),
        "readout_energy": float(readout.square().mean().item()),
        "layer20_state_mse": float(aux_error.square().mean().item()),
        "layer20_readout_energy": float(aux_readout.square().mean().item()),
    }


def historical_sensitivity(torch, error, covariance, auxiliary_index):
    # Mean over layer, head and value; covariance is a mean over history tokens.
    value = torch.einsum("lhij,lhiv,lhjv->", covariance.double(), error.double(), error.double())
    denom = error.shape[0] * error.shape[1] * error.shape[3]
    ae = error[auxiliary_index].double()
    ac = covariance[auxiliary_index].double()
    av = torch.einsum("hij,hiv,hjv->", ac, ae, ae)
    adenom = ae.shape[0] * ae.shape[2]
    return float((value / denom).item()), float((av / adenom).item())


def load_tokens_and_rotations(torch, transport, config, document, selected):
    dm = json.loads((BRIDGE / "document_manifest.json").read_text())
    doc = next(x for x in dm["documents"] if x["document_id"] == document)
    token_path = Path(transport.mapped(doc["token_file"]["path"]))
    if sha256(token_path) != doc["token_file"]["sha256"]:
        raise RuntimeError("TOKEN_HASH_MISMATCH")
    tokens = json.loads(token_path.read_text())
    rm = json.loads((BRIDGE / "rotation_manifest.json").read_text())
    manifest = {x["condition_id"]: x for x in rm["bank"]["conditions"]}
    loaded = {}
    identities = {}
    for condition in sorted(set(selected)):
        item = manifest[condition]
        path = Path(transport.mapped(item["artifact"]["path"]))
        digest = sha256(path)
        if digest != item["artifact"]["sha256"]:
            raise RuntimeError(f"ROTATION_HASH_MISMATCH:{condition}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact["layer_ids"] != config["layers"]:
            raise RuntimeError(f"ROTATION_LAYER_MISMATCH:{condition}")
        mapping = artifact["rotations"]
        matrices = []
        for layer in config["layers"]:
            matrix = mapping[layer] if layer in mapping else mapping[str(layer)]
            matrices.append(matrix.float())
        loaded[condition] = torch.stack(matrices, dim=0)
        identities[condition] = {"path": str(path), "sha256": digest}
    return doc, [int(x) for x in tokens], token_path, loaded, identities


def extract_fp(torch, runner, capture, tokens, end, checkpoints, layers):
    cache = None
    records = None
    fp_logits = {}
    fp_states = {}
    scoring = set()
    for start in checkpoints:
        scoring.update(range(start, start + 128))
    started = time.perf_counter()
    for pos in range(end):
        cache, logits, layer_records = runner.model_step(torch, capture, cache, tokens[pos], pos)
        rec = {
            key: torch.cat([layer_records[int(layer)][key] for layer in layers], dim=0)
            for key in ("q", "k", "v", "beta", "log_decay")
        }
        if records is None:
            records = {}
            for key, tensor in rec.items():
                shape = (tensor.shape[0], end) + tuple(tensor.shape[2:])
                records[key] = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        for key, tensor in rec.items():
            records[key][:, pos].copy_(tensor[:, 0])
        if pos in scoring:
            fp_logits[pos] = logits.detach().clone()
        if pos in {x - 1 for x in checkpoints} | {x + 31 for x in checkpoints} | {x + 127 for x in checkpoints}:
            fp_states[pos] = stack_cache(torch, runner, capture, cache, layers).detach().clone()
    torch.cuda.synchronize()
    return records, fp_logits, fp_states, time.perf_counter() - started


def replay_parity(torch, core, records, fp_states):
    state = None
    rows = []
    last = max(fp_states)
    for pos in range(last + 1):
        state = core.update_pre_q(state, rec_at(records, pos))
        if pos in fp_states:
            reference = fp_states[pos]
            diff = state.float() - reference.float()
            rows.append({
                "position": pos,
                "max_abs": float(diff.abs().max().item()),
                "relative_l2": float(torch.linalg.vector_norm(diff).item() / (torch.linalg.vector_norm(reference.float()).item() + EPS)),
                "finite": bool(torch.isfinite(diff).all()),
            })
    # This is an implementation parity gate, not a scientific acceptance threshold.
    passed = all(x["finite"] and x["relative_l2"] <= 1e-5 for x in rows)
    return {"passed": passed, "checks": rows}


def build_covariances(torch, records, starts):
    result = {}
    for start in starts:
        q = records["q"][:, start - 128 : start].float()
        q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        q = q * (128.0 ** -0.5)
        result[start] = torch.einsum("lthk,lthj->lhkj", q, q) / q.shape[1]
    return result


def run_condition(torch, runner, core, capture, tokens, records, fp_logits, covariances,
                  rotations, condition, run_index, starts, horizons, layers, auxiliary_index):
    cache = None
    fp_state = None
    end = max(starts) + max(horizons)
    windows = {start: {h: [] for h in horizons} for start in starts}
    instants = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for pos in range(end):
        fp_rec = rec_at(records, pos)
        fp_state = core.update_pre_q(fp_state, fp_rec)
        cache, logits, _ = runner.model_step(torch, capture, cache, tokens[pos], pos)
        runner.quantize_l6_cache(torch, core, capture, cache, rotations)
        active = [start for start in starts if start <= pos < start + max(horizons)]
        if not active:
            continue
        branch_state = stack_cache(torch, runner, capture, cache, layers)
        error = branch_state - fp_state.float()
        qhat = query_hat(torch, fp_rec)
        base = state_metrics(torch, error, qhat, auxiliary_index)
        kl = runner.kl_metrics(torch, fp_logits[pos], logits)["KL"]
        base["functional_kl"] = kl
        for start in active:
            offset = pos - start
            for horizon in horizons:
                if offset < horizon:
                    windows[start][horizon].append(base)
            if offset == 0:
                p2, p2_aux = historical_sensitivity(torch, error, covariances[start], auxiliary_index)
                instants[start] = {
                    "B0": base["state_mse"],
                    "P1": base["readout_energy"],
                    "P2_W128": p2,
                    "layer20_B0": base["layer20_state_mse"],
                    "layer20_P1": base["layer20_readout_energy"],
                    "layer20_P2_W128": p2_aux,
                }
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_windows = []
    for start in starts:
        row = {"start": start, **instants[start]}
        for horizon in horizons:
            values = windows[start][horizon]
            if len(values) != horizon:
                raise RuntimeError(f"WINDOW_LENGTH_MISMATCH:{start}:{horizon}:{len(values)}")
            row[f"P3_H{horizon}"] = sum(x["readout_energy"] for x in values) / horizon
            row[f"B1_H{horizon}"] = sum(x["state_mse"] for x in values) / horizon
            row[f"B2_H{horizon}"] = sum(x["functional_kl"] for x in values) / horizon
            row[f"layer20_P3_H{horizon}"] = sum(x["layer20_readout_energy"] for x in values) / horizon
            row[f"layer20_B1_H{horizon}"] = sum(x["layer20_state_mse"] for x in values) / horizon
        output_windows.append(row)
    metric_keys = [k for k in output_windows[0] if k != "start"]
    aggregate = {key: sum(x[key] for x in output_windows) / len(output_windows) for key in metric_keys}
    return {
        "condition": condition,
        "run_index": run_index,
        "windows": output_windows,
        "aggregate": aggregate,
        "elapsed_seconds": elapsed,
        "tokens_processed": end,
        "tokens_per_second": end / elapsed,
        "gpu_hours": elapsed / 3600.0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--conditions", required=True, help="comma separated; duplicates retained for determinism tests")
    ap.add_argument("--starts", default="256,1024")
    ap.add_argument("--horizons", default="32,128")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out_path = ROOT / "runs" / f"{args.tag}.json"
    stop_path = ROOT / "runs" / f"{args.tag}.STOP.json"
    if out_path.exists() or stop_path.exists():
        raise FileExistsError("refusing to overwrite existing run evidence")
    started_wall = time.time()
    capture_instance = None
    try:
        import torch
        config = json.loads(CONFIG_PATH.read_text())
        starts = [int(x) for x in args.starts.split(",")]
        horizons = [int(x) for x in args.horizons.split(",")]
        conditions = args.conditions.split(",")
        if args.document not in config["documents"] or args.gpu != config["document_gpu_map"][args.document]:
            raise RuntimeError("FROZEN_DOCUMENT_GPU_MAP_VIOLATION")
        if any(x not in config["conditions"] for x in conditions):
            raise RuntimeError("FROZEN_ROSTER_VIOLATION")
        if not set(starts).issubset(set(config["windows"])) or not set(horizons).issubset(set(config["horizons"])):
            raise RuntimeError("FROZEN_WINDOW_VIOLATION")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
            raise RuntimeError("CUDA_VISIBLE_DEVICES_MISMATCH")
        direction = import_file(DIRECTION_RUNNER, f"direction_runner_proxy_{args.gpu}_{os.getpid()}")
        transport, _, capture_module, core = direction.load_assets(args.gpu)
        host = direction.gpu_guard(torch, args.gpu)
        doc, tokens, token_path, rotations_cpu, rotation_ids = load_tokens_and_rotations(
            torch, transport, config, args.document, conditions
        )
        end = max(starts) + max(horizons)
        if len(tokens) < end or min(starts) < 128 or end > 2048:
            raise RuntimeError("CALIBRATION_RANGE_INVALID")
        torch.manual_seed(0)
        capture_instance = capture_module.Capture()
        capture = capture_instance
        records, fp_logits, fp_states, fp_seconds = extract_fp(
            torch, direction, capture, tokens, end, starts, config["layers"]
        )
        parity = replay_parity(torch, core, records, fp_states)
        if not parity["passed"]:
            raise RuntimeError("FP_REPLAY_PARITY_FAIL:" + json.dumps(parity))
        covariances = build_covariances(torch, records, starts)
        aux_index = config["layers"].index(config["auxiliary_layer"])
        results = []
        counts = {}
        for condition in conditions:
            counts[condition] = counts.get(condition, 0) + 1
            rotations = rotations_cpu[condition].to(capture.device)
            result = run_condition(
                torch, direction, core, capture, tokens, records, fp_logits, covariances,
                rotations, condition, counts[condition], starts, horizons,
                config["layers"], aux_index,
            )
            results.append(result)
            print(json.dumps({"event": "condition_complete", "tag": args.tag, "condition": condition,
                              "run_index": counts[condition], "elapsed_seconds": result["elapsed_seconds"],
                              "gpu_hours": result["gpu_hours"]}), flush=True)
        total_seconds = time.time() - started_wall
        output = {
            "schema": "short_horizon_proxy_run_v1",
            "tag": args.tag,
            "document": args.document,
            "physical_gpu": args.gpu,
            "gpu_uuid": config["gpu_uuids"][str(args.gpu)],
            "starts": starts,
            "horizons": horizons,
            "conditions": conditions,
            "config_sha256": sha256(CONFIG_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "direction_runner_sha256": sha256(DIRECTION_RUNNER),
            "transport_sha256": sha256(TRANSPORT),
            "token_file": {"path": str(token_path), "sha256": sha256(token_path)},
            "rotation_artifacts": rotation_ids,
            "document_manifest_entry": doc,
            "host": host,
            "fp_reference": {
                "source": "same-run current-runtime early-prefix teacher",
                "tokens": end,
                "elapsed_seconds": fp_seconds,
                "gpu_hours": fp_seconds / 3600.0,
                "replay_parity": parity,
                "archived_long_trace_used": False,
            },
            "results": results,
            "total_wall_seconds": total_seconds,
            "accounted_gpu_hours": fp_seconds / 3600.0 + sum(x["gpu_hours"] for x in results),
            "started_epoch": started_wall,
            "completed_epoch": time.time(),
            "valid": True,
        }
        if not finite_tree(output):
            raise RuntimeError("NONFINITE_OUTPUT")
        output["content_sha256_without_self"] = canonical_hash(output)
        atomic_json(out_path, output)
        capture_instance.close()
        capture_instance = None
        print(json.dumps({"event": "complete", "path": str(out_path), "sha256": sha256(out_path),
                          "accounted_gpu_hours": output["accounted_gpu_hours"]}), flush=True)
    except Exception as exc:
        if capture_instance is not None:
            try:
                capture_instance.close()
            except Exception:
                pass
        evidence = {
            "schema": "short_horizon_proxy_stop_v1",
            "tag": args.tag,
            "document": args.document,
            "gpu": args.gpu,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "started_epoch": started_wall,
            "stopped_epoch": time.time(),
            "host": socket.gethostname(),
            "kernel": platform.release(),
        }
        atomic_json(stop_path, evidence)
        print(json.dumps({"event": "stop", "path": str(stop_path), "error": repr(exc)}), flush=True)
        raise


if __name__ == "__main__":
    main()
