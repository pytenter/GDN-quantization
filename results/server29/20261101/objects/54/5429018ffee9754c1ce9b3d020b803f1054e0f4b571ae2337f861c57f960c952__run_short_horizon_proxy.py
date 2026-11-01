#!/usr/bin/env python3
"""Frozen forward-only evaluator for natural KDA rotation surrogates."""

from __future__ import annotations

import argparse
import ast
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


ROOT = Path("/data/zypan/experiment_runs/LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1__resume20261010_v2")
CONFIG_PATH = ROOT / "frozen_config.json"
BRIDGE = Path("/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/experiments/LING_LONG_HORIZON_OBSERVABILITY_BRIDGE_V1")
DIRECTION_RUNNER = Path("/data/zypan/experiment_runs/LING_KDA_LONG_HORIZON_DIRECTION_CAUSAL_V1/scripts/run_direction_causal.py")
TRANSPORT = Path("/data/zypan/worktrees/ling-long-horizon-propagation-confirm-v1/experiments/LING_LONG_HORIZON_PROPAGATION_CONFIRM_V1/scripts/transport.py")
EPS = 1e-30
HIST_SOURCE = Path("/data/zypan/worktrees/ling-long-horizon-observability-bridge-v3/provenance_capsule_v3/source_snapshot/data01/user2/.codex/worktrees/GDN-quantization/ling-kda-int8-rc-granularity/experiments/ling")
_OPERATORS = None


def operators():
    global _OPERATORS
    if _OPERATORS is None:
        path = HIST_SOURCE / "run_ling_kda_query_readout_causal_closure_v1.py"
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in {"query_readout", "prepared_query"}]
        assert len(nodes) == 2
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        reference = import_file(HIST_SOURCE / "pytorch_kda_reference.py", "proxy_historical_kda_reference")
        _OPERATORS = (namespace, reference)
    return _OPERATORS


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
    os.link(tmp, path)  # atomic, exclusive publication: never replaces evidence
    tmp.unlink()


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
    ns, reference = operators()
    old = dict(rec, g=rec["log_decay"], use_qk_l2norm_in_kernel=True,
               use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    return ns["prepared_query"](reference, old)


def functional_kl(torch, ref_logits, branch_logits):
    # Canonical FP32 full-vocabulary KL; model outputs may be [B,V] or [B,T,V].
    def last(x):
        if x.ndim == 3:
            return x[:, -1, :].float()
        if x.ndim == 2:
            return x.float()
        raise RuntimeError(f"UNEXPECTED_LOGIT_LAYOUT:{tuple(x.shape)}")
    a, b = last(ref_logits), last(branch_logits)
    assert a.shape == b.shape
    logp, logq = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return float((torch.softmax(a, -1) * (logp - logq)).sum(-1).mean().item())


def stack_cache(torch, runner, capture, cache, layers):
    return torch.cat([runner.cache_state(capture, cache, int(layer)).float() for layer in layers], dim=0)


def state_metrics(torch, error, qhat, auxiliary_index):
    ns, _ = operators()
    readout = ns["query_readout"](error, qhat)[:, 0]
    aux_error = error[auxiliary_index]
    aux_q = qhat[auxiliary_index]
    aux_readout = readout[auxiliary_index]
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
    timing = {"readout_and_state_seconds": 0.0, "functional_kl_seconds": 0.0,
              "P2_seconds": 0.0}
    previous = {}
    consumed = [0]
    original_kernel = capture.kernel
    def audited_kernel(**kwargs):
        layer = int(capture.probe.current_layer)
        initial = kwargs.get("initial_state")
        if layer in previous:
            if initial is None or not torch.equal(initial, previous[layer]):
                raise RuntimeError(f"PERSISTENT_WRITEBACK_CONSUMPTION_FAIL:{condition}:{layer}")
            consumed[0] += 1
        elif initial is not None and bool(initial.count_nonzero()):
            raise RuntimeError("DOCUMENT_START_NOT_ZERO")
        return original_kernel(**kwargs)
    capture.kernel = audited_kernel
    for pos in range(end):
        fp_rec = rec_at(records, pos)
        fp_state = core.update_pre_q(fp_state, fp_rec)
        cache, logits, _ = runner.model_step(torch, capture, cache, tokens[pos], pos)
        summaries = runner.quantize_l6_cache(torch, core, capture, cache, rotations)
        if sorted(summaries) != layers:
            raise RuntimeError("QDQ_LAYER_COVERAGE_FAIL")
        previous = {int(layer): runner.cache_state(capture, cache, layer).detach().clone()
                    for layer in layers}
        active = [start for start in starts if start <= pos < start + max(horizons)]
        if not active:
            continue
        branch_state = stack_cache(torch, runner, capture, cache, layers)
        error = branch_state - fp_state.float()
        qhat = query_hat(torch, fp_rec)
        ts = time.perf_counter()
        base = state_metrics(torch, error, qhat, auxiliary_index)
        timing["readout_and_state_seconds"] += time.perf_counter() - ts
        ts = time.perf_counter()
        kl = functional_kl(torch, fp_logits[pos], logits)
        timing["functional_kl_seconds"] += time.perf_counter() - ts
        base["functional_kl"] = kl
        for start in active:
            offset = pos - start
            for horizon in horizons:
                if offset < horizon:
                    windows[start][horizon].append(base)
            if offset == 0:
                ts = time.perf_counter()
                p2, p2_aux = historical_sensitivity(torch, error, covariances[start], auxiliary_index)
                timing["P2_seconds"] += time.perf_counter() - ts
                instants[start] = {
                    "B0": base["state_mse"],
                    "P1": base["readout_energy"],
                    "P2_W128": p2,
                    "layer20_B0": base["layer20_state_mse"],
                    "layer20_P1": base["layer20_readout_energy"],
                    "layer20_P2_W128": p2_aux,
                }
    torch.cuda.synchronize()
    capture.kernel = original_kernel
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
        "metric_overhead_seconds": timing,
        "mechanical_gates": {"every_token_layer_qdq_count": end * len(layers),
                             "actual_next_kernel_equal_post_qdq_count": consumed[0],
                             "expected_consumption_count": (end - 1) * len(layers),
                             "passed": consumed[0] == (end - 1) * len(layers)},
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
        torch.set_grad_enabled(False)
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
        load_started = time.perf_counter()
        capture_instance = capture_module.Capture()
        capture = capture_instance
        load_seconds = time.perf_counter() - load_started
        atomic_json(ROOT / "runs" / f"{args.tag}.STARTED.json",
                    {"started_epoch": started_wall, "pid": os.getpid(), "gpu_uuid": host["gpu_uuid"],
                     "document": args.document, "runner_sha256": sha256(Path(__file__))})
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
            atomic_json(ROOT / "runs" / f"{args.tag}.{condition}.{counts[condition]}.json", result)
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
            "model_loading_seconds": load_seconds,
            "historical_operator_sources": {str(p): sha256(p) for p in
                [HIST_SOURCE / "run_ling_kda_query_readout_causal_closure_v1.py",
                 HIST_SOURCE / "pytorch_kda_reference.py"]},
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
            "accounted_gpu_hours": total_seconds / 3600.0,
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
