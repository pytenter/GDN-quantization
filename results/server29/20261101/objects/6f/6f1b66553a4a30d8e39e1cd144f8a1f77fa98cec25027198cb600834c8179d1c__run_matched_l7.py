#!/usr/bin/env python3
"""Forward-only Phase A extension of the immutable verified surrogate evaluator.

No training, historical file writes, process signals, or long-horizon execution.
GPU qualification is NOT implied by successful compilation/CPU helper tests.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from objective import original_helpers, combined_scores, digest

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
PRIOR = Path("/data/zypan/experiment_runs/LING_KDA_SHORT_HORIZON_SURROGATE_VALIDATION_V1__resume20261010_v2")
TRAINER = Path("/data/zypan/cleanroom/LING_L6_L7_NEW_MEMBER_HANDOFF_V1_e1256bf_full/experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/recurrent_dense_train.py")


def load_module(path, name):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def publish(path, value):
    # Same-filesystem exclusive atomic publication. Never replaces even STOPs.
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


class OutputTap:
    """Read-only hooks at the exact historical post-o_proj target."""
    def __init__(self, torch, capture, layers):
        self.torch, self.capture, self.layers = torch, capture, layers
        self.enabled, self.outputs, self.norm_inputs = True, {}, {}
        self.handles = []
        for layer in layers:
            attention = capture.model.model.layers[layer].attention
            self.handles.append(attention.o_proj.register_forward_hook(self.output_hook(layer)))
            self.handles.append(attention.o_norm.register_forward_pre_hook(self.norm_hook(layer)))

    def output_hook(self, layer):
        def hook(module, inputs, output):
            if self.enabled:
                if layer in self.outputs:
                    raise RuntimeError("DUPLICATE_POST_O_PROJ_CAPTURE")
                self.outputs[layer] = output.detach().clone()
        return hook

    def norm_hook(self, layer):
        def hook(module, inputs):
            if self.enabled:
                if len(inputs) != 2:
                    raise RuntimeError("UNEXPECTED_REAL_O_NORM_INPUT_LAYOUT")
                # Only retain the current token. No detached values enter the model.
                self.norm_inputs[layer] = tuple(x.detach().clone() for x in inputs)
        return hook

    def begin(self):
        self.outputs, self.norm_inputs = {}, {}

    def validate(self):
        if set(self.outputs) != set(self.layers) or set(self.norm_inputs) != set(self.layers):
            raise RuntimeError("ACTUAL_POST_O_PROJ_CAPTURE_COVERAGE_FAIL")
        if not all(self.torch.isfinite(x).all() for x in self.outputs.values()):
            raise RuntimeError("NONFINITE_LOCAL_FUNCTIONAL_OUTPUT")

    def close(self):
        for h in self.handles:
            h.remove()


class ObservedRunner:
    """Delegates original full-model stepping/QDQ; observes without changing it."""
    def __init__(self, base, direction, torch, core, capture, tap, helpers, starts, layers):
        self.base, self.direction, self.torch, self.core = base, direction, torch, core
        self.capture, self.tap, self.helpers, self.starts, self.layers = capture, tap, helpers, starts, layers
        self.mode = "fp"
        self.fp_outputs, self.fp_output_hashes, self.fp_logit_hashes = {}, {}, {}
        self.helper_forward_checks = []
        self.reference_states = None
        self.scores = {}
        self.position = None

    def __getattr__(self, name):
        return getattr(self.direction, name)

    def active(self, pos):
        return any(s <= pos < s+128 for s in self.starts)

    def model_step(self, torch, capture, cache, token, pos):
        self.position = pos
        self.tap.begin()
        result = self.direction.model_step(torch, capture, cache, token, pos)
        self.tap.validate()
        if not torch.isfinite(result[1]).all():
            raise RuntimeError("NONFINITE_FULL_LOGITS")
        if self.mode == "fp" and self.active(pos):
            self.fp_outputs[pos] = dict(self.tap.outputs)
            self.fp_output_hashes[pos] = {layer: self.direction.tensor_sha(x)
                                          for layer, x in self.tap.outputs.items()}
            self.fp_logit_hashes[pos] = self.direction.tensor_sha(result[1])
            if pos in {s for s in self.starts} | {s+1 for s in self.starts}:
                self.tap.enabled = False
                try:
                    for layer in self.layers:
                        core, gate = self.tap.norm_inputs[layer]
                        rebuilt = self.helpers["local_output"](capture.model, layer, core, gate)
                        target = self.tap.outputs[layer]
                        same = bool(torch.equal(rebuilt, target))
                        identity_loss = float(self.helpers["relative_mse"](rebuilt, target))
                        if not same or identity_loss != 0:
                            raise RuntimeError("REAL_L7_POST_O_PROJ_HELPER_PARITY_FAIL")
                        self.helper_forward_checks.append({"token": pos, "layer": layer,
                            "bitwise_equal": same, "identity_loss": identity_loss,
                            "post_o_proj_sha256": self.direction.tensor_sha(target)})
                finally:
                    self.tap.enabled = True
        return result

    def quantize_l6_cache(self, torch, core, capture, cache, rotations):
        summaries = self.direction.quantize_l6_cache(torch, core, capture, cache, rotations)
        if self.active(self.position):
            if self.reference_states is None:
                raise RuntimeError("MATCHED_TEACHER_STATE_MISSING")
            branch = self.base.stack_cache(torch, self.direction, capture, cache, self.layers)
            values = combined_scores(torch, self.helpers, self.tap.outputs,
                self.fp_outputs[self.position], branch, self.reference_states, self.layers)
            self.scores[self.position] = [float(x) for x in values]
        return summaries


class ObservedCore:
    def __init__(self, core, runner):
        self.core, self.runner = core, runner

    def __getattr__(self, name):
        return getattr(self.core, name)

    def update_pre_q(self, state, rec):
        result = self.core.update_pre_q(state, rec)
        self.runner.reference_states = result
        return result


def augment(result, observer, starts):
    keys = ["L7_FUNCTIONAL_ORIGINAL", "L7_STATE_ORIGINAL", "L7_TOTAL_ORIGINAL"]
    for row in result["windows"]:
        values = [observer.scores[p] for p in range(row["start"], row["start"]+128)]
        assert len(values) == 128
        for i, key in enumerate(keys):
            row[key] = sum(v[i] for v in values)/128
    for key in keys:
        result["aggregate"][key] = sum(x[key] for x in result["windows"])/len(starts)
    result["l7_loss_application"] = "EXACT_HISTORICAL_HELPERS_ON_MATCHED_FULL_MODEL_DEPLOYMENT_TENSORS"
    result["historical_training_sampling_protocol_replayed"] = False
    result["l7_scored_token_count"] = len(observer.scores)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True)
    ap.add_argument("--gpu", type=int, required=True, choices=[3,6])
    ap.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = ap.parse_args()
    tag = f"{args.mode}_{args.document}_gpu{args.gpu}"
    path = ROOT / "runs" / f"{tag}.json"
    stop = ROOT / "runs" / f"{tag}.STOP.json"
    if any((ROOT / "runs").glob("*.STOP.json")) or path.exists():
        raise RuntimeError("FAIL_CLOSED_OR_ALREADY_EXECUTED_NO_RETRY")
    capture, tap = None, None
    start_wall = time.time()
    try:
        import torch
        torch.set_grad_enabled(False)
        config = json.loads((ROOT / "PHASE_A_CONFIG.json").read_text())
        if args.document not in config["documents"] or config["document_gpu_map"][args.document] != args.gpu:
            raise RuntimeError("FROZEN_DOCUMENT_GPU_MAP_VIOLATION")
        freeze = json.loads((ROOT / "PREMEASUREMENT_CODE_FREEZE.json").read_text())
        for name, sha in freeze["files"].items():
            if digest(ROOT/name) != sha:
                raise RuntimeError(f"FROZEN_FILE_HASH_CHANGED:{name}")
        for name, sha in freeze["dependencies"].items():
            if digest(name) != sha:
                raise RuntimeError(f"DEPENDENCY_HASH_CHANGED:{name}")
        if Path("/var/run/reboot-required").exists():
            raise RuntimeError("HOST_REBOOT_REQUIRED_MAINTENANCE_GATE")
        if args.mode == "formal":
            gate = json.loads((ROOT / "PHASE_A_BUDGET_GATE.json").read_text())
            if gate["status"] != "PASS" or gate["conservative_projected_gpu_hours"] > 2:
                raise RuntimeError("PHASE_A_BUDGET_GATE_FAIL")
            for doc in config["documents"]:
                gpu = config["document_gpu_map"][doc]
                smoke = ROOT / "runs" / f"smoke_{doc}_gpu{gpu}.json"
                if digest(smoke) != gate["smoke_sha256"][doc]:
                    raise RuntimeError("SMOKE_EVIDENCE_CHANGED")
                if not json.loads(smoke.read_text())["valid"]:
                    raise RuntimeError("MATCHED_GPU_SMOKE_NOT_PASS")
            wall_limit = gate["per_document_wall_cap_seconds"][args.document]
        else:
            wall_limit = 600
        base = load_module(PRIOR / "run_short_horizon_proxy_v2.py", "l7_verified_proxy_base")
        base._DEADLINE = start_wall + wall_limit - 10
        direction = load_module(base.DIRECTION_RUNNER, "l7_verified_direction_dependencies")
        host = direction.gpu_guard(torch, args.gpu)
        if any(host[k] != config["runtime"][k] for k in ["kernel", "driver", "torch", "cuda_runtime", "selector"]):
            raise RuntimeError("FROZEN_RUNTIME_MISMATCH")
        transport, _, capture_module, core = direction.load_assets(args.gpu)
        conditions = config["conditions"] if args.mode == "formal" else ["IDENTITY", "IDENTITY"]
        starts = config["windows"] if args.mode == "formal" else [256]
        layers = config["layers"]
        doc, tokens, token_path, rotations, rotation_ids = base.load_tokens_and_rotations(
            torch, transport, config, args.document, conditions)
        torch.manual_seed(config["seed"])
        load_started = time.perf_counter()
        capture = capture_module.Capture()
        for parameter in capture.model.parameters():
            parameter.requires_grad_(False)
        capture.model.eval()
        helpers = original_helpers(torch, TRAINER)
        tap = OutputTap(torch, capture, layers)
        observer = ObservedRunner(base, direction, torch, core, capture, tap, helpers, starts, layers)
        observed_core = ObservedCore(core, observer)
        load_seconds = time.perf_counter()-load_started
        publish(ROOT / "runs" / f"{tag}.STARTED.json", {
            "pid": os.getpid(), "started_epoch": start_wall, "gpu_uuid": host["gpu_uuid"],
            "runner_sha256": digest(__file__), "code_freeze_sha256": digest(ROOT/"PREMEASUREMENT_CODE_FREEZE.json"),
            "document": args.document, "mode": args.mode, "max_wall_seconds": wall_limit})
        end = max(starts)+128
        records, fp_logits, fp_states, fp_seconds = base.extract_fp(
            torch, observer, capture, tokens, end, starts, layers)
        parity = base.replay_parity(torch, core, records, fp_states)
        if not parity["passed"]:
            raise RuntimeError("FP_REPLAY_PARITY_FAIL")
        fp_identity = {"post_o_proj": observer.fp_output_hashes, "full_logits": observer.fp_logit_hashes,
            "drivers": {k: direction.tensor_sha(v) for k,v in records.items()},
            "checkpoint_states": {p: direction.tensor_sha(v) for p,v in fp_states.items()}}
        covariances = base.build_covariances(torch, records, starts)
        observer.mode = "student"
        results, counts = [], {}
        for condition in conditions:
            base.budget_guard()
            counts[condition] = counts.get(condition,0)+1
            observer.scores, observer.reference_states = {}, None
            result = base.run_condition(torch, observer, observed_core, capture, tokens, records,
                fp_logits, covariances, rotations[condition].to(capture.device), condition,
                counts[condition], starts, [128], layers, layers.index(20))
            result = augment(result, observer, starts)
            if not result["mechanical_gates"]["passed"]:
                raise RuntimeError("REAL_QDQ_WRITEBACK_GATE_FAIL")
            publish(ROOT / "runs" / f"{tag}.{condition}.{counts[condition]}.json", result)
            results.append(result)
            print(json.dumps({"event": "condition_complete", "document": args.document,
                "condition": condition, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        if args.mode == "smoke" and results[0]["windows"] != results[1]["windows"]:
            raise RuntimeError("MATCHED_FORWARD_REPEAT_DETERMINISM_FAIL")
        tap.close(); tap = None
        capture.close(); capture = None
        total_wall = time.time()-start_wall
        output = {"schema": "matched_original_l7_v1", "document": args.document, "mode": args.mode,
            "gpu_uuid": host["gpu_uuid"], "host": host, "runner_sha256": digest(__file__),
            "historical_trainer_sha256": digest(TRAINER), "config_sha256": digest(ROOT/"PHASE_A_CONFIG.json"),
            "token_file": {"path": str(token_path), "sha256": digest(token_path)},
            "rotation_artifacts": rotation_ids, "model_loading_seconds": load_seconds,
            "fp_reference": {"source": "SHARED_CURRENT_RUNTIME_SAME_GPU_REFERENCE", "tokens": end,
                "tensor_identities": fp_identity, "content_sha256": base.canonical_hash(fp_identity),
                "replay_parity": parity, "helper_real_forward_checks": observer.helper_forward_checks,
                "elapsed_seconds": fp_seconds}, "results": results, "valid": True,
            "started_epoch": start_wall, "completed_epoch": time.time(),
            "accounted_gpu_hours": total_wall/3600, "total_wall_seconds": total_wall,
            "prior_score_set_joined": False, "gradients_or_optimizer_used": False}
        if not base.finite_tree(output):
            raise RuntimeError("NONFINITE_RESULT_TREE")
        publish(path, output)
        publish(path.with_suffix(".sha256.json"), {"path": str(path), "sha256": digest(path)})
        print(json.dumps({"event": "complete", "sha256": digest(path), "path": str(path)}), flush=True)
    except Exception as exc:
        if tap is not None:
            tap.close()
        if capture is not None:
            capture.close()
        publish(stop, {"status": "FAIL_AND_STOP_NO_RETRY", "error": repr(exc),
            "traceback": traceback.format_exc(), "started_epoch": start_wall, "stopped_epoch": time.time(),
            "accounted_gpu_hours_upper_bound": (time.time()-start_wall)/3600,
            "runner_sha256": digest(__file__)})
        raise


if __name__ == "__main__":
    main()
