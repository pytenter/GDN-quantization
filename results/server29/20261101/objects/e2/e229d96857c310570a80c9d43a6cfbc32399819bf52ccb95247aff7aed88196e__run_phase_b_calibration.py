"""Independent-input adapter around unchanged, real Phase A calibration functions.

Short calibration only: end=384 smoke or end=1152 full16; no 32K outcome.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
from objective import original_helpers, digest
from run_matched_l7 import (ROOT, PRIOR, TRAINER, publish, load_module, OutputTap,
                            ObservedRunner, ObservedCore, augment)
from phase_b_calibration_common import RUNS, CONFIG, FREEZE, read, checked_json, verify


def load_inputs(torch, cfg, document, conditions):
    selection = checked_json(ROOT / "PROSPECTIVE_DOCUMENT_SELECTION.json")
    entry = next(x for x in selection["selected_documents"] if x["document_id"] == document)
    token_path = Path(entry["token_file"]["path"])
    assert digest(token_path) == entry["token_file"]["sha256"]
    assert digest(entry["raw_text"]["path"]) == entry["raw_text"]["sha256"]
    tokens = read(token_path)
    assert isinstance(tokens, list) and len(tokens) == 32769
    assert all(type(x) is int and x >= 0 for x in tokens), "TOKEN_LAYOUT_CHANGED"
    identities = read(ROOT / "SOURCE_IDENTITIES.json")["rotations"]
    rotations, ids = {}, {}
    for condition in sorted(set(conditions)):
        item = identities[condition]
        assert digest(item["path"]) == item["sha256"], "ROTATION_CHANGED"
        artifact = torch.load(item["path"], map_location="cpu", weights_only=False)
        assert artifact["layer_ids"] == cfg["layers"]
        mapping = artifact["rotations"]
        matrices = [mapping[layer] if layer in mapping else mapping[str(layer)] for layer in cfg["layers"]]
        tensor = torch.stack([x.float() for x in matrices], dim=0)
        assert tuple(tensor.shape) == (18, 16, 128, 128) and torch.isfinite(tensor).all()
        rotations[condition], ids[condition] = tensor, item
    return entry, tokens, token_path, rotations, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True)
    ap.add_argument("--gpu", type=int, choices=(3, 6), required=True)
    ap.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = ap.parse_args()
    tag = f"{args.mode}_{args.document}_gpu{args.gpu}"
    path, stop = RUNS / f"{tag}.json", RUNS / f"{tag}.STOP.json"
    assert not path.exists() and not stop.exists(), "ALREADY_EXECUTED_NO_RETRY"
    start, capture, tap = time.time(), None, None
    try:
        cfg = verify()
        assert args.document in cfg["documents"] and cfg["document_gpu_map"][args.document] == args.gpu
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu)
        receipt = read(RUNS / f"{tag}.LAUNCH_AUTHORIZATION.json")
        assert receipt["runner_sha256"] == digest(__file__)
        assert receipt["freeze_sha256"] == digest(FREEZE)
        wall_limit = receipt["max_wall_seconds"]
        if args.mode == "formal":
            gate = checked_json(ROOT / "PHASE_B_CALIBRATION_BUDGET_GATE.json")
            assert gate["status"] == "PASS" and gate["conservative_projected_total_gpu_hours"] <= 16
            for doc in cfg["documents"]:
                gpu = cfg["document_gpu_map"][doc]
                smoke_path = RUNS / f"smoke_{doc}_gpu{gpu}.json"
                smoke = checked_json(smoke_path)
                assert digest(smoke_path) == gate["smoke_sha256"][doc] and smoke["valid"]
                assert smoke["results"][0]["windows"] == smoke["results"][1]["windows"]
        publish(RUNS / f"{tag}.STARTED.json", dict(pid=os.getpid(), started_epoch=start,
            gpu_uuid=cfg["gpu_uuids"][str(args.gpu)], document=args.document, mode=args.mode,
            max_wall_seconds=wall_limit, runner_sha256=digest(__file__), freeze_sha256=digest(FREEZE)))
        import torch
        torch.set_grad_enabled(False)
        base = load_module(PRIOR / "run_short_horizon_proxy_v2.py", "l7_b_verified_proxy")
        base._DEADLINE = start + wall_limit - 10
        original_budget_guard = base.budget_guard
        def fail_closed_budget_guard():
            assert not (ROOT / "PHASE_B_FAIL_CLOSED_STOP.json").exists(), "PEER_OR_MONITOR_STOP_NO_RETRY"
            original_budget_guard()
        base.budget_guard = fail_closed_budget_guard
        direction = load_module(base.DIRECTION_RUNNER, "l7_b_verified_direction")
        host = direction.gpu_guard(torch, args.gpu)
        assert all(host[k] == cfg["runtime"][k] for k in ("kernel", "driver", "torch", "cuda_runtime", "selector"))
        _, _, capture_module, core = direction.load_assets(args.gpu)
        conditions = cfg["conditions"] if args.mode == "formal" else ["IDENTITY"] * 2
        starts = cfg["windows"] if args.mode == "formal" else [256]
        layers = cfg["layers"]
        entry, tokens, token_path, rotations, rotation_ids = load_inputs(torch, cfg, args.document, conditions)
        torch.manual_seed(cfg["seed"])
        load_start = time.perf_counter()
        capture = capture_module.Capture()
        for parameter in capture.model.parameters():
            parameter.requires_grad_(False)
        capture.model.eval()
        tap = OutputTap(torch, capture, layers)
        helpers = original_helpers(torch, TRAINER)
        observer = ObservedRunner(base, direction, torch, core, capture, tap, helpers, starts, layers)
        observed_core = ObservedCore(core, observer)
        load_seconds = time.perf_counter() - load_start
        end = max(starts) + 128
        assert end in (384, 1152), "CALIBRATION_ONLY_LIMIT"
        records, fp_logits, fp_states, fp_seconds = base.extract_fp(torch, observer, capture, tokens, end, starts, layers)
        parity = base.replay_parity(torch, core, records, fp_states)
        assert parity["passed"], "FP_REPLAY_PARITY_FAIL"
        fp_identity = dict(post_o_proj=observer.fp_output_hashes, full_logits=observer.fp_logit_hashes,
            drivers={k: direction.tensor_sha(v) for k, v in records.items()},
            checkpoint_states={p: direction.tensor_sha(v) for p, v in fp_states.items()})
        covariances = base.build_covariances(torch, records, starts)
        observer.mode = "student"
        results, counts = [], {}
        for condition in conditions:
            base.budget_guard()
            counts[condition] = counts.get(condition, 0) + 1
            observer.scores, observer.reference_states = {}, None
            result = base.run_condition(torch, observer, observed_core, capture, tokens, records,
                fp_logits, covariances, rotations[condition].to(capture.device), condition,
                counts[condition], starts, [128], layers, layers.index(cfg["auxiliary_layer"]))
            augment(result, observer, starts)
            assert result["mechanical_gates"]["passed"], "REAL_QDQ_WRITEBACK_GATE_FAIL"
            publish(RUNS / f"{tag}.{condition}.{counts[condition]}.json", result)
            results.append(result)
            print(json.dumps(dict(event="condition_complete", document=args.document,
                condition=condition, elapsed_seconds=result["elapsed_seconds"])), flush=True)
        if args.mode == "smoke":
            assert results[0]["windows"] == results[1]["windows"], "REPEAT_DETERMINISM_FAIL"
        tap.close(); tap = None
        capture.close(); capture = None
        for condition, item in rotation_ids.items():
            assert digest(item["path"]) == item["sha256"], "ROTATION_MUTATION_FAIL"
        total = time.time() - start
        output = dict(schema="independent_matched_original_l7_calibration_v1", document=args.document,
            mode=args.mode, gpu_uuid=host["gpu_uuid"], host=host, runner_sha256=digest(__file__),
            config_sha256=digest(CONFIG), freeze_sha256=digest(FREEZE), historical_trainer_sha256=digest(TRAINER),
            token_file=dict(path=str(token_path), sha256=digest(token_path)), document_identity=entry,
            rotation_artifacts=rotation_ids, model_loading_seconds=load_seconds,
            fp_reference=dict(source="SHARED_CURRENT_RUNTIME_SAME_GPU_REFERENCE", tokens=end,
                tensor_identities=fp_identity, content_sha256=base.canonical_hash(fp_identity),
                replay_parity=parity, helper_real_forward_checks=observer.helper_forward_checks,
                elapsed_seconds=fp_seconds), results=results, valid=True,
            started_epoch=start, completed_epoch=time.time(), accounted_gpu_hours=total / 3600,
            total_wall_seconds=total, historical_training_sampling_protocol_replayed=False,
            prior_score_set_joined=False, gradients_or_optimizer_used=False,
            long_horizon_outcome_generated_or_used=False)
        assert base.finite_tree(output), "NONFINITE_RESULT_TREE"
        publish(path, output)
        publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
        print(json.dumps(dict(event="complete", path=str(path), sha256=digest(path))), flush=True)
    except Exception as exc:
        if tap is not None:
            tap.close()
        if capture is not None:
            capture.close()
        publish(stop, dict(status="FAIL_AND_STOP_NO_RETRY", error=repr(exc),
            traceback=traceback.format_exc(), started_epoch=start, stopped_epoch=time.time(),
            accounted_gpu_hours_upper_bound=(time.time() - start) / 3600,
            runner_sha256=digest(__file__)))
        latch = ROOT / "PHASE_B_FAIL_CLOSED_STOP.json"
        try:
            publish(latch, dict(status="FAIL_CLOSED_NO_RETRY", run_stop_path=str(stop),
                epoch=time.time(), no_signals_used=True))
        except FileExistsError:
            pass
        raise


if __name__ == "__main__":
    main()
