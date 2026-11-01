"""Qualification/micro/full worker; distinct frozen authority required per mode."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from objective import digest
from run_matched_l7 import ROOT, PRIOR, load_module, publish
from run_phase_b_calibration_v2 import load_inputs
from persistent_engine_v1 import stream
from persistent_stage_common_v1 import RUNS, FREEZE, LATCH, MODES, verify, checked_json, read, complete, fingerprint, latch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, choices=(3, 6), required=True)
    ap.add_argument("--document", required=True)
    ap.add_argument("--mode", choices=tuple(MODES), required=True)
    args = ap.parse_args(); start = time.time(); capture = None
    tag = f"{args.mode}_{args.document}_gpu{args.gpu}"
    path = RUNS / (tag + ".json"); stop = RUNS / (tag + ".STOP.json")
    assert not path.exists() and not stop.exists(), "ALREADY_EXECUTED_NO_RETRY"
    try:
        cfg, selection, freeze = verify()
        entry = next(x for x in selection["documents"] if x["document"] == args.document)
        assert entry["gpu"] == args.gpu and os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu)
        receipt = read(RUNS / (tag + ".LAUNCH_AUTHORIZATION.json"))
        assert receipt["runner_sha256"] == digest(__file__) and receipt["freeze_sha256"] == digest(FREEZE)
        mode = MODES[args.mode]
        if args.mode in ("micro", "formal"):
            for item in selection["documents"]:
                complete("qualification", item["document"], item["gpu"])
        if args.mode == "formal":
            gate = checked_json(ROOT / "PERSISTENT_COMPLETE_COST_GATE_V1.json")
            assert gate["status"] == "PASS" and gate["conservative_entire_task_gpu_hours"] <= 16
            assert receipt["cost_gate_sha256"] == digest(ROOT / "PERSISTENT_COMPLETE_COST_GATE_V1.json")
            for item in selection["documents"]:
                _, p = complete("micro", item["document"], item["gpu"])
                assert digest(p) == gate["micro_sha256"][item["document"]]
        deadline = receipt["epoch"] + receipt["max_wall_seconds"] - 20
        assert time.time() < deadline
        publish(RUNS / (tag + ".STARTED.json"), dict(pid=os.getpid(), started_epoch=start,
            document=args.document, mode=args.mode, gpu_uuid=receipt["gpu_uuid"],
            runner_sha256=digest(__file__), freeze_sha256=digest(FREEZE), max_wall_seconds=receipt["max_wall_seconds"]))
        import torch
        torch.set_grad_enabled(False)
        base = load_module(PRIOR / "run_short_horizon_proxy_v2.py", "persistent_verified_proxy")
        direction = load_module(base.DIRECTION_RUNNER, "persistent_verified_direction")
        host = direction.gpu_guard(torch, args.gpu)
        assert all(host[k] == cfg["runtime"][k] for k in ("kernel", "driver", "torch", "cuda_runtime", "selector"))
        def guard():
            assert not LATCH.exists() and not (ROOT / "PHASE_B_FAIL_CLOSED_STOP.json").exists(), "PEER_STOP_NO_RETRY"
            assert time.time() < deadline, "SAFE_TOKEN_BOUNDARY_WALL_BUDGET_STOP"
        guard()
        _, _, capture_module, core = direction.load_assets(args.gpu)
        conditions = [entry["pair"]["condition_a"], entry["pair"]["condition_b"]]
        document, tokens, token_path, rotations, rotation_ids = load_inputs(torch, cfg, args.document, conditions)
        torch.manual_seed(cfg["seed"])
        tick = time.perf_counter(); capture = capture_module.Capture()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        capture.model.eval()
        assert len(capture.model.model.layers) == 24
        assert not any(p.requires_grad for p in capture.model.parameters())
        rotations = {c: r.to(capture.device) for c, r in rotations.items()}
        loading = time.perf_counter() - tick
        def progress(pos, end, elapsed):
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
            occupants = [x for x in apps.splitlines() if receipt["gpu_uuid"] in x]
            assert occupants and all(int(x.split(",")[1]) == os.getpid() for x in occupants), "MIDRUN_UUID_EXCLUSIVITY_FAIL"
            assert not Path("/var/run/reboot-required").exists(), "MIDRUN_MAINTENANCE_FAIL"
            assert digest(__file__) == receipt["runner_sha256"] and digest(FREEZE) == receipt["freeze_sha256"]
            print(json.dumps(dict(event="persistent_progress", mode=args.mode, document=args.document,
                position=pos, endpoint=end, elapsed_seconds=elapsed, pid=os.getpid(), gpu_uuid=receipt["gpu_uuid"])), flush=True)
        torch.cuda.reset_peak_memory_stats()
        results = []
        for repeat in range(mode["repeats"]):
            guard()
            torch.manual_seed(cfg["seed"])
            result = stream(torch, direction, core, capture, tokens, rotations, cfg["layers"], mode["endpoint"],
                mode["primary"], mode["secondary"], guard, fp_replay=args.mode == "qualification", progress=progress)
            results.append(result)
            capture.current = {}; capture.post = {}; gc.collect(); torch.cuda.empty_cache()
        repeat_pass = args.mode != "qualification" or fingerprint(results[0]) == fingerprint(results[1])
        assert repeat_pass, "PERSISTENT_REPEAT_BITWISE_FAIL"
        for condition, asset in rotation_ids.items():
            assert digest(asset["path"]) == asset["sha256"], "ROTATION_MUTATION_FAIL"
        memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), total_bytes=torch.cuda.get_device_properties(0).total_memory)
        capture.close(); capture = None
        ended = time.time()
        output = dict(schema="prospective_full_hybrid_persistent_qdq_v1", valid=True,
            document=args.document, mode=args.mode, endpoint=mode["endpoint"], gpu_uuid=host["gpu_uuid"], host=host,
            runner_sha256=digest(__file__), engine_sha256=digest(ROOT / "persistent_engine_v1.py"),
            freeze_sha256=digest(FREEZE), pair_selection_sha256=digest(ROOT / "PROSPECTIVE_DOCUMENT_AND_PAIR_SELECTION.json"),
            token_file=dict(path=str(token_path), sha256=digest(token_path)), document_identity=document,
            conditions=conditions, rotation_artifacts=rotation_ids, predicted_winners=entry["predicted_winners"],
            stream=results[0], repeat_bitwise_pass=repeat_pass,
            repeat_numeric_fingerprints=[fingerprint(r) for r in results], gpu_memory=memory,
            model_loading_seconds=loading, started_epoch=start, completed_epoch=ended,
            total_wall_seconds=ended-start, accounted_gpu_hours=(ended-start)/3600,
            scientific_primary_result=args.mode == "formal", qualification_or_micro_not_confirmatory=args.mode != "formal",
            persistent_qdq_disabled_after_anchor=False, gradients_or_optimizer_used=False,
            historical_training_sampling_protocol_replayed=False)
        assert base.finite_tree(output), "NONFINITE_CANONICAL_RESULT"
        publish(path, output); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
        print(json.dumps(dict(event="complete", mode=args.mode, path=str(path), sha256=digest(path))), flush=True)
    except Exception as exc:
        if capture is not None:
            try: capture.close()
            except Exception: pass
        publish(stop, dict(status="FAIL_AND_STOP_NO_RETRY", error=repr(exc), traceback=traceback.format_exc(),
            document=args.document, mode=args.mode, started_epoch=start, stopped_epoch=time.time(),
            accounted_gpu_hours_upper_bound=(time.time()-start)/3600, runner_sha256=digest(__file__)))
        latch(repr(exc), args.mode, run_stop=str(stop))
        raise


if __name__ == "__main__":
    main()
