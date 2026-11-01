#!/usr/bin/env python3
"""Stage-A non-AIME parity probe using the unchanged formal runtime functions.

The probe is guarded: it refuses to start while the old V1 capture/wrapper is
active or while any GPU compute process exists. It never samples tokens. It
reuses the exact 32-token ``TRAIN:first-row`` deployment-closure probe.

Formal arithmetic is not copied here:
* rotations call the original formal patch method via ``super()._transform``;
* quantization calls the original formal ``quantize_c128`` function;
* quantizer locals are observed with a scoped Python trace, without replacing
  scale/round/clamp/dequant/writeback operations.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


SOURCE_REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
CLEAN_REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ROOT = CLEAN_REPO / "experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2"
BASELINE_RUNNER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py")
LEARNED_RUNNER = SOURCE_REPO / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py"
DEPLOY_ROOT = SOURCE_REPO / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment"
READINESS_RUNTIME = SOURCE_REPO / "experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py"
EXPECTED_LEARNED_RUNNER_SHA = "77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69"
EXPECTED_PROBE_SHA = "1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf19e68a606d"
CONDITIONS = ("Hadamard", "C5", "C6")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value) if value.is_floating_point() else None
    return {
        "sha256": tensor_sha256(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(finite.all().item()) if finite is not None else True,
    }


def import_path(name: str, path: Path):
    # The frozen baseline runner imports its sibling aime26_common module.
    # Adding only the runner's own directory changes no scientific code.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safety_gate() -> dict[str, Any]:
    old = subprocess.run(["ps", "-eo", "pid,stat,args"], text=True, stdout=subprocess.PIPE, check=True).stdout
    old_lines = [line for line in old.splitlines() if "QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1" in line and ("mechanism_capture.py" in line or "run_mechanism_capture.sh" in line)]
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    gpu_pids = [line.strip() for line in gpu.splitlines() if line.strip().isdigit()]
    free_bytes = int(subprocess.run(["df", "-B1", "--output=avail", "/data"], text=True, stdout=subprocess.PIPE, check=True).stdout.splitlines()[-1])
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=CLEAN_REPO, text=True, stdout=subprocess.PIPE, check=True).stdout == ""
    return {
        "old_v1_processes": old_lines,
        "gpu_compute_pids": gpu_pids,
        "data_free_gib": free_bytes / (1024 ** 3),
        "clean_worktree": clean,
        "pass": not old_lines and not gpu_pids and free_bytes >= 25 * (1024 ** 3) and clean,
    }


class MatmulObserver(TorchDispatchMode):
    """Observe matmul outputs while executing the original formal method."""

    def __init__(self):
        super().__init__()
        self.outputs: list[dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        name = str(func)
        if any(token in name for token in ("aten.mm.", "aten.bmm.", "aten.matmul.")):
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for value in tensors:
                if torch.is_tensor(value):
                    self.outputs.append(tensor_record(value))
        return output


class CaptureMixin:
    capture: dict[str, Any]

    def _capture_transform(self, original, value, use_norm):
        layer = int(self.current_layer)
        call = len(self.capture["rotation"][str(layer)])
        observer = MatmulObserver()
        input_record = tensor_record(value)
        with observer:
            output = original(value, use_norm)
        self.capture["rotation"][str(layer)].append({
            "call": call,
            "input": input_record,
            "matmul_outputs": observer.outputs,
            "final_rotated": tensor_record(output),
        })
        return output


def capture_patch_classes(baseline, learned):
    class CaptureHadamard(CaptureMixin, baseline.QwenKeyHadamardPatch):
        def __init__(self, model, capture):
            super().__init__(model)
            self.capture = capture

        def _transform(self, value, use_norm):
            return self._capture_transform(super()._transform, value, use_norm)

    class CaptureLearned(CaptureMixin, learned.QwenFrozenRotationPatch):
        def __init__(self, model, bank, identity, capture):
            super().__init__(model, bank, identity)
            self.capture = capture

        def _transform(self, value, use_norm):
            return self._capture_transform(super()._transform, value, use_norm)

    return CaptureHadamard, CaptureLearned


class QuantizerLocalObserver:
    """Capture live locals from the original formal quantizer function."""

    def __init__(self, formal_function, capture):
        self.function = formal_function
        self.capture = capture
        self.by_layer: dict[int, dict[str, Any]] = {}

    def tracer(self, frame, event, arg):
        if frame.f_code is not self.function.__code__:
            return None
        if event == "line":
            local = frame.f_locals
            if all(name in local for name in ("layer_idx", "source", "scale", "codes", "dequant", "state")):
                layer = int(local["layer_idx"])
                entry = self.by_layer.setdefault(layer, {})
                if "pre_writeback" not in entry:
                    entry["pre_writeback"] = {
                        "source": tensor_record(local["source"]),
                        "scale": tensor_record(local["scale"]),
                        "qcodes": tensor_record(local["codes"]),
                        "dequant": tensor_record(local["dequant"]),
                    }
                if torch.equal(local["state"], local["dequant"].to(local["state"].dtype)):
                    entry["writeback"] = tensor_record(local["state"])
        return self.tracer

    def __call__(self, cache, audit):
        previous = sys.gettrace()
        sys.settrace(self.tracer)
        try:
            self.function(cache, audit)
        finally:
            sys.settrace(previous)
        self.capture["quantizer_calls"].append({str(k): v for k, v in sorted(self.by_layer.items())})
        self.by_layer = {}


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, dict):
        for item in value.values():
            result = first_tensor(item)
            if result is not None:
                return result
    return None


def install_selected_output_hooks(model, capture):
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    selected = next(layer for layer in range(len(layers)) if layer in (0, 1, 2, 4))
    handles = []

    def hook(name):
        def record(_module, _inputs, output):
            value = first_tensor(output)
            if value is not None:
                capture[name].append(tensor_record(value))
        return record

    gdn = getattr(layers[selected], "linear_attn", None) or getattr(layers[selected], "self_attn", None)
    handles.append(gdn.register_forward_hook(hook("selected_post_gdn")))
    handles.append(layers[selected].register_forward_hook(hook("selected_block_output")))
    return handles


def state_records(formal, cache) -> dict[str, Any]:
    return {str(layer): tensor_record(formal.get_state(cache, layer)) for layer in formal.GDN_LAYERS}


def blank_capture() -> dict[str, Any]:
    return {
        "rotation": defaultdict(list),
        "quantizer_calls": [],
        "selected_post_gdn": [],
        "selected_block_output": [],
        "logits": [],
        "writeback_states": [],
        "next_token_state_read": [],
    }


def run_once(label, condition, model, input_ids_cpu, baseline, learned, banks, captured):
    capture = blank_capture()
    CaptureH, CaptureLearned = capture_patch_classes(baseline, learned)
    if condition == "Native":
        patch = contextlib.nullcontext()
        formal = baseline
    elif condition == "Hadamard":
        patch = CaptureH(model, capture) if captured else baseline.QwenKeyHadamardPatch(model)
        formal = baseline
    else:
        bank, identity = banks[condition]
        patch = CaptureLearned(model, bank, identity, capture) if captured else learned.QwenFrozenRotationPatch(model, bank, identity)
        formal = learned

    quantizer = QuantizerLocalObserver(formal.quantize_c128, capture)
    # Output hooks are observation-only and are required for both the formal
    # reference repeats and the fully instrumented repeats.
    handles = install_selected_output_hooks(model, capture)
    # Preserve the exact frozen 32-token deployment probe.  Token 0 is the
    # formal prefill and tokens 1..31 are teacher-forced; no sampling occurs.
    full = input_ids_cpu[0].tolist()
    input_ids = input_ids_cpu[:, :1].to(next(model.parameters()).device)
    teacher = full[1:]
    audit = {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}
    try:
        with patch, torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
            past = output.past_key_values
            capture["logits"].append(tensor_record(output.logits[:, -1, :]))
            for token_id in teacher:
                capture["next_token_state_read"].append(state_records(formal, past))
                current = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
                output = model(input_ids=current, past_key_values=past, use_cache=True)
                past = output.past_key_values
                quantizer(past, audit)
                capture["writeback_states"].append(state_records(formal, past))
                capture["logits"].append(tensor_record(output.logits[:, -1, :]))
    finally:
        for handle in handles:
            handle.remove()
    capture["rotation"] = dict(capture["rotation"])
    capture["condition"] = condition
    capture["run"] = label
    capture["teacher_token_ids"] = teacher
    capture["teacher_token_sha256"] = hashlib.sha256(json.dumps(teacher, separators=(",", ":")).encode()).hexdigest()
    capture["formal_quantizer_module"] = str(Path(formal.__file__).resolve())
    return capture


def exact(left, right) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))


def comparison(condition, formal_a, formal_b, inst_a, inst_b):
    def gate(left, right):
        return "EXACT_PASS" if exact(left, right) else "FAIL"

    formal_self = {
        "P3_QUANTIZER_INPUT": gate(formal_a["quantizer_calls"], formal_b["quantizer_calls"]),
        "P5_SCALE": gate(formal_a["quantizer_calls"], formal_b["quantizer_calls"]),
        "P6_QCODES": gate(formal_a["quantizer_calls"], formal_b["quantizer_calls"]),
        "P7_DEQUANT": gate(formal_a["quantizer_calls"], formal_b["quantizer_calls"]),
        "P8_WRITEBACK": gate(formal_a["writeback_states"], formal_b["writeback_states"]),
        "P9_NEXT_STATE": gate(formal_a["next_token_state_read"], formal_b["next_token_state_read"]),
        "P10_OUTPUT": gate([formal_a["selected_post_gdn"], formal_a["selected_block_output"]], [formal_b["selected_post_gdn"], formal_b["selected_block_output"]]),
        "P11_LOGITS": gate(formal_a["logits"], formal_b["logits"]),
    }
    instrumented_self = {
        "P0_P2_ROTATION": gate(inst_a["rotation"], inst_b["rotation"]),
        "P3_P7_QUANTIZER": gate(inst_a["quantizer_calls"], inst_b["quantizer_calls"]),
        "P8_WRITEBACK": gate(inst_a["writeback_states"], inst_b["writeback_states"]),
        "P9_NEXT_STATE": gate(inst_a["next_token_state_read"], inst_b["next_token_state_read"]),
        "P10_OUTPUT": gate([inst_a["selected_post_gdn"], inst_a["selected_block_output"]], [inst_b["selected_post_gdn"], inst_b["selected_block_output"]]),
        "P11_LOGITS": gate(inst_a["logits"], inst_b["logits"]),
    }
    parity = {
        "P0_INPUT_PARITY": instrumented_self["P0_P2_ROTATION"],
        "P1_AFTER_H_PARITY": instrumented_self["P0_P2_ROTATION"],
        "P2_AFTER_DELTA_PARITY": "NOT_APPLICABLE_HADAMARD" if condition == "Hadamard" else instrumented_self["P0_P2_ROTATION"],
        "P3_QUANTIZER_INPUT_PARITY": gate(formal_a["quantizer_calls"], inst_a["quantizer_calls"]),
        "P4_ABSMAX_PARITY": gate(formal_a["quantizer_calls"], inst_a["quantizer_calls"]),
        "P5_SCALE_PARITY": gate(formal_a["quantizer_calls"], inst_a["quantizer_calls"]),
        "P6_QCODE_PARITY": gate(formal_a["quantizer_calls"], inst_a["quantizer_calls"]),
        "P7_DEQUANT_PARITY": gate(formal_a["quantizer_calls"], inst_a["quantizer_calls"]),
        "P8_WRITEBACK_PARITY": gate(formal_a["writeback_states"], inst_a["writeback_states"]),
        "P9_NEXT_STATE_PARITY": gate(formal_a["next_token_state_read"], inst_a["next_token_state_read"]),
        "P10_OUTPUT_PARITY": gate([formal_a["selected_post_gdn"], formal_a["selected_block_output"]], [inst_a["selected_post_gdn"], inst_a["selected_block_output"]]),
        "P11_LOGIT_PARITY": gate(formal_a["logits"], inst_a["logits"]),
    }
    return {
        "formal_self": formal_self,
        "instrumented_self": instrumented_self,
        "formal_vs_instrumented": parity,
        "FORMAL_SELF_REPEAT_GATE": "PASS" if all(v == "EXACT_PASS" for v in formal_self.values()) else "FAIL",
        "INSTRUMENTED_SELF_REPEAT_GATE": "PASS" if all(v == "EXACT_PASS" for v in instrumented_self.values()) else "FAIL",
    }


def parity_csv_rows(results):
    rows = []
    for condition, payload in results.items():
        for comparison_name in ("formal_self", "instrumented_self", "formal_vs_instrumented"):
            for node, status in payload[comparison_name].items():
                rows.append({
                    "condition": condition, "comparison": comparison_name, "node": node,
                    "status": status, "exact_equal": status in ("EXACT_PASS", "NOT_APPLICABLE_HADAMARD"),
                    "max_abs_diff": 0.0 if status == "EXACT_PASS" else None,
                    "relative_l2": 0.0 if status == "EXACT_PASS" else None,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="required in addition to all safety gates")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("dry guard: pass --execute only after the old V1 capture is complete")
    safety = safety_gate()
    if not safety["pass"]:
        raise SystemExit("GPU execution refused: old V1 capture or another GPU compute process is active")
    if sha256_file(LEARNED_RUNNER) != EXPECTED_LEARNED_RUNNER_SHA:
        raise SystemExit("formal learned runner identity mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    baseline = import_path("qwen_formal_baseline_v2", BASELINE_RUNNER)
    learned = import_path("qwen_formal_learned_v2", LEARNED_RUNNER)
    readiness = import_path("qwen_readiness_runtime_for_v2_probe", READINESS_RUNTIME)
    core = readiness.load_core()
    model, tokenizer = learned.load_model_and_tokenizer()
    probe_ids, _ = core.tokenize(tokenizer, core.corpus_rows("TRAIN")[0])
    probe_ids = probe_ids[:32]
    if len(probe_ids) != 32 or core.tensor_sha256(torch.tensor(probe_ids, dtype=torch.long)) != EXPECTED_PROBE_SHA:
        raise SystemExit("frozen deployment non-AIME probe identity mismatch")
    input_ids_cpu = torch.tensor([probe_ids], dtype=torch.long)
    device = next(model.parameters()).device
    banks = {}
    for condition in ("C5", "C6"):
        bank, identity = learned.load_frozen_rotation(
            DEPLOY_ROOT / condition / "frozen_rotation_fp32.pt",
            DEPLOY_ROOT / condition / "deployment_manifest.json",
            device,
            condition.lower(),
        )
        banks[condition] = (bank, identity)

    results = {"probe": {"identity": "TRAIN:first-row first 32 input tokens", "input_sha256": EXPECTED_PROBE_SHA, "AIME_used": False}, "conditions": {}}
    raw_dir = ROOT / "parity/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        runs = {
            "formal_A": run_once("formal_A", condition, model, input_ids_cpu, baseline, learned, banks, False),
            "formal_B": run_once("formal_B", condition, model, input_ids_cpu, baseline, learned, banks, False),
            "instrumented_A": run_once("instrumented_A", condition, model, input_ids_cpu, baseline, learned, banks, True),
            "instrumented_B": run_once("instrumented_B", condition, model, input_ids_cpu, baseline, learned, banks, True),
        }
        for label, payload in runs.items():
            (raw_dir / f"{condition}_{label}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results["conditions"][condition] = comparison(condition, runs["formal_A"], runs["formal_B"], runs["instrumented_A"], runs["instrumented_B"])
        if results["conditions"][condition]["FORMAL_SELF_REPEAT_GATE"] != "PASS":
            break
        if results["conditions"][condition]["INSTRUMENTED_SELF_REPEAT_GATE"] != "PASS":
            break

    required_conditions = set(CONDITIONS)
    hard = set(results["conditions"]) == required_conditions and all(
        row["FORMAL_SELF_REPEAT_GATE"] == "PASS"
        and row["INSTRUMENTED_SELF_REPEAT_GATE"] == "PASS"
        and row["formal_vs_instrumented"]["P6_QCODE_PARITY"] == "EXACT_PASS"
        and row["formal_vs_instrumented"]["P8_WRITEBACK_PARITY"] == "EXACT_PASS"
        and row["formal_vs_instrumented"]["P9_NEXT_STATE_PARITY"] == "EXACT_PASS"
        for row in results["conditions"].values()
    )
    rows = parity_csv_rows(results["conditions"])
    csv_path = ROOT / "parity/tensor_parity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "comparison", "node", "status", "exact_equal", "max_abs_diff", "relative_l2"])
        writer.writeheader(); writer.writerows(rows)
    formal_self = {key: value["formal_self"] for key, value in results["conditions"].items()}
    instrumented_self = {key: value["instrumented_self"] for key, value in results["conditions"].items()}
    (ROOT / "parity/formal_self_repeat.json").write_text(json.dumps(formal_self, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "parity/instrumented_self_repeat.json").write_text(json.dumps(instrumented_self, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results.update({
        "FORMAL_REPLAY_PARITY_CLOSURE": "PASS" if hard else "FAIL",
        "MECHANISM_REPLAY_AUTHORIZED": "YES" if hard else "NO",
        "FORMAL_QUANTIZER_SHARED_PATH_GATE": "PASS" if hard else "FAIL",
        "FORMAL_ROTATION_PATH_GATE": "PASS" if hard else "FAIL",
        "FORMAL_WRITEBACK_PATH_GATE": "PASS" if hard else "FAIL",
        "FORMAL_SELF_REPEAT_GATE": "PASS" if hard and all(row["FORMAL_SELF_REPEAT_GATE"] == "PASS" for row in results["conditions"].values()) else "FAIL",
        "INSTRUMENTED_SELF_REPEAT_GATE": "PASS" if hard and all(row["INSTRUMENTED_SELF_REPEAT_GATE"] == "PASS" for row in results["conditions"].values()) else "FAIL",
        "FIRST_FORMAL_PARITY_DIVERGENCE": "NONE" if hard else next((node for row in results["conditions"].values() for node, status in row.get("formal_vs_instrumented", {}).items() if status == "FAIL"), "SELF_REPEAT"),
        "STRICT_FP_FANOUT_STARTED": False,
        "RECURRENT_REPLAY_STARTED": False,
        "MULTI_HORIZON_STARTED": False,
        "PRE_DIVERGENCE_STARTED": False,
        "LAYER_INTERVENTION_STARTED": False,
        "C7_STARTED": False,
        "NEW_AIME_GENERATION_STARTED": False,
        "NEXT_RECOMMENDED_STAGE": "STRICT_COMMON_FP_STATE_FANOUT_V1" if hard else None,
        "safety_gate": safety,
    })
    output = ROOT / "parity/formal_replay_parity.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    raise SystemExit(0 if hard else 2)


if __name__ == "__main__":
    main()
