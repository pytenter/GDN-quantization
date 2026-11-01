#!/usr/bin/env python3
"""Stage-A non-AIME parity probe using the unchanged formal runtime functions.

The probe is guarded: it refuses to start while the old V1 capture is active or
while any GPU compute process exists.  It never samples tokens.  A fixed text
continuation supplies teacher-forced token IDs.

Formal arithmetic is not copied here:
* rotations call the original formal patch method via ``super()._transform``;
* quantization calls the original formal ``quantize_c128`` function;
* quantizer locals are observed with a scoped Python trace, without replacing
  scale/round/clamp/dequant/writeback operations.
"""

from __future__ import annotations

import argparse
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


REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
ROOT = REPO / "experiments/qwen_gdn/QWEN_GDN_FORMAL_EQUIVALENT_MECHANISM_CLOSURE_V2"
BASELINE_RUNNER = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/experiments/aime26/run_qwen_aime26_formal.py")
LEARNED_RUNNER = REPO / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/scripts/run_qwen_c5_c6_formal.py"
DEPLOY_ROOT = REPO / "experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1/deployment"
EXPECTED_LEARNED_RUNNER_SHA = "77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69"
FIXED_PROMPT = "Give the capital of France in one short sentence. This is a deterministic software parity probe, not a mathematics contest problem."
FIXED_CONTINUATION = "The capital of France is Paris."
CONDITIONS = ("Native", "Hadamard", "C5", "C6")


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
        "finite": bool(finite.all().item()) if finite is not None else True,
    }


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safety_gate() -> dict[str, Any]:
    old = subprocess.run(["pgrep", "-af", "mechanism_capture.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    old_lines = [line for line in old.splitlines() if "QWEN_GDN_HADAMARD_VS_LEARNED_ROTATION_MECHANISM_CLOSURE_V1" in line]
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    gpu_pids = [line.strip() for line in gpu.splitlines() if line.strip().isdigit()]
    return {"old_v1_processes": old_lines, "gpu_compute_pids": gpu_pids, "pass": not old_lines and not gpu_pids}


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


def run_once(label, condition, model, tokenizer, baseline, learned, banks, captured):
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
    handles = install_selected_output_hooks(model, capture) if captured else []
    messages = [{"role": "user", "content": FIXED_PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(next(model.parameters()).device)
    teacher = tokenizer(FIXED_CONTINUATION, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    if len(teacher) < 4:
        raise RuntimeError("fixed continuation unexpectedly too short")
    teacher = teacher[:4]
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


def comparison(condition, a, b, c):
    # A uses untouched formal rotation path plus read-only quantizer-local observation.
    # B/C add full read-only rotation/module capture; B-vs-C establishes internal repeatability.
    gates = {
        "ROTATION_INTERMEDIATE_PARITY": "PASS_EXACT" if exact(b["rotation"], c["rotation"]) else "FAIL",
        "ROTATED_TENSOR_PARITY": "PASS_EXACT" if exact(b["rotation"], c["rotation"]) else "FAIL",
        "SCALE_PARITY": "PASS_EXACT" if exact(a["quantizer_calls"], b["quantizer_calls"]) else "FAIL",
        "QCODE_PARITY": "PASS_EXACT" if exact(a["quantizer_calls"], b["quantizer_calls"]) else "FAIL",
        "DEQUANT_PARITY": "PASS_EXACT" if exact(a["quantizer_calls"], b["quantizer_calls"]) else "FAIL",
        "WRITEBACK_PARITY": "PASS_EXACT" if exact(a["writeback_states"], b["writeback_states"]) else "FAIL",
        "NEXT_TOKEN_STATE_PARITY": "PASS_EXACT" if exact(a["next_token_state_read"], b["next_token_state_read"]) else "FAIL",
        "SELECTED_OUTPUT_PARITY": "PASS_EXACT" if exact(b["selected_post_gdn"], c["selected_post_gdn"]) and exact(b["selected_block_output"], c["selected_block_output"]) else "FAIL",
        "LOGIT_PARITY": "PASS_EXACT" if exact(a["logits"], b["logits"]) else "FAIL",
    }
    if condition == "Native":
        gates["ROTATION_INTERMEDIATE_PARITY"] = "NOT_APPLICABLE_IDENTITY"
        gates["ROTATED_TENSOR_PARITY"] = "NOT_APPLICABLE_IDENTITY"
    return gates


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
    model, tokenizer = learned.load_model_and_tokenizer()
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

    results = {"probe": {"prompt": FIXED_PROMPT, "continuation": FIXED_CONTINUATION}, "conditions": {}}
    raw_dir = ROOT / "parity/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        runs = {
            "A_formal": run_once("A_formal", condition, model, tokenizer, baseline, learned, banks, False),
            "B_instrumented": run_once("B_instrumented", condition, model, tokenizer, baseline, learned, banks, True),
            "C_instrumented_repeat": run_once("C_instrumented_repeat", condition, model, tokenizer, baseline, learned, banks, True),
        }
        for label, payload in runs.items():
            (raw_dir / f"{condition}_{label}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results["conditions"][condition] = comparison(condition, runs["A_formal"], runs["B_instrumented"], runs["C_instrumented_repeat"])

    hard = all(
        gates["QCODE_PARITY"] == "PASS_EXACT" and gates["WRITEBACK_PARITY"] == "PASS_EXACT" and gates["LOGIT_PARITY"] == "PASS_EXACT"
        for gates in results["conditions"].values()
    )
    results.update({
        "FORMAL_REPLAY_PARITY_CLOSURE": "PASS" if hard else "FAIL",
        "MECHANISM_REPLAY_AUTHORIZED": "YES" if hard else "NO",
        "FORMAL_QUANTIZER_SHARED_PATH_GATE": "PASS" if hard else "FAIL",
        "FORMAL_ROTATION_PATH_GATE": "PASS" if hard else "FAIL",
        "FORMAL_WRITEBACK_PATH_GATE": "PASS" if hard else "FAIL",
        "safety_gate": safety,
    })
    output = ROOT / "parity/formal_replay_parity.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    raise SystemExit(0 if hard else 2)


if __name__ == "__main__":
    main()
