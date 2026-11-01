"""Bounded post-calibration check of every real layer mask in both periods."""
import importlib.util
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pace_buffer import PACEBuffer
from pace_runtime import GDN_LAYERS, load_head_mask


def run_actual_mask_check(root, *, device="cuda:0", progress=None,
                          native_reference=None, head_mask_path=None):
    root = Path(root)
    path = Path(head_mask_path or root / "calibration/head_mask.json")
    masks, mask_sha, _ = load_head_mask(path, GDN_LAYERS)
    spec = importlib.util.spec_from_file_location("pace_actual_mask_scalar_oracle", ROOT / "tests/test_pace_math.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    if str(device).startswith("cuda") and native_reference is None:
        raise RuntimeError("Actual CUDA mask gate requires the pinned native helper callback")
    reference = native_reference or oracle._direct_reference
    cases = []
    # The calibrated32-head mask routing is the subject of this check. G32 K32
    # and V3 keep all24masks inexpensive; model128x128 was covered by GateB.
    values = oracle._tokens(10, h=32, u=16, k=32, v=3, seed=4816, device=device)
    with torch.inference_mode():
        for layer in GDN_LAYERS:
            mask = masks[layer]
            for period in (8, 1):
                buffer = PACEBuffer(period, mask, key_heads=16)
                buffer.set_prompt_length(1)
                state = torch.zeros((1, 32, 32, 3), dtype=torch.float32, device=device)
                for index, tensors in enumerate(values):
                    q, key, value, beta, g = tensors
                    recovered = key if index == 0 else oracle._scalar_key(key[:, ::2]).repeat_interleave(2, dim=1)
                    expected, state = reference(q, recovered, value, beta, g, state)
                    actual = buffer.step(q, key, value, beta, g)
                    if not torch.equal(actual, expected):
                        raise AssertionError("Actual-mask output differs from independent scalar-codec/native golden")
                    if index == 0 or index % period == 0:
                        state = oracle._scalar_state(state, mask)
                    oracle._state_close(buffer.state_view_for_test(), state)
                    if any(t.is_floating_point() and tuple(t.shape) == (1, 32, 32, 3)
                           for t in oracle._all_owned_tensors(buffer)):
                        raise AssertionError("Actual-mask production buffer retains a dense FP state")
                report = buffer.finalize("preserve")
                if (report["quantization_count"] != 1+9//period
                        or report["pending_tokens"] != 9 % period
                        or report["base_nbytes"] != 2172
                        or report["base_bpv"] != 5.65625):
                    raise AssertionError("Actual-mask writeback/tail/physical storage differs")
                if sum(t.numel()*t.element_size() for t in buffer.resident_tensors().values()) != report["total_resident_tensor_nbytes"]:
                    raise AssertionError("Actual-mask tensor inventory differs")
                cases.append(dict(layer=layer, decode_period=period, tokens=10,
                                  quantization_count=report["quantization_count"],
                                  pending_tokens=report["pending_tokens"], base_nbytes=report["base_nbytes"],
                                  base_bpv=report["base_bpv"], status="PASS"))
            if progress:
                progress("ACTUAL_HEAD_MASK_LAYER_CHECKED", layer=layer, cases=len(cases))
    return dict(status="PASS", actual_cuda=str(device).startswith("cuda"), device=str(device),
                scope="postcalibration_actual24masks_bothperiods_engineering_only_not_formal",
                head_mask_sha256=mask_sha, layers=24, cases=cases,
                state_shape=[1, 32, 32, 3], tokens_per_case=10,
                native_source_sha256=getattr(native_reference, "source_sha256", None),
                native_callback_provided=native_reference is not None)
