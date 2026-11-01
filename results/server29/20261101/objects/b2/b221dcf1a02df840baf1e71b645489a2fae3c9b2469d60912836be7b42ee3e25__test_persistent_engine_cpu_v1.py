"""CPU-only synthetic engine checks. Never claims real GPU qualification."""
import hashlib
import json
import os
import types
from pathlib import Path
import torch
from persistent_engine_v1 import full_kl, summarize, PersistentAudit, hybrid_inventory, stream


def qdq(state):
    source = state.float()
    scale = source.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.
    codes = torch.round(source / scale).clamp(-127, 127)
    return (codes * scale).to(state.dtype), codes.to(torch.int8), scale


class Cache:
    def __init__(self):
        self.layers = [types.SimpleNamespace(keys=None, values=None) for _ in range(3)]

    def get_seq_length(self, layer_idx):
        return self.layers[layer_idx].keys.shape[-2]


class Capture:
    def __init__(self):
        self.model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[None] * 3))
        self.probe = types.SimpleNamespace(current_layer=None)
        self.kernel = lambda initial_state=None: torch.zeros((1, 16, 128, 128)) if initial_state is None else initial_state + .007


def cache_state(capture, cache, layer):
    return cache.layers[layer].keys


def quantize(torch, core, capture, cache, rotations):
    result = {}
    for index, layer in enumerate((0, 2)):
        state = cache_state(capture, cache, layer)
        dequant, codes, scale = core.qdq_r128(state.float().matmul(rotations[index][None]))
        state.copy_(dequant.matmul(rotations[index][None].transpose(-1, -2)))
        result[layer] = dict(scale_min=float(scale.min()), scale_max=float(scale.max()), scale_mean=float(scale.mean()))
    return result


def step(torch, capture, cache, token, position):
    cache = Cache() if cache is None else cache
    for layer in (0, 2):
        capture.probe.current_layer = layer
        cache.layers[layer].keys = capture.kernel(initial_state=cache.layers[layer].keys)
        cache.layers[layer].values = tuple(torch.ones((1, 2, 3)) * position for _ in range(3))
    cache.layers[1].keys = torch.ones((1, 2, position + 1, 4))
    cache.layers[1].values = torch.ones((1, 2, position + 1, 4)) * 2
    x = float(cache.layers[0].keys.sum())
    logits = torch.tensor([[[x / 2000., .2, -.4, .9]]])
    return cache, logits, {}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    torch.manual_seed(0)
    core = types.SimpleNamespace(qdq_r128=qdq)
    direction = types.SimpleNamespace(cache_state=cache_state, quantize_l6_cache=quantize, model_step=step)
    rotations = {"A": torch.eye(128).repeat(2, 1, 1), "B": torch.eye(128).repeat(2, 1, 1)}
    a, b = torch.randn(1, 1, 101), torch.randn(1, 1, 101)
    expected = float((torch.softmax(a.float()[:, -1], -1) *
                      (torch.log_softmax(a.float()[:, -1], -1) - torch.log_softmax(b.float()[:, -1], -1))).sum(-1).mean())
    assert full_kl(torch, a, b) == expected and full_kl(torch, a, a) == 0.
    assert summarize([1., 2., 3., 4.], 8, 12)["auc"] == 10
    runs = [stream(torch, direction, core, Capture(), [1] * 8, rotations, [0, 2], 8, 4, 2, lambda: None) for _ in range(2)]
    assert runs[0]["fp_reference_full_logit_sha256"] == runs[1]["fp_reference_full_logit_sha256"]
    for x, y in zip(runs[0]["results"], runs[1]["results"]):
        assert x["full_logit_sha256"] == y["full_logit_sha256"]
        assert x["kl_trajectory"] == y["kl_trajectory"]
        assert x["persistent_integrity"]["every_token_layer_qdq_count"] == 16
        assert x["persistent_integrity"]["actual_next_kernel_equal_post_qdq_count"] == 14
    cap = Capture(); audit = PersistentAudit(torch, direction, core, cap, [0, 2], ["A", "B"])
    audit.active = "A"; audit.previous["A"][0] = torch.ones((1, 16, 128, 128))
    cap.probe.current_layer = 0
    caught = False
    try:
        cap.kernel(initial_state=torch.zeros((1, 16, 128, 128)))
    except AssertionError:
        caught = True
    assert caught, "CONSUMPTION_FAILURE_MUST_BE_DETECTED"
    audit.close()
    cache, _, _ = step(torch, Capture(), None, 1, 0)
    cache.layers[1].keys = torch.ones((1, 2, 2, 4))
    caught = False
    try:
        hybrid_inventory(torch, cache, 0, [0, 2], 3)
    except AssertionError:
        caught = True
    assert caught, "WRONG_MLA_LENGTH_MUST_BE_DETECTED"
    assert not torch.cuda.is_initialized()
    output = dict(status="CPU_SYNTHETIC_ENGINE_TESTS_PASS_GPU_NOT_QUALIFIED", gpu_work=False,
        actual_model_executed=False, optimizer_step=False, original_frozen_code_changed=False,
        checks=["full_vocab_kl_direction_exact_formula", "identity_kl", "window_count_auc_quantiles",
                "next_kernel_consumption", "true_int8_inverse_writeback", "mla_conv_not_modified_by_qdq",
                "hybrid_cache_coverage", "cross_branch_cache_non_aliasing", "repeat_hash_identity",
                "corrupted_consumption_rejected", "wrong_mla_length_rejected"],
        engine_sha256=hashlib.sha256(Path(__file__).with_name("persistent_engine_v1.py").read_bytes()).hexdigest())
    # CPU evidence goes to own new file; no output is a real GPU PASS.
    from run_matched_l7 import publish, ROOT
    from objective import digest
    path = ROOT / "PERSISTENT_ENGINE_CPU_TESTS_V1.json"
    publish(path, output); publish(path.with_suffix(".sha256.json"), dict(path=str(path), sha256=digest(path)))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
