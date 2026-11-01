"""CPU mock-model binding checks; no model weights or CUDA queries.

The small reference functions are AST-extracted from the hash-pinned native
source. These checks validate scoped integration and bookkeeping, not GPU or
full-model qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pace_runtime import (CalibrationTap, MARKER_MAGIC, NATIVE_SHA256, PACEAdapter,
                          PACEEngine, canonical_sha, load_head_mask, rank_head_scores, verify_vendor)


def native_reference_module():
    path = ROOT / "vendor/modeling_qwen3_5.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SHA256:
        raise RuntimeError("native fixture source differs")
    names = {"l2norm", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule"}
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{n: namespace[n] for n in names})


class MockCache:
    def __init__(self):
        self.layers = [SimpleNamespace(recurrent_states={0: None},
                                       conv_states={0: torch.zeros(1, dtype=torch.bfloat16)})]

    def update_recurrent_state(self, state, layer_idx):
        old = self.layers[layer_idx].recurrent_states[0]
        if old is None:
            old = torch.zeros_like(state)
            self.layers[layer_idx].recurrent_states[0] = old
        old.copy_(state)


class MockLinear(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, token, cache):
        # K32/V4 keeps the meaningful32-head layout inexpensive on CPU.
        base = torch.arange(16*32, dtype=torch.float32).reshape(1, 1, 16, 32)
        q = ((base % 17 - 8) / 19 + token / 100).to(torch.bfloat16).repeat_interleave(2, dim=2)
        k = ((base % 13 - 6) / 23 - token / 200).to(torch.bfloat16).repeat_interleave(2, dim=2)
        v = (torch.arange(32*4).reshape(1, 1, 32, 4) % 7 / 10 + token / 100).to(torch.bfloat16)
        beta = torch.full((1, 1, 32), .5, dtype=torch.bfloat16)
        g = torch.full((1, 1, 32), -.01, dtype=torch.float32)
        state = cache.layers[0].recurrent_states[0]
        helper = (self.native.torch_chunk_gated_delta_rule if state is None
                  else self.native.torch_recurrent_gated_delta_rule)
        out, state = helper(q, k, v, g=g, beta=beta, initial_state=state,
                            output_final_state=True, use_qk_l2norm_in_kernel=True)
        cache.update_recurrent_state(state, 0)
        cache.layers[0].conv_states[0].add_(1)
        return out


class MockModel(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.linear = MockLinear(native)
        self.weight = torch.nn.Parameter(torch.zeros((64, 1), dtype=torch.bfloat16))
        self.model = SimpleNamespace(layers=[SimpleNamespace(linear_attn=self.linear)])
        self.caches = []

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        cache = past_key_values or MockCache()
        self.linear(int(input_ids.item()), cache)
        self.caches.append(cache)
        return SimpleNamespace(logits=torch.zeros((1, 1, 64), dtype=torch.bfloat16),
                               past_key_values=cache)


class MockTokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def sampler(sequence):
    position = [0]

    def sample_next(_warpers, logits, mask, generator):
        token = sequence[position[0]]
        position[0] += 1
        return torch.tensor([[token]], dtype=torch.long), 20

    return SimpleNamespace(setup_seed=lambda seed: torch.manual_seed(seed), build_warpers=lambda: None,
                           stop_ids=lambda tokenizer: {63}, sample_next=sample_next,
                           ids_sha256=canonical_sha)


class RuntimeBindings(unittest.TestCase):
    def setUp(self):
        self.native = native_reference_module()
        self.model = MockModel(self.native)
        self.masks = {0: [i < 8 for i in range(32)]}

    def test_vendor_pin_and_scoped_marker_restore(self):
        verify_vendor(ROOT)
        old = self.native.torch_recurrent_gated_delta_rule
        cache = MockCache()
        adapter = PACEAdapter(self.model, self.native, self.masks, 8, 2)
        with adapter:
            for token in (1, 2, 3):
                self.model.linear(token, cache)
                adapter.assert_cache(cache, token)
            state = cache.layers[0].recurrent_states[0]
            self.assertEqual(tuple(state.shape), (2,))
            self.assertEqual(state.tolist(), [MARKER_MAGIC, 3.0])
            self.assertEqual(cache.layers[0].conv_states[0].item(), 3)
            self.assertEqual(adapter.buffers[0].pending_tokens, 1)
            self.assertEqual(adapter.buffers[0].quantization_count, 1)
            self.assertFalse(adapter.audit()["dense_recurrent_carrier"])
        self.assertIs(self.native.torch_recurrent_gated_delta_rule, old)
        self.assertFalse(self.native._pace48_active)

    def test_marker_tamper_and_exception_restore(self):
        original = self.native.torch_chunk_gated_delta_rule
        cache = MockCache()
        with self.assertRaisesRegex(RuntimeError, "marker changed"):
            with PACEAdapter(self.model, self.native, self.masks, 8, 1):
                self.model.linear(1, cache)
                cache.layers[0].recurrent_states[0][1] += 1
                self.model.linear(2, cache)
        self.assertIs(self.native.torch_chunk_gated_delta_rule, original)
        self.assertFalse(self.native._pace48_active)

    def test_nested_patch_rejected_without_damaging_outer(self):
        with PACEAdapter(self.model, self.native, self.masks, 8, 1):
            patched = self.native.torch_recurrent_gated_delta_rule
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with PACEAdapter(self.model, self.native, self.masks, 8, 1):
                    pass
            self.assertIs(self.native.torch_recurrent_gated_delta_rule, patched)

    def test_generation_eos_and_length_forward_counts(self):
        for condition, sequence, cap, expected, finish in (
                ("R2", [4, 5, 63], 9, 5, "eos"),
                ("R3", [4, 5, 6], 3, 5, "length"),
                ("R2", [63], 9, 3, "eos")):
            native = native_reference_module()
            model = MockModel(native)
            engine = PACEEngine(ROOT, condition, model=model, tokenizer=MockTokenizer(),
                                native_module=native, masks=self.masks, device="cpu",
                                sampler_functions=sampler(sequence), require_native_pin=False)
            result = engine.generate([1, 2, 3], problem_id="NON_AIME_MOCK", max_new_tokens=cap)
            self.assertEqual(result["actual_model_forward_calls"], expected)
            self.assertEqual(result["finish_reason"], finish)
            self.assertEqual(result["generated_token_ids"], sequence)
            self.assertEqual(tuple(model.caches[-1].layers[0].recurrent_states[0].shape), (2,))
            self.assertEqual(model.caches[-1].layers[0].conv_states[0].item(), expected)
            self.assertFalse(native._pace48_active)

    def test_calibration_preserves_original_state_and_collects_boundaries(self):
        cache = None
        original = self.native.torch_recurrent_gated_delta_rule
        with torch.inference_mode(), CalibrationTap(self.model, self.native, layer_ids=(0,)) as tap:
            for token in range(256):
                out = self.model(torch.tensor([[token % 7]], dtype=torch.long), cache)
                cache = out.past_key_values
            self.assertEqual(tap.tokens[0], 256)
            self.assertEqual(tap.error_counts[0], 1)
            self.assertEqual(tap.decay_counts[0], 256)
            self.assertEqual(tuple(cache.layers[0].recurrent_states[0].shape), (1, 32, 32, 4))
            self.assertTrue(torch.isfinite(tap.error_sums[0]).all())
        self.assertIs(self.native.torch_recurrent_gated_delta_rule, original)

    def test_headscore_aggregate_then_fixed_period_rank_and_stable_ties(self):
        errors = [64.0] * 32
        errors[30] = 128.0
        decays = [.9 * 16384] * 32
        decays[31] = .99 * 16384
        totals = {0: dict(error=errors, decay=decays, error_count=64, decay_count=16384)}
        rows, masks = rank_head_scores(totals, layer_ids=(0,))
        self.assertEqual(masks["0"]["int8_heads"], [0, 1, 2, 3, 4, 5, 30, 31])
        self.assertAlmostEqual(rows[31]["E_h"], 1.0 / (1.0 - .99**16), places=12)
        self.assertEqual(sum(row["selected_bits"] == 8 for row in rows), 8)
        totals[0]["decay_count"] -= 1
        with self.assertRaisesRegex(RuntimeError, "aggregate64"):
            rank_head_scores(totals, layer_ids=(0,))

    def test_257_prompt_then_decode_boundaries_and_common_prefix(self):
        outputs = {}
        for period in (8, 1):
            native = native_reference_module()
            model, cache = MockModel(native), MockCache()
            adapter = PACEAdapter(model, native, self.masks, period, 257)
            selected = []
            with adapter:
                for index in range(266):
                    out = model.linear(index % 7, cache)
                    adapter.assert_cache(cache, index + 1)
                    if index in (256, 257):
                        selected.append(out.clone())
                report = adapter.audit()["storage_by_layer"]["0"]
                self.assertEqual(report["quantization_count"], 2 + 9 // period)
                self.assertEqual(report["pending_tokens"], 9 % period)
                self.assertGreater(report["peak_prefill_buffer_nbytes"], 0)
                self.assertGreater(report["peak_decode_buffer_nbytes"], 0)
            outputs[period] = selected
        # Both arms have identical prompt semantics and identical current-token
        # output before their first different decode writeback.
        for a, b in zip(outputs[1], outputs[8]):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args, rest = parser.parse_known_args()
    torch.set_num_threads(1)
    unittest.main(argv=[sys.argv[0]] + rest)
