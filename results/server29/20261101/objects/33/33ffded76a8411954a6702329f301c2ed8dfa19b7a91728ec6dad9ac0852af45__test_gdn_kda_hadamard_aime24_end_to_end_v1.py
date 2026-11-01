import importlib.util
from pathlib import Path

import torch


REPO_RUNNER = Path(__file__).resolve().parents[2] / "experiments" / "rotation" / "run_gdn_kda_hadamard_aime24_end_to_end_v1.py"
RUNNER = REPO_RUNNER if REPO_RUNNER.is_file() else Path(__file__).with_name("run_gdn_kda_hadamard_aime24_end_to_end_v1.py")
SPEC = importlib.util.spec_from_file_location("hadamard_runner", RUNNER)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_h128_is_exact_deterministic_normalized_sylvester():
    first = MOD.deterministic_normalized_hadamard(128)
    second = MOD.deterministic_normalized_hadamard(128)
    assert first.shape == (128, 128)
    assert torch.equal(first, second)
    assert set(torch.unique(first * (128 ** 0.5)).tolist()) == {-1.0, 1.0}


def test_h128_is_orthogonal_on_both_sides():
    matrix = MOD.deterministic_normalized_hadamard(128)
    identity = torch.eye(128, dtype=torch.float64)
    assert torch.allclose(matrix.t() @ matrix, identity, atol=1e-12, rtol=0)
    assert torch.allclose(matrix @ matrix.t(), identity, atol=1e-12, rtol=0)


def test_axis_roundtrips_and_algebra_parity():
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(1, 2, 128, 128, generator=generator, dtype=torch.float64)
    q = torch.randn(1, 2, 128, generator=generator, dtype=torch.float64)
    k = torch.randn(1, 2, 128, generator=generator, dtype=torch.float64)
    v = torch.randn(1, 2, 128, generator=generator, dtype=torch.float64)
    rotation = MOD.deterministic_normalized_hadamard(128)
    key_state = MOD.rotate_state_key_axis(state, rotation)
    value_state = MOD.rotate_state_value_axis(state, rotation)
    assert torch.allclose(MOD.inverse_rotate_state_key_axis(key_state, rotation), state, atol=1e-12, rtol=0)
    assert torch.allclose(MOD.inverse_rotate_state_value_axis(value_state, rotation), state, atol=1e-12, rtol=0)
    native = torch.einsum("...kv,...k->...v", state, q)
    key_readout = torch.einsum("...av,...a->...v", key_state, torch.einsum("ak,...k->...a", rotation.t(), q))
    value_readout = torch.einsum("...ku,...k->...u", value_state, q) @ rotation.t()
    assert torch.allclose(key_readout, native, atol=1e-12, rtol=0)
    assert torch.allclose(value_readout, native, atol=1e-12, rtol=0)
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128


def test_quantizer_shapes_are_unchanged():
    state = torch.randn(1, 4, 128, 128)
    c_qdq, c_scale, _ = MOD.quantize_state(state, "C128")
    r_qdq, r_scale, _ = MOD.quantize_state(state, "R128")
    assert c_qdq.shape == r_qdq.shape == state.shape
    assert c_scale.shape == (1, 4, 1, 128)
    assert r_scale.shape == (1, 4, 128, 1)


def test_gdn_cache_compat_supports_dynamic_and_legacy_layouts():
    class Harness:
        @staticmethod
        def get_state(cache, layer_idx, state_idx=0):
            return cache.layers[layer_idx].recurrent_states[state_idx]

    class Object:
        pass

    harness = Harness()
    MOD.install_gdn_cache_compat(harness)
    expected = torch.randn(1, 2, 128, 128)
    dynamic = Object()
    dynamic.recurrent_states = [expected]
    assert harness.get_state(dynamic, 0) is expected
    legacy = Object()
    layer = Object()
    layer.recurrent_states = [expected]
    legacy.layers = [layer]
    assert harness.get_state(legacy, 0) is expected


def test_output_names_are_unique_by_arch_condition_scope_and_shard(tmp_path):
    old = MOD.RESULT_DIR
    MOD.RESULT_DIR = tmp_path
    try:
        paths = {
            MOD.output_path(arch, condition, scope, shard, 3)
            for arch, condition, scope, shard in (
                ("gdn", "INT8_C128_KEY_HADAMARD", "formal", 0),
                ("gdn", "INT8_C128_KEY_HADAMARD", "smoke", 0),
                ("gdn", "INT8_C128_NATIVE", "formal", 0),
                ("kda", "INT8_R128_VALUE_HADAMARD", "formal", 0),
                ("kda", "INT8_R128_VALUE_HADAMARD", "formal", 1),
            )
        }
        assert len(paths) == 5
    finally:
        MOD.RESULT_DIR = old
