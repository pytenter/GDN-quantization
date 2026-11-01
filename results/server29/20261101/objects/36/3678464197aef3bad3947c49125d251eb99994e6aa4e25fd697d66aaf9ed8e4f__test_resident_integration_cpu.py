"""CPU-only actual-GDN adapter integration, never a 9B or GPU qualification.

Runs the actual pinned Qwen GDN class with its full 16/32 x 128 head geometry,
but a 32-wide synthetic outer hidden layer and a small explicit cache fixture.
The forward, convolution, gates, norm, GDN helper and readout are production
code.  Deterministic synthetic token ids are not AIME or calibration data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU integration requires explicitly empty CUDA_VISIBLE_DEVICES")
    sys.path[:0] = [str(root / "src"), str(root / "vendor"), "/data/zypan/transformers-qwen35"]
    import torch
    from torch import nn
    import torch.nn.functional as F
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    import resident_rotation as resident
    import group_codecs as codec

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261022)
    assert resident.file_sha256(qmod.__file__) == resident.MODEL_SOURCE_SHA256
    assets = resident.ResidentMatrices.from_root(root, device="cpu")
    checks, audits = [], []

    def check(name, value, **details):
        if not bool(value):
            raise AssertionError(name + " " + json.dumps(details, sort_keys=True))
        checks.append(dict(name=name, status="PASS", **details))

    def same(actual, expected):
        return (actual.shape == expected.shape and actual.dtype == expected.dtype
                and torch.equal(actual, expected)
                and codec.tensor_bytes(actual) == codec.tensor_bytes(expected))

    class Cache:
        def __init__(self):
            self.layers = [SimpleNamespace(conv_states=[None], recurrent_states=[None], record_past=False)]

        def has_previous_state(self, layer):
            return self.layers[layer].recurrent_states[0] is not None

        def update_conv_state(self, value, layer, conv_kernel_size):
            padded = F.pad(value, (max(0, conv_kernel_size - value.shape[-1]), 0))
            self.layers[layer].conv_states[0] = padded[:, :, -conv_kernel_size:].clone()
            return value

        def update_recurrent_state(self, value, layer):
            self.layers[layer].recurrent_states[0] = value

    config = SimpleNamespace(hidden_size=32, linear_num_value_heads=32,
                             linear_num_key_heads=16, linear_key_head_dim=128,
                             linear_value_head_dim=128, linear_conv_kernel_dim=4,
                             hidden_act="silu", rms_norm_eps=1e-6,
                             layer_types=["linear_attention"])

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = qmod.Qwen3_5GatedDeltaNet(config, 0)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.embedding = nn.Embedding(64, 32)
            self.lm_head = nn.Linear(32, 64, bias=False)

        def forward(self, input_ids, past_key_values=None, use_cache=True):
            if not use_cache:
                raise RuntimeError("Fixture protocol always uses its fresh persistent cache")
            hidden = self.embedding(input_ids)
            output = self.model.layers[0].linear_attn(hidden, cache_params=past_key_values)
            return SimpleNamespace(logits=self.lm_head(output), past_key_values=past_key_values)

    template = Model().to(dtype=torch.bfloat16).eval()
    ids = (3, 17, 29)
    baseline = []
    old = {name: getattr(qmod, name) for name in (
        "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}

    def capture(fn):
        def call(query, key, value, *args, **kwargs):
            baseline.append(dict(query=query.detach().clone(), key=key.detach().clone(),
                                 value=value.detach().clone(), g=kwargs["g"].detach().clone(),
                                 beta=kwargs["beta"].detach().clone()))
            return fn(query, key, value, *args, **kwargs)
        return call

    # This process owns these temporary Python globals; no shared file or
    # foreign running process is touched. Restore them before adapter install.
    cache = Cache()
    try:
        for name, fn in old.items():
            setattr(qmod, name, capture(fn))
        with torch.inference_mode():
            for token in ids:
                template(torch.tensor([[token]]), past_key_values=cache)
    finally:
        for name, fn in old.items():
            setattr(qmod, name, fn)
    check("actual_native_fixture_three_forwards", len(baseline) == len(ids))
    for token, record in enumerate(baseline):
        for name in ("query", "key"):
            check(f"native_shared_head_pairs_token{token}_{name}",
                  same(record[name][:, :, 0::2], record[name][:, :, 1::2]))

    all_results = {}
    for condition in ("H", "C1", "S3"):
        for policy in ("immediate", "token"):
            model, cache = copy.deepcopy(template), Cache()
            module = model.model.layers[0].linear_attn
            class_forward = type(module).forward
            pending, outputs, states, encoded_fields = {}, [], [], []

            def observe(event, token, layer, payload):
                check(f"{condition}_{policy}_{event}_layer{token}", layer == 0)
                if event == "PREPARED_QK":
                    raw = baseline[token]
                    expected_q, expected_k = resident.prepare_qk(
                        raw["query"][:, :, 0::2], raw["key"][:, :, 0::2], assets.matrix(0, condition))
                    check(f"{condition}_{policy}_prepared_query_token{token}", same(payload["query"], expected_q))
                    check(f"{condition}_{policy}_prepared_key_token{token}", same(payload["key"], expected_k))
                    for name in ("value", "g", "beta"):
                        check(f"{condition}_{policy}_unchanged_{name}_token{token}", same(payload[name], raw[name]))
                elif event == "PRE_QDQ":
                    expected = codec.encode(payload["state"], "B1").cpu()
                    pending["encoded"] = expected
                    pending["decoded"] = codec.decode(expected)
                elif event == "POST_QDQ":
                    expected = pending["encoded"]
                    for name in ("codes", "scales", "negative_zero_flags"):
                        check(f"{condition}_{policy}_canonical_{name}_token{token}",
                              same(payload[name].cpu(), getattr(expected, name)))
                    check(f"{condition}_{policy}_canonical_writeback_token{token}",
                          same(payload["writeback"].cpu(), pending["decoded"]))
                    encoded_fields.append({name: codec.tensor_bytes(payload[name])
                                           for name in ("codes", "scales", "negative_zero_flags")})

            patch = resident.ResidentRotationPatch(model, assets, condition, qdq=True,
                        observer=observe, audit_next_consumption=True,
                        finite_policy=policy, layers=(0,))
            with torch.inference_mode(), patch:
                check(f"{condition}_{policy}_instance_local_forward_installed", "forward" in module.__dict__)
                for token in ids:
                    output = model(torch.tensor([[token]]), past_key_values=cache)
                    outputs.append(output.logits.detach().clone())
                    state = cache.layers[0].recurrent_states[0]
                    check(f"{condition}_{policy}_cache_is_qdq_token{patch.token_index}",
                          same(state.cpu(), pending["decoded"]))
                    states.append(state.detach().clone())
                audit = patch.audit()
                check(f"{condition}_{policy}_exact_gdn_qdq_counts",
                      audit["original_gdn_helper_calls"] == 3 and audit["qdq_calls"] == 3)
                check(f"{condition}_{policy}_actual_next_consumption_checks",
                      audit["next_consumption_full_value_byte_checks"] == 2)
                check(f"{condition}_{policy}_one_chunk_two_recurrent",
                      audit["chunk_prefill_calls"] == 1 and audit["recurrent_decode_calls"] == 2)
                check(f"{condition}_{policy}_token_finite_barriers",
                      audit["full_token_finite_checks"] == 3 and not patch.finite_flags)
                check(f"{condition}_{policy}_no_state_rotation_or_qdq_permutation",
                      audit["state_rotation_matmul_calls"] == 0 and not audit["P_inside_QDQ"])
                audits.append(audit)
            check(f"{condition}_{policy}_instance_forward_restored",
                  "forward" not in module.__dict__ and type(module).forward is class_forward)
            check(f"{condition}_{policy}_global_helpers_untouched",
                  all(getattr(qmod, name) is fn for name, fn in old.items()))
            all_results[(condition, policy)] = (outputs, states, encoded_fields)

    for condition in ("H", "C1", "S3"):
        immediate = all_results[(condition, "immediate")]
        deferred = all_results[(condition, "token")]
        check(condition + "_finite_policies_exact_output_state_and_fields",
              all(same(a, b) for a, b in zip(immediate[0], deferred[0]))
              and all(same(a, b) for a, b in zip(immediate[1], deferred[1]))
              and immediate[2] == deferred[2])

    # NoQDQ route still rotates Q/K and keeps resident State without the codec.
    model, cache = copy.deepcopy(template), Cache()
    with torch.inference_mode(), resident.ResidentRotationPatch(
            model, assets, "C1", qdq=False, layers=(0,)) as patch:
        model(torch.tensor([[3]]), past_key_values=cache)
        check("noqdq_real_helper_route", patch.audit()["qdq_calls"] == 0
              and patch.audit()["original_gdn_helper_calls"] == 1)

    # A cache already in some other coordinate may not be adopted silently.
    rejected = False
    try:
        with torch.inference_mode(), resident.ResidentRotationPatch(model, assets, "C1", layers=(0,)):
            model(torch.tensor([[17]]), past_key_values=cache)
    except RuntimeError as exc:
        rejected = "fresh empty recurrent cache" in str(exc)
    check("reject_adopting_nonempty_external_cache", rejected)
    check("restore_after_cache_rejection", "forward" not in model.model.layers[0].linear_attn.__dict__)

    # An existing H/U prehook must not be nested or bypassed silently.
    model = copy.deepcopy(template)
    handle = model.model.layers[0].linear_attn.register_forward_pre_hook(lambda *_: None)
    rejected = False
    try:
        with resident.ResidentRotationPatch(model, assets, "C1", layers=(0,)):
            pass
    except RuntimeError as exc:
        rejected = "Existing instance forward/hooks" in str(exc)
    finally:
        handle.remove()
    check("reject_nested_existing_rotation_hook", rejected)

    model, cache = copy.deepcopy(template), Cache()
    model.embedding.weight.data[3, 0] = float("nan")
    rejected = False
    try:
        with torch.inference_mode(), resident.ResidentRotationPatch(model, assets, "C1", layers=(0,)):
            model(torch.tensor([[3]]), past_key_values=cache)
    except FloatingPointError as exc:
        rejected = "Nonfinite" in str(exc)
    check("deferred_finite_policy_rejects_nonfinite_before_return", rejected)
    check("restore_after_nonfinite_rejection", "forward" not in model.model.layers[0].linear_attn.__dict__)

    return dict(status="PASS_CPU_ACTUAL_GDN_FORWARD_RESIDENT_INTEGRATION",
                scope="Actual production GDN class/helper CPU; synthetic hidden32 wrapper, not 9B or GPU",
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                new_gpu_dispatches=0, new_aime_generations=0,
                frozen_model_source_sha256=resident.MODEL_SOURCE_SHA256,
                adapter_source_sha256=resident.file_sha256(root / "src/resident_rotation.py"),
                scripts_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                actual_head_geometry=[16, 32, 128, 128], fixed_synthetic_token_ids=list(ids),
                conditions=["H", "C1", "S3"], finite_policies=["immediate", "token"],
                checks=checks, audit_rows=audits,
                does_not_establish=["real 9B numerical behavior", "CUDA arithmetic equivalence",
                                    "GPU memory feasibility", "GPU TPS", "81920-token stability"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.root.resolve())
    except BaseException as exc:
        write_new(args.output.with_name(args.output.stem + "_FAILURE.json"),
                  dict(status="FAIL_CPU_INTEGRATION", error=repr(exc), traceback=traceback.format_exc()))
        raise
    write_new(args.output, result)
    print(json.dumps(dict(status=result["status"], checks=len(result["checks"])), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
