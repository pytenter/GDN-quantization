#!/usr/bin/env python3
"""Reduced, passing FSDP2 H1 hook/profiler noninterference control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh


ROOT = Path(__file__).resolve().parents[1]
REENTRY_SCRIPTS = ROOT.parent / "QWEN_GDN_FSDP2_RECURRENT_REENTRY_CLOSURE_V1" / "scripts"
sys.path.insert(0, str(REENTRY_SCRIPTS))
from run_canonical_gdn_reentry import import_v1  # noqa: E402
from run_real_gdn_block_reentry import mem  # noqa: E402
from run_small_stack_reentry import SmallStack, load_layers  # noqa: E402


def sha_tensor(t):
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(data, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def run(condition):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2 or torch.cuda.device_count() != 2 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("exactly two safe physical GPUs 0/1 and two ranks required")
    # v1 files record the diagnostic scalar-serializer error; never overwrite them.
    tag = f"hook_h1_v2_{condition}_rank{rank}"
    if (ROOT / "analysis" / f"{tag}.json").exists() or (ROOT / "analysis" / f"{tag}.error.json").exists():
        raise RuntimeError("single-use evidence already exists")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(20260927)
        torch.cuda.manual_seed_all(20260927)
        layers, config, conv_hashes = load_layers(rank, 2)
        stack = SmallStack(layers)
        mesh = init_device_mesh("cuda", (world,))
        for layer in stack.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(stack, mesh=mesh, reshard_after_forward=True)
        v1 = import_v1()
        device = torch.device("cuda", rank)
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
        optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
        theta_before = {name: sha_tensor(p) for name, p in bank.named_parameters()}
        hidden = torch.randn((1, 1, config.hidden_size), device=device,
                             dtype=torch.bfloat16, requires_grad=True)
        initial = hidden
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=config)
        hooks = []
        count = {f"layer{i}": Counter() for i in range(2)}
        count["root"] = Counter()
        if condition == "instrumented":
            for name, module in [(f"layer{i}", layer) for i, layer in enumerate(stack.model.layers)] + [("root", stack)]:
                def pre(_mod, _inputs, unit=name):
                    count[unit]["forward_enter"] += 1

                def post(_mod, _inputs, output, unit=name):
                    count[unit]["forward_exit"] += 1
                    if torch.is_tensor(output) and output.requires_grad:
                        count[unit]["output_requires_grad"] += 1

                        def observe(grad, label=unit):
                            count[label]["output_grad_hook"] += 1
                            return grad

                        output.register_hook(observe)
                hooks.append(module.register_forward_pre_hook(pre))
                hooks.append(module.register_forward_hook(post))
        torch.cuda.empty_cache()
        before = mem(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        optimizer.zero_grad(set_to_none=True)
        fsdp_events = []
        try:
            with v1.RecurrentExperimentPatch(stack, bank, v1.hadamard(device)) as patch:
                patch.mode = "student"
                patch.capture = False
                output = stack(hidden, cache)
                v1.enable_differentiable_cache(cache)
                loss = output.float().square().mean()
                if condition == "instrumented":
                    with profile(activities=[ProfilerActivity.CPU], record_shapes=False,
                                 profile_memory=False, with_stack=False) as prof:
                        loss.backward()
                    fsdp_events = [event.name for event in prof.events()
                                   if "FSDP::pre_backward" in event.name or
                                      "FSDP::post_backward" in event.name]
                else:
                    loss.backward()
                after_backward = mem(rank)
                if initial.grad is None or not bool(torch.isfinite(initial.grad).all()):
                    raise RuntimeError("missing/nonfinite input gradient")
                input_grad_hash = sha_tensor(initial.grad)
                bank_grads = {name: sha_tensor(p.grad) if p.grad is not None else None
                              for name, p in bank.named_parameters()}
                optimizer.step()
                theta_after = {name: sha_tensor(p) for name, p in bank.named_parameters()}
        finally:
            for hook in hooks:
                hook.remove()
        data = {
            "condition": condition, "rank": rank, "status": "PASS",
            "blocks": 2, "horizon": 1, "reshard_after_forward": True,
            "checkpoint_conv_hashes": conv_hashes,
            "loss_sha256": sha_tensor(loss), "loss": float(loss.detach()),
            "input_gradient_sha256": input_grad_hash,
            "rotation_gradient_sha256": bank_grads,
            "theta_before_sha256": theta_before, "theta_after_sha256": theta_after,
            "hook_counts": {key: dict(value) for key, value in count.items()},
            "fsdp_profiler_event_names": fsdp_events,
            "memory_before": before, "memory_after_backward": after_backward,
            "PRIVATE_FSDP_API_READ_ONLY_DIAGNOSTIC": False,
            "instrumentation_caveat": "public output hooks and CPU profiler may perturb runtime; compare exact noninstrumented control",
        }
        save_once(ROOT / "analysis" / f"{tag}.json", data)
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({"condition": condition, "status": "PASS",
                              "peak_allocated_bytes": after_backward["peak_allocated_bytes"],
                              "fsdp_event_count": len(fsdp_events)}), flush=True)
    except Exception as exc:
        save_once(ROOT / "analysis" / f"{tag}.error.json", {
            "rank": rank, "condition": condition, "type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["plain", "instrumented"])
    args = parser.parse_args()
    run(args.condition)
