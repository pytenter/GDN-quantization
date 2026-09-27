#!/usr/bin/env python3
"""One authorized full-9B FSDP2 H4 C5+0*logits causal intervention."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


ROOT = Path(__file__).resolve().parents[1]
PARENT_SCRIPTS = ROOT.parent / "QWEN_GDN_FSDP2_RECURRENT_TRAINING_FEASIBILITY_V1" / "scripts"
sys.path.insert(0, str(PARENT_SCRIPTS))
from run_fsdp_load import import_v1, memory, save_once  # noqa: E402
from run_fsdp_forward_gate import load_model, tensor_sha  # noqa: E402
from run_zero_anchor_single import forward_student_with_ordinary_output  # noqa: E402


def local_sample_hash(p):
    local = p.to_local() if isinstance(p, DTensor) else p
    return tensor_sha(local.detach().reshape(-1)[:4096])


def safe_memory():
    try:
        return memory()
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def run():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 2 or torch.cuda.device_count() != 2 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("exactly the frozen two-rank physical GPU0/1 topology required")
    if not (ROOT / "analysis" / "zero_anchor_single_semantics.json").exists():
        raise RuntimeError("single-GPU zero-anchor semantics gate missing")
    gate = json.loads((ROOT / "analysis" / "zero_anchor_single_semantics.json").read_text())
    if gate.get("ZERO_ANCHOR_SINGLE_GPU_SEMANTICS_GATE") != "PASS":
        raise RuntimeError("single-GPU zero-anchor exact semantics gate not PASS")
    if any((ROOT / "analysis" / f"h4_anchor_rank{i}.json").exists() or
           (ROOT / "analysis" / f"h4_anchor_rank{i}.error.json").exists() for i in (0, 1)):
        raise RuntimeError("H4 anchor diagnostic is single-use; evidence file already exists")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    phase = "initialization"
    after_forward = None
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        v1 = import_v1()
        model, tokenizer = load_model(True, rank, world)
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("frozen backbone has trainable parameters")
        device = torch.device("cuda", rank)
        bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
        hadamard = v1.hadamard(device)
        row = v1.corpus_rows("TRAIN")[0]
        ids, _ = v1.tokenize(tokenizer, row)
        ids = ids[:64]
        if len(ids) != 64:
            raise RuntimeError("frozen sample shorter than 64")
        sample_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        sample_keys = ["model.layers.0.linear_attn.in_proj_qkv.weight",
                       "model.layers.16.linear_attn.in_proj_qkv.weight", "lm_head.weight"]
        model_params = dict(model.named_parameters())
        before_model_hash = {name: local_sample_hash(model_params[name]) for name in sample_keys}
        theta_before = {name: tensor_sha(p) for name, p in bank.named_parameters()}
        optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
        if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in bank.parameters()}:
            raise RuntimeError("optimizer is not rotation-only")
        qdq = {}
        original_qdq = v1.CAYLEY.qwen_c128_ste
        with v1.RecurrentExperimentPatch(model, bank, hadamard) as patch:
            phase = "teacher_targets"
            targets = v1.collect_teacher_targets(model, patch, ids[:32], [31])

            def capture_qdq(value):
                result = original_qdq(value)
                if patch.mode == "student" and patch.capture:
                    layer = int(patch.current_layer)
                    qdq[layer] = {"pre": value.detach().cpu().clone(),
                                  "scale": result.scale.detach().cpu().clone(),
                                  "codes": result.codes.detach().cpu().clone(),
                                  "post": result.dequant.detach().cpu().clone()}
                return result

            v1.CAYLEY.qwen_c128_ste = capture_qdq
            try:
                phase = "H4_student_forward"
                cache = None
                ordinary_logits = None
                before = memory()
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                for index in range(32):
                    capture = index == 31
                    if capture:
                        v1.install_teacher_target(patch, targets[31], device)
                    cache, logits = forward_student_with_ordinary_output(
                        v1, model, patch, ids[index], cache, capture)
                    if capture:
                        ordinary_logits = logits
                    if (index + 1) % 4 == 0 and index < 31:
                        v1.detach_cache(cache)
                if sorted(qdq) != list(v1.GDN_LAYERS):
                    raise RuntimeError(f"QDQ captured layers mismatch: {sorted(qdq)}")
                if ordinary_logits is None or not ordinary_logits.requires_grad:
                    raise RuntimeError("ordinary model logits have no grad path")
                if not bool(torch.isfinite(ordinary_logits).all().detach().cpu()):
                    raise FloatingPointError("nonfinite ordinary logits; anchor forbidden")
                anchor_sum = ordinary_logits.float().sum()
                if not bool(torch.isfinite(anchor_sum).detach().cpu()):
                    raise FloatingPointError("nonfinite anchor sum")
                state, functional, c5_loss, c6_loss = patch.captured_losses()
                if not bool(torch.isfinite(c5_loss).detach().cpu()):
                    raise FloatingPointError("nonfinite C5 diagnostic loss")
                writeback = {str(layer): bool(torch.equal(
                    cache.layers[layer].recurrent_states[0], patch.student_post[layer]))
                    for layer in v1.GDN_LAYERS}
                if not all(writeback.values()):
                    raise RuntimeError("recurrent state writeback mismatch")
                anchored_loss = c5_loss + 0.0 * anchor_sum
                if not torch.equal(c5_loss.detach(), anchored_loss.detach()):
                    raise RuntimeError("zero anchor changed C5 scalar")
                after_forward = memory()
                phase = "H4_anchored_backward"
                anchored_loss.backward()
                after_backward = memory()
                phase = "gradient_gate"
                gradients = {}
                for name, p in bank.named_parameters():
                    g = p.grad
                    gradients[name] = {
                        "exists": g is not None,
                        "finite": bool(torch.isfinite(g).all()) if g is not None else False,
                        "sha256": tensor_sha(g) if g is not None else None,
                        "norm": float(torch.linalg.vector_norm(g.float())) if g is not None else None,
                    }
                if len(gradients) != 24 or not all(g["exists"] and g["finite"] for g in gradients.values()):
                    raise RuntimeError("missing/nonfinite 24/24 rotation gradient")
                phase = "optimizer"
                optimizer.step()
                after_optimizer = memory()
                theta_after = {name: tensor_sha(p) for name, p in bank.named_parameters()}
                after_model_hash = {name: local_sample_hash(model_params[name]) for name in sample_keys}
                if after_model_hash != before_model_hash:
                    raise RuntimeError("frozen backbone sample hash changed")
                phase = "record"
                result = {
                    "condition": "C5 + 0.0 * finite ordinary final-token model logits sum",
                    "baseline": "frozen parent H4 zero-storage failure; not rerun",
                    "horizon": 4, "world_size": world, "rank": rank,
                    "physical_visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"],
                    "sample_id": str(row.get("id", "TRAIN:first-row")),
                    "sample_token_hash": sample_hash,
                    "anchor_logits_shape": list(ordinary_logits.shape),
                    "anchor_logits_finite_requires_grad": True,
                    "loss": {"state": float(state.detach()), "functional": float(functional.detach()),
                             "C5": float(c5_loss.detach()), "C6": float(c6_loss.detach()),
                             "anchored_C5": float(anchored_loss.detach())},
                    "QDQ": {str(layer): {field: {"sha256": tensor_sha(tensor),
                                                  "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                                          for field, tensor in fields.items()}
                            for layer, fields in qdq.items()},
                    "cache_writeback_exact_by_layer": writeback,
                    "theta0_before": theta_before, "theta_after": theta_after,
                    "gradient": gradients,
                    "model_sample_hash_before": before_model_hash,
                    "model_sample_hash_after": after_model_hash,
                    "model_sample_unchanged": True,
                    "memory_before_rollout": before,
                    "memory_before_backward": after_forward,
                    "memory_after_backward": after_backward,
                    "memory_after_optimizer": after_optimizer,
                    "status": "COMPLETE",
                }
                save_once(ROOT / "analysis" / f"h4_anchor_rank{rank}.json", result)
            finally:
                v1.CAYLEY.qwen_c128_ste = original_qdq
        dist.barrier(device_ids=[rank])
        if rank == 0:
            print(json.dumps({"condition": "H4_anchor", "status": "COMPLETE",
                              "C5": result["loss"]["C5"],
                              "peak_allocated_bytes": after_backward["peak_allocated_bytes"],
                              "peak_reserved_bytes": after_backward["peak_reserved_bytes"]}), flush=True)
    except Exception as exc:
        save_once(ROOT / "analysis" / f"h4_anchor_rank{rank}.error.json", {
            "rank": rank, "phase": phase, "type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "memory_before_backward": after_forward,
            "memory_at_failure": safe_memory(),
        })
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
