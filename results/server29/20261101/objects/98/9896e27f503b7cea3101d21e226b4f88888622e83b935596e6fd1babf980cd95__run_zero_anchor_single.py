#!/usr/bin/env python3
"""One-process exact H1 C5 versus C5+0*ordinary-logits semantics gate."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PARENT_SCRIPTS = ROOT.parent / "QWEN_GDN_FSDP2_RECURRENT_TRAINING_FEASIBILITY_V1" / "scripts"
sys.path.insert(0, str(PARENT_SCRIPTS))
from run_fsdp_load import import_v1, memory, save_once  # noqa: E402
from run_fsdp_forward_gate import load_model, tensor_sha  # noqa: E402


def forward_student_with_ordinary_output(v1, model, patch, token_id, cache, capture):
    """Same frozen V1 model call/cache writeback; expose discarded logits only."""
    patch.mode = "student"
    patch.capture = capture
    device = model.get_input_embeddings().weight.device
    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
    output = model(input_ids=token, past_key_values=cache, use_cache=True)
    result = output.past_key_values
    v1.enable_differentiable_cache(result)
    return result, output.logits if capture else None


def run_condition(v1, model, ids, targets, device, anchored):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    bank = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
    hadamard = v1.hadamard(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.003, weight_decay=0.0)
    theta_before = {name: tensor_sha(p) for name, p in bank.named_parameters()}
    if len(theta_before) != 24:
        raise RuntimeError(f"expected 24 rotation tensors, got {len(theta_before)}")
    optimizer.zero_grad(set_to_none=True)
    before = memory()
    torch.cuda.reset_peak_memory_stats()
    qdq_hashes = {}
    with v1.RecurrentExperimentPatch(model, bank, hadamard) as patch:
        original_qdq = v1.CAYLEY.qwen_c128_ste

        def audit_qdq(value):
            result = original_qdq(value)
            if patch.mode == "student" and patch.capture:
                layer = int(patch.current_layer)
                qdq_hashes[str(layer)] = {
                    "codes": tensor_sha(result.codes),
                    "scale": tensor_sha(result.scale),
                    "dequant": tensor_sha(result.dequant),
                }
            return result

        v1.CAYLEY.qwen_c128_ste = audit_qdq
        try:
            cache = None
            ordinary_logits = None
            for index in range(32):
                capture = index == 31
                if capture:
                    v1.install_teacher_target(patch, targets[31], device)
                cache, logits = forward_student_with_ordinary_output(
                    v1, model, patch, ids[index], cache, capture)
                if capture:
                    ordinary_logits = logits
                elif index < 31:
                    v1.detach_cache(cache)
            if ordinary_logits is None or not ordinary_logits.requires_grad:
                raise RuntimeError("ordinary logits lacks live autograd path")
            if not bool(torch.isfinite(ordinary_logits).all().detach().cpu()):
                raise FloatingPointError("nonfinite ordinary logits; zero anchor forbidden")
            anchor_sum = ordinary_logits.float().sum()
            if not bool(torch.isfinite(anchor_sum).detach().cpu()):
                raise FloatingPointError("nonfinite ordinary logits sum; zero anchor forbidden")
            state, functional, c5_loss, c6_loss = patch.captured_losses()
            if not bool(torch.isfinite(c5_loss).detach().cpu()):
                raise FloatingPointError("nonfinite C5 loss")
            writeback = {str(layer): bool(torch.equal(
                cache.layers[layer].recurrent_states[0], patch.student_post[layer]))
                for layer in v1.GDN_LAYERS}
            if not all(writeback.values()):
                raise RuntimeError("frozen differentiable writeback mismatch")
            if len(qdq_hashes) != 24:
                raise RuntimeError(f"expected 24 QDQ captures, got {len(qdq_hashes)}")
            loss = c5_loss + 0.0 * anchor_sum if anchored else c5_loss
            if not torch.equal(loss.detach(), c5_loss.detach()):
                raise RuntimeError("zero anchor changed C5 scalar value")
            after_forward = memory()
            loss.backward()
            after_backward = memory()
            gradients = {}
            for name, p in bank.named_parameters():
                if p.grad is None or not bool(torch.isfinite(p.grad).all().detach().cpu()):
                    raise RuntimeError(f"missing/nonfinite rotation gradient {name}")
                gradients[name] = tensor_sha(p.grad)
            optimizer.step()
            after_optimizer = memory()
            theta_after = {name: tensor_sha(p) for name, p in bank.named_parameters()}
            result = {
                "condition": "anchored" if anchored else "original",
                "horizon": 1,
                "model": "full canonical Qwen3.5-9B on one GPU, frozen backbone",
                "ordinary_anchor_tensor": "final captured-token model output logits",
                "ordinary_logits_shape": list(ordinary_logits.shape),
                "ordinary_logits_requires_grad": ordinary_logits.requires_grad,
                "ordinary_logits_finite": True,
                "anchor_sum_finite": True,
                "anchor_coefficient": 0.0 if anchored else None,
                "C5_loss": float(c5_loss.detach()),
                "C5_loss_sha256": tensor_sha(c5_loss),
                "loss_sha256": tensor_sha(loss),
                "state_loss": float(state.detach()),
                "functional_loss": float(functional.detach()),
                "C6_loss": float(c6_loss.detach()),
                "theta_before_sha256": theta_before,
                "rotation_gradient_sha256": gradients,
                "theta_after_sha256": theta_after,
                "QDQ_final_token_sha256": qdq_hashes,
                "cache_writeback_exact_by_layer": writeback,
                "memory_before": before,
                "memory_after_forward": after_forward,
                "memory_after_backward": after_backward,
                "memory_after_optimizer": after_optimizer,
                "status": "COMPLETE",
            }
        finally:
            v1.CAYLEY.qwen_c128_ste = original_qdq
    del optimizer, bank, hadamard, cache, ordinary_logits, loss
    gc.collect()
    return result


def main():
    if torch.cuda.device_count() != 1 or os.getenv("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("exactly one explicitly selected idle GPU required")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    v1 = import_v1()
    model, tokenizer = load_model(False, 0, 1)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("backbone unexpectedly trainable")
    row = v1.corpus_rows("TRAIN")[0]
    ids, _ = v1.tokenize(tokenizer, row)
    ids = ids[:64]
    if len(ids) != 64:
        raise RuntimeError("frozen sample shorter than 64")
    input_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    device = torch.device("cuda", 0)
    bank_for_teacher = v1.CAYLEY.PerLayerCayleyRotations(v1.GDN_LAYERS).to(device)
    with v1.RecurrentExperimentPatch(model, bank_for_teacher, v1.hadamard(device)) as patch:
        targets = v1.collect_teacher_targets(model, patch, ids[:32], [31])
    del bank_for_teacher
    gc.collect()
    original = run_condition(v1, model, ids, targets, device, anchored=False)
    anchored = run_condition(v1, model, ids, targets, device, anchored=True)
    for name, record in (("original", original), ("anchored", anchored)):
        save_once(ROOT / "analysis" / f"single_h1_{name}.json", record)
    comparisons = {
        "C5_loss_bitwise_equal": original["C5_loss_sha256"] == anchored["C5_loss_sha256"]
        and original["loss_sha256"] == anchored["loss_sha256"],
        "theta_before_exact": original["theta_before_sha256"] == anchored["theta_before_sha256"],
        "rotation_gradient_24_of_24_exact": original["rotation_gradient_sha256"] == anchored["rotation_gradient_sha256"]
        and len(original["rotation_gradient_sha256"]) == 24,
        "theta_one_step_Adam_24_of_24_exact": original["theta_after_sha256"] == anchored["theta_after_sha256"]
        and len(original["theta_after_sha256"]) == 24,
        "QDQ_24_of_24_exact": original["QDQ_final_token_sha256"] == anchored["QDQ_final_token_sha256"]
        and len(original["QDQ_final_token_sha256"]) == 24,
        "cache_writeback_exact_in_both": all(original["cache_writeback_exact_by_layer"].values())
        and all(anchored["cache_writeback_exact_by_layer"].values()),
    }
    gate = "PASS" if all(comparisons.values()) else "FAIL"
    save_once(ROOT / "analysis" / "zero_anchor_single_semantics.json", {
        "ZERO_ANCHOR_SINGLE_GPU_SEMANTICS_GATE": gate,
        "input_token_sha256": input_hash,
        "sample_id": str(row.get("id", "TRAIN:first-row")),
        "comparisons": comparisons,
        "conditions": ["single_h1_original.json", "single_h1_anchored.json"],
    })
    print(json.dumps({"gate": gate, "comparisons": comparisons}, sort_keys=True), flush=True)
    if gate != "PASS":
        raise RuntimeError("zero-anchor exact semantics gate failed; H4 intervention forbidden")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save_once(ROOT / "analysis" / "zero_anchor_single.error.json", {
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
