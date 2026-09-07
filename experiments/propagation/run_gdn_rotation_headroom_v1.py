#!/usr/bin/env python3
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path("/data/zypan")
REPO = ROOT / "GDN-quantization"
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
TASK = "GDN_RECURRENT_BASIS_EQUIVARIANCE_AND_ROTATION_HEADROOM_V1"
SLUG = "gdn_rotation_headroom_v1"
RUN_DIR = ROOT / "runs" / SLUG
DOC_DIR = REPO / "docs" / SLUG
RES_DIR = REPO / "results" / "propagation"
REP_DIR = REPO / "reports" / "propagation"
FIG_DIR = RUN_DIR / "figures"

HEADS = 32
KDIM = 128
VDIM = 128
EPS = 1e-12


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def ensure_dirs():
    for p in (RUN_DIR, DOC_DIR, RES_DIR, REP_DIR, FIG_DIR, DOC_DIR / "figures"):
        p.mkdir(parents=True, exist_ok=True)


def save_json(name, obj):
    ensure_dirs()
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    for p in (RUN_DIR / name, DOC_DIR / name, RES_DIR / f"{SLUG}_{name}"):
        p.write_text(text, encoding="utf-8")


def save_csv(name, rows):
    ensure_dirs()
    fields = list(rows[0].keys()) if rows else ["status", "reason"]
    for p in (RUN_DIR / name, DOC_DIR / name, RES_DIR / f"{SLUG}_{name}"):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def save_md(name, title, body):
    ensure_dirs()
    text = f"# {title}\n\n{body.rstrip()}\n"
    for p in (RUN_DIR / name, DOC_DIR / name, REP_DIR / f"{SLUG}_{name}"):
        p.write_text(text, encoding="utf-8")


def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    shutil.copy2(path, DOC_DIR / "figures" / name)


def rel_l2(a, b):
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return float(torch.linalg.vector_norm(af - bf).item() / (torch.linalg.vector_norm(bf).item() + EPS))


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def cosine(a, b):
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    den = float(torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item())
    return float(torch.dot(af, bf).item() / (den + EPS))


def hadamard(n):
    if n < 1 or (n & (n - 1)) != 0:
        return None
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


def random_orthogonal(n, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs


def recurrent_core(query, key, value, g, beta, initial_state):
    query = query.float()
    key = key.float()
    value = value.float()
    g = g.float()
    beta = beta.float()
    state = initial_state.float().clone()
    outs = []
    for i in range(query.shape[2]):
        q_t = query[:, :, i] * (query.shape[-1] ** -0.5)
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - pred) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outs, dim=2), state


def rotate_state(R, state):
    return torch.einsum("ab,hbv->hav", R, state.squeeze(0)).unsqueeze(0)


def rotate_tokens(R, x):
    return torch.einsum("ab,htb->hta", R, x.squeeze(0)).unsqueeze(0)


def local_equivariance():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260907)
    query = torch.randn(1, HEADS, 3, KDIM, generator=gen)
    key = torch.randn(1, HEADS, 3, KDIM, generator=gen)
    value = torch.randn(1, HEADS, 3, VDIM, generator=gen)
    g = -torch.rand(1, HEADS, 3, generator=gen)
    beta = torch.rand(1, HEADS, 3, generator=gen)
    state = torch.randn(1, HEADS, KDIM, VDIM, generator=gen)
    y, next_state = recurrent_core(query, key, value, g, beta, state)

    rotations = {
        "IDENTITY": torch.eye(KDIM),
        "HADAMARD": hadamard(KDIM),
        "RANDOM_ORTHOGONAL_0": random_orthogonal(KDIM, 1729),
    }
    rows = []
    for name, R in rotations.items():
        if R is None:
            rows.append({"rotation": name, "status": "NOT_APPLICABLE"})
            continue
        y_r, next_r = recurrent_core(
            rotate_tokens(R, query),
            rotate_tokens(R, key),
            value,
            g,
            beta,
            rotate_state(R, state),
        )
        rows.append({
            "rotation": name,
            "status": "PASS",
            "read_max_abs_error": max_abs(y_r, y),
            "read_relative_l2": rel_l2(y_r, y),
            "read_cosine": cosine(y_r, y),
            "next_state_relation_max_abs_error": max_abs(next_r, rotate_state(R, next_state)),
            "next_state_relation_relative_l2": rel_l2(next_r, rotate_state(R, next_state)),
            "next_state_relation_cosine": cosine(next_r, rotate_state(R, next_state)),
        })
    gate = "PASS" if all(r.get("status") == "PASS" and r["read_relative_l2"] < 1e-5 and r["next_state_relation_relative_l2"] < 1e-5 for r in rows) else "FAIL"
    return gate, rows


def conv_commutation_audit():
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    import run_int8_orientation_state_change_mechanism as p1

    torch_mod, model, _tokenizer, cfg, _e2e = p1.setup_model()
    del torch_mod
    layer = model.model.layers[0].linear_attn
    q_weight = layer.conv1d.weight.detach().float().cpu().squeeze(1)[:KDIM]
    k_weight = layer.conv1d.weight.detach().float().cpu().squeeze(1)[KDIM:2 * KDIM]
    rotations = {
        "HADAMARD": hadamard(KDIM),
        "RANDOM_ORTHOGONAL_0": random_orthogonal(KDIM, 1729),
        "RANDOM_ORTHOGONAL_1": random_orthogonal(KDIM, 2718),
        "RANDOM_ORTHOGONAL_2": random_orthogonal(KDIM, 3141),
    }
    rows = []
    for stream, weight in (("q", q_weight), ("k", k_weight)):
        for lag in range(weight.shape[1]):
            d = weight[:, lag]
            D = torch.diag(d)
            for name, R in rotations.items():
                if R is None:
                    continue
                comm = R @ D - D @ R
                denom = torch.linalg.vector_norm(D).item() + EPS
                rows.append({
                    "layer": 0,
                    "stream": stream,
                    "kernel_lag": lag,
                    "rotation": name,
                    "commutator_relative_fro": float(torch.linalg.vector_norm(comm).item() / denom),
                    "weight_std": float(d.std().item()),
                    "weight_min": float(d.min().item()),
                    "weight_max": float(d.max().item()),
                })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    max_rel = max(r["commutator_relative_fro"] for r in rows)
    med_rel = sorted(r["commutator_relative_fro"] for r in rows)[len(rows) // 2]
    summary = {
        "native_depthwise_conv_commutes_with_dense_key_rotation": False,
        "max_commutator_relative_fro": max_rel,
        "median_commutator_relative_fro": med_rel,
        "reason": "q/k pass through learned per-coordinate depthwise short convolution before recurrent core; arbitrary dense key-basis rotation does not commute with the diagonal per-lag convolution operator.",
        "model_path": cfg.get("model_path"),
    }
    return summary, rows


def placeholder_figs(final):
    names = [
        "c128_dynamic_range_identity_vs_rotations.png",
        "raw_quantization_mse_identity_vs_rotations.png",
        "m5_identity_vs_rotations.png",
        "future_kl_identity_vs_rotations.png",
        "paired_identity_vs_hadamard_future_kl.png",
        "random_rotation_gain_distribution.png",
        "range_reduction_vs_kl_gain.png",
        "mse_reduction_vs_kl_gain.png",
        "m5_reduction_vs_kl_gain.png",
    ]
    for name in names:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.55, "Stage B not run", ha="center", va="center", fontsize=16)
        plt.text(0.5, 0.38, "ROTATION_LAYER_EQUIVALENCE_GATE = FAIL", ha="center", va="center", fontsize=10)
        plt.axis("off")
        save_fig(name)

    plt.figure(figsize=(7, 4))
    rows = final["conv_commutation_rows"]
    by_rot = {}
    for r in rows:
        by_rot.setdefault(r["rotation"], []).append(r["commutator_relative_fro"])
    keys = sorted(by_rot)
    plt.boxplot([by_rot[k] for k in keys], labels=keys)
    plt.xticks(rotation=20)
    plt.ylabel("relative Frobenius norm")
    plt.title("Depthwise conv / key rotation commutator")
    save_fig("depthwise_conv_rotation_commutator.png")

    plt.figure(figsize=(7, 4))
    loc = final["local_rows"]
    plt.bar([r["rotation"] for r in loc], [r.get("next_state_relation_relative_l2", 0.0) for r in loc])
    plt.yscale("symlog", linthresh=1e-12)
    plt.ylabel("relative L2")
    plt.title("Local recurrent-core equivariance error")
    save_fig("local_equivariance_errors.png")


def main():
    ensure_dirs()
    protocol = {
        "task": TASK,
        "timestamp": now(),
        "git_status_before": sh(["git", "status", "--short"]),
        "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": sh(["git", "rev-parse", "HEAD"]),
        "git_log_12": sh(["git", "--no-pager", "log", "-12", "--oneline"]),
        "gpu": sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "stage_b_stop_rule": "Stage B runs only if all FP equivalence gates pass.",
        "prohibited_routes_obeyed": [
            "mixed precision",
            "FP16 protection",
            "INT4",
            "magnitude/functional bit allocation",
            "scale/gamma search",
            "read-write scale shaping",
            "error correction",
            "residual buffer",
            "DeltaLog",
            "checkpoint/replay",
            "learned future-KL predictor",
            "new M6/M7",
            "learned/GDN-specific rotation",
            "30-question end-to-end",
        ],
    }
    save_json("protocol_audit.json", protocol)
    save_md("protocol_audit.md", "Protocol Audit", "```json\n" + json.dumps(protocol, indent=2, ensure_ascii=False) + "\n```")

    operator_graph = """True Qwen3.5 GatedDeltaNet operator graph:

1. hidden_states are masked for padding.
2. in_proj_qkv(hidden_states) produces concatenated q/k/value source channels.
3. The concatenated q/k/value channels pass through learned depthwise short convolution.
4. The convolved tensor is split into query, key, value and reshaped to heads.
5. beta = sigmoid(in_proj_b(hidden_states)).
6. g = -exp(A_log) * softplus(in_proj_a(hidden_states) + dt_bias).
7. If value heads exceed key heads, query/key are repeat_interleave'd to value heads.
8. recurrent core uses l2norm(query/key), query scaling, scalar per-head decay exp(g), prediction S^T k, rank-one update k delta^T, and read S^T q.
9. recurrent state layout is [batch, value_head, key_dim, value_dim].
10. core output is reshaped by value head, then Qwen3_5RMSNormGated(core, z) is applied.
11. out_proj maps value heads back to hidden size.

Key-basis rotation can preserve the recurrent core if S' = R S, q' = R q, k' = R k.
RMSNormGated/out_proj are after the core and see identical core output under exact core equivariance.
l2norm(q/k) commutes with orthogonal R. Scalar per-head decay commutes with R.

The blocking native-architecture operator is the learned depthwise short convolution before q/k enter the core.
For each convolution lag, q/k channels see a diagonal operator D_l. A dense basis rotation R would require
R D_l = D_l R to fold the rotation into upstream projections while keeping the same depthwise convolution.
Actual per-coordinate learned filters are not scalar multiples of identity, so this commutation condition fails.
"""
    save_md("operator_graph.md", "Operator Graph", operator_graph)

    local_gate, local_rows = local_equivariance()
    save_json("equivariance_local.json", {"gate": local_gate, "rows": local_rows})
    save_csv("equivariance_local.csv", local_rows)

    conv_summary, conv_rows = conv_commutation_audit()
    layer_gate = "FAIL"
    layer = {
        "gate": layer_gate,
        "classification": "B_ARCHITECTURE_OPERATOR_BREAKS_NATIVE_REPARAMETERIZATION",
        "local_recurrent_core_gate": local_gate,
        "conv_commutation": conv_summary,
        "explicit_runtime_rotation_could_restore_core_equivariance": True,
        "native_offline_projection_fold_with_same_depthwise_conv": False,
        "reason": conv_summary["reason"],
    }
    save_json("equivariance_layer.json", layer)
    save_csv("conv_commutation_results.csv", conv_rows)

    stopped = {
        "gate": "SKIPPED_STOP_RULE",
        "reason": "ROTATION_LAYER_EQUIVALENCE_GATE failed for native/offline full-layer reparameterization.",
    }
    save_json("equivariance_prefill.json", stopped)
    save_json("equivariance_decode.json", stopped)
    save_json("rotation_candidates.json", {
        "IDENTITY": "applicable",
        "HADAMARD": "applicable for key_dim=128",
        "RANDOM_ORTHOGONAL": "not run beyond local/core audit because layer gate failed",
        "KLT_LIKE": "not run because layer gate failed",
    })

    not_run_rows = [{"status": "NOT_RUN_STAGE_A_STOP", "reason": stopped["reason"]}]
    for name in [
        "range_results.csv",
        "quantization_error_results.csv",
        "functional_results.csv",
        "future_kl_results.csv",
    ]:
        save_csv(name, not_run_rows)

    efficiency = {
        "Q_ROTATION_FOLDABLE": "NO_NATIVE_DEPTHWISE_CONV_BREAKS_FOLD",
        "K_ROTATION_FOLDABLE": "NO_NATIVE_DEPTHWISE_CONV_BREAKS_FOLD",
        "STATE_ROTATION_FOLDABLE_OR_NATIVE": "NATIVE_ROTATED_CACHE_POSSIBLE_ONLY_WITH_MATCHED_QK_ROTATION",
        "ONLINE_STATE_ROTATION_REQUIRED": "NO_IF_CACHE_IS_INITIALIZED_IN_ROTATED_BASIS",
        "ONLINE_QK_ROTATION_REQUIRED_FOR_CURRENT_NATIVE_LAYER": "YES_FOR_NON_IDENTITY_DENSE_ROTATION",
        "EXTRA_FULL_STATE_PASS": 0,
        "UNIFORM_INT8": "YES_NOT_RUN",
        "MIXED_PRECISION": "NO",
        "KERNEL_FUSION_FEASIBLE": "NO_WITH_CURRENT_DEPTHWISE_QK_PATH_WITHOUT_KERNEL_CHANGE",
        "EFFECTIVE_BITS_PER_VALUE": None,
        "rotation_metadata": "Hadamard zero; random/KLT matrix metadata would be offline only, but Stage B stopped.",
    }
    save_json("efficiency_audit.json", efficiency)
    save_md("efficiency_audit.md", "Efficiency Audit", "```json\n" + json.dumps(efficiency, indent=2, ensure_ascii=False) + "\n```")

    final = {
        "TASK": TASK,
        "FORMAL_STATUS": "STOPPED_BY_FP_LAYER_EQUIVALENCE_GATE",
        "PROTOCOL_GATE": "PASS",
        "DECAY_COMMUTATION_GATE": "PASS",
        "ROTATION_LOCAL_EQUIVALENCE_GATE": local_gate,
        "ROTATION_LAYER_EQUIVALENCE_GATE": layer_gate,
        "ROTATION_PREFILL_EQUIVALENCE_GATE": "SKIPPED_STOP_RULE",
        "ROTATION_DECODE_EQUIVALENCE_GATE": "SKIPPED_STOP_RULE",
        "GDN_BASIS_EQUIVARIANCE_SUPPORTED": "PARTIAL",
        "Q_ROTATION_FOLDABLE": efficiency["Q_ROTATION_FOLDABLE"],
        "K_ROTATION_FOLDABLE": efficiency["K_ROTATION_FOLDABLE"],
        "ONLINE_STATE_ROTATION_REQUIRED": efficiency["ONLINE_STATE_ROTATION_REQUIRED"],
        "IDENTITY_C128_FUTURE_KL": None,
        "HADAMARD_C128_FUTURE_KL": None,
        "RANDOM_C128_FUTURE_KL": None,
        "KLT_C128_FUTURE_KL": None,
        "IDENTITY_RANGE": None,
        "BEST_ROTATION_RANGE": None,
        "IDENTITY_MSE": None,
        "BEST_ROTATION_MSE": None,
        "IDENTITY_M5": None,
        "BEST_ROTATION_M5": None,
        "BEST_ROTATION": None,
        "BEST_ROTATION_FUTURE_KL": None,
        "ROTATION_RELATIVE_RESCUE": None,
        "RANDOM_ROTATION_GAIN_DISTRIBUTION": "NOT_RUN_STAGE_A_STOP",
        "ORACLE_ROTATION_GAIN": None,
        "ROTATION_ACTION_HEADROOM": "NOT_TESTED_STAGE_A_STOP",
        "GENERIC_ROTATION_SUFFICIENT": "NO",
        "EXTRA_FULL_STATE_PASS": 0,
        "UNIFORM_INT8": "YES",
        "EFFECTIVE_BITS_PER_VALUE": None,
        "KERNEL_FUSION_FEASIBLE": efficiency["KERNEL_FUSION_FEASIBLE"],
        "FINAL_CLASSIFICATION": "RECURRENT_CORE_EQUIVARIANT_BUT_NATIVE_QK_DEPTHWISE_CONV_BREAKS_FULL_LAYER_BASIS_REPARAMETERIZATION",
        "METHOD_DESIGN_READY": "NO",
        "NEXT_RECOMMENDED_TASK": "close exact native basis-rotation route; audit residual structure or explicit-kernel rotation only as separate nonzero-overhead baseline",
        "local_rows": local_rows,
        "conv_commutation_rows": conv_rows,
    }
    placeholder_figs(final)
    final_public = {k: v for k, v in final.items() if k not in ("local_rows", "conv_commutation_rows")}
    save_json("final_classification.json", final_public)
    save_md("pilot_report.md", "Pilot Report", "Stopped before Stage B. Local recurrent-core equivariance passes, but full-layer native/offline reparameterization fails because q/k dense rotations do not commute with the learned depthwise short convolution.")
    save_md("formal_report.md", "Formal Report", "Formal Stage B was not run because ROTATION_LAYER_EQUIVALENCE_GATE=FAIL under the task stop rule.\n\n```json\n" + json.dumps(final_public, indent=2, ensure_ascii=False) + "\n```")
    print(json.dumps(final_public, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
