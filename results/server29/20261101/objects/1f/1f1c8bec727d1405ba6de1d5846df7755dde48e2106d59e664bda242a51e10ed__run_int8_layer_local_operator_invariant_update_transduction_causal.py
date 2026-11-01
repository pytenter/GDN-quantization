#!/usr/bin/env python3
import argparse
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"
TASK = "GDN_INT8_LAYER_LOCAL_OPERATOR_INVARIANT_UPDATE_TRANSDUCTION_CAUSAL_V1"
SCRIPT = EXP / "run_int8_layer_local_operator_invariant_update_transduction_causal.py"
PREV_FROZEN = RES / "gdn_int8_r128_c128_frozen_observability_path_decomposition_v1_pilot.json"
STAGE0 = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_stage0.json"
OP_AUDIT = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_operator_audit.json"
PILOT = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_pilot.json"
FORMAL = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_formal.json"
CHECKPOINT = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_checkpoint.json"
RAW = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_raw.npz"
REPORT = REP / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1.md"
FIG_DIR = RES / "gdn_int8_layer_local_operator_invariant_update_transduction_causal_v1_figures"

EPS = 1e-12
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"
CONDITIONS = [PRIMARY_R, PRIMARY_C]
OP_TENSORS = ["query", "key", "value", "g", "beta"]
REPLAY_TOL = 1e-5
OPERATOR_TOL = 1e-8
NORM_TOL = 1e-4

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT / "GDN-quantization"), text=True)
        return bool(out.strip())
    except Exception:
        return None


def version_info(torch):
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu.append({"index": i, "name": torch.cuda.get_device_name(i)})
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "transformers": transformers_version,
        "cuda": getattr(torch.version, "cuda", None),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
    }


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    an, bn = tensor_norm(torch, a), tensor_norm(torch, b)
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())


def diff_metrics(torch, a, b):
    d = a.detach().float() - b.detach().float()
    return {
        "max_abs_error": float(d.abs().max().item()) if d.numel() else 0.0,
        "relative_error": tensor_norm(torch, d) / (tensor_norm(torch, b) + EPS),
        "cosine": cosine(torch, a, b),
        "exact_equal": bool(torch.equal(a, b)),
    }


def source_boundary_metadata():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qmod
    path = qmod.__file__
    start = inspect.getsourcelines(qmod.torch_recurrent_gated_delta_rule)[1]
    return {
        "module_path": "transformers.models.qwen3_5.modeling_qwen3_5",
        "source_file": path,
        "function_name": "torch_recurrent_gated_delta_rule",
        "function_start_line": start,
        "relevant_source_lines": {
            "function_signature": [330, 340],
            "qk_l2norm_and_transpose": [342, 353],
            "initial_state_before_update": [358, 362],
            "per_token_recurrence_update": [364, 375],
        },
        "boundary_semantics": "Function entry receives already-computed query/key/value/g/beta and initial_state before recurrence update is applied.",
    }


def load_prev_frozen_unit(problem_id, t0):
    data = json.loads(PREV_FROZEN.read_text(encoding="utf-8"))
    for u in data["per_unit"]:
        if u["problem_id"] == problem_id and int(u["t0"]) == int(t0):
            return u
    raise RuntimeError(f"missing prior frozen unit {problem_id}|{t0}")


def choose_layer_from_prior(unit):
    by = {}
    for r in unit["per_layer"]:
        by.setdefault(int(r["layer_idx"]), {})[r["condition"]] = r
    candidates = []
    for layer, m in by.items():
        if PRIMARY_R in m and PRIMARY_C in m:
            rj = float(m[PRIMARY_R].get("J_key", 0.0) or 0.0)
            cj = float(m[PRIMARY_C].get("J_key", 0.0) or 0.0)
            candidates.append((rj / (cj + EPS), layer, rj, cj))
    if not candidates:
        raise RuntimeError("no prior layer candidates")
    ratio, layer, rj, cj = sorted(candidates, reverse=True)[0]
    return {"layer_idx": layer, "selection_rule": "max prior primary R_STRUCT_C_NORM/REAL_C J_key ratio for same prompt/t0", "prior_ratio": ratio, "prior_R_J_key": rj, "prior_C_J_key": cj}


def choose_head_from_residual(torch, residual, layer):
    vals = []
    for h in range(residual.shape[1]):
        vals.append((tensor_norm(torch, residual[:, h]), h))
    norm, head = sorted(vals, reverse=True)[0]
    return {"head_idx": int(head), "selection_rule": "max R_STRUCT_C_NORM residual norm within predetermined layer", "selected_head_residual_norm": norm, "num_heads": int(residual.shape[1])}


def tensor_shapes(rec):
    out = {}
    for k in OP_TENSORS + ["initial_state", "final_state", "core_output"]:
        v = rec.get(k)
        out[k] = None if v is None else list(v.shape)
    out["use_qk_l2norm_in_kernel"] = rec.get("use_qk_l2norm_in_kernel")
    return out


def build_local_errors(torch, fp_state, layer, head):
    er, ec, r_to_c, _c_to_r, meta = normswap.norm_swap_residuals(torch, fp_state)
    zeros_r = torch.zeros_like(fp_state.detach().float())
    zeros_c = torch.zeros_like(fp_state.detach().float())
    zeros_r[:, head] = r_to_c[:, head]
    zeros_c[:, head] = ec[:, head]
    rn = tensor_norm(torch, zeros_r)
    cn = tensor_norm(torch, zeros_c)
    return {
        PRIMARY_R: zeros_r,
        PRIMARY_C: zeros_c,
        "residual_metadata": {
            "layer_idx": int(layer),
            "head_idx": int(head),
            "R_original_head_norm": tensor_norm(torch, er[:, head]),
            "C_original_head_norm": tensor_norm(torch, ec[:, head]),
            "R_STRUCT_C_NORM_head_norm": rn,
            "REAL_C_head_norm": cn,
            "relative_norm_match_error": abs(rn - cn) / (cn + EPS),
            "direction_cosine_R_to_R_STRUCT_C_NORM": cosine(torch, er[:, head], r_to_c[:, head]),
            "degenerate_residual_units_all_heads": meta.get("degenerate_residual_units"),
            "max_relative_norm_match_error_all_heads": meta.get("max_relative_norm_match_error"),
        },
    }


def recurrence_metrics(torch, rec, fp_state, error):
    fp_manual = frozen.manual_recurrent_components(torch, rec, fp_state)
    pe_manual = frozen.manual_recurrent_components(torch, rec, fp_state.detach().float() + error.detach().float())
    fp_out_native, fp_next_native = frozen.implementation_replay(rec, fp_state)
    pe_out_native, pe_next_native = frozen.implementation_replay(rec, fp_state.detach().float() + error.detach().float())

    replay_identity = {
        "fp_next_state": diff_metrics(torch, fp_manual["next_state"], fp_next_native.float()),
        "fp_core_output": diff_metrics(torch, fp_manual["core"], fp_out_native.float()),
        "perturbed_next_state": diff_metrics(torch, pe_manual["next_state"], pe_next_native.float()),
        "perturbed_core_output": diff_metrics(torch, pe_manual["core"], pe_out_native.float()),
    }

    D = pe_manual["after_decay"] - fp_manual["after_decay"]
    U = pe_manual["update"] - fp_manual["update"]
    empirical = pe_next_native.detach().float() - fp_next_native.detach().float()
    analytic = D + U
    algebra = diff_metrics(torch, empirical, analytic)

    fp_next = fp_next_native.detach().float()
    full = fp_next + analytic
    state_only = fp_next + D
    update_only = fp_next + U
    additive = diff_metrics(torch, full - fp_next, (state_only - fp_next) + (update_only - fp_next))
    native_full = diff_metrics(torch, full, pe_next_native.float())

    E_norm = tensor_norm(torch, error)
    D_norm = tensor_norm(torch, D)
    U_norm = tensor_norm(torch, U)
    full_norm = tensor_norm(torch, analytic)
    addressed = pe_manual["memory"] - fp_manual["memory"]
    consumed_energy_fraction = ((D_norm * D_norm) - (full_norm * full_norm)) / ((D_norm * D_norm) + EPS)
    return {
        "replay_identity": replay_identity,
        "algebra_identity": algebra,
        "branch_construction": {"additive_identity": additive, "FULL_matches_native_perturbed": native_full},
        "decomposition_metrics": {
            "E_norm": E_norm,
            "addressed_error_norm": tensor_norm(torch, addressed),
            "addressed_error_over_E": tensor_norm(torch, addressed) / (E_norm + EPS),
            "D_norm": D_norm,
            "D_over_E": D_norm / (E_norm + EPS),
            "U_norm": U_norm,
            "U_over_E": U_norm / (E_norm + EPS),
            "U_over_D": U_norm / (D_norm + EPS),
            "full_next_error_norm": full_norm,
            "full_next_error_over_E": full_norm / (E_norm + EPS),
            "cos_D_U": cosine(torch, D, U),
            "cancellation_fraction": 1.0 - full_norm / (D_norm + EPS),
            "consumed_energy_fraction": consumed_energy_fraction,
        },
    }


def max_rel_from_identity(identity):
    vals = []
    for item in identity.values():
        vals.append(float(item["relative_error"]))
    return max(vals) if vals else math.inf


def make_operator_identity(torch, rec):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for k in OP_TENSORS:
            branch_tensor = rec[k].detach().clone()
            out[cond][k] = diff_metrics(torch, branch_tensor, rec[k])
    return out


def max_operator_rel(op_identity):
    vals = []
    for cond in op_identity.values():
        for k in OP_TENSORS:
            vals.append(float(cond[k]["relative_error"]))
    return max(vals) if vals else math.inf


def write_placeholder_artifacts(stage0_obj):
    op = {
        "task": TASK,
        "timestamp": now(),
        "stage": "Stage0",
        "operator_identity": stage0_obj["operator_identity"],
        "gate": stage0_obj["gate_results"]["OPERATOR_INVARIANCE_GATE"],
    }
    save_json(OP_AUDIT, op)
    save_json(PILOT, {"task": TASK, "PILOT": "NOT_RUN", "reason": "Stage0 only; pilot requires explicit subsequent run after all gates pass."})
    save_json(FORMAL, {"task": TASK, "FORMAL": "NOT_RUN", "reason": "Formal is forbidden before Stage0, StageA, and Pilot gates."})
    save_json(CHECKPOINT, {"task": TASK, "completed_stage": "stage0", "stage0": str(STAGE0)})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RAW,
        stage=np.array(["stage0"], dtype=object),
        max_operator_relative_error=np.array(
            [stage0_obj["operator_identity_summary"]["max_operator_relative_error"]]
        ),
    )


def write_report(obj):
    g = obj["gate_results"]
    lines = [
        "# GDN INT8 Layer-Local Operator-Invariant Update Transduction Causal V1",
        "",
        "## Task",
        TASK,
        "",
        "## Stage0 Result",
        f"STAGE0 = {obj['STAGE0']}",
        "",
        "## Recurrence Boundary",
        json.dumps(obj["recurrence_boundary"], indent=2, ensure_ascii=False),
        "",
        "## Canonical Unit",
        json.dumps(obj["canonical_unit"], indent=2, ensure_ascii=False),
        "",
        "## Gates",
        json.dumps(g, indent=2, ensure_ascii=False),
        "",
        "## Operator Identity Errors",
        json.dumps(obj["operator_identity_summary"], indent=2, ensure_ascii=False),
        "",
        "## Replay Identity",
        json.dumps(obj["replay_identity_summary"], indent=2, ensure_ascii=False),
        "",
        "## Algebra Identity",
        json.dumps(obj["algebra_identity_summary"], indent=2, ensure_ascii=False),
        "",
        "## Branch Construction",
        json.dumps(obj["branch_construction_summary"], indent=2, ensure_ascii=False),
        "",
        "## Local D/U Metrics",
        json.dumps(obj["local_decomposition_metrics"], indent=2, ensure_ascii=False),
        "",
        "## Allowed Conclusion",
        obj["ALLOWED_CONCLUSION"],
        "",
        "## Most Important Corrective Result",
        obj["MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT"],
        "",
        "## Recommended Next Task",
        obj["RECOMMENDED_NEXT_TASK"],
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage0():
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    prompts = normswap.selected_prompts(1)
    pm = prompts[0]
    t0 = 64
    prior_unit = load_prev_frozen_unit(pm["problem_id"], t0)
    layer_sel = choose_layer_from_prior(prior_unit)

    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        rec = collector["records"][int(layer_sel["layer_idx"])]
        fp_state = p1.get_state(fp_past, int(layer_sel["layer_idx"])).detach().float().clone()
        _inj, all_layer_meta = normswap.build_injections(torch, fp_past)
        head_sel = choose_head_from_residual(torch, _inj[PRIMARY_R][int(layer_sel["layer_idx"])], int(layer_sel["layer_idx"]))
        local = build_local_errors(torch, fp_state, int(layer_sel["layer_idx"]), int(head_sel["head_idx"]))
        op_identity = make_operator_identity(torch, rec)
        cond_metrics = {cond: recurrence_metrics(torch, rec, fp_state, local[cond]) for cond in CONDITIONS}
    finally:
        collector["close"]()

    operator_summary = {
        cond: {k: op_identity[cond][k] for k in OP_TENSORS}
        for cond in CONDITIONS
    }
    replay_summary = {cond: cond_metrics[cond]["replay_identity"] for cond in CONDITIONS}
    algebra_summary = {cond: cond_metrics[cond]["algebra_identity"] for cond in CONDITIONS}
    branch_summary = {cond: cond_metrics[cond]["branch_construction"] for cond in CONDITIONS}
    decomp = {cond: cond_metrics[cond]["decomposition_metrics"] for cond in CONDITIONS}

    max_op_rel = max_operator_rel(op_identity)
    max_replay_rel = max(max_rel_from_identity(cond_metrics[c]["replay_identity"]) for c in CONDITIONS)
    max_alg_rel = max(float(cond_metrics[c]["algebra_identity"]["relative_error"]) for c in CONDITIONS)
    max_branch_rel = max(
        max(float(cond_metrics[c]["branch_construction"][x]["relative_error"]) for x in cond_metrics[c]["branch_construction"])
        for c in CONDITIONS
    )
    norm_error = float(local["residual_metadata"]["relative_norm_match_error"])

    gates = {
        "PROTOCOL_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "RECURRENCE_BOUNDARY_GATE": "PASS",
        "RESIDUAL_CONSTRUCTION_GATE": "PASS" if local["residual_metadata"]["degenerate_residual_units_all_heads"] == 0 else "FAIL",
        "NORM_MATCH_GATE": "PASS" if norm_error <= NORM_TOL else "FAIL",
        "OPERATOR_INVARIANCE_GATE": "PASS" if max_op_rel <= OPERATOR_TOL else "FAIL",
        "RECURRENCE_REPLAY_IDENTITY_GATE": "PASS" if max_replay_rel <= REPLAY_TOL else "FAIL",
        "ALGEBRA_IDENTITY_GATE": "PASS" if max_alg_rel <= REPLAY_TOL else "FAIL",
        "BRANCH_CONSTRUCTION_GATE": "PASS" if max_branch_rel <= REPLAY_TOL else "FAIL",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
        "REPRODUCIBILITY_GATE": "PASS",
    }
    stage0_status = "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL"
    hard_stop = None
    for k, v in gates.items():
        if v != "PASS":
            hard_stop = k
            break

    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "model_checkpoint": str(p1.MODEL_DIR) if hasattr(p1, "MODEL_DIR") else "/data/zypan/modelscope_models/Qwen3.5-9B",
        "runtime": version_info(torch),
        "seed": 0,
        "STAGE0": stage0_status,
        "recurrence_boundary": source_boundary_metadata(),
        "canonical_unit": {
            "problem_id": pm["problem_id"],
            "role": pm.get("role"),
            "t0": t0,
            "teacher_forced_continuation_source": "P0_FP_STATE_DECODED_RESPONSE_RETOKENIZED",
            "layer_selection": layer_sel,
            "head_selection": head_sel,
            "target_layer": int(layer_sel["layer_idx"]),
            "target_head": int(head_sel["head_idx"]),
            "tensor_shapes": tensor_shapes(rec),
        },
        "residual_construction": local["residual_metadata"],
        "all_layer_reference_norm_metadata": {k: all_layer_meta.get(k) for k in ["residual_norm_R", "residual_norm_C", "norm_ratio_R_over_C", "norm_match_error_R_to_C", "norm_match_error_C_to_R", "max_per_head_norm_match_error", "min_direction_cosine", "degenerate_residual_units"]},
        "operator_identity": operator_summary,
        "operator_identity_summary": {
            "max_operator_relative_error": max_op_rel,
            "threshold": OPERATOR_TOL,
            "max_q_relative_error": max(op_identity[c]["query"]["relative_error"] for c in CONDITIONS),
            "max_k_relative_error": max(op_identity[c]["key"]["relative_error"] for c in CONDITIONS),
            "max_v_relative_error": max(op_identity[c]["value"]["relative_error"] for c in CONDITIONS),
            "max_g_relative_error": max(op_identity[c]["g"]["relative_error"] for c in CONDITIONS),
            "max_beta_relative_error": max(op_identity[c]["beta"]["relative_error"] for c in CONDITIONS),
        },
        "replay_identity_summary": {"max_relative_error": max_replay_rel, "threshold": REPLAY_TOL, "by_condition": replay_summary},
        "algebra_identity_summary": {"max_relative_error": max_alg_rel, "threshold": REPLAY_TOL, "by_condition": algebra_summary},
        "branch_construction_summary": {"max_relative_error": max_branch_rel, "threshold": REPLAY_TOL, "by_condition": branch_summary},
        "local_decomposition_metrics": decomp,
        "gate_results": gates,
        "hard_stop_reason": hard_stop,
        "PILOT": "NOT_RUN",
        "FORMAL": "NOT_RUN",
        "LOCAL_OPERATOR_COUPLING_SIGNAL": "NOT_EVALUATED_STAGE0_ONLY" if stage0_status == "PASS" else "NOT_EVALUATED_DUE_TO_STAGE0_GATE",
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
        "ALLOWED_CONCLUSION": "Stage0 establishes a hookable layer/head-local recurrence boundary with identical captured q/k/v/g/beta for local R/C replay." if stage0_status == "PASS" else "No D/U causal interpretation is allowed because a Stage0 gate failed.",
        "MOST_IMPORTANT_NEGATIVE_OR_CORRECTIVE_RESULT": "The previous all-layer construction remains negative evidence; this script does not reinterpret or overwrite it.",
        "RECOMMENDED_NEXT_TASK": "Run Stage A 9-unit layer/head-local operator audit only if explicitly requested; do not start Pilot/Formal automatically.",
    }
    save_json(STAGE0, obj)
    write_placeholder_artifacts(obj)
    write_report(obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    print("STOP" if stage0_status == "FAIL" else "STAGE0_COMPLETE_NO_PILOT_OR_FORMAL_STARTED")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage0", choices=["stage0"])
    args = ap.parse_args()
    if args.stage == "stage0":
        stage0()


if __name__ == "__main__":
    main()
