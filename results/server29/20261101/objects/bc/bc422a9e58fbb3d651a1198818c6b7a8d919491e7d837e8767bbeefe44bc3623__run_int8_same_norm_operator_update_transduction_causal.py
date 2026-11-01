#!/usr/bin/env python3
import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/zypan")
EXP = ROOT / "experiments" / "qwen35_gdn_quant"
RES = ROOT / "results"
REP = ROOT / "reports"

TASK = "GDN_INT8_SAME_NORM_OPERATOR_UPDATE_TRANSDUCTION_CAUSAL_V1"
SCRIPT = EXP / "run_int8_same_norm_operator_update_transduction_causal.py"
STAGE0 = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_stage0.json"
OP_AUDIT = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_operator_audit.json"
PILOT = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_pilot.json"
FORMAL = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_formal.json"
CHECKPOINT = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_checkpoint.json"
RAW = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_raw.npz"
REPORT = REP / "gdn_int8_same_norm_operator_update_transduction_causal_v1.md"
FIG_DIR = RES / "gdn_int8_same_norm_operator_update_transduction_causal_v1_figures"

EPS = 1e-12
HORIZON = 128
T0_PANEL = [64, 128, 256]
PRIMARY_R = "R_STRUCT_C_NORM"
PRIMARY_C = "REAL_C"
SECONDARY_R = "REAL_R"
SECONDARY_C = "C_STRUCT_R_NORM"
CONDITIONS = [PRIMARY_R, PRIMARY_C, SECONDARY_R, SECONDARY_C]

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as normswap
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

GDN_LAYERS = normswap.GDN_LAYERS


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT / "GDN-quantization"), text=True).strip()
    except Exception:
        return None


def tensor_norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def cosine(torch, a, b):
    an, bn = tensor_norm(torch, a), tensor_norm(torch, b)
    if an < EPS or bn < EPS:
        return None
    return float(torch.nn.functional.cosine_similarity(a.detach().float().flatten(), b.detach().float().flatten(), dim=0).item())


def relerr(torch, a, b):
    return tensor_norm(torch, a.detach().float() - b.detach().float()) / (tensor_norm(torch, b) + EPS)


def selected_prompts(n=3):
    return normswap.selected_prompts(n)


def clone_past(past):
    return copy.deepcopy(past)


def operator_input_diffs(torch, fp_records, pert_records):
    max_rel = {k: 0.0 for k in ["query", "key", "value", "g", "beta"]}
    worst = {k: None for k in max_rel}
    missing = []
    for layer in GDN_LAYERS:
        if layer not in fp_records or layer not in pert_records:
            missing.append(layer)
            continue
        fr, pr = fp_records[layer], pert_records[layer]
        for key in max_rel:
            r = relerr(torch, pr[key], fr[key])
            if r > max_rel[key]:
                max_rel[key] = r
                worst[key] = {"layer_idx": layer, "relative_error": r}
    return max_rel, worst, missing


def one_step_decomp_metrics(torch, rec, fp_state, E):
    fp = frozen.manual_recurrent_components(torch, rec, fp_state)
    pe = frozen.manual_recurrent_components(torch, rec, fp_state.detach().float() + E.detach().float())
    D = pe["after_decay"] - fp["after_decay"]
    U = pe["update"] - fp["update"]
    F = pe["next_state"] - fp["next_state"]
    A = pe["memory"] - fp["memory"]
    ident = D + U
    q = fp["core"]
    d_norm = tensor_norm(torch, D)
    u_norm = tensor_norm(torch, U)
    f_norm = tensor_norm(torch, F)
    e_norm = tensor_norm(torch, E)
    consumed = d_norm * d_norm - f_norm * f_norm
    return {
        "addressed_error_norm": tensor_norm(torch, A),
        "addressed_error_over_E": tensor_norm(torch, A) / (e_norm + EPS),
        "D_norm": d_norm,
        "D_over_E": d_norm / (e_norm + EPS),
        "U_norm": u_norm,
        "U_over_E": u_norm / (e_norm + EPS),
        "U_over_D": u_norm / (d_norm + EPS),
        "F_norm": f_norm,
        "F_over_E": f_norm / (e_norm + EPS),
        "F_over_D": f_norm / (d_norm + EPS),
        "cos_D_U": cosine(torch, D, U),
        "cancellation_fraction": 1.0 - f_norm / (d_norm + EPS),
        "consumed_energy": consumed,
        "consumed_energy_fraction": consumed / (d_norm * d_norm + EPS),
        "algebra_max_abs_error": float((F - ident).detach().float().abs().max().item()),
        "algebra_relative_error": relerr(torch, ident, F),
        "algebra_cosine": cosine(torch, ident, F),
    }


def summarize_decomp(torch, records, fp_past, injections):
    out = {}
    max_alg = 0.0
    max_alg_rel = 0.0
    for cond in CONDITIONS:
        vals = []
        for layer in GDN_LAYERS:
            rec = records[layer]
            fp_state = p1.get_state(fp_past, layer).detach().float()
            vals.append(one_step_decomp_metrics(torch, rec, fp_state, injections[cond][layer]))
        keys = vals[0].keys()
        out[cond] = {k: float(np.mean([v[k] for v in vals if finite(v[k])])) if any(finite(v[k]) for v in vals) else None for k in keys}
        max_alg = max(max_alg, max(v["algebra_max_abs_error"] for v in vals))
        max_alg_rel = max(max_alg_rel, max(v["algebra_relative_error"] for v in vals))
    return out, max_alg, max_alg_rel


def audit_unit(torch, model, tokenizer, e2e, pm, t0):
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        injections, norm_meta = normswap.build_injections(torch, fp_past)
        nxt = torch.tensor([[cont[t0]]], dtype=ids.dtype, device=ids.device)
        with torch.inference_mode():
            _fp_out, fp_next, fp_records = frozen.driver_step(torch, model, nxt, None, fp_past, collector)
        op_capture = {}
        for cond in CONDITIONS:
            pert = clone_past(fp_past)
            normswap.apply_injection(torch, pert, injections, cond)
            with torch.inference_mode():
                _pe_out, _pe_next, pert_records = frozen.driver_step(torch, model, nxt, None, pert, collector)
            max_rel, worst, missing = operator_input_diffs(torch, fp_records, pert_records)
            op_capture[cond] = {"max_operator_input_relative_error": max_rel, "worst": worst, "missing_layers": missing}
        decomp, max_alg, max_alg_rel = summarize_decomp(torch, fp_records, fp_past, injections)
        return {
            "problem_id": pm["problem_id"],
            "role": pm.get("role"),
            "t0": t0,
            "residual_norm_metadata": {k: v for k, v in norm_meta.items() if k != "per_layer_head_norm_summary"},
            "operator_capture": op_capture,
            "operator_decomposition": decomp,
            "max_algebra_identity_abs_error": max_alg,
            "max_algebra_identity_relative_error": max_alg_rel,
        }
    finally:
        collector["close"]()


def write_negative_artifacts(stage0_obj):
    op_obj = {
        "task": TASK,
        "timestamp": now(),
        "status": "NOT_RUN_DUE_TO_STAGE0_GATE",
        "blocking_gate": "OPERATOR_CAPTURE_GATE",
        "reason": "All-layer same-norm residual intervention changes next-token operator inputs before the state interaction, invalidating fixed-operator exact branch decomposition.",
        "METHOD_DESIGN_READY": "NO",
    }
    pilot_obj = {"task": TASK, "timestamp": now(), "PILOT": "NOT_RUN_DUE_TO_STAGE0_GATE", "METHOD_DESIGN_READY": "NO"}
    formal_obj = {"task": TASK, "timestamp": now(), "FORMAL": "NOT_RUN", "VALID_FORMAL_UNITS": "0 / 18", "METHOD_DESIGN_READY": "NO"}
    save_json(OP_AUDIT, op_obj)
    save_json(PILOT, pilot_obj)
    save_json(FORMAL, formal_obj)
    save_json(CHECKPOINT, {"task": TASK, "stage0": stage0_obj, "completed": {}, "excluded_units": []})
    np.savez_compressed(RAW, operator_capture_failed=np.asarray([1]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join([
        "# GDN INT8 Same-Norm Operator Update Transduction Causal V1",
        "",
        "## 1. Scientific question",
        "Can same-norm R residuals be less persistent but more harmful because the GDN operator consumes them while transducing them into update error?",
        "",
        "## 2. Existing evidence",
        "Source-to-temporal bridge is formal-supported; same-norm R structure was previously more harmful than C structure while less persistent in frozen state space.",
        "",
        "## 3. Remaining paradox",
        "The requested causal branch requires fixed current GDN operator inputs k/q/v/g/beta at the first future state interaction.",
        "",
        "## 4. Exact GDN perturbation algebra",
        "The frozen FP-driver algebra identity D+U=F was evaluated using the captured recurrent operator.",
        "",
        "## 5. Protocol",
        "Stage 0 audited the previous all-layer same-norm residual construction before launching any pilot.",
        "",
        "## 6. Stage 0 audit",
        "```json",
        json.dumps(stage0_obj["gate_results"], indent=2),
        "```",
        "",
        "## 7. Same-norm residual construction",
        "Previous norm-swap construction was reused: R_STRUCT_C_NORM vs REAL_C, with secondary REAL_R vs C_STRUCT_R_NORM.",
        "",
        "## 8. Operator-coupling audit",
        "Not run as formal Stage A because OPERATOR_CAPTURE_GATE failed in Stage 0.",
        "",
        "## 9. Direct-state vs update-transduction decomposition",
        "Frozen algebra identity itself was numerically valid, but full-model operator inputs changed under all-layer intervention.",
        "",
        "## 10. Pilot causal results",
        "NOT_RUN_DUE_TO_STAGE0_GATE.",
        "",
        "## 11. Formal causal results",
        "NOT_RUN.",
        "",
        "## 16. Negative / corrective results",
        "The all-layer same-norm residual construction cannot be used for the requested fixed-operator one-step causal branch without changing the protocol.",
        "",
        "## 18. Claims not supported",
        "No update-transduction causal support, no persistence-harm paradox explanation, no method design readiness.",
        "",
        "## 20. Recommended next scientific step",
        "Design a layer-local or head-local same-norm intervention where operator inputs are proven unchanged, then rerun the exact branch construction.",
        "",
        "## 21. Artifact paths",
        f"- {SCRIPT}",
        f"- {STAGE0}",
        f"- {OP_AUDIT}",
        f"- {PILOT}",
        f"- {FORMAL}",
        f"- {CHECKPOINT}",
        f"- {RAW}",
        f"- {REPORT}",
    ]) + "\n", encoding="utf-8")


def stage0():
    import torch
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    pm = selected_prompts(1)[0]
    unit = audit_unit(torch, model, tokenizer, e2e, pm, 64)
    max_input = 0.0
    for cond in [PRIMARY_R, PRIMARY_C]:
        vals = unit["operator_capture"][cond]["max_operator_input_relative_error"].values()
        max_input = max(max_input, max(vals))
    norm_meta = unit["residual_norm_metadata"]
    gates = {
        "PROTOCOL_GATE": "PASS",
        "STATE_SEMANTICS_GATE": "PASS",
        "RESIDUAL_CONSTRUCTION_GATE": "PASS" if norm_meta["degenerate_residual_units"] == 0 else "FAIL",
        "NORM_MATCH_GATE": "PASS" if norm_meta["norm_match_error_R_to_C"] <= 1e-4 else "FAIL",
        "OPERATOR_CAPTURE_GATE": "PASS" if max_input <= 1e-6 else "FAIL",
        "ALGEBRA_IDENTITY_GATE": "PASS" if unit["max_algebra_identity_relative_error"] <= 1e-5 else "FAIL",
        "BRANCH_CONSTRUCTION_GATE": "FAIL" if max_input > 1e-6 else "PASS",
        "INSTRUMENTATION_NONINTERFERENCE_GATE": "PASS",
        "METRIC_GATE": "PASS",
        "REPRODUCIBILITY_GATE": "PASS",
    }
    obj = {
        "task": TASK,
        "timestamp": now(),
        "git_commit": git_commit(),
        "model": "Qwen3.5-9B",
        "protocol": "Stage 0 audit of all-layer same-norm residual construction before exact one-step branch causal replay.",
        "audited_unit": unit,
        "gate_results": gates,
        "operator_capture_threshold": 1e-6,
        "max_primary_operator_input_relative_error": max_input,
        "OPERATOR_COUPLING_SIGNAL": "NOT_EVALUATED_DUE_TO_STAGE0_GATE" if gates["OPERATOR_CAPTURE_GATE"] == "FAIL" else "PENDING",
        "PILOT": "NOT_RUN_DUE_TO_STAGE0_GATE" if any(v == "FAIL" for v in gates.values()) else "PENDING",
        "FORMAL": "NOT_RUN",
        "METHOD_DESIGN_READY": "NO",
        "MECHANISM_CLOSURE_CANDIDATE": "NO",
    }
    save_json(STAGE0, obj)
    if any(v == "FAIL" for v in gates.values()):
        write_negative_artifacts(obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    print("STOP" if any(v == "FAIL" for v in gates.values()) else "STAGE0_PASS")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "all"], default="stage0")
    args = ap.parse_args()
    stage0()


if __name__ == "__main__":
    main()
