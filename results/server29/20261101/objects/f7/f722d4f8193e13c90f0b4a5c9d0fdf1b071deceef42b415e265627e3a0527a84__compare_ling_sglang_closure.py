#!/usr/bin/env python3
"""Score Ling SGLang closure gates B-D and the FP+H diagnostic."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


EPS = 1e-30


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def summarize(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "min": None}
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "n": len(values),
        "mean": float(x.mean()),
        "median": float(x.quantile(0.5)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
        "min": float(x.min()),
    }


def tensor_metrics(actual, reference):
    a, b = actual.float().reshape(-1), reference.float().reshape(-1)
    return {
        "relative_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(EPS)),
        "cosine": float(F.cosine_similarity(a[None], b[None]).item()),
        "max_abs": float((a - b).abs().max().item()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def logit_metrics(actual, reference):
    a, b = actual.float(), reference.float()
    if a.shape[-1] != b.shape[-1]:
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :n], b[..., :n]
    row = tensor_metrics(a, b)
    loga, logb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    row["kl_reference_actual"] = float((logb.exp() * (logb - loga)).sum(-1).mean().item())
    row["top1_match"] = bool(a.argmax(-1).item() == b.argmax(-1).item())
    ka, kb = set(a.topk(20, -1).indices.reshape(-1).tolist()), set(b.topk(20, -1).indices.reshape(-1).tolist())
    row["top20_overlap"] = len(ka & kb) / 20.0
    return row


def aggregate(rows, key):
    return summarize([row[key] for row in rows])


def hadamard(device="cpu"):
    value = torch.ones((1, 1), dtype=torch.float32, device=device)
    while value.shape[0] < 128:
        value = torch.cat((torch.cat((value, value), 1), torch.cat((value, -value), 1)), 0)
    return value / math.sqrt(128)


def load_trace_pair(reference_dir, actual_dir, transform_state=None):
    logit_rows, state_rows, missing = [], [], []
    for prompt_path in sorted(reference_dir.glob("prompt_*/prompt.json")):
        p = int(json.loads(prompt_path.read_text())["prompt_index"])
        ref_steps = sorted((reference_dir / f"prompt_{p:02d}").glob("manual_step*.pt"))
        for ref_path in ref_steps:
            step = int(ref_path.stem.split("step")[-1])
            actual_logits = actual_dir / f"prompt_{p:02d}" / f"logits_step{step:04d}.pt"
            if not actual_logits.exists():
                missing.append(str(actual_logits))
                continue
            ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            act_logits = torch.load(actual_logits, map_location="cpu", weights_only=True)
            lm = logit_metrics(act_logits, ref["logits"])
            lm.update({"prompt_index": p, "step": step, "phase": "prefill" if step == 0 else "decode"})
            logit_rows.append(lm)
            for layer, reference_state in ref["states"].items():
                state_path = actual_dir / f"prompt_{p:02d}" / f"state_step{step:04d}_layer{int(layer):02d}_{'prefill' if step == 0 else 'decode'}.pt"
                if not state_path.exists():
                    missing.append(str(state_path))
                    continue
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                if transform_state is not None:
                    state = transform_state(state)
                sm = tensor_metrics(state, reference_state)
                sm.update({"prompt_index": p, "step": step, "layer": int(layer)})
                state_rows.append(sm)
    return logit_rows, state_rows, missing


def load_sglang_pair(prompt_source, reference_dir, actual_dir, transform_state=None):
    logit_rows, state_rows, missing = [], [], []
    for prompt_path in sorted(prompt_source.glob("prompt_*/prompt.json")):
        p = int(json.loads(prompt_path.read_text())["prompt_index"])
        ref_prompt = reference_dir / f"prompt_{p:02d}"
        for ref_logits_path in sorted(ref_prompt.glob("logits_step*.pt")):
            step = int(ref_logits_path.stem.split("step")[-1])
            actual_logits_path = actual_dir / f"prompt_{p:02d}" / f"logits_step{step:04d}.pt"
            if not actual_logits_path.exists():
                missing.append(str(actual_logits_path))
                continue
            ref_logits = torch.load(ref_logits_path, map_location="cpu", weights_only=True)
            act_logits = torch.load(actual_logits_path, map_location="cpu", weights_only=True)
            lm = logit_metrics(act_logits, ref_logits)
            lm.update({"prompt_index": p, "step": step, "phase": "prefill" if step == 0 else "decode"})
            logit_rows.append(lm)
            phase = "prefill" if step == 0 else "decode"
            for ref_state_path in sorted(ref_prompt.glob(f"state_step{step:04d}_layer*_{phase}.pt")):
                layer = int(ref_state_path.name.split("_layer")[1].split("_")[0])
                actual_state_path = actual_dir / f"prompt_{p:02d}" / ref_state_path.name
                if not actual_state_path.exists():
                    missing.append(str(actual_state_path))
                    continue
                ref_state = torch.load(ref_state_path, map_location="cpu", weights_only=True)
                act_state = torch.load(actual_state_path, map_location="cpu", weights_only=True)
                if transform_state is not None:
                    act_state = transform_state(act_state)
                sm = tensor_metrics(act_state, ref_state)
                sm.update({"prompt_index": p, "step": step, "layer": layer})
                state_rows.append(sm)
    return logit_rows, state_rows, missing


def score_runtime_gate(logits, states, missing, label, policy):
    per_layer = {}
    for layer in sorted({row["layer"] for row in states}):
        layer_rows = [row for row in states if row["layer"] == layer]
        per_layer[str(layer)] = {
            "relative_l2": aggregate(layer_rows, "relative_l2"),
            "cosine": aggregate(layer_rows, "cosine"),
        }
    summary = {
        "logit_relative_l2": aggregate(logits, "relative_l2"),
        "logit_cosine": aggregate(logits, "cosine"),
        "kl": aggregate(logits, "kl_reference_actual"),
        "top1_agreement": sum(x["top1_match"] for x in logits) / max(1, len(logits)),
        "top20_overlap": aggregate(logits, "top20_overlap"),
        "state_relative_l2": aggregate(states, "relative_l2"),
        "state_cosine": aggregate(states, "cosine"),
        "state_by_layer": per_layer,
        "missing_files": missing,
    }
    finite = all(row["finite"] for row in logits + states)
    checks = {
        "complete_3x129_logits": len(logits) == 3 * 129,
        "complete_3x129x3_states": len(states) == 3 * 129 * 3,
        "no_missing_files": not missing,
        "all_finite": finite,
    }
    for name, (value, relation, threshold) in policy["checks"].items():
        checks[name] = value(summary)
        if relation == "max":
            checks[name] = checks[name] <= threshold
        else:
            checks[name] = checks[name] >= threshold
    passed = bool(logits) and bool(states) and all(checks.values())
    return {
        "gate": "PASS" if passed else "FAIL",
        "comparison": label,
        "policy_name": policy["name"],
        "policy_rationale": policy["rationale"],
        "tolerances": {k: {"relation": v[1], "threshold": v[2]} for k, v in policy["checks"].items()},
        "checks": checks,
        "summary": summary,
        "per_step_logits": logits,
        "per_step_states": states,
    }


def load_audit(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", required=True)
    parser.add_argument("--fp-dir", required=True)
    parser.add_argument("--r128-dir", required=True)
    parser.add_argument("--h-dir", required=True)
    parser.add_argument("--r128-h-dir", required=True)
    parser.add_argument("--r128-dump-dir", required=True)
    parser.add_argument("--r128-audit", required=True)
    parser.add_argument("--h-audit", required=True)
    parser.add_argument("--r128-h-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manual, fp, r128, hdir, r128_hdir, out = map(
        Path, (args.manual_dir, args.fp_dir, args.r128_dir, args.h_dir, args.r128_h_dir, args.output_dir)
    )

    gate_b_policy = {
        "name": "native-runtime-semantic-parity-v1",
        "rationale": (
            "This gate checks native SGLang cache/token semantics rather than bitwise arithmetic parity. "
            "The immutable reference closure already established a BF16/optimized-KDA numerical floor. "
            "Layer 0 recurrent state is the direct cache/position sentinel; deeper-layer drift is reported "
            "per layer and constrained through output-distribution agreement."
        ),
        "checks": {
            "logit_relative_l2_median_max": (lambda s: s["logit_relative_l2"]["median"], "max", 0.10),
            "logit_cosine_median_min": (lambda s: s["logit_cosine"]["median"], "min", 0.995),
            "kl_p95_max": (lambda s: s["kl"]["p95"], "max", 0.15),
            "top1_agreement_min": (lambda s: s["top1_agreement"], "min", 0.95),
            "top20_overlap_mean_min": (lambda s: s["top20_overlap"]["mean"], "min", 0.90),
            "layer0_state_relative_l2_p95_max": (lambda s: s["state_by_layer"]["0"]["relative_l2"]["p95"], "max", 0.01),
            "layer0_state_cosine_min_min": (lambda s: s["state_by_layer"]["0"]["cosine"]["min"], "min", 0.999),
        },
    }
    # HF exposes the canonical state as [H,K,V], whereas SGLang's KDA cache is
    # physically [H,V,K].  Normalize SGLang evidence to canonical [H,K,V].
    gate_b = score_runtime_gate(
        *load_trace_pair(manual, fp, transform_state=lambda state: state.transpose(-1, -2)),
        "canonical manual BF16 vs SGLang FP",
        gate_b_policy,
    )
    save_json(out / "gate_b_fp_runtime.json", gate_b)

    quant_rows = []
    for path in sorted(Path(args.r128_dump_dir).glob("runtime_*_layer*_head00.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        source = record["prequant_state"].float()
        scale = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        qcode = torch.round(source / scale).clamp(-127, 127).to(torch.int8)
        # CPU and GPU reductions can differ in the last FP32 bit.  Rounding is
        # checked exactly through qcode; dequant is reconstructed exactly from
        # the runtime-recorded scale.
        dequant = record["qcode"].float() * record["scale"]
        scale_error = float((scale - record["scale"]).abs().max().item())
        quant_rows.append({
            "file": str(path),
            "phase": record["phase"],
            "layer": int(record["layer"]),
            "head": int(record["head"]),
            "scale_shape": list(record["scale"].shape),
            "scale_exact": torch.equal(scale, record["scale"]),
            "scale_max_abs_error": scale_error,
            "scale_close_rtol_1e-6_atol_1e-12": torch.allclose(scale, record["scale"], rtol=1e-6, atol=1e-12),
            "qcode_exact": torch.equal(qcode, record["qcode"]),
            "dequant_exact": torch.equal(dequant, record["dequant_state"]),
            "qrange": [int(record["qcode"].min()), int(record["qcode"].max())],
        })
    r128_audit = load_audit(Path(args.r128_audit))
    quant_events = [x for x in r128_audit if x.get("event") == "kda_state_qdq"]
    phases = {x.get("phase") for x in quant_events}
    gate_c_pass = (
        bool(quant_rows) and all(
            x["scale_shape"] == [128, 1]
            and x["scale_close_rtol_1e-6_atol_1e-12"]
            and x["qcode_exact"]
            and x["dequant_exact"]
            and x["qrange"][0] >= -127 and x["qrange"][1] <= 127
            for x in quant_rows
        )
        and phases == {"prefill", "decode"}
        and all(
            x.get("grouping") == "canonical R128 scale=[B,H,K,1]; SGLang physical scale=[B,H,1,K]"
            for x in quant_events
        )
    )
    gate_c = {"gate": "PASS" if gate_c_pass else "FAIL", "rows": quant_rows, "audit_events": quant_events, "verified": ["symmetric zero-point 0", "round", "[-127,127]", "R128 grouping", "prefill QDQ", "decode requant"]}
    save_json(out / "gate_c_r128.json", gate_c)

    hmat = hadamard()
    # SGLang physically stores KDA state as [H,V,K].  Logical S -> S H is
    # therefore H^T S_storage; recover with the left inverse before comparison.
    h_logits, h_states, h_missing = load_sglang_pair(
        manual, fp, hdir, transform_state=lambda state: hmat.T @ state.float()
    )
    h_policy = {
        "name": "optimized-kda-hadamard-drift-v1",
        "rationale": (
            "The diagnostic rejects large KL, meaningful top-1/top-20 trajectory drift, or cache-basis "
            "failure while allowing the BF16 dense-H cast-back floor documented by reference closure."
        ),
        "checks": {
            "logit_relative_l2_p95_max": (lambda s: s["logit_relative_l2"]["p95"], "max", 0.075),
            "logit_cosine_median_min": (lambda s: s["logit_cosine"]["median"], "min", 0.999),
            "kl_p95_max": (lambda s: s["kl"]["p95"], "max", 0.02),
            "top1_agreement_min": (lambda s: s["top1_agreement"], "min", 0.98),
            "top20_overlap_mean_min": (lambda s: s["top20_overlap"]["mean"], "min", 0.95),
            "state_relative_l2_p95_max": (lambda s: s["state_relative_l2"]["p95"], "max", 0.05),
            "state_cosine_min_min": (lambda s: s["state_cosine"]["min"], "min", 0.98),
        },
    }
    fp_h = score_runtime_gate(h_logits, h_states, h_missing, "SGLang FP vs SGLang FP+Value-H", h_policy)
    save_json(out / "fp_hadamard_diagnostic.json", fp_h)

    r128_h_logits, r128_h_states, r128_h_missing = load_sglang_pair(
        manual, r128, r128_hdir, transform_state=lambda state: hmat.T @ state.float()
    )
    r128_h_policy = {
        "name": "quantized-kda-hadamard-trajectory-v1",
        "rationale": (
            "R128 is deliberately basis-dependent, so R128 and R128+Value-H recurrent states are not "
            "expected to be FP-equivalent after inverse rotation. This diagnostic rejects nonfinite or "
            "catastrophic state/trajectory drift while requiring strong logit-distribution, top-1 and "
            "top-20 agreement under identical teacher forcing."
        ),
        "checks": {
            "logit_relative_l2_p95_max": (lambda s: s["logit_relative_l2"]["p95"], "max", 0.10),
            "logit_cosine_median_min": (lambda s: s["logit_cosine"]["median"], "min", 0.999),
            "kl_p95_max": (lambda s: s["kl"]["p95"], "max", 0.02),
            "top1_agreement_min": (lambda s: s["top1_agreement"], "min", 0.98),
            "top20_overlap_mean_min": (lambda s: s["top20_overlap"]["mean"], "min", 0.95),
            "state_relative_l2_p95_max": (lambda s: s["state_relative_l2"]["p95"], "max", 0.15),
            "state_cosine_min_min": (lambda s: s["state_cosine"]["min"], "min", 0.98),
        },
    }
    r128_h = score_runtime_gate(
        r128_h_logits,
        r128_h_states,
        r128_h_missing,
        "SGLang INT8-R128 vs SGLang INT8-R128+Value-H",
        r128_h_policy,
    )
    save_json(out / "r128_hadamard_diagnostic.json", r128_h)

    basis_rows, basis_missing = [], []
    for prompt_path in sorted(manual.glob("prompt_*/prompt.json")):
        prompt_index = int(json.loads(prompt_path.read_text())["prompt_index"])
        prompt_dir = hdir / f"prompt_{prompt_index:02d}"
        for layer in (0, 12, 22):
            paths = {
                kind: prompt_dir / f"basis_{kind}_layer{layer:02d}.pt"
                for kind in ("kernel_return", "prefill_end_cache", "first_decode_input")
            }
            absent = [str(path) for path in paths.values() if not path.exists()]
            if absent:
                basis_missing.extend(absent)
                continue
            kernel = torch.load(paths["kernel_return"], map_location="cpu", weights_only=True).float()
            cache = torch.load(paths["prefill_end_cache"], map_location="cpu", weights_only=True).float()
            first_decode = torch.load(paths["first_decode_input"], map_location="cpu", weights_only=True).float()
            kernel_cache = tensor_metrics(cache, kernel)
            cache_decode = tensor_metrics(first_decode, cache)
            redundant_h = tensor_metrics(hmat @ kernel, cache)
            basis_rows.append(
                {
                    "prompt_index": prompt_index,
                    "layer": layer,
                    "kernel_return_to_cache": kernel_cache,
                    "cache_to_first_decode_input": cache_decode,
                    "redundant_hypothetical_H_kernel_to_cache": redundant_h,
                    "kernel_cache_exact": torch.equal(cache, kernel),
                    "cache_first_decode_exact": torch.equal(first_decode, cache),
                }
            )
    basis_pass = (
        len(basis_rows) == 3 * 3
        and not basis_missing
        and all(row["kernel_cache_exact"] and row["cache_first_decode_exact"] for row in basis_rows)
        and all(row["redundant_hypothetical_H_kernel_to_cache"]["relative_l2"] > 0.5 for row in basis_rows)
    )
    basis_gate = {
        "gate": "PASS" if basis_pass else "FAIL",
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO" if basis_pass else "UNRESOLVED",
        "comparison": "kernel-return -> prefill-end cache -> first-decode input",
        "missing_files": basis_missing,
        "rows": basis_rows,
    }
    save_json(out / "gate_e_prefill_endpoint_basis.json", basis_gate)
    h_audit = load_audit(Path(args.h_audit))
    r128_h_audit = load_audit(Path(args.r128_h_audit))
    rotation_events = [x for x in h_audit + r128_h_audit if x.get("event") == "kda_value_hadamard"]
    phases = {x.get("phase") for x in rotation_events}
    placement_ok = bool(rotation_events) and phases == {"prefill", "decode"} and all(
        "ShortConv+SiLU" in x.get("forward_placement", "") and "before RMSNorm" in x.get("inverse_placement", "")
        and x.get("cache_basis") == "S H retained across prefill and decode" for x in rotation_events
    )
    gate_d = {
        "gate": "PASS" if placement_ok and fp_h["gate"] == "PASS" and r128_h["gate"] == "PASS" and basis_pass else "FAIL",
        "placement_ok": placement_ok,
        "rotation_events": rotation_events,
        "fp_h_diagnostic_gate": fp_h["gate"],
        "r128_h_diagnostic_gate": r128_h["gate"],
        "prefill_endpoint_basis_gate": basis_gate["gate"],
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO" if basis_pass else "UNRESOLVED",
    }
    save_json(out / "gate_d_hadamard.json", gate_d)
    overall = {
        "KDA_ROTATION_SEMANTICS_VERSION": "CORRECTED_PREFILL_ENDPOINT_V2",
        "LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE": basis_gate["gate"],
        "REDUNDANT_PREFILL_ENDPOINT_ROTATION": "NO" if basis_pass else "UNRESOLVED",
        "gate_a": json.loads((fp / "gate_a_prompt.json").read_text())["gate"],
        "gate_b": gate_b["gate"],
        "gate_c": gate_c["gate"],
        "gate_d": gate_d["gate"],
        "fp_h_diagnostic": fp_h["gate"],
        "r128_h_diagnostic": r128_h["gate"],
    }
    pass_fields = ("LING_KDA_PREFILL_ENDPOINT_STATE_BASIS_GATE", "gate_a", "gate_b", "gate_c", "gate_d", "fp_h_diagnostic", "r128_h_diagnostic")
    overall["LING_SGLANG_RUNTIME_CLOSURE"] = "PASS" if all(overall[key] == "PASS" for key in pass_fields) and overall["REDUNDANT_PREFILL_ENDPOINT_ROTATION"] == "NO" else "FAIL"
    save_json(out / "closure_summary.json", overall)
    print(json.dumps(overall, indent=2))
    raise SystemExit(0 if overall["LING_SGLANG_RUNTIME_CLOSURE"] == "PASS" else 2)


if __name__ == "__main__":
    main()
