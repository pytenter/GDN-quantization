import json, math, sys
from pathlib import Path

EXP = Path("/data/zypan/experiments/qwen35_gdn_quant")
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_int8_orientation_state_change_mechanism as p1
import run_int8_r128_c128_natural_residual_norm_swap_causal as ns
import run_int8_r128_c128_frozen_observability_path_decomposition as frozen

GDN_LAYERS = ns.GDN_LAYERS

def norm(torch, x):
    return float(torch.linalg.vector_norm(x.detach().float()).item())

def clone_past(past):
    import copy
    return copy.deepcopy(past)

def main():
    torch, model, tokenizer, _cfg, e2e = p1.setup_model()
    pm = ns.selected_prompts(1)[0]
    t0 = 64
    fp_past, ids, cont, collector = frozen.run_fp_to_t0(torch, model, tokenizer, e2e, pm, t0)
    try:
        inj, meta = ns.build_injections(torch, fp_past)
        pert = clone_past(fp_past)
        ns.apply_injection(torch, pert, inj, "R_STRUCT_C_NORM")
        nxt = torch.tensor([[cont[t0]]], dtype=ids.dtype, device=ids.device)
        with torch.inference_mode():
            _fp_out, fp_next, fp_records = frozen.driver_step(torch, model, nxt, None, fp_past, collector)
            _pe_out, pe_next, pe_records = frozen.driver_step(torch, model, nxt, None, pert, collector)
        max_rel = {k: 0.0 for k in ["query", "key", "value", "g", "beta"]}
        worst = []
        for layer in GDN_LAYERS:
            fr, pr = fp_records[layer], pe_records[layer]
            for k in max_rel:
                rel = norm(torch, fr[k].float() - pr[k].float()) / (norm(torch, fr[k]) + 1e-12)
                if rel > max_rel[k]:
                    max_rel[k] = rel
                    worst.append((k, layer, rel))
        print(json.dumps({"max_operator_input_relerr": max_rel, "worst": worst[-10:], "norm_meta": {k: meta[k] for k in meta if k != "per_layer_head_norm_summary"}}, indent=2))
    finally:
        collector["close"]()

if __name__ == "__main__":
    main()
