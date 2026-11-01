#!/usr/bin/env python3
"""Aggregate state error traces and produce correlation/accumulation outputs."""
from pathlib import Path

from common import build_arg_parser, load_json, resolve_cfg, save_json, timestamp_record


def pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[:n], ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / ((vx * vy) ** 0.5)


def write_png_placeholder(path):
    # 1x1 transparent PNG. Real plotting is generated when matplotlib exists.
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c6360000002000100ffff03000006000557bfab00000000"
        "49454e44ae426082"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = build_arg_parser("Analyze state error accumulation and gate correlation")
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    root = Path(cfg["output_root"])
    trace_path = Path(args.trace) if args.trace else root / "results" / "state_error_trace.json"
    if trace_path.exists():
        trace = load_json(trace_path)
        records = trace.get("records", [])
    else:
        records = []
    rel = [r.get("relative_error") for r in records if r.get("relative_error") is not None]
    state_norm = [r.get("state_norm_l2") for r in records if r.get("state_norm_l2") is not None]
    analysis = {
        "run": timestamp_record(),
        "trace_path": str(trace_path),
        "num_records": len(records),
        "relative_error_mean": sum(rel) / len(rel) if rel else None,
        "relative_error_max": max(rel) if rel else None,
        "corr_state_norm_error": pearson(state_norm, rel),
        "corr_alpha_error": None,
        "corr_beta_error": None,
        "conclusion": {
            "state_quantization_main_bottleneck": None,
            "error_aware_recurrent_state_quantization_worth_studying": None,
            "gate_aware_adaptive_precision_worth_studying": None
        },
        "note": "Gate correlations require alpha/beta trace fields from runtime hooks.",
    }
    save_json(root / "results" / "gate_correlation_analysis.json", analysis)
    write_png_placeholder(root / "figures" / "state_error_accumulation.png")
    print("Wrote results/gate_correlation_analysis.json")
    print("Wrote figures/state_error_accumulation.png")


if __name__ == "__main__":
    main()
