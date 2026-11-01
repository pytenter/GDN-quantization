#!/usr/bin/env python3
"""Summarize state sensitivity and create an error accumulation figure."""
from pathlib import Path

from common import build_arg_parser, load_json, resolve_cfg, save_json, timestamp_record


def aggregate_by_token(records):
    buckets = {}
    for rec in records:
        token = rec.get("token_index")
        if token is None:
            continue
        buckets.setdefault(token, []).append(rec.get("relative_error", 0.0))
    return [{"token_index": k, "relative_error_mean": sum(v) / len(v), "count": len(v)} for k, v in sorted(buckets.items())]


def main():
    parser = build_arg_parser("Analyze state quantization error")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    root = Path(cfg["output_root"])
    data = load_json(root / "results" / "state_sensitivity.json")
    summary = {"run": timestamp_record(), "experiments": []}
    for exp in data.get("experiments", []):
        traj = aggregate_by_token(exp.get("records", []))
        summary["experiments"].append({
            "bits": exp.get("bits"),
            "matches_fp32_generation": exp.get("matches_fp32_generation"),
            "relative_error_mean": exp.get("relative_error_mean"),
            "relative_error_max": exp.get("relative_error_max"),
            "trajectory": traj,
        })
    save_json(root / "results" / "state_error_analysis.json", summary)
    try:
        import matplotlib.pyplot as plt
        for exp in summary["experiments"]:
            if exp["bits"] == "FP32":
                continue
            xs = [p["token_index"] for p in exp["trajectory"]]
            ys = [p["relative_error_mean"] for p in exp["trajectory"]]
            plt.plot(xs, ys, marker="o", label="INT%s" % exp["bits"])
        plt.xlabel("Decode step")
        plt.ylabel("Mean relative state error")
        plt.legend()
        plt.tight_layout()
        (root / "figures").mkdir(exist_ok=True)
        plt.savefig(root / "figures" / "state_error_accumulation.png", dpi=160)
    except Exception as exc:
        summary["plot_error"] = str(exc)
        save_json(root / "results" / "state_error_analysis.json", summary)
    print("Wrote results/state_error_analysis.json")


if __name__ == "__main__":
    main()
