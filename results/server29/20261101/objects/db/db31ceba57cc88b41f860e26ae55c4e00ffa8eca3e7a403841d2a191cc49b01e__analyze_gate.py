#!/usr/bin/env python3
"""Static and hook preparation for Qwen3.5 GDN gate analysis."""
from pathlib import Path

from common import build_arg_parser, resolve_cfg, save_json, timestamp_record


def main():
    parser = build_arg_parser("Prepare gate/state correlation analysis")
    args = parser.parse_args()
    cfg = resolve_cfg(args)
    result = {
        "run": timestamp_record(),
        "status": "prepared_not_executed",
        "note": "Gate tensors are implementation-specific. Source inspection shows Qwen3.5 GDN uses q/k/v, g, beta and gated delta rule internals. Runtime gate hooks are not executed in the first milestone.",
        "candidate_gate_names": ["g", "beta", "decay gate", "update gate"],
        "model_path": cfg["model_path"],
    }
    save_json(Path(cfg["output_root"]) / "results" / "gate_state_correlation.json", result)
    print("Wrote results/gate_state_correlation.json")


if __name__ == "__main__":
    main()
