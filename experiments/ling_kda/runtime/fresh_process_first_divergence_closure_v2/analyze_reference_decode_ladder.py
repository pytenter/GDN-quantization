#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_variant(values):
    reference = values[0]
    return next((index for index, value in enumerate(values[1:], 1) if value != reference), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("run_*.json"))]
    if len(rows) < 5:
        raise RuntimeError(f"expected at least 5 runs, got {len(rows)}")

    events = []
    for layer in rows[0]["layers"]:
        key = str(layer)
        values = [row["prefill_cache_layer_sha256"][key] for row in rows]
        events.append({"location": "prefill_cache", "step": 0, "layer": layer, "field": "cache", "variant_run": first_variant(values)})
    for step_index, first_step in enumerate(rows[0]["steps"]):
        step = first_step["step"]
        common_hooks = set.intersection(*(set(row["steps"][step_index].get("hook_tensor_sha256", {})) for row in rows))
        for field in ("layer8_output_hidden", "layer9_input_hidden", "layer9_pre_norm_output", "layer9_q_projection_raw", "layer9_q_short_conv_output"):
            if field in common_hooks:
                values = [row["steps"][step_index]["hook_tensor_sha256"][field] for row in rows]
                events.append({"location": "decode_hook", "step": step, "layer": 9, "field": field, "variant_run": first_variant(values)})
        values = [row["steps"][step_index]["logits_sha256"] for row in rows]
        events.append({"location": "decode", "step": step, "layer": None, "field": "logits", "variant_run": first_variant(values)})
        for layer in rows[0]["layers"]:
            key = str(layer)
            values = [row["steps"][step_index]["cache_layer_sha256"][key] for row in rows]
            events.append({"location": "decode", "step": step, "layer": layer, "field": "cache", "variant_run": first_variant(values)})
            common = set.intersection(*(
                set(row["steps"][step_index]["record_tensor_sha256"].get(key, {}))
                for row in rows
            ))
            for field in sorted(common):
                values = [row["steps"][step_index]["record_tensor_sha256"][key][field] for row in rows]
                events.append({"location": "decode_record", "step": step, "layer": layer, "field": field, "variant_run": first_variant(values)})
    divergent = [event for event in events if event["variant_run"] is not None]
    result = {
        "task": "LING_FRESH_PROCESS_FIRST_DIVERGENCE_CLOSURE_V2",
        "runs": len(rows),
        "steps": len(rows[0]["steps"]),
        "FIRST_DIVERGENT_REFERENCE_DECODE_EVENT": divergent[0] if divergent else None,
        "divergent_events": divergent,
        "REFERENCE_DECODE_REPEATABLE": not divergent,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"REFERENCE_DECODE_REPEATABLE": result["REFERENCE_DECODE_REPEATABLE"], "FIRST_DIVERGENT_REFERENCE_DECODE_EVENT": result["FIRST_DIVERGENT_REFERENCE_DECODE_EVENT"]}, indent=2))


if __name__ == "__main__":
    main()
