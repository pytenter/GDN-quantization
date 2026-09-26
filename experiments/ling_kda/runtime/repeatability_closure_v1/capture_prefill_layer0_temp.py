#!/usr/bin/env python3
"""Temporarily capture one pre-quant layer-0 recurrent state for an exact diff."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MAIN = import_file(HERE / "run_ling_repeatability.py", "ling_repeatability_temp_capture_import")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-repo", default="/data01/user2/repos/GDN-quantization")
    parser.add_argument("--max-memory-gib", type=int, default=22)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--state-checkpoint", required=True)
    parser.add_argument("--functional-checkpoint", required=True)
    parser.add_argument("--condition", choices=("Hadamard", "Dense_State"), default="Dense_State")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    F, model, tokenizer, layers, device, h, delta_mats, runtime_mats, _observed = MAIN.load_context(args)
    documents = sorted(MAIN.R.load_corpus(Path(args.corpus), "HELDOUT"), key=lambda row: int(row["panel_index"]))
    ids = tokenizer(documents[0]["raw_text"], add_special_tokens=False).input_ids[:1024]
    method = args.condition
    probe = MAIN.E.PerLayerHistoryProbe.build(F, runtime_mats[method])
    probe.install(model)
    try:
        basis = "rotated" if MAIN.E.rotated(method) else "native"
        item = MAIN.E.prefill(F, model, ids, probe, basis, layers, device, method)
        stack = F.BASE.cache_stack(item["past"], layers)
        layer = min(stack)
        state = stack[layer].detach().float().cpu().clone()
    finally:
        probe.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"document_id": documents[0]["document_id"], "condition": method, "layer": int(layer), "state": state}, args.output)
    print(f"temporary_capture={args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
