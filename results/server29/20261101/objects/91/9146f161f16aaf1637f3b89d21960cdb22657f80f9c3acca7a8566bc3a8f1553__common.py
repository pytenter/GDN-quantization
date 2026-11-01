#!/usr/bin/env python3
"""Shared utilities for Qwen3.5 GDN/KDA sensitivity experiments."""
import argparse
import importlib
import json
import os
import time
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("run_config.json")


def require(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency '%s'. Run in an environment with the required ML stack. "
            "Original error: %s" % (module_name, exc)
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def load_run_config(path=None):
    cfg = load_json(path or DEFAULT_CONFIG)
    cfg["model_path"] = os.path.expanduser(cfg["model_path"])
    cfg["output_root"] = os.path.expanduser(cfg.get("output_root", "."))
    return cfg


def load_model_config(model_path):
    cfg = load_json(Path(model_path) / "config.json")
    return cfg, cfg.get("text_config", cfg)


def layer_summary(text_cfg):
    types = list(text_cfg.get("layer_types") or [])
    if not types:
        types = ["unknown"] * int(text_cfg.get("num_hidden_layers", 0))
    return {
        "total_layers": len(types),
        "attention_layers": sum(1 for t in types if t == "full_attention"),
        "gdn_layers": sum(1 for t in types if t == "linear_attention"),
        "layer_types": types,
    }


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", default=None)
    return parser


def resolve_cfg(args):
    cfg = load_run_config(args.config)
    if args.model:
        cfg["model_path"] = args.model
    if args.output_root:
        cfg["output_root"] = args.output_root
    return cfg


def timestamp_record():
    return {"unix_time": time.time(), "local_time": time.strftime("%Y-%m-%d %H:%M:%S %z")}
