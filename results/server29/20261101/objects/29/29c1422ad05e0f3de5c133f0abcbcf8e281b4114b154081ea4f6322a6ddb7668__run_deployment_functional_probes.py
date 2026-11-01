#!/usr/bin/env python3
"""Run the existing fixed 32-token non-AIME probe for C5 and C6 sequentially."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/qwen_gdn/QWEN_C5_C6_DEPLOYMENT_CLOSURE_V1")
RUNTIME_PATH = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_GDN_48GB_FORMAL_TRAINING_READINESS_CLOSURE_V2/scripts/runtime.py")
PROBE_PATH = ROOT / "scripts/deployment_frozen_probe.py"
CANONICAL_SOURCE = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py")
ROTATION_SOURCE = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen/experiments/shared/rotation/cayley_rotation.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    runtime = load_module("runtime", RUNTIME_PATH)
    core = runtime.load_core()
    probe = load_module("deployment_frozen_probe", PROBE_PATH)
    source_hash = sha256(CANONICAL_SOURCE)
    rotation_source_hash = sha256(ROTATION_SOURCE)
    if source_hash != "48bd12d252c827ae57c48b2228d95e30de9f92baad47ebc2e8b64405b49e516e":
        raise RuntimeError("canonical recurrent source hash mismatch")
    if rotation_source_hash != "62e769dfad73170279daf0bdb56855b7d7e4aaf77478de897958fa87b2608067":
        raise RuntimeError("canonical rotation source hash mismatch")

    model, tokenizer, device = runtime.load_model_and_tokenizer(core)
    gates = {}
    for condition in ("C5", "C6"):
        path = ROOT / "deployment" / condition / "frozen_rotation_fp32.pt"
        manifest_path = ROOT / "deployment" / condition / "deployment_manifest.json"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload["condition"] != condition or manifest["condition"] != condition:
            raise RuntimeError(f"{condition} identity mismatch")
        if payload["canonical_source_sha256"] != source_hash or payload["rotation_source_sha256"] != rotation_source_hash:
            raise RuntimeError(f"{condition} source provenance mismatch")
        if sha256(path) != manifest["file_sha256"]:
            raise RuntimeError(f"{condition} file SHA mismatch")
        bank = runtime.FrozenMatrixRotationBank(payload["layer_ids"], payload["matrices"]).to(device)
        result = probe.run_fixed_probe(core, model, tokenizer, bank)
        result.update({
            "condition": condition,
            "rotation_file": str(path),
            "rotation_sha256": sha256(path),
            "deployment_manifest": str(manifest_path),
            "deployment_manifest_sha256": sha256(manifest_path),
            "canonical_source_sha256": source_hash,
            "rotation_source_sha256": rotation_source_hash,
            "model_loaded_once_for_both_conditions": True,
            "AIME_used": False,
        })
        gate = all([
            result["sample"] == "TRAIN:first-row",
            result["token_count"] == 32,
            result["sample_token_sha256"] == "1ab158a4061d9f8bf3b75dcede826b0f5451bdcf9b58394310c0cf119e68a606d",
            result["C128_QDQ_calls"] == 24 * 32,
            result["C128_QDQ_layers"] == list(core.GDN_LAYERS),
            result["frozen_rotation_layer_ids"] == list(core.GDN_LAYERS),
            len(result["writeback_exact_by_layer"]) == 24,
            all(result["writeback_exact_by_layer"].values()),
            result["direct_frozen_load"] is True,
            result["theta_reconstruction"] is False,
            result["hadamard_fallback"] is False,
            result["identity_fallback"] is False,
            result["no_grad"] is True,
            result["formal_inference"] is False,
        ])
        result["FUNCTIONAL_PROBE_GATE"] = "PASS" if gate else "FAIL"
        write_once(ROOT / "analysis" / f"{condition}_functional_probe.json", result)
        gates[f"{condition}_FUNCTIONAL_PROBE_GATE"] = result["FUNCTIONAL_PROBE_GATE"]
        if not gate:
            raise RuntimeError(f"{condition} fixed non-AIME probe failed")
        del bank
        torch.cuda.empty_cache()
    write_once(ROOT / "analysis/functional_probe_gate_summary.json", gates)
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
