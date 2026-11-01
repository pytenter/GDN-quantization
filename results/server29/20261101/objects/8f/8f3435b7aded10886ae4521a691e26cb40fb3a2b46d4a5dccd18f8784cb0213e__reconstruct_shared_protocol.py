#!/usr/bin/env python3
"""CPU-only reconstruction and identity audit of the formal H64 validation path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path("experiments/QWEN_GDN_SHARED_OBJECTIVE_PARETO_AND_DEV_SCREEN_V1")
PROV = Path("experiments/QWEN_GDN_C5_C6_EVIDENCE_GUIDED_DIAGNOSIS_V1/training_provenance")
SOURCE_ROOT = Path("/data/zypan")
MODEL = SOURCE_ROOT / "modelscope_models/Qwen3.5-9B"
CANONICAL_REPO = SOURCE_ROOT / "worktrees/hadamard-init-dense-oracle-v1-qwen"
SOURCE = CANONICAL_REPO / "experiments/QWEN_RECURRENT_DENSE_C5_C6_V1/run_recurrent_dense.py"
ROTATION = CANONICAL_REPO / "experiments/shared/rotation/cayley_rotation.py"
CORPUS = CANONICAL_REPO / "CALIBRATION_RAW_TEXTS.jsonl"
TRANSFORMERS_ROOT = SOURCE_ROOT / "transformers-qwen35/src"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.environ.setdefault("QWEN_TRANSFORMERS_SRC", str(TRANSFORMERS_ROOT))
    os.environ.setdefault("QWEN_MODEL_PATH", str(MODEL))
    os.environ.setdefault("HF_HOME", "/data/zypan/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(TRANSFORMERS_ROOT))

    model_prov = json.loads((PROV / "configs/model_provenance.json").read_text(encoding="utf-8"))
    source_prov = json.loads((PROV / "configs/source_provenance.json").read_text(encoding="utf-8"))
    validation_panel = json.loads((PROV / "configs/validation_panel.json").read_text(encoding="utf-8"))
    protocol = json.loads((PROV / "configs/formal_training_protocol.json").read_text(encoding="utf-8"))

    source_checks = {
        "canonical_c5_c6": {"path": str(SOURCE), "expected": source_prov["canonical_c5_c6"]["sha256"], "actual": sha256(SOURCE)},
        "canonical_rotation": {"path": str(ROTATION), "expected": source_prov["canonical_rotation"]["sha256"], "actual": sha256(ROTATION)},
        "canonical_modeling": {
            "path": str(TRANSFORMERS_ROOT / "transformers/models/qwen3_5/modeling_qwen3_5.py"),
            "expected": source_prov["canonical_modeling"]["sha256"],
            "actual": sha256(TRANSFORMERS_ROOT / "transformers/models/qwen3_5/modeling_qwen3_5.py"),
        },
        "corpus": {"path": str(CORPUS), "expected": protocol["training"]["common"]["corpus_sha256"], "actual": sha256(CORPUS)},
    }
    for item in source_checks.values():
        item["pass"] = item["expected"] == item["actual"]

    model_checks = []
    for original in model_prov["inventory"]:
        local = MODEL / Path(original["path"]).name
        actual_size = local.stat().st_size
        actual_hash = sha256(local)
        model_checks.append({
            "path": str(local), "expected_size_bytes": original["size_bytes"], "actual_size_bytes": actual_size,
            "expected_sha256": original["sha256"], "actual_sha256": actual_hash,
            "pass": actual_size == original["size_bytes"] and actual_hash == original["sha256"],
        })

    core = import_file(SOURCE, "qwen_shared_objective_protocol_core")
    if Path(core.CORPUS).resolve() != CORPUS.resolve():
        raise RuntimeError(f"core corpus path drift: {core.CORPUS}")
    if tuple(core.GDN_LAYERS) != tuple(model_prov["gdn_layer_ids"]):
        raise RuntimeError("GDN layer mapping drift")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    rows = core.corpus_rows("VALIDATION")
    frozen_by_id = {x["document_id"]: x for x in validation_panel["panel"]}
    document_checks = []
    for row in rows:
        frozen = frozen_by_id[row["document_id"]]
        ids, positions = core.tokenize(tokenizer, row)
        digest = hashlib.sha256(b"".join(int(v).to_bytes(8, "little", signed=True) for v in ids)).hexdigest()
        check = {
            "document_id": row["document_id"], "raw_text_sha256_expected": frozen["raw_text_sha256"],
            "raw_text_sha256_actual": row["raw_text_sha256"], "token_count_expected": frozen["token_count"],
            "token_count_actual": len(ids), "token_ids_sha256_expected": frozen["token_ids_le_int64_sha256"],
            "token_ids_sha256_actual": digest, "capture_positions_expected": frozen["capture_positions_zero_based"],
            "capture_positions_actual": positions,
        }
        check["pass"] = (
            check["raw_text_sha256_expected"] == check["raw_text_sha256_actual"]
            and check["token_count_expected"] == check["token_count_actual"]
            and check["token_ids_sha256_expected"] == check["token_ids_sha256_actual"]
            and check["capture_positions_expected"] == check["capture_positions_actual"]
        )
        document_checks.append(check)

    # CPU identity check for the exact theta=0 Cayley initialization.
    import torch
    bank = core.CAYLEY.PerLayerCayleyRotations(core.GDN_LAYERS)
    zero_parameter_count = sum(int(torch.count_nonzero(p)) for p in bank.parameters())
    identity = torch.eye(128, dtype=torch.float32)
    per_layer_identity_max_abs = {str(layer): float((bank.layer(layer).matrix().detach() - identity).abs().max()) for layer in core.GDN_LAYERS}
    hadamard_cpu = core.hadamard(torch.device("cpu"))
    final_rotation_identity_max_abs = {
        str(layer): float((hadamard_cpu @ bank.layer(layer).matrix().detach() - hadamard_cpu).abs().max())
        for layer in core.GDN_LAYERS
    }
    hadamard_identity = {
        "label": "CANONICAL_HADAMARD_REEVALUATED_BASELINE",
        "theta_zero_nonzero_parameter_count": zero_parameter_count,
        "delta_identity_max_abs_by_layer": per_layer_identity_max_abs,
        "final_rotation_equals_h128_max_abs_by_layer": final_rotation_identity_max_abs,
        "pass": zero_parameter_count == 0 and max(per_layer_identity_max_abs.values()) == 0.0 and max(final_rotation_identity_max_abs.values()) == 0.0,
        "interpretation": "This verifies the reevaluated canonical H128 initializer; it is not a recovered historical update-0 row.",
    }

    all_pass = (
        all(x["pass"] for x in source_checks.values())
        and all(x["pass"] for x in model_checks)
        and [r["document_id"] for r in rows] == validation_panel["document_ids"]
        and len(document_checks) == 16
        and all(x["pass"] for x in document_checks)
        and hadamard_identity["pass"]
    )
    result = {
        "status": "PASS" if all_pass else "FAIL",
        "ORIGINAL_VALIDATION_PROTOCOL_MATCH": "YES" if all_pass else "NO",
        "baseline_label": "CANONICAL_HADAMARD_REEVALUATED_BASELINE",
        "source_checks": source_checks,
        "model_inventory_checks": model_checks,
        "validation_documents": 16,
        "captures_per_document": 8,
        "total_captures": 128,
        "sequence_length": 1024,
        "document_checks": document_checks,
        "hadamard_identity": hadamard_identity,
        "semantics": {
            "teacher_reference": "FP teacher trajectory collected with collect_teacher_targets on identical tokenized documents and frozen capture positions",
            "student_state": "independent persistent recurrent cache initialized None per document; every token consumes prior student state",
            "rotation": "q/k.float() @ H128 @ Delta; theta=0 implies Delta=I and final rotation H128",
            "recovery": "Delta then H128 on key axis; at theta=0 this is H128",
            "quantization": protocol["quantization"],
            "state_objective": "per layer: sum((recovered_state-FP_teacher_prev)^2 over K,V) / clamp_min(sum(FP_teacher_prev^2 over K,V),1e-12), then mean across B,H; mean across 24 GDN layers; mean across 128 captures",
            "functional_objective": "relative_mse of each full linear_attn.out_proj output versus FP teacher output; mean across 24 GDN layers; mean across 128 captures",
            "prefill_decode_boundary": "FP teacher advances in chunks between captures and one captured token; student replays every token one-by-one with persistent QDQ/writeback",
            "horizon": "H64 only detaches cache tensors every 64 student tokens; no numerical reset; no effect on no-grad validation values",
            "aggregation": "unweighted arithmetic mean over 128 capture losses (16 documents x 8 captures)",
        },
    }
    atomic_json(ROOT / "analysis/shared_validation_protocol.json", result)
    print(json.dumps({"status": result["status"], "protocol_match": result["ORIGINAL_VALIDATION_PROTOCOL_MATCH"], "model_files": len(model_checks)}, sort_keys=True))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
