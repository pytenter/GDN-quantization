#!/usr/bin/env python3
"""Read-only source audit and frozen anchor inventory; no outcome-driven selection."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
MODEL = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
AIME = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1/artifacts/aime26_v2/official_sampling_81920/qwen/formal/shards")
CAPTURE = ROOT / "experiments/QWEN_TEACHER_FORCED_REPLAY_MECHANISM_CAPTURE_V1"
M5 = ROOT / "experiments/QWEN_ROTATION_AWARE_M5_VALIDATION_V1"
LABELS = ROOT / "results/rotation/qwen_metric_alignment_audit_v1/canonical_trajectory_metrics/sample_metrics.jsonl"
H = (1, 4, 8, 16, 32, 64)
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
BAD_PAIR = ("aime26_28", 2)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entry(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    source_files = sorted(AIME.glob("worker*/fp_state.jsonl"))
    rows = []
    for path in source_files:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    rows.sort(key=lambda r: (int(r["canonical_index"]), int(r["seed"])))
    keys = [(r["problem_id"], int(r["seed"])) for r in rows]
    if len(rows) != 60 or len(set(keys)) != 60:
        raise RuntimeError("FP source trajectory count or unique ID audit failed")
    labels = {}
    with LABELS.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                labels.setdefault((item["problem_id"], int(item["seed"])), {})[item["condition"]] = bool(item["v4_correct"])
    if any(key not in labels for key in keys):
        raise RuntimeError("Frozen V4 outcome mapping lacks a source trajectory")
    groups = Counter()
    for key in keys:
        if key == BAD_PAIR:
            continue
        item = labels[key]
        identity, hadamard = item["int8_c128"], item["int8_c128_key_h"]
        group = "stable_correct" if identity and hadamard else "hadamard_rescue" if hadamard else "stable_failure" if not identity else "identity_only_correct"
        groups[group] += 1
    if sum(groups.values()) != 59:
        raise RuntimeError("clean59 label pairing failed")
    anchor_rows = []
    for r in rows:
        tokens = list(map(int, r["generated_token_ids"]))
        n = len(tokens)
        if n < 65:
            raise RuntimeError(f"trajectory too short for H=64: {r['problem_id']}/{r['seed']}")
        anchors = list(dict.fromkeys(math.floor(f * (n - 65)) for f in FRACTIONS))
        if len(anchors) != 4 or any(a + 64 >= n for a in anchors):
            raise RuntimeError(f"invalid frozen anchor inventory: {r['problem_id']}/{r['seed']}")
        token_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        anchor_rows.append({"problem_id": r["problem_id"], "seed": int(r["seed"]), "canonical_index": int(r["canonical_index"]), "generated_tokens": n, "token_ids_sha256": token_hash, "anchors": anchors, "clean59": (r["problem_id"], int(r["seed"])) != BAD_PAIR})
    capture_config = json.loads((CAPTURE / "capture_config.json").read_text())
    m5_manifest = json.loads((M5 / "artifact_manifest.json").read_text())
    if capture_config["execution"]["chunk_size"] != 64 or len(capture_config["gdn_layers"]) != 24 or not m5_manifest["token_pairing_exact"]:
        raise RuntimeError("Frozen capture semantics audit failed")
    checkpoints = sorted(MODEL.glob("model*.safetensors"))
    if not checkpoints:
        raise RuntimeError("model checkpoint shards missing")
    shard_entries = [entry(path) for path in checkpoints]
    combined = hashlib.sha256()
    for item in shard_entries:
        combined.update((Path(item["path"]).name + ":" + item["sha256"] + "\n").encode())
    manifest = {
        "ARTIFACT_AUDIT": "PASS", "model": "Qwen3.5-9B/GDN", "checkpoint_combined_sha256": combined.hexdigest(),
        "checkpoint_shards": shard_entries, "model_config": entry(MODEL / "config.json"),
        "checkpoint_index": entry(MODEL / "model.safetensors.index.json"),
        "fp_trajectory_files": [entry(path) for path in source_files], "fp_trajectory_count": len(rows),
        "frozen_outcomes": entry(LABELS), "frozen_outcome_group_counts_clean59": dict(groups),
        "capture_config": entry(CAPTURE / "capture_config.json"), "capture_script": entry(CAPTURE / "run_capture.py"),
        "prior_m5_manifest": entry(M5 / "artifact_manifest.json"), "prior_m5_token_pairing_exact": True,
        "preregistration": entry(OUT / "preregistration.json"),
        "amendments": [entry(OUT / f"execution_amendment_v{i}.json") for i in (2, 3)],
        "previous_order_amendment": entry(OUT / "preregistration_amendment_v1.json"),
        "replay_script": entry(OUT / "replay_persistent_functional_v3.py"),
        "batch_rows": capture_config["execution"]["batch_rows"], "chunk_size": 64,
        "rotation": "normalized symmetric key-side H128; Hadamard state remains in rotated runtime coordinates",
        "quantizer": "frozen symmetric C128 state QDQ, axis -2, integer range [-127,127]",
        "recurrent_state_dtype": "FP32 compute/QDQ; model parameters BF16",
        "gdn_layer_count": len(capture_config["gdn_layers"]), "gdn_layers": capture_config["gdn_layers"],
        "kv_cache": capture_config["execution"]["kv_cache"],
        "determinism": "Frozen FP token IDs; temperature 1 metric reduction; no sampling or gold-conditioned anchor selection.",
    }
    protocol = {
        "source": "Frozen preregistration plus execution_amendment_v2.json", "horizons": list(H),
        "fractions": list(FRACTIONS), "formula": "floor(f*(N-65)), deduplicate while preserving order",
        "minimum_generated_tokens": 65, "max_horizon": 64,
        "anchor_count_per_trajectory": 4, "valid_trajectory_count": len(anchor_rows),
        "sequence_end": "Feed token at anchor+h-1; compare logits predicting FP token at anchor+h.",
        "invalid_anchor_policy": "Exclude only incomplete/nonfinite/unreconstructable anchor with explicit reason; never replace using labels or metrics.",
        "trajectory_anchors": anchor_rows,
    }
    write_new(OUT / "artifact_manifest.json", manifest)
    write_new(OUT / "anchor_protocol.json", protocol)
    print("ARTIFACT_AUDIT = PASS")
    print("CHECKPOINT_COMBINED_SHA256 = " + manifest["checkpoint_combined_sha256"])
    print("FP_TRAJECTORIES = 60; CLEAN_PAIRS = 59; ANCHORS = 240")


if __name__ == "__main__":
    main()
