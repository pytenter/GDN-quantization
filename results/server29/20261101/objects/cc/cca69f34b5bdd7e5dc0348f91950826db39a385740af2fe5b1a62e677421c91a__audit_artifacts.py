#!/usr/bin/env python3
"""Read-only source/provenance audit before the formal stability replay."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1"
PRIOR = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
M5 = ROOT / "experiments/QWEN_ROTATION_AWARE_M5_VALIDATION_V1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    target = OUT / "artifact_manifest.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite artifact manifest")
    prior = read(PRIOR / "artifact_manifest.json")
    parity = read(OUT / "replay_parity/parity_results.json")
    previous = read(PRIOR / "results.json")
    m5 = read(M5 / "results.json")
    if prior["ARTIFACT_AUDIT"] != "PASS" or parity["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("source gates not passed")
    if previous["PERSISTENT_FUNCTIONAL_SURROGATE_GATE"] != "FAIL" or m5["ROTATION_AWARE_M5_GATE"] != "FAIL":
        raise RuntimeError("expected prior result provenance mismatch")
    files = [prior["checkpoint_index"], prior["model_config"], prior["capture_script"], prior["frozen_outcomes"]]
    files += prior["checkpoint_shards"] + prior["fp_trajectory_files"]
    checked = []
    for item in files:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"missing/size mismatch: {path}")
        actual = sha(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"sha mismatch: {path}")
        checked.append({"path": str(path), "bytes": item["bytes"], "sha256": actual})
    protocol = read(PRIOR / "anchor_protocol.json")
    inventory = {(x["problem_id"], int(x["seed"])): x["token_ids_sha256"] for x in protocol["trajectory_anchors"]}
    if len(inventory) != 60:
        raise RuntimeError("expected 60 distinct source trajectory IDs")
    seen = set()
    for source in prior["fp_trajectory_files"]:
        with Path(source["path"]).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["problem_id"], int(row["seed"]))
                tokens = list(map(int, row["generated_token_ids"]))
                actual = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
                if key in seen or actual != inventory.get(key):
                    raise RuntimeError(f"source token hash/ID mismatch: {key}")
                seen.add(key)
    if len(seen) != 60:
        raise RuntimeError("source trajectory count mismatch")
    if parity["provenance"]["checkpoint_combined_sha256"] != prior["checkpoint_combined_sha256"]:
        raise RuntimeError("checkpoint combined hash mismatch")
    if parity["provenance"]["quantizer_source_sha256"] != sha(Path(prior["capture_script"]["path"])):
        raise RuntimeError("quantizer source hash mismatch")
    output = {
        "task": "QWEN_HADAMARD_TRAJECTORY_STABILITY_ANALYSIS_V1",
        "ARTIFACT_AUDIT": "PASS",
        "REPLAY_PARITY_GATE": "PASS",
        "source_trajectories": 60,
        "source_token_hashes_verified": 60,
        "checkpoint_combined_sha256": prior["checkpoint_combined_sha256"],
        "quantizer_function_sha256": parity["provenance"]["quantizer_function_sha256"],
        "quantizer_source_sha256": parity["provenance"]["quantizer_source_sha256"],
        "rotation_h128_sha256": parity["provenance"]["rotation_h128_sha256"],
        "rotation_orthogonality_max_abs": parity["provenance"]["rotation_orthogonality_max_abs"],
        "source_files": checked,
        "preregistration_sha256": sha(OUT / "preregistration.json"),
        "parity_amendment_sha256": sha(OUT / "parity_amendment_v1.json"),
        "replay_script_sha256": sha(OUT / "replay_stability_v2.py"),
        "prior_results_sha256": sha(PRIOR / "results.json"),
        "prior_m5_results_sha256": sha(M5 / "results.json"),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "new_generation": False
    }
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ARTIFACT_AUDIT": "PASS", "source_trajectories": 60, "checkpoint_combined_sha256": output["checkpoint_combined_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
