#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


root = Path.cwd()
source_repo = Path("/data/zypan/GDN-quantization")
worktree = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
artifact_v2 = worktree / "artifacts/aime26_v2/official_sampling_81920"
artifact_v1 = worktree / "artifacts/aime26_v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def add_file(entries, original, relative, purpose, family, protocol, status,
             safe, reason, notes=""):
    path = Path(original)
    entries.append({
        "original_path": str(path),
        "relative_project_path": relative,
        "size": path.stat().st_size,
        "mtime": stamp(path),
        "sha256": sha256(path),
        "purpose": purpose,
        "model": "Qwen3.5-9B / GDN",
        "experiment_family": family,
        "protocol": protocol,
        "status": status,
        "safe_to_commit": safe,
        "reason": reason,
        "notes": notes,
    })


entries = []

# Files deliberately copied into the isolated consolidation clone.
for path in sorted((root / "experiments/qwen_gdn/aime26/81920").glob("*")):
    if path.is_file():
        original = worktree / "experiments/aime26" / path.name
        add_file(entries, original, str(path.relative_to(root)),
                 "Canonical 81920 Qwen formal runner/launcher/audit code",
                 "aime26", "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2",
                 "canonical", True, "Small executable provenance; no model or raw generations")

scorer = artifact_v2 / "long_generation_audit_static_v1/qwen/code/aime26_scorer_v4.py"
add_file(entries, scorer, "experiments/shared/scoring/aime26_scorer_v4.py",
         "Frozen V4 AIME answer extraction and scoring", "aime26_scoring",
         "AIME26_STRICT_V4_CANDIDATE", "canonical", True,
         "Canonical cross-server scorer; hash independently verified")

evidence_map = {
    "results/aime26/81920/qwen_gdn/evidence/launch/deterministic_replay.json": artifact_v2 / "qwen/launch/deterministic_replay.json",
    "results/aime26/81920/qwen_gdn/evidence/launch/preflight_report.json": artifact_v2 / "qwen/launch/preflight_report.json",
    "results/aime26/81920/qwen_gdn/evidence/integrity/qwen_protocol_manifest.json": artifact_v2 / "integrity/qwen_protocol_manifest.json",
    "results/aime26/81920/qwen_gdn/evidence/integrity/dataset_manifest.json": artifact_v1 / "formal/dataset/dataset_manifest.json",
    "results/aime26/81920/qwen_gdn/evidence/reference_closure/hadamard_math_tests.json": artifact_v1 / "reference_closure/hadamard_math_tests.json",
    "results/aime26/81920/qwen_gdn/evidence/reference_closure/qwen_operator_equivalence.json": artifact_v1 / "reference_closure/qwen_operator_equivalence.json",
    "results/aime26/81920/qwen_gdn/evidence/reference_closure/qwen_precision_ablation.json": artifact_v1 / "reference_closure/qwen_precision_ablation.json",
    "results/aime26/81920/qwen_gdn/evidence/dtype/dtype_audit_fp_state.json": artifact_v1 / "formal/qwen/dtype_audit_fp_state.json",
}
for relative, original in evidence_map.items():
    add_file(entries, original, relative, "Compact formal gate/equivalence/dtype evidence",
             "aime26_evidence", "81920 formal reference closure", "canonical", True,
             "Compact evidence with reproducibility value")

for name, purpose in [
    ("frozen_v4_per_sample.csv", "Frozen V4 per-sample decisions without response text"),
    ("frozen_v4_summary.json", "Frozen V4 aggregate and paired comparison"),
    ("PROVENANCE.json", "Immutable raw-shard hashes and paths"),
]:
    relative = f"results/aime26/81920/qwen_gdn/{name}"
    add_file(entries, root / relative, relative, purpose, "aime26_scoring",
             "81920 formal + Frozen V4", "canonical", True,
             "Derived by reading immutable formal shards; no fabricated samples",
             "Original raw shard identities are recorded in PROVENANCE.json")

mechanism_scripts = sorted((root / "experiments/qwen_gdn/mechanism").glob("*.py"))
for path in mechanism_scripts:
    original = source_repo / "experiments/propagation" / path.name
    add_file(entries, original, str(path.relative_to(root)), "Selected final mechanism experiment code",
             "gdn_mechanism", "mechanism closure", "useful", True,
             "Selected script corresponding to a compact final result/report")

for path in sorted((root / "results/mechanism/qwen_gdn").glob("*")):
    if "persistent_source" in path.name:
        original = source_repo / "runs/qwen_gdn_persistent_source_error_magnitude_equalization_transfer_v1/aggregate_summary.json"
    elif "mag_matched" in path.name:
        original = source_repo / "runs/qwen_gdn_mag_matched_residual_functional_geometry_closure_v1/aggregate_summary.json"
    else:
        original = source_repo / "results/propagation" / path.name
    add_file(entries, original, str(path.relative_to(root)), "Compact final mechanism classification",
             "gdn_mechanism", "mechanism closure", "useful", True,
             "Compact summary selected instead of raw event/layer traces")

for path in sorted((root / "reports/mechanism").glob("*.md")):
    original = source_repo / "reports/propagation" / path.name
    add_file(entries, original, str(path.relative_to(root)), "Final scientific mechanism report",
             "gdn_mechanism", "mechanism closure", "useful", True,
             "Human-readable final report")

for path in sorted((root / "experiments/qwen_gdn/rotation/prototypes").glob("*.py")):
    original = source_repo / "experiments/propagation" / path.name
    add_file(entries, original, str(path.relative_to(root)), "Historical structured-rotation prototype",
             "rotation_method_design", "prototype audit", "failed", True,
             "Retained as negative evidence; not canonical and not method-design-ready")

for path in sorted((root / "results/rotation/qwen_key_hadamard/prototypes").glob("*")):
    original = source_repo / "results/propagation" / path.name
    add_file(entries, original, str(path.relative_to(root)), "Prototype final stop-gate result",
             "rotation_method_design", "prototype audit", "failed", True,
             "FP equivalence gate stopped the route; no formal rotation gain claim")

for path in sorted((root / "reports/rotation").glob("*.md")):
    original = source_repo / "reports/propagation" / path.name
    add_file(entries, original, str(path.relative_to(root)), "Prototype failure report",
             "rotation_method_design", "prototype audit", "failed", True,
             "Negative evidence, explicitly separated from canonical Key-Hadamard")

# Formal raw generations are provenance-only and intentionally excluded.
raw_base = artifact_v2 / "qwen/formal/shards"
for path in sorted(raw_base.glob("worker*/*.jsonl")):
    add_file(entries, path, None, "Raw 81920 formal generation shard", "aime26",
             "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2",
             "canonical", False, "Large raw generation JSONL; provenance-only",
             "Contains response text and token IDs; compact scoring output is archived")

# Representative legacy scorer/smoke assets: retain in place, never mix with formal.
legacy_scorer = Path("/data/zypan/worktrees/aime26-scorer-diagnostic-v1/experiments/aime26/aime26_diagnostic_scorer.py")
add_file(entries, legacy_scorer, None, "Historical diagnostic scorer (pre-V4)",
         "aime26_scoring", "diagnostic pre-freeze", "legacy", False,
         "Not the canonical formal scorer", "V2/V3 standalone executable files were not found on qwen3090")
for path in sorted((artifact_v1 / "formal/qwen/smoke").glob("*.summary.json")):
    add_file(entries, path, None, "Historical 65536-era smoke summary", "aime26_smoke",
             "old 65536 smoke", "legacy", False,
             "Old protocol; deterministic/radix semantics differ and must not mix with 81920 formal")

# Model snapshot fingerprints, but never model content.
model_root = Path("/data/zypan/modelscope_models/Qwen3.5-9B")
for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
    path = model_root / name
    add_file(entries, path, None, "Local model snapshot fingerprint", "model_provenance",
             "local snapshot", "NEEDS_REVIEW", False,
             "Model/checkpoint material is excluded from Git",
             "Runner pins a local path, not an explicit upstream revision")

inventory = {
    "server_name": "qwen3090",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "branch": "codex/server-consolidation-2026-09-21-qwen3090",
    "entries": entries,
    "cross_server_keys": {
        "scorer_v4_sha256": "fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b",
        "common_utils_sha256": "34a320d20a48cda8d0a2a194914f3fa04f8c32bf8c6c533f5f9916fc5d6f9df7",
        "canonical_runner_and_embedded_quantizer_sha256": "9ff56d00b6b2dbaf9be3c3186bd30be8d3b9cc9d9b580a680316bcc8c01247dd",
        "aime26_dataset_manifest_sha256": "5cdda617c7bbf73a0277de75a0f25ed52976e385b01e1264b806085d33649b0f",
        "qwen_protocol_manifest_sha256": "4edf137ab275005683063a873a13f20552caa4bec0d8fa93bbf31fabcfff770e",
        "fp_state_dtype_audit_sha256": "1d3ab8e82978359707a988a1cd3bbc2c8b4babd537e5e58adb8b5f05e5ee0cd5",
    },
}

out = root / "docs/server_inventory"
out.mkdir(parents=True, exist_ok=True)
(out / "qwen3090.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")

raw_bytes = sum(e["size"] for e in entries if e["purpose"] == "Raw 81920 formal generation shard")
md = f"""# qwen3090 server inventory

Generated by a read-only audit of source artifacts; copied files live only in the isolated consolidation clone.

## Canonical AIME26 result

- Protocol: `GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2`
- FP_STATE: 52 correct, 8 wrong, 3 abstain, 86.67% (wrong includes abstain).
- INT8_C128: 18 correct, 42 wrong, 39 abstain, 30.00%.
- INT8_C128 + Key-Hadamard: 41 correct, 19 wrong, 6 abstain, 68.33%.
- Paired Key-Hadamard vs INT8_C128: 23 rescued, 0 lost, net +23.
- Frozen V4 SHA256: `fa5dd904c7d1887df4b8c23613f37b8d438ea494cb72607b89be8e07d55a343b` (cross-server match PASS).
- Twelve immutable raw shards ({raw_bytes} bytes) are hashed in the JSON inventory and `PROVENANCE.json`, but are not copied into Git.

## Formal protocol audit

- Manual Hugging Face autoregressive runtime, not SGLang. Therefore `enable_deterministic_inference`, `disable_radix_cache`, and `uses_mamba_radix_cache` are **N/A as runtime flags**. Deterministic seeded replay independently passed with exact token-ID agreement; no radix/Mamba cache exists in this path.
- `max_new_tokens=81920`, seeds 1/2, sampling enabled, temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 1.5 on generated output tokens only, repetition penalty 1.0, thinking enabled.
- Chat template uses `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` followed by tokenization with `add_special_tokens=False`.
- Stop IDs are tokenizer EOS plus valid `<|im_end|>` and `<|endoftext|>` IDs.
- The runner uses a local Qwen3.5-9B snapshot in BF16; it does not pin an upstream model revision. Config/index/tokenizer hashes are inventory-only. FP_STATE recurrent storage and kernel state are actual FP32, while q/k/v/beta and core output are BF16 and decay `g` is FP32.
- Historical `aime26_v1/formal/qwen/smoke` artifacts are classified legacy and excluded. They are not mixed with the 81920 formal results.

## Quantizer and Key-Hadamard semantics

- State shape is `[B, 32, 128, 128]`. C128 computes FP32 source and scale `[B,H,1,V]` with `amax(abs(state), dim=K) / 127`, epsilon floor `1e-12`, `torch.round` (ties-to-even), clamp `[-127,127]`, dequantize in FP32, then copy to FP32 state storage.
- Prefill is not quantized. QDQ is applied to all 24 GDN recurrent states after each decode kernel return, before the next decode consumes the cache.
- Key-Hadamard is Qwen/GDN-only: normalized H128 is applied on the key dimension to both q and k immediately before both chunk-prefill and recurrent-decode GDN kernels. The kernel therefore creates/updates recurrent state in the rotated key basis. There is no separate inverse/recovery on recurrent state; the paired q transform makes the GDN core/readout equivalent. Value coordinates are unchanged.
- Formal audit rows record calls, all GDN layers, prefill/decode phase counts, scale shapes and observed qrange. Reference-closure Hadamard math, operator equivalence and precision-ablation evidence is archived.

## Mechanism and method-design classification

- Selected compact mechanism chain: orientation/low-rank stability, RMS/value geometry, out-proj anisotropy, real R/C recurrent mediation, residual-geometry principle extraction, persistent-source/magnitude equalization transfer, and magnitude-matched functional-geometry closure.
- Large event JSONL, per-token/layer/head traces, CSV panels and full run directories remain provenance-only.
- `gdn_rotation_headroom_v1`: **failed/negative prototype**, stopped by full-layer FP equivalence because native q/k depthwise convolution breaks the basis reparameterization.
- `gdn_postconv_structured_rotation_v1`: **failed/negative prototype**, prefill/decode FP equivalence failed; route closed.
- No canonical learned rotation, Butterfly, HARP, Givens, Householder, Cayley, dense-orthogonal optimizer, or trainable rotation implementation was found. Mentions in candidate/protocol files are exclusions or unexecuted design options, not methods.

## Scorer versions and conflicts

- V4 is the only canonical executable used for the reproduced formal score.
- A historical diagnostic scorer (`2618f92c...`) is classified legacy and not copied.
- Standalone V2/V3 scorer executables were not found on qwen3090; their historical derived scores are not promoted. This remains `NEEDS_REVIEW` against the third server or an eventual merged ling4090 inventory.
- `aime26_common.py` and `run_qwen_aime26_formal.py` appear in multiple worktrees. The canonical 81920 hashes differ from the older diagnostic worktree and are recorded in `qwen3090.json`.

## Safety

- No running zypan GPU/tmux/screen experiment was modified; all four RTX 3090s were idle at audit start.
- No checkpoints, caches, datasets, raw generation JSONL, raw token/layer/head traces, credentials, or host-specific secrets are selected for staging.
- No commit, push, PR, merge, reset, deletion of source data, or expensive experiment rerun was performed.
"""
(out / "qwen3090.md").write_text(md)
print(json.dumps({"entries": len(entries), "raw_bytes": raw_bytes, "json": str(out / 'qwen3090.json'), "md": str(out / 'qwen3090.md')}, indent=2))
