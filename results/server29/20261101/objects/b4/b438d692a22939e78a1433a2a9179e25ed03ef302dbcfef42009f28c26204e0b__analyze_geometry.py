#!/usr/bin/env python3
"""Frozen phase-1 state-error geometry analysis; no model or logits loaded."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
OUT = ROOT / "experiments/QWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1"
PRIOR = ROOT / "experiments/QWEN_PERSISTENT_FUNCTIONAL_SURROGATE_V1"
BOOT = 5000
PERM = 10000
SEED = 20260925
LAYERS = 24
SCALARS = (
    "error_norm_identity", "error_norm_hadamard", "fp_state_norm",
    "relative_error_identity", "relative_error_hadamard",
    "cos_error_fp_identity", "cos_error_fp_hadamard",
    "cos_error_identity_hadamard", "readout_error_norm_identity",
    "readout_error_norm_hadamard", "cos_readout_error_fp_identity",
    "cos_readout_error_fp_hadamard",
)
PRIMARY = (
    ("delta_error_norm", "error_norm_hadamard", "error_norm_identity"),
    ("delta_relative_error", "relative_error_hadamard", "relative_error_identity"),
    ("delta_cos_error_fp", "cos_error_fp_hadamard", "cos_error_fp_identity"),
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def boot_ci(x, seed):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = np.median(x[rng.integers(0, len(x), size=(BOOT, len(x)))], axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def sign_p(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    n = len(x)
    if n == 0:
        return None
    positives = int((x > 0).sum())
    tail = min(positives, n - positives)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n))


def summary(x, seed):
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    return {
        "n": int(len(valid)), "missing_or_undefined": int(len(x) - len(valid)),
        "median": finite_median(valid), "bootstrap_95ci": boot_ci(valid, seed),
        "p_two_sided_sign_vs_zero": sign_p(valid),
        "positive_fraction": float((valid > 0).mean()) if len(valid) else None,
    }


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def group_contrast(a, b, seed):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {"n_rescue": int(len(a)), "n_stable_failure": int(len(b)), "test": "UNDEFINED"}
    observed = float(np.median(a) - np.median(b))
    rng = np.random.default_rng(seed)
    pool = np.concatenate((a, b))
    extreme = 0
    for _ in range(PERM):
        perm = rng.permutation(pool)
        delta = float(np.median(perm[:len(a)]) - np.median(perm[len(a):]))
        extreme += abs(delta) >= abs(observed) - 1e-15
    return {
        "n_rescue": int(len(a)), "n_stable_failure": int(len(b)),
        "rescue_median_minus_failure_median": observed,
        "p_two_sided_permutation": float((extreme + 1) / (PERM + 1)),
        "permutations": PERM,
        "cliffs_delta": cliffs_delta(a, b),
    }


def group_for(labels):
    i, h = labels["int8_c128"], labels["int8_c128_key_h"]
    if i and h:
        return "stable_correct"
    if not i and h:
        return "hadamard_rescue"
    if not i and not h:
        return "stable_failure"
    return "identity_only_correct"


def frozen_outcomes():
    source = Path(load(PRIOR / "artifact_manifest.json")["frozen_outcomes"]["path"])
    labels = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            labels.setdefault((item["problem_id"], int(item["seed"])), {})[item["condition"]] = bool(item["v4_correct"])
    exclude = load(PRIOR / "preregistration.json")["primary_aime_panel"]["excluded_pair"]
    return labels, (exclude["problem_id"], int(exclude["seed"])), source


def validate_frame(frame, item):
    n = int(item["generated_tokens"]) - 1
    if len(frame) != n * LAYERS or int(item["rows"]) != len(frame):
        raise RuntimeError(f"row count mismatch {item['problem_id']} seed{item['seed']}")
    frame.sort_values(["token_position", "layer"], inplace=True)
    if not np.array_equal(frame.token_position.to_numpy(), np.repeat(np.arange(n), LAYERS)):
        raise RuntimeError("token grid mismatch")
    unique_layers = np.sort(frame.layer.unique())
    if len(unique_layers) != LAYERS or not np.array_equal(frame.layer.to_numpy(), np.tile(unique_layers, n)):
        raise RuntimeError("layer grid mismatch")
    if frame[["problem_id", "seed"]].drop_duplicates().shape[0] != 1:
        raise RuntimeError("mixed trajectory IDs")
    if frame.problem_id.iloc[0] != item["problem_id"] or int(frame.seed.iloc[0]) != int(item["seed"]):
        raise RuntimeError("metadata trajectory ID mismatch")
    tokens = frame.drop_duplicates("token_position").sort_values("token_position")
    if len(tokens) != n or not np.array_equal(tokens.target_token_id.to_numpy()[:-1], tokens.input_token_id.to_numpy()[1:]):
        raise RuntimeError("token trajectory alignment mismatch")
    token_ids = tokens.input_token_id.astype(int).tolist() + [int(tokens.target_token_id.iloc[-1])]
    source_hash = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
    if source_hash != item["token_ids_sha256"]:
        raise RuntimeError("source token hash mismatch")
    if frame[["error_norm_identity", "error_norm_hadamard", "fp_state_norm",
              "relative_error_identity", "relative_error_hadamard"]].isna().to_numpy().any():
        raise RuntimeError("nonfinite core geometry")
    for name in SCALARS:
        values = frame[name].to_numpy(dtype=float)
        if name.startswith("cos_"):
            if np.isinf(values).any() or (np.abs(values[np.isfinite(values)]) > 1.000001).any():
                raise RuntimeError(f"invalid cosine {name}")
        elif not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite scalar {name}")
    return n, unique_layers


def plot_figures(traj, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title in ((axes[0], "error_norm", "State error L2"),
                              (axes[1], "relative_error", "Relative state error")):
        x = traj[f"{metric}_identity"].to_numpy(float)
        y = traj[f"{metric}_hadamard"].to_numpy(float)
        mask = traj.clean59.to_numpy(bool)
        ax.scatter(x[mask], y[mask], s=16, alpha=0.65)
        lo, hi = float(min(x[mask].min(), y[mask].min())), float(max(x[mask].max(), y[mask].max()))
        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1)
        ax.set(xlabel="Identity trajectory median", ylabel="Hadamard trajectory median", title=title)
    fig.savefig(directory / "state_error_distribution.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"stable_correct": "#4C78A8", "hadamard_rescue": "#F58518", "stable_failure": "#54A24B"}
    for ax, metric, title in ((axes[0], "delta_cos_error_fp", "H-I error / FP-state cosine"),
                              (axes[1], "cos_error_identity_hadamard", "I-H error cosine")):
        groups = ("stable_correct", "hadamard_rescue", "stable_failure")
        for ix, group in enumerate(groups):
            values = paired.loc[paired.group == group, metric].dropna().to_numpy(float)
            ax.scatter(np.full(len(values), ix), values, s=18, alpha=0.7, color=colors[group])
            if len(values):
                ax.plot([ix - 0.2, ix + 0.2], [np.median(values)] * 2, color="black", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(xticks=range(3), xticklabels=["stable\ncorrect", "Hadamard\nrescue", "stable\nfailure"], ylabel="Trajectory median", title=title)
    fig.savefig(directory / "geometry_by_outcome_group.png", dpi=160)
    plt.close(fig)


def main():
    prereg = load(OUT / "preregistration.json")
    audit = load(OUT / "analysis/replay_audit.json")
    parity = load(OUT / "analysis/replay_parity.json")
    if prereg["status"] != "FROZEN_BEFORE_FORMAL_COMPARISON" or audit["hash_verification"] != "PASS" or parity["REPLAY_PARITY_GATE"] != "PASS":
        raise RuntimeError("preregistration/audit/parity gate missing")
    for index in range(4):
        status = load(OUT / "analysis" / f"shard_{index:02d}" / "status.json")
        if status["status"] != "COMPLETE" or status["completed"] != status["assigned"] or status["assigned"] != 15:
            raise RuntimeError(f"shard {index} incomplete")
    labels, excluded, outcome_source = frozen_outcomes()
    records = []
    invalid_cos = {name: 0 for name in SCALARS if name.startswith("cos_")}
    all_layers = None
    total_rows = total_positions = 0
    for index in range(60):
        directory = OUT / "analysis" / f"shard_{index % 4:02d}"
        found = list(directory.glob(f"{index:03d}_*_meta.json"))
        if len(found) != 1:
            raise RuntimeError(f"missing/ambiguous unit metadata {index}")
        meta = load(found[0])
        path = Path(str(found[0])[:-10] + ".parquet")
        if digest(path) != meta["parquet_sha256"] or int(meta["global_index"]) != index:
            raise RuntimeError(f"unit data/hash mismatch {index}")
        frame = pq.read_table(path).to_pandas()
        n, layers = validate_frame(frame, meta)
        if all_layers is None:
            all_layers = layers
        elif not np.array_equal(layers, all_layers):
            raise RuntimeError("cross-unit GDN layer set mismatch")
        total_rows += len(frame)
        total_positions += n
        for name in invalid_cos:
            invalid_cos[name] += int((~np.isfinite(frame[name].to_numpy(float))).sum())
        key = (meta["problem_id"], int(meta["seed"]))
        if key not in labels:
            raise RuntimeError(f"outcome label missing {key}")
        group = group_for(labels[key])
        record = {"problem_id": key[0], "seed": key[1], "canonical_index": index,
                  "token_count": n, "clean59": key != excluded, "group": group}
        for name in SCALARS:
            record[name] = finite_median(frame[name].to_numpy(float))
        records.append(record)
        if (index + 1) % 5 == 0:
            print(json.dumps({"event": "geometry_analysis_progress", "completed": index + 1, "total": 60}), flush=True)
    if total_rows != audit["expected_paired_layer_rows"] or total_positions != audit["token_positions_per_condition"]:
        raise RuntimeError("total row/token counts differ from audit")
    traj = pd.DataFrame(records).sort_values(["canonical_index"])
    clean = traj[traj.clean59].copy()
    if len(clean) != 59 or clean.duplicated(["problem_id", "seed"]).any() or (traj[~traj.clean59][["problem_id", "seed"]].iloc[0].tolist() != list(excluded)):
        raise RuntimeError("clean59 real-ID pairing invalid")
    groups = clean.group.value_counts().to_dict()
    if groups != {"stable_correct": 18, "hadamard_rescue": 23, "stable_failure": 18}:
        raise RuntimeError(f"frozen group counts changed: {groups}")
    paired = clean[["problem_id", "seed", "group", "token_count", "cos_error_identity_hadamard"]].copy()
    for name, h, i in PRIMARY:
        paired[name] = clean[h] - clean[i]
    paired["delta_cos_readout_error_fp"] = clean.cos_readout_error_fp_hadamard - clean.cos_readout_error_fp_identity
    paired["delta_readout_error_norm"] = clean.readout_error_norm_hadamard - clean.readout_error_norm_identity
    analysis = {"task": prereg["task"], "phase": prereg["phase"], "clean59_n": 59,
                "token_positions_per_condition": total_positions, "paired_token_layer_rows": total_rows,
                "gdn_layers": list(map(int, all_layers)), "group_counts": groups,
                "undefined_cosine_rows": invalid_cos,
                "paired_clean59": {}, "cross_error_cosine": {}, "secondary_readout_alignment": {}}
    for j, (name, _, _) in enumerate(PRIMARY):
        analysis["paired_clean59"][name] = summary(paired[name], SEED + j)
    analysis["cross_error_cosine"] = summary(paired.cos_error_identity_hadamard, SEED + 11)
    analysis["secondary_readout_alignment"]["delta_cos_readout_error_fp"] = summary(paired.delta_cos_readout_error_fp, SEED + 12)
    analysis["secondary_readout_alignment"]["delta_readout_error_norm"] = summary(paired.delta_readout_error_norm, SEED + 13)
    rescue = {"task": prereg["task"], "groups": groups, "primary_family": {}, "secondary_readout_alignment": {},
              "test": "two-sided 10000-label-permutation on trajectory medians; Holm across primary family"}
    family = list(prereg["aggregation_and_tests"]["rescue_primary_family"])
    for j, metric in enumerate(family + ["delta_cos_readout_error_fp", "delta_readout_error_norm"]):
        block = {"groups": {group: summary(paired.loc[paired.group == group, metric], SEED + 100 + 10*j + k)
                             for k, group in enumerate(("stable_correct", "hadamard_rescue", "stable_failure"))},
                 "rescue_vs_stable_failure": group_contrast(
                     paired.loc[paired.group == "hadamard_rescue", metric],
                     paired.loc[paired.group == "stable_failure", metric], SEED + 200 + j)}
        target = "primary_family" if metric in family else "secondary_readout_alignment"
        rescue[target][metric] = block
    pvals = {name: rescue["primary_family"][name]["rescue_vs_stable_failure"].get("p_two_sided_permutation") for name in family}
    sortable = sorted(((p, name) for name, p in pvals.items() if p is not None))
    running = 0.0
    for rank, (p, name) in enumerate(sortable):
        running = max(running, min(1.0, p * (len(sortable) - rank)))
        rescue["primary_family"][name]["rescue_vs_stable_failure"]["p_holm"] = float(running)
    traj_path = OUT / "analysis/trajectory_summary.csv"
    pair_path = OUT / "analysis/clean59_paired.csv"
    traj.to_csv(traj_path, index=False)
    paired.to_csv(pair_path, index=False)
    save_json(OUT / "state_error_geometry.json", analysis)
    save_json(OUT / "rescue_statistics.json", rescue)
    save_json(OUT / "functional_error_statistics.json", {"status": "DEFERRED_BY_USER", "reason": "Phase 1 only; no full-logits JVP or finite difference was run"})
    save_json(OUT / "nonlinear_filtering.json", {"status": "DEFERRED_BY_USER", "reason": "No RMSNorm/gate/out-projection functional intervention in Phase 1"})
    plot_figures(traj, paired)
    primary_text = []
    for name in family:
        item = rescue["primary_family"][name]["rescue_vs_stable_failure"]
        primary_text.append(f"| {name} | {item['rescue_median_minus_failure_median']:.6g} | {item['cliffs_delta']:.3g} | {item['p_two_sided_permutation']:.4g} | {item['p_holm']:.4g} |")
    report = [
        "# Qwen Hadamard Functional Error Geometry — Phase 1",
        "", "This is an observational deterministic teacher-forced error-geometry audit, not a functional-sensitivity or causality test.", "",
        f"Replay parity: PASS. Source trajectories: 60; clean59: 59; GDN layers: 24; token positions per condition: {total_positions}; paired token-layer rows: {total_rows}.",
        f"Clean59 real-ID groups: {groups}. No new generation, training, model/quantizer/scorer change, or stored full state/logit tensors.", "",
        "## Paired clean59 geometry", "",
        "Values below use equal-weight trajectory medians over all tokens and all 24 layers; bootstrap resamples trajectories.", "",
        "| Metric (Hadamard minus Identity) | Median | Bootstrap 95% CI | Sign-test p | n |", "| --- | ---: | --- | ---: | ---: |",
    ]
    for name in ("delta_error_norm", "delta_relative_error", "delta_cos_error_fp"):
        item = analysis["paired_clean59"][name]
        report.append(f"| {name} | {item['median']:.6g} | {item['bootstrap_95ci']} | {item['p_two_sided_sign_vs_zero']:.4g} | {item['n']} |")
    cc = analysis["cross_error_cosine"]
    report += ["", f"Identity-Hadamard error-direction cosine: trajectory median {cc['median']:.6g}, bootstrap 95% CI {cc['bootstrap_95ci']}, n={cc['n']}. This measures directional difference, not functional benefit.",
               "", "## Rescue versus stable failure", "",
               "Two-sided permutation tests compare trajectory-level geometry; stable-correct is retained as a control. Holm correction covers the four preregistered primary comparisons.", "",
               "| Geometry feature | Rescue minus stable-failure median | Cliff's delta | Permutation p | Holm p |", "| --- | ---: | ---: | ---: | ---: |", *primary_text,
               "", "## Scope and interpretation", "",
               "The readout-space cosine is a separate secondary diagnostic; a state error vector and a readout vector are not in the same coordinate space, so no undefined state/query cosine is fabricated.",
               "Full-logits JVP, full-vocabulary finite differences, nonlinear filtering and any functional-amplification conclusion remain deferred. A geometry difference alone cannot explain accuracy or justify a learnable-rotation objective.",
               "", f"Undefined cosine row counts: {invalid_cos}. All mandatory norms and relative errors were finite.",
               "", "Figures: `figures/state_error_distribution.png`, `figures/geometry_by_outcome_group.png`.", ""]
    (OUT / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {"task": prereg["task"], "phase": "PHASE_1_COMPLETE", "REPLAY_PARITY_GATE": "PASS",
              "source_trajectories": 60, "clean59": 59, "groups": groups,
              "state_error_geometry": "MEASURED", "functional_sensitivity": "DEFERRED_BY_USER",
              "new_generation": False, "new_rotation_training": False,
              "model_modified": False, "quantizer_modified": False, "scorer_modified": False,
              "no_causal_claim": True}
    save_json(OUT / "results.json", result)
    outputs = [OUT / x for x in (
        "preregistration.json", "analysis/replay_audit.json", "analysis/replay_parity.json",
        "analysis/trajectory_summary.csv", "analysis/clean59_paired.csv",
        "state_error_geometry.json", "rescue_statistics.json",
        "functional_error_statistics.json", "nonlinear_filtering.json",
        "figures/state_error_distribution.png", "figures/geometry_by_outcome_group.png",
        "final_report.md", "results.json")]
    manifest = {"task": prereg["task"], "status": "COMPLETE", "phase": "PHASE_1",
                "capture_script_sha256": digest(OUT / "capture_geometry.py"),
                "analysis_script_sha256": digest(Path(__file__)),
                "frozen_outcomes_sha256": digest(outcome_source),
                "outputs": {str(path.relative_to(OUT)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in outputs}}
    save_json(OUT / "run_manifest.json", manifest)
    print("\nQWEN_HADAMARD_FUNCTIONAL_ERROR_GEOMETRY_ANALYSIS_V1")
    print("PHASE_1_GEOMETRY = COMPLETE")
    print("REPLAY_PARITY_GATE = PASS")
    print(f"SOURCE_TRAJECTORIES = 60; CLEAN59 = 59; PAIRED_TOKEN_LAYER_ROWS = {total_rows}")
    print("FUNCTIONAL_SENSITIVITY = DEFERRED_BY_USER")
    print("NEW_FREE_GENERATION = NO; ROTATION_TRAINING = NO")


if __name__ == "__main__":
    main()
