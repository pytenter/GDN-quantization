"""CPU-only postmeasurement analysis; never launches or changes a measurement.

Requires BOTH complete, SHA-paired formal rosters. All route decisions use only
matched short scores. Historical pulse labels are opened afterwards and are
DISCOVERY_ONLY, never new persistent outcomes or independent anchor samples.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
METRICS = ["P1", "P2_W128", "P3_H128", "B0", "L7_FUNCTIONAL_ORIGINAL",
           "L7_STATE_ORIGINAL", "L7_TOTAL_ORIGINAL"]
COMPARISONS = list(itertools.combinations(METRICS, 2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    # Same identity serialization as the inherited scorer, not file SHA.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def restored_fp_identity(identity):
    """Restore the frozen runner's integer token/layer keys after JSON reading.

    JSON stores object keys as strings. The original identity digest was computed
    before serialization, with numeric sort order. Do not substitute a lexical
    string-key rehash, especially for token1024 or layer10 and higher.
    """
    return dict(drivers=identity["drivers"],
        checkpoint_states={int(k): v for k, v in identity["checkpoint_states"].items()},
        full_logits={int(k): v for k, v in identity["full_logits"].items()},
        post_o_proj={int(token): {int(layer): value for layer, value in layers.items()}
                     for token, layers in identity["post_o_proj"].items()})


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(x) for x in value)
    return True


def midranks(values):
    assert all(math.isfinite(x) for x in values), "NONFINITE_RANK_INPUT"
    ranks = [0.0] * len(values)
    indices = sorted(range(len(values)), key=lambda i: values[i])
    begin = 0
    while begin < len(indices):
        end = begin + 1
        while end < len(indices) and values[indices[begin]] == values[indices[end]]:
            end += 1
        for index in indices[begin:end]:
            ranks[index] = (begin + end + 1) / 2
        begin = end
    return ranks


def pearson(a, b):
    assert len(a) == len(b)
    if len(a) < 2:
        return None
    ac = [x - statistics.mean(a) for x in a]
    bc = [x - statistics.mean(b) for x in b]
    aa, bb = math.fsum(x*x for x in ac), math.fsum(x*x for x in bc)
    if not aa or not bb:
        return None
    return max(-1.0, min(1.0, math.fsum(x*y for x, y in zip(ac, bc)) / math.sqrt(aa*bb)))


def spearman(a, b):
    return pearson(midranks(a), midranks(b))


def sign(value):
    return (value > 0) - (value < 0)


def pairwise_concordance(a, b):
    counts = dict(concordant=0, discordant=0, tied_a_only=0, tied_b_only=0, tied_both=0)
    for i, j in itertools.combinations(range(len(a)), 2):
        sa, sb = sign(a[i]-a[j]), sign(b[i]-b[j])
        key = ("tied_both" if not sa and not sb else "tied_a_only" if not sa else
               "tied_b_only" if not sb else "concordant" if sa == sb else "discordant")
        counts[key] += 1
    n = counts["concordant"] + counts["discordant"]
    return dict(counts, eligible_untied_pairs=n,
                concordance=counts["concordant"] / n if n else None)


def rank_residual(values, control):
    x, z = midranks(values), midranks(control)
    xm, zm = statistics.mean(x), statistics.mean(z)
    xx, zz = [v-xm for v in x], [v-zm for v in z]
    variance = math.fsum(v*v for v in zz)
    slope = math.fsum(v*w for v, w in zip(xx, zz)) / variance if variance else 0.0
    residual = [v-slope*w for v, w in zip(xx, zz)]
    # OLS may leave roundoff when x is entirely determined by rank(control).
    # This numerical guard reports unavailable association, never a healthy zero.
    norm = math.fsum(v*v for v in residual)
    floor = (32 * sys.float_info.epsilon)**2 * max(1.0, math.fsum(v*v for v in xx))
    return dict(values=residual, intercept=xm-slope*zm, slope=slope,
                residual_sum_squares=norm, numerically_nonconstant=norm > floor)


def partial_spearman(a, b, control):
    ra, rb = rank_residual(a, control), rank_residual(b, control)
    value = pearson(ra["values"], rb["values"]) if (
        ra["numerically_nonconstant"] and rb["numerically_nonconstant"]) else None
    return dict(partial_spearman=value, predictor=ra, target=rb,
                missing_reason=None if value is not None else "NO_RESIDUAL_RANK_VARIANCE")


def loo(a, b, conditions, control=None):
    result = []
    for index, name in enumerate(conditions):
        indices = [i for i in range(len(a)) if i != index]
        aa, bb = [a[i] for i in indices], [b[i] for i in indices]
        row = dict(omitted_condition=name, n=len(indices), spearman_rho=spearman(aa, bb),
                   **pairwise_concordance(aa, bb))
        if control is not None:
            row["partial_spearman_controlling_L7_TOTAL"] = partial_spearman(
                aa, bb, [control[i] for i in indices])["partial_spearman"]
        result.append(row)
    return result


def opposite_pairs(conditions, a, b, minimum_gap):
    ra, rb = midranks(a), midranks(b)
    rows = []
    for i, j in itertools.combinations(range(len(conditions)), 2):
        ga, gb = abs(ra[i]-ra[j]), abs(rb[i]-rb[j])
        if sign(a[i]-a[j]) * sign(b[i]-b[j]) != -1:
            continue
        if ga < minimum_gap or gb < minimum_gap:
            continue
        ca, cb = sorted([conditions[i], conditions[j]])
        rows.append(dict(condition_a=ca, condition_b=cb, rank_gap_a=ga, rank_gap_b=gb,
                         minimum_rank_gap=min(ga, gb),
                         metric_a_winner=conditions[i] if a[i] < a[j] else conditions[j],
                         metric_b_winner=conditions[i] if b[i] < b[j] else conditions[j]))
    return sorted(rows, key=lambda row: (-row["minimum_rank_gap"],
                                        row["condition_a"], row["condition_b"]))


def validate_runs(root, run_dir, cfg):
    if list(run_dir.glob("*.STOP.json")):
        raise RuntimeError("RETAINED_STOP_NO_ANALYSIS_AS_PASS")
    runs, identities = {}, {}
    for doc in cfg["documents"]:
        gpu = cfg["document_gpu_map"][doc]
        path = run_dir / f"formal_{doc}_gpu{gpu}.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        sidecar_path = path.with_suffix(".sha256.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sha(path) == sidecar["sha256"], "FORMAL_SIDECAR_MISMATCH"
        assert run["valid"] and finite_tree(run) and run["mode"] == "formal"
        assert run["document"] == doc and run["gpu_uuid"] == cfg["gpu_uuids"][str(gpu)]
        assert run["config_sha256"] == sha(root / "PHASE_A_CONFIG.json")
        assert run["runner_sha256"] == sha(root / "run_matched_l7.py")
        assert not run["prior_score_set_joined"] and not run["gradients_or_optimizer_used"]
        assert [x["condition"] for x in run["results"]] == cfg["conditions"]
        assert run["fp_reference"]["tokens"] == 1152
        assert run["fp_reference"]["replay_parity"]["passed"]
        assert content_sha(restored_fp_identity(run["fp_reference"]["tensor_identities"])) == run["fp_reference"]["content_sha256"]
        checks = run["fp_reference"]["helper_real_forward_checks"]
        assert len(checks) == len(cfg["layers"]) * len(cfg["windows"]) * 2
        assert all(x["bitwise_equal"] and x["identity_loss"] == 0 for x in checks)
        for condition in run["results"]:
            gates = condition["mechanical_gates"]
            assert gates["passed"] and gates["every_token_layer_qdq_count"] == 1152*18
            assert gates["actual_next_kernel_equal_post_qdq_count"] == gates["expected_consumption_count"] == 1151*18
            assert [x["start"] for x in condition["windows"]] == cfg["windows"]
            assert condition["l7_scored_token_count"] == 256
            assert not condition["historical_training_sampling_protocol_replayed"]
            for metric in METRICS:
                average = statistics.mean(x[metric] for x in condition["windows"])
                assert math.isclose(condition["aggregate"][metric], average, rel_tol=1e-14, abs_tol=0)
            for scores in condition["windows"] + [condition["aggregate"]]:
                assert math.isclose(scores["L7_TOTAL_ORIGINAL"],
                    scores["L7_FUNCTIONAL_ORIGINAL"] + 0.1*scores["L7_STATE_ORIGINAL"],
                    rel_tol=2e-6, abs_tol=0), "ORIGINAL_COMBINATION_CONSISTENCY_FAIL"
        runs[doc] = run
        identities[doc] = dict(canonical_sha256=sha(path), sidecar_sha256=sha(sidecar_path),
                               fp_reference_sha256=run["fp_reference"]["content_sha256"],
                               gpu_uuid=run["gpu_uuid"], input_sha256=run["token_file"]["sha256"])
    return runs, identities


def matched_tables(runs, cfg):
    scores, ranks, pairs, comparisons, sensitivity, panels = [], [], [], [], [], {}
    conditions = cfg["conditions"]
    for doc, run in runs.items():
        by = {x["condition"]: x for x in run["results"]}
        for window in [str(x) for x in cfg["windows"]] + ["MEAN_256_1024"]:
            values = {metric: [] for metric in METRICS}
            for name in conditions:
                item = by[name]
                value = (item["aggregate"] if window == "MEAN_256_1024" else
                         next(x for x in item["windows"] if x["start"] == int(window)))
                scores.append(dict(document=doc, condition=name, window=window, horizon=128,
                    gpu_uuid=run["gpu_uuid"], input_sha256=run["token_file"]["sha256"],
                    rotation_sha256=run["rotation_artifacts"][name]["sha256"],
                    fp_reference_sha256=run["fp_reference"]["content_sha256"],
                    runner_sha256=run["runner_sha256"], matched_definition_not_training_replay=True,
                    **{m: value[m] for m in METRICS}))
                for metric in METRICS:
                    values[metric].append(value[metric])
            panels[(doc, window)] = values
            rank = {m: midranks(v) for m, v in values.items()}
            for m in METRICS:
                for i, name in enumerate(conditions):
                    ranks.append(dict(document=doc, window=window, metric=m, condition=name,
                                      value=values[m][i], midrank=rank[m][i]))
            for ma, mb in COMPARISONS:
                a, b = values[ma], values[mb]
                comparisons.append(dict(document=doc, window=window, metric_a=ma, metric_b=mb,
                    n=16, spearman_rho=spearman(a, b), **pairwise_concordance(a, b)))
                sensitivity.extend(dict(document=doc, window=window, metric_a=ma, metric_b=mb,
                                        **x) for x in loo(a, b, conditions))
                for i, j in itertools.combinations(range(16), 2):
                    sa, sb = sign(a[i]-a[j]), sign(b[i]-b[j])
                    pairs.append(dict(document=doc, window=window, metric_a=ma, metric_b=mb,
                        condition_a=conditions[i], condition_b=conditions[j],
                        score_a_i=a[i], score_a_j=a[j], score_b_i=b[i], score_b_j=b[j],
                        midrank_a_i=rank[ma][i], midrank_a_j=rank[ma][j],
                        midrank_b_i=rank[mb][i], midrank_b_j=rank[mb][j],
                        rank_gap_a=abs(rank[ma][i]-rank[ma][j]),
                        rank_gap_b=abs(rank[mb][i]-rank[mb][j]), opposite_order=sa*sb == -1,
                        tied_a=sa == 0, tied_b=sb == 0))
    return dict(scores=scores, ranks=ranks, pairwise_rankings=pairs,
                redundancy_statistics=comparisons, redundancy_leave_one_out=sensitivity), panels


def choose_route(panels, cfg):
    eligible = {}
    for route, a, b in [("A", "P1", "L7_TOTAL_ORIGINAL"),
                         ("B", "L7_TOTAL_ORIGINAL", "B0")]:
        eligible[route] = []
        for doc in cfg["documents"]:
            values = panels[(doc, "MEAN_256_1024")]
            pairs = opposite_pairs(cfg["conditions"], values[a], values[b],
                                   cfg["route_rule"]["minimum_midrank_gap_both_metrics"])
            eligible[route].append(dict(document=doc, metric_a=a, metric_b=b, eligible_pairs=pairs))
    route = ("A" if any(x["eligible_pairs"] for x in eligible["A"]) else
             "B" if any(x["eligible_pairs"] for x in eligible["B"]) else "C")
    return dict(route=route, status="INFORMATIVE_SHORT_RANKING_CONTRAST" if route != "C" else
                "NO_INFORMATIVE_NATURAL_ROTATION_CONTRAST",
                basis="FROZEN_EQUAL_TWO_WINDOW_AGGREGATE_PER_DOCUMENT",
                historical_window_panels_are_reported_not_selected=True,
                eligible_pairs=eligible, minimum_midrank_gap_both_metrics=3,
                used_historical_or_new_future_outcomes=False,
                phase_b_long_runs_authorized_by_this_file=False)


def discovery_tables(panels, cfg, label_path, baseline_path):
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    identities = {tuple(x["key"]): x for x in baseline["files"]}
    labels = list(csv.DictReader(label_path.open(encoding="utf-8", newline="")))
    statistics_rows, residual_rows, sensitivity, source_ids = [], [], [], []
    for doc in cfg["documents"]:
        for role, anchor in [("Primary", 8192), ("Secondary", 2048)]:
            records = [x for x in labels if x["document"] == doc and x["anchor_role"] == role]
            by = {x["condition"]: x for x in records}
            assert len(records) == 16 and sorted(by) == sorted(cfg["conditions"])
            for name in cfg["conditions"]:
                row = by[name]
                assert int(row["anchor"]) == anchor
                identity = identities[(doc, name, anchor, 32768)]
                assert row["canonical_sha256"] == identity["canonical"]["sha256"]
                assert row["sidecar_sha256"] == identity["sidecar"]["sha256"]
                source_ids.append(dict(document=doc, condition=name, anchor=anchor,
                                       canonical=identity["canonical"], sidecar=identity["sidecar"]))
            target = [float(by[c]["FutureKL"]) for c in cfg["conditions"]]
            assert all(math.isfinite(x) for x in target)
            for window in ["256", "1024", "MEAN_256_1024"]:
                values = panels[(doc, window)]
                control = values["L7_TOTAL_ORIGINAL"]
                for metric in METRICS:
                    partial = partial_spearman(values[metric], target, control)
                    statistics_rows.append(dict(document=doc, window=window, anchor=anchor,
                        anchor_role=role, metric=metric, scope="DISCOVERY_ONLY", n_rotations=16,
                        spearman_rho=spearman(values[metric], target),
                        partial_spearman_controlling_L7_TOTAL=partial["partial_spearman"],
                        partial_missing_reason=partial["missing_reason"],
                        **pairwise_concordance(values[metric], target)))
                    for i, name in enumerate(cfg["conditions"]):
                        residual_rows.append(dict(document=doc, window=window, anchor=anchor,
                            metric=metric, condition=name, scope="DISCOVERY_ONLY",
                            predictor_rank=midranks(values[metric])[i], target_rank=midranks(target)[i],
                            predictor_rank_residual=partial["predictor"]["values"][i],
                            target_rank_residual=partial["target"]["values"][i],
                            controlling_rank_L7_TOTAL=midranks(control)[i]))
                    sensitivity.extend(dict(document=doc, window=window, anchor=anchor,
                        metric=metric, scope="DISCOVERY_ONLY", **x) for x in
                        loo(values[metric], target, cfg["conditions"], control))
    return dict(discovery_statistics=statistics_rows, discovery_rank_residuals=residual_rows,
                discovery_leave_one_out=sensitivity,
                label_csv_sha256=sha(label_path), label_canonical_identities=source_ids,
                independent_documents=2, historical_anchor_panels=4,
                labels="PULSE_PREFIX_THEN_NO_FURTHER_QDQ_NOT_PERSISTENT_VALIDATION",
                cross_runtime="EXPLORATORY_ASSOCIATION_ONLY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--historical-labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "PHASE_A_CONFIG.json").read_text(encoding="utf-8"))
    runs, identities = validate_runs(root, args.runs, cfg)
    tables, panels = matched_tables(runs, cfg)
    route = choose_route(panels, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    short = args.output / "PHASE_A_MATCHED_TABLES.json"
    write_json(short, tables)
    route_path = args.output / "PHASE_A_ROUTE.json"
    write_json(route_path, route)
    # Freeze short inputs, exact analysis code and Route before opening labels.
    write_json(args.output / "PHASE_A_SHORT_SCORE_AND_ROUTE_FREEZE.json", dict(
        epoch=time.time(), formal_identities=identities, short_table_sha256=sha(short),
        route_sha256=sha(route_path), analysis_code_sha256=sha(__file__),
        historical_labels_not_yet_opened_by_this_execution=True,
        frozen_scoring_code_unchanged=True, run_data_complete=True))
    discovery = discovery_tables(panels, cfg, args.historical_labels,
                                 root / "HISTORICAL_IMMUTABILITY_BASELINE.json")
    write_json(args.output / "PHASE_A_DISCOVERY_TABLES.json", discovery)
    write_json(args.output / "PHASE_A_ANALYSIS.json", dict(
        experiment=cfg["experiment"], route=route, formal_identities=identities,
        scoring_commit="a74255b1b173d27a3e51f16dbc4cb26b46104d08",
        analysis_code_sha256=sha(__file__), scope="PHASE_A_ACTUAL_MATCHED_FORWARD_SCORES",
        redundancy_statistics=tables["redundancy_statistics"],
        incremental_information=discovery["discovery_statistics"],
        sample_units=dict(independent_documents=2, rotations_per_document=16,
                          calibration_windows_per_document=2, historical_anchor_panels=4),
        no_gradient_equivalence_claim=True, no_training_advantage_claim=True,
        phase_b_executed=False, phase_c_executed=False))
    print(json.dumps(dict(status="PHASE_A_ANALYSIS_COMPLETE", route=route["route"],
                          output=str(args.output), score_rows=len(tables["scores"]))))


if __name__ == "__main__":
    main()
