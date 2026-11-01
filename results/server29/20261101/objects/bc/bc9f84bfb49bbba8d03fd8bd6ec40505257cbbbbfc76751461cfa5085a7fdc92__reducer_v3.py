#!/usr/bin/env python3
"""Prospective per-sample Stage-B streaming reducer V3 (CPU-only implementation)."""
from __future__ import annotations

import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "stage_b_streaming_reducer_v3"
LAYERS = (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
PRIMARY = ("scale", "papr_linear", "qdq_relative_l2")
SECONDARY = ("absmax", "range")
METRICS = PRIMARY + SECONDARY
PAIRS = (("C6", "Hadamard"), ("C5", "Hadamard"), ("C6", "C5"), ("Native", "Hadamard"))
THRESHOLDS = (1.00, 1.10, 1.25, 1.50, 2.00)
TOP_M = 42
N_GROUPS = 4096
EPS = 1e-12


def _load_v2():
    path = Path(__file__).resolve().parents[2] / "QWEN_GDN_STAGE_B_STREAMING_REDUCER_V2_CLOSURE/scripts/reducer_v2.py"
    spec = importlib.util.spec_from_file_location("frozen_reducer_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_v2()
DeterministicTopK = V2.DeterministicTopK
MultiplicativeHistogram = V2.MultiplicativeHistogram
Moments = V2.Moments
canonical_values = V2.canonical_values


def candidate_top42_tail(values):
    x = canonical_values(values)
    if x.size != N_GROUPS:
        raise ValueError(f"expected {N_GROUPS} groups")
    selected = np.partition(x, x.size - TOP_M)[-TOP_M:]
    selected.sort()
    h = (x.size - 1) * .99
    lo, hi = int(math.floor(h)), int(math.ceil(h))
    base = x.size - TOP_M
    if lo < base or hi < base:
        raise AssertionError("TOP42_SUFFICIENCY_GATE=FAIL")
    p99 = float(selected[lo-base] + (h-lo) * (selected[hi-base] - selected[lo-base]))
    k = max(1, int(math.ceil(.01*x.size)))
    return {"p99": p99, "cvar99": float(selected[-k:].mean(dtype=np.float64)),
            "max": float(selected[-1]), "k_cvar99": k}


def _new_pair():
    return {"total_count": 0, **{f"count_gt_{str(t).replace('.', '_')}": 0 for t in THRESHOLDS},
            "sum_log_ratio": np.float64(0.0), "sum_sq_log_ratio": np.float64(0.0), "max_ratio": 0.0}


def moment_summary(state):
    count = int(state["count"])
    mean = float(state["sum"] / count) if count else None
    variance = max(0.0, float(state["sum_sq"] / count - mean * mean)) if count else None
    return {**state, "mean": mean, "variance": variance,
            "std": math.sqrt(variance) if variance is not None else None,
            "accumulator_dtype": "float64"}


class SampleReducerV3:
    def __init__(self, identity):
        self.identity = dict(identity)
        self.moments = defaultdict(Moments)
        self.tail_moments = defaultdict(Moments)
        self.layer_hist = defaultdict(MultiplicativeHistogram)
        self.global_hist = defaultdict(MultiplicativeHistogram)
        self.topk = defaultdict(DeterministicTopK)
        self.paired = defaultdict(_new_pair)
        self.last_committed_token_idx = -1
        self.tokens_seen = []

    def update_token(self, token_idx, batches):
        token_idx = int(token_idx)
        expected = self.last_committed_token_idx + 1
        if token_idx < expected:
            raise ValueError("duplicate token after resume")
        if token_idx > expected:
            raise ValueError("token gap after resume")
        expected_keys = {(layer, cond) for layer in LAYERS for cond in CONDITIONS}
        if set(batches) != expected_keys:
            raise ValueError("token key coverage mismatch")
        for layer in LAYERS:
            scales = {}
            for cond in CONDITIONS:
                rec = batches[(layer, cond)]
                gids = np.asarray(rec["group_ids"], np.int64).reshape(-1)
                if gids.size != N_GROUPS or set(gids.tolist()) != set(range(N_GROUPS)):
                    raise ValueError("group-linear identity mismatch")
                if set(rec["metrics"]) != set(METRICS):
                    raise ValueError("metric set mismatch")
                dense = {}
                for metric in METRICS:
                    x = canonical_values(rec["metrics"][metric])
                    if x.size != N_GROUPS:
                        raise ValueError("metric cardinality mismatch")
                    ordered = np.empty(N_GROUPS, np.float64); ordered[gids] = x
                    dense[metric] = ordered
                    key = (layer, cond, metric)
                    self.moments[key].update(ordered)
                    self.layer_hist[key].update(ordered)
                    self.global_hist[(cond, metric)].update(ordered)
                    if metric in PRIMARY:
                        tails = candidate_top42_tail(ordered)
                        for stat in ("p99", "cvar99", "max"):
                            self.tail_moments[(layer, cond, metric, stat)].update([tails[stat]])
                        self.topk[key].update(ordered, token_idx, np.arange(N_GROUPS, dtype=np.int64))
                scales[cond] = dense["scale"]
            for numerator, denominator in PAIRS:
                ratio = (scales[numerator] + EPS) / (scales[denominator] + EPS)
                p = self.paired[(layer, numerator, denominator)]
                p["total_count"] += int(ratio.size)
                for threshold in THRESHOLDS:
                    p[f"count_gt_{str(threshold).replace('.', '_')}"] += int(np.count_nonzero(ratio > threshold))
                logs = np.log(ratio)
                p["sum_log_ratio"] = np.float64(p["sum_log_ratio"] + logs.sum(dtype=np.float64))
                p["sum_sq_log_ratio"] = np.float64(p["sum_sq_log_ratio"] + np.square(logs).sum(dtype=np.float64))
                p["max_ratio"] = max(p["max_ratio"], float(ratio.max()))
        self.last_committed_token_idx = token_idx
        self.tokens_seen.append(token_idx)

    @staticmethod
    def _records(mapping, serializer):
        return [{"key": list(k), "state": serializer(v)} for k, v in sorted(mapping.items())]

    def to_state(self):
        return {
            "reducer_schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "processing_cursor": {"sample_id": self.identity["sample_id"],
                                  "last_committed_token_idx": self.last_committed_token_idx},
            "tokens_seen": list(self.tokens_seen),
            "moments": self._records(self.moments, lambda x:x.state()),
            "tail_moments": self._records(self.tail_moments, lambda x:x.state()),
            "layer_histograms": self._records(self.layer_hist, lambda x:x.state()),
            "global_histograms": self._records(self.global_hist, lambda x:x.state()),
            "extreme_topk": self._records(self.topk, lambda x:x.state()),
            "paired": self._records(self.paired, lambda x:{k:(float(v) if isinstance(v,np.floating) else v) for k,v in x.items()}),
        }

    @classmethod
    def from_state(cls, state, expected_identity):
        if state.get("reducer_schema_version") != SCHEMA_VERSION:
            raise ValueError("wrong reducer_schema_version")
        if state.get("identity") != expected_identity:
            raise ValueError("checkpoint identity mismatch")
        cursor = state.get("processing_cursor", {})
        if cursor.get("sample_id") != expected_identity["sample_id"]:
            raise ValueError("cursor sample mismatch")
        obj = cls(expected_identity)
        obj.last_committed_token_idx = int(cursor["last_committed_token_idx"])
        obj.tokens_seen = list(map(int, state["tokens_seen"]))
        if obj.tokens_seen != list(range(obj.last_committed_token_idx+1)):
            raise ValueError("corrupted processing cursor")
        for row in state["moments"]: obj.moments[tuple(row["key"])] = Moments.from_state(row["state"])
        for row in state["tail_moments"]: obj.tail_moments[tuple(row["key"])] = Moments.from_state(row["state"])
        for section,target in (("layer_histograms",obj.layer_hist),("global_histograms",obj.global_hist)):
            for row in state[section]: target[tuple(row["key"])] = MultiplicativeHistogram.from_state(row["state"])
        for row in state["extreme_topk"]: obj.topk[tuple(row["key"])] = DeterministicTopK.from_state(row["state"])
        for row in state["paired"]:
            rec=dict(row["state"])
            if int(rec["total_count"]) < 0 or any(int(rec[k]) < 0 for k in rec if k.startswith("count_gt_")):
                raise ValueError("corrupted paired state")
            obj.paired[tuple(row["key"])] = rec
        return obj

    def discrete_signature(self):
        s=self.to_state()
        return {
            "minmax":[{"key":r["key"],"count":r["state"]["count"],"min":r["state"]["min"],"max":r["state"]["max"]} for r in s["moments"]],
            "histograms":s["layer_histograms"],"extreme_topk":s["extreme_topk"],
            "paired_counts":[{"key":r["key"],"total_count":r["state"]["total_count"],
               "thresholds":{k:v for k,v in r["state"].items() if k.startswith("count_gt_")}} for r in s["paired"]],
            "cursor":s["processing_cursor"],"tokens_seen":s["tokens_seen"],
        }

    def sample_artifact(self):
        s=self.to_state()
        units={}
        for layer in LAYERS:
            for cond in CONDITIONS:
                key=f"layer_{layer}/{cond}"
                units[key]={
                    "metrics":{m:moment_summary(next(r["state"] for r in s["moments"] if r["key"]==[layer,cond,m])) for m in METRICS},
                    "token_tail_summaries":{m:{stat:moment_summary(next(r["state"] for r in s["tail_moments"] if r["key"]==[layer,cond,m,stat])) for stat in ("p99","cvar99","max")} for m in PRIMARY},
                }
        return {"reducer_schema_version":SCHEMA_VERSION,"identity":self.identity,
                "processing_cursor":s["processing_cursor"],"summary_units":units,
                "paired":s["paired"],"extreme_topk":s["extreme_topk"],
                "descriptive_histogram_counts":{"layer":s["layer_histograms"],"global":s["global_histograms"]}}
