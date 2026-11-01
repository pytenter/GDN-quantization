#!/usr/bin/env python3
"""Deterministic full-coverage Stage-B streaming reducer.

Raw group observations are consumed exactly once and discarded.  The reducer
keeps exact float64 moments, exact token-local tails, exact paired threshold
counts, deterministic top-K event ledgers, and a mergeable deterministic
DDSketch-style log histogram for approximate global group tails.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np


VERSION = "QWEN_STAGE_B_STREAMING_REDUCER_V1"
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
PRIMARY = ("scale", "absmax", "papr_linear", "qdq_relative_l2")
SECONDARY = ("range", "std", "qdq_relative_mse", "unique_qcode_count", "saturation_fraction")
TAIL_METRICS = ("scale", "papr_linear", "qdq_relative_l2")
PAIRS = (("C6", "Hadamard"), ("C5", "Hadamard"), ("C6", "C5"), ("Native", "Hadamard"))
THRESHOLDS = (1.00, 1.10, 1.25, 1.50, 2.00)
TOP_K = 32
EPS = 1e-12
PERCENTILE_METHOD = "linear"
SKETCH_RELATIVE_ACCURACY = 0.0025


class Moments:
    def __init__(self):
        self.count = 0
        self.sum = np.float64(0.0)
        self.sum_sq = np.float64(0.0)
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values):
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if not x.size:
            return
        self.count += int(x.size)
        self.sum = np.float64(self.sum + x.sum(dtype=np.float64))
        self.sum_sq = np.float64(self.sum_sq + np.square(x, dtype=np.float64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))

    def as_dict(self):
        mean = float(self.sum / self.count) if self.count else None
        variance = max(0.0, float(self.sum_sq / self.count - mean * mean)) if self.count else None
        return {
            "count": self.count, "sum": float(self.sum), "sum_sq": float(self.sum_sq),
            "min": self.minimum if self.count else None, "max": self.maximum if self.count else None,
            "mean": mean, "variance": variance,
            "std": math.sqrt(variance) if variance is not None else None,
            "accumulator_dtype": "float64",
        }


class DeterministicLogSketch:
    """Positive-value DDSketch mapping with deterministic integer bins."""
    def __init__(self, relative_accuracy=SKETCH_RELATIVE_ACCURACY):
        self.alpha = float(relative_accuracy)
        self.gamma = (1.0 + self.alpha) / (1.0 - self.alpha)
        self.log_gamma = math.log(self.gamma)
        self.bins = defaultdict(int)
        self.zero_count = 0
        self.count = 0

    def update(self, values):
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if np.any(x < 0) or np.any(~np.isfinite(x)):
            raise ValueError("log sketch accepts finite nonnegative values only")
        zeros = int((x == 0).sum())
        self.zero_count += zeros
        self.count += int(x.size)
        positive = x[x > 0]
        if positive.size:
            keys = np.ceil(np.log(positive) / self.log_gamma).astype(np.int64)
            unique, counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique.tolist(), counts.tolist()):
                self.bins[int(key)] += int(count)

    def value(self, key):
        return float(2.0 * (self.gamma ** key) / (self.gamma + 1.0))

    def quantile(self, q):
        if not self.count:
            return None
        rank = max(1, int(math.ceil(float(q) * self.count)))
        if rank <= self.zero_count:
            return 0.0
        seen = self.zero_count
        for key in sorted(self.bins):
            seen += self.bins[key]
            if seen >= rank:
                return self.value(key)
        return self.value(max(self.bins))

    def cvar(self, q):
        if not self.count:
            return None
        target = max(1, int(math.ceil((1.0 - float(q)) * self.count)))
        remaining = target
        total = 0.0
        for key in sorted(self.bins, reverse=True):
            take = min(remaining, self.bins[key])
            total += take * self.value(key)
            remaining -= take
            if remaining == 0:
                break
        return total / target

    def as_dict(self):
        return {
            "implementation": "deterministic_DDSketch_style_log_histogram",
            "relative_accuracy": self.alpha, "gamma": self.gamma,
            "count": self.count, "zero_count": self.zero_count,
            "bins": [[k, self.bins[k]] for k in sorted(self.bins)],
            "p95": self.quantile(.95), "p99": self.quantile(.99),
            "p999": self.quantile(.999), "cvar95": self.cvar(.95),
            "cvar99": self.cvar(.99),
        }


def exact_token_tail(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size:
        raise ValueError("empty token-local group values")
    k95 = max(1, int(math.ceil(.05 * x.size)))
    k99 = max(1, int(math.ceil(.01 * x.size)))
    # Partition gives exact top-k values without sorting the entire vector.
    top95 = np.partition(x, x.size - k95)[-k95:]
    top99 = np.partition(x, x.size - k99)[-k99:]
    return {
        "median": float(np.quantile(x, .50, method=PERCENTILE_METHOD)),
        "p95": float(np.quantile(x, .95, method=PERCENTILE_METHOD)),
        "p99": float(np.quantile(x, .99, method=PERCENTILE_METHOD)),
        "cvar95": float(top95.mean(dtype=np.float64)),
        "cvar99": float(top99.mean(dtype=np.float64)),
        "max": float(x.max()),
        "k95": k95, "k99": k99, "group_count": int(x.size),
    }


class StreamingReducer:
    def __init__(self):
        self.group_moments = defaultdict(Moments)
        self.token_tail_moments = defaultdict(Moments)
        self.sketches = defaultdict(DeterministicLogSketch)
        self.events = defaultdict(list)
        self.paired = defaultdict(lambda: {
            "total_count": 0, **{f"count_gt_{str(t).replace('.', '_')}": 0 for t in THRESHOLDS},
            "sum_log_ratio": np.float64(0), "sum_sq_log_ratio": np.float64(0), "max_ratio": 0.0,
        })
        self.pending = {}
        self.keys_seen = set()
        self.tokens_seen = set()

    def _events_update(self, sample_id, token_idx, layer, condition, metric, values, group_ids):
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        select = np.argpartition(x, -min(TOP_K, x.size))[-min(TOP_K, x.size):]
        heap = self.events[(sample_id, int(layer), condition, metric)]
        for idx in select.tolist():
            value = float(x[idx]); group_id = int(ids[idx])
            key = (value, -int(token_idx), -group_id)
            item = (key, int(token_idx), group_id, value)
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, item)

    def update(self, sample_id, token_idx, layer, condition, metrics, group_ids):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        needed = set(PRIMARY + SECONDARY)
        if not needed.issubset(metrics):
            raise KeyError(f"missing metrics: {sorted(needed-set(metrics))}")
        group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        n = group_ids.size
        for metric in PRIMARY + SECONDARY:
            values = np.asarray(metrics[metric], dtype=np.float32).reshape(-1)
            if values.size != n:
                raise ValueError(f"group cardinality mismatch for {metric}")
            self.group_moments[(sample_id, int(layer), condition, metric)].update(values)
            if metric in PRIMARY:
                self.sketches[(condition, metric)].update(values)
            if metric in TAIL_METRICS:
                tail = exact_token_tail(values)
                for name in ("median", "p95", "p99", "cvar95", "cvar99", "max"):
                    self.token_tail_moments[(sample_id, int(layer), condition, metric, name)].update([tail[name]])
                self._events_update(sample_id, token_idx, layer, condition, metric, values, group_ids)
        token_key = (sample_id, int(token_idx), int(layer))
        pending = self.pending.setdefault(token_key, {})
        if condition in pending:
            raise RuntimeError(f"duplicate condition for {token_key}: {condition}")
        pending[condition] = np.asarray(metrics["scale"], dtype=np.float64).reshape(-1).copy()
        if set(pending) == set(CONDITIONS):
            for numerator, denominator in PAIRS:
                ratio = (pending[numerator] + EPS) / (pending[denominator] + EPS)
                rec = self.paired[(sample_id, int(layer), numerator, denominator)]
                rec["total_count"] += int(ratio.size)
                for threshold in THRESHOLDS:
                    rec[f"count_gt_{str(threshold).replace('.', '_')}"] += int((ratio > threshold).sum())
                log_ratio = np.log(ratio)
                rec["sum_log_ratio"] = np.float64(rec["sum_log_ratio"] + log_ratio.sum(dtype=np.float64))
                rec["sum_sq_log_ratio"] = np.float64(rec["sum_sq_log_ratio"] + np.square(log_ratio).sum(dtype=np.float64))
                rec["max_ratio"] = max(rec["max_ratio"], float(ratio.max()))
            del self.pending[token_key]
        self.keys_seen.add((sample_id, int(layer), condition))
        self.tokens_seen.add((sample_id, int(token_idx)))

    def finalize(self):
        if self.pending:
            raise RuntimeError(f"unpaired condition buffers remain: {list(self.pending)[:3]}")
        group_rows = []
        for (sample, layer, condition, metric), moment in sorted(self.group_moments.items()):
            group_rows.append({"sample_id": sample, "layer": layer, "condition": condition, "metric": metric, **moment.as_dict()})
        tail_rows = []
        for (sample, layer, condition, metric, statistic), moment in sorted(self.token_tail_moments.items()):
            tail_rows.append({
                "sample_id": sample, "layer": layer, "condition": condition,
                "metric": metric, "token_local_statistic": statistic, **moment.as_dict(),
            })
        events = []
        for (sample, layer, condition, metric), heap in sorted(self.events.items()):
            for rank, item in enumerate(sorted(heap, key=lambda z: z[0], reverse=True), 1):
                _, token_idx, group_id, value = item
                events.append({
                    "sample_id": sample, "layer": layer, "condition": condition,
                    "metric": metric, "rank": rank, "token_idx": token_idx,
                    "group_idx": group_id, "metric_value": value,
                })
        paired_rows = []
        for (sample, layer, numerator, denominator), rec in sorted(self.paired.items()):
            row = {"sample_id": sample, "layer": layer, "numerator": numerator, "denominator": denominator}
            row.update({k: float(v) if isinstance(v, np.floating) else v for k, v in rec.items()})
            paired_rows.append(row)
        sketches = [
            {"condition": condition, "metric": metric, **sketch.as_dict()}
            for (condition, metric), sketch in sorted(self.sketches.items())
        ]
        return {
            "reducer_version": VERSION,
            "definitions": {
                "statistical_unit": "sample", "accumulator_dtype": "float64",
                "percentile_method": PERCENTILE_METHOD,
                "cvar95": "mean(top max(1,ceil(0.05*N)))",
                "cvar99": "mean(top max(1,ceil(0.01*N)))",
                "top_k_extreme_events": TOP_K,
                "primary_metrics": list(PRIMARY), "secondary_metrics": list(SECONDARY),
                "exact_tails": "token-local", "approximate_tails": "global group-level deterministic log sketch",
            },
            "group_moments": group_rows, "token_local_tail_moments": tail_rows,
            "paired_scale_ratios": paired_rows, "extreme_events": events,
            "global_group_sketches": sketches,
            "key_coverage": [list(x) for x in sorted(self.keys_seen)],
            "tokens_seen": [list(x) for x in sorted(self.tokens_seen)],
        }
