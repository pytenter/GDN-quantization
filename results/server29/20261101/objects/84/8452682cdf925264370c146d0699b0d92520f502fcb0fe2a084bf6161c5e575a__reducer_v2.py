#!/usr/bin/env python3
"""Deterministic, per-sample Stage-B streaming reducer V2 (CPU only)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


SCHEMA_VERSION = "stage_b_streaming_reducer_v2"
TOP_K = 32
GAMMA = 1.001
CONDITIONS = ("Native", "Hadamard", "C5", "C6")
PRIMARY = ("scale", "absmax", "papr_linear", "qdq_relative_l2")
PAIRS = (("C6", "Hadamard"), ("C5", "Hadamard"), ("C6", "C5"), ("Native", "Hadamard"))
THRESHOLDS = (1.00, 1.10, 1.25, 1.50, 2.00)
EPS = 1e-12


def canonical_values(values, nonnegative=True):
    x = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if np.any(~np.isfinite(x)):
        raise ValueError("NONFINITE_METRIC_GATE=FAIL")
    if nonnegative and np.any(x < 0):
        raise ValueError("negative value in non-negative primary metric")
    x[x == 0.0] = 0.0  # canonicalize -0.0 to +0.0
    return x


def event_sort_key(event):
    value, token_idx, group_idx = event
    return (-float(value), int(token_idx), int(group_idx))


class DeterministicTopK:
    def __init__(self, k=TOP_K):
        self.k = int(k)
        self.events = []

    def update(self, values, token_idx, group_ids):
        x = canonical_values(values)
        gids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if x.size != gids.size:
            raise ValueError("value/id cardinality mismatch")
        if len(set(map(int, gids.tolist()))) != gids.size:
            raise ValueError("duplicate EVENT_ID within batch")
        incoming = [(float(v), int(token_idx), int(g)) for v, g in zip(x.tolist(), gids.tolist())]
        existing_ids = {(t, g) for _, t, g in self.events}
        if any((t, g) in existing_ids for _, t, g in incoming):
            raise ValueError("duplicate EVENT_ID across batches")
        # A full total-order selection is intentional: partition tie behaviour is never used.
        self.events = sorted(self.events + incoming, key=event_sort_key)[: self.k]

    def merge(self, other):
        if self.k != other.k:
            raise ValueError("TopK K mismatch")
        result = DeterministicTopK(self.k)
        ids = [(t, g) for _, t, g in self.events + other.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate EVENT_ID during TopK merge")
        result.events = sorted(self.events + other.events, key=event_sort_key)[: self.k]
        return result

    def rows(self):
        return [
            {"rank": rank, "metric_value": value, "token_idx": token, "group_linear_idx": group}
            for rank, (value, token, group) in enumerate(sorted(self.events, key=event_sort_key), 1)
        ]

    def state(self):
        return {"k": self.k, "events": self.rows()}

    @classmethod
    def from_state(cls, state):
        obj = cls(int(state["k"]))
        rows = state["events"]
        events = [(float(r["metric_value"]), int(r["token_idx"]), int(r["group_linear_idx"])) for r in rows]
        if any(not math.isfinite(v) for v, _, _ in events):
            raise ValueError("corrupted TopK: nonfinite")
        if len({(t, g) for _, t, g in events}) != len(events):
            raise ValueError("corrupted TopK: duplicate identity")
        if events != sorted(events, key=event_sort_key) or len(events) > obj.k:
            raise ValueError("corrupted TopK: order or size")
        obj.events = events
        return obj


class MultiplicativeHistogram:
    def __init__(self, gamma=GAMMA):
        self.gamma = float(gamma)
        self.log_gamma = math.log(self.gamma)
        self.zero_count = 0
        self.count = 0
        self.bins = defaultdict(int)

    def index(self, value):
        x = float(value)
        if not math.isfinite(x) or x <= 0:
            raise ValueError("positive finite value required")
        i = int(math.floor(math.log(x) / self.log_gamma))
        lower = self.gamma ** i
        upper = self.gamma ** (i + 1)
        while x < lower:
            i -= 1
            lower = self.gamma ** i
            upper = self.gamma ** (i + 1)
        while x >= upper:
            i += 1
            lower = self.gamma ** i
            upper = self.gamma ** (i + 1)
        if not (lower <= x < upper):
            raise AssertionError("histogram boundary correction failed")
        return i

    def update(self, values):
        x = canonical_values(values)
        self.zero_count += int(np.count_nonzero(x == 0.0))
        self.count += int(x.size)
        for value in x[x > 0.0].tolist():
            self.bins[self.index(value)] += 1

    def merge(self, other):
        if self.gamma != other.gamma:
            raise ValueError("histogram gamma mismatch")
        out = MultiplicativeHistogram(self.gamma)
        out.zero_count = self.zero_count + other.zero_count
        out.count = self.count + other.count
        for i, n in self.bins.items():
            out.bins[i] += n
        for i, n in other.bins.items():
            out.bins[i] += n
        return out

    def quantile(self, q):
        if self.count == 0:
            return None
        # Same target position as NumPy's frozen linear method; select the upper
        # order statistic and return its conservative bin upper edge.
        target = int(math.ceil((self.count - 1) * float(q))) + 1
        if target <= self.zero_count:
            return 0.0
        seen = self.zero_count
        for i in sorted(self.bins):
            seen += self.bins[i]
            if seen >= target:
                return float(self.gamma ** (i + 1))
        raise AssertionError("histogram count inconsistency")

    def state(self):
        return {
            "gamma": self.gamma,
            "count": self.count,
            "zero_count": self.zero_count,
            "bins": [[int(i), int(self.bins[i])] for i in sorted(self.bins)],
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(float(state["gamma"]))
        obj.count = int(state["count"])
        obj.zero_count = int(state["zero_count"])
        if obj.count < 0 or obj.zero_count < 0 or obj.zero_count > obj.count:
            raise ValueError("corrupted histogram counts")
        for i, n in state["bins"]:
            i, n = int(i), int(n)
            if n <= 0 or i in obj.bins:
                raise ValueError("corrupted histogram bins")
            obj.bins[i] = n
        if obj.zero_count + sum(obj.bins.values()) != obj.count:
            raise ValueError("corrupted histogram total")
        return obj


class Moments:
    def __init__(self):
        self.count = 0
        self.total = np.float64(0.0)
        self.total_sq = np.float64(0.0)
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values):
        x = canonical_values(values)
        self.count += int(x.size)
        self.total = np.float64(self.total + x.sum(dtype=np.float64))
        self.total_sq = np.float64(self.total_sq + np.square(x).sum(dtype=np.float64))
        if x.size:
            self.minimum = min(self.minimum, float(x.min()))
            self.maximum = max(self.maximum, float(x.max()))

    def state(self):
        return {"count": self.count, "sum": float(self.total), "sum_sq": float(self.total_sq),
                "min": None if self.count == 0 else self.minimum,
                "max": None if self.count == 0 else self.maximum}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.count = int(state["count"])
        obj.total = np.float64(state["sum"])
        obj.total_sq = np.float64(state["sum_sq"])
        obj.minimum = math.inf if state["min"] is None else float(state["min"])
        obj.maximum = -math.inf if state["max"] is None else float(state["max"])
        if obj.count < 0 or any(not math.isfinite(float(v)) for v in (obj.total, obj.total_sq)):
            raise ValueError("corrupted moments")
        return obj


def _new_pair():
    return {"total_count": 0, **{f"count_gt_{str(t).replace('.', '_')}": 0 for t in THRESHOLDS},
            "sum_log_ratio": np.float64(0.0), "sum_sq_log_ratio": np.float64(0.0), "max_ratio": 0.0}


class SampleReducer:
    """One independent reducer state for one sample, updated in canonical token order."""
    def __init__(self, identity):
        self.identity = dict(identity)
        self.moments = defaultdict(Moments)
        self.layer_hist = defaultdict(MultiplicativeHistogram)
        self.global_hist = defaultdict(MultiplicativeHistogram)
        self.topk = defaultdict(DeterministicTopK)
        self.paired = defaultdict(_new_pair)
        self.pending = {}
        self.last_token = {}
        self.keys_seen = set()
        self.batch_count = 0

    def update_batch(self, token_idx, layer, condition, metrics, group_ids):
        if condition not in CONDITIONS:
            raise ValueError("unknown condition")
        if set(metrics) != set(PRIMARY):
            raise ValueError("primary metric set mismatch")
        gids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        n = gids.size
        if n == 0 or set(map(int, gids.tolist())) != set(range(n)):
            raise ValueError("GROUP_LINEAR_IDENTITY_GATE=FAIL")
        stream_key = (int(layer), condition)
        if int(token_idx) <= self.last_token.get(stream_key, -1):
            raise ValueError("non-canonical or duplicate token batch")
        self.last_token[stream_key] = int(token_idx)
        dense_metrics = {}
        for metric in PRIMARY:
            values = canonical_values(metrics[metric])
            if values.size != n:
                raise ValueError("metric cardinality mismatch")
            dense = np.empty(n, dtype=np.float64)
            dense[gids] = values
            dense_metrics[metric] = dense
            key = (int(layer), condition, metric)
            self.moments[key].update(dense)
            self.layer_hist[key].update(dense)
            self.global_hist[(condition, metric)].update(dense)
            self.topk[key].update(dense, int(token_idx), np.arange(n, dtype=np.int64))
        pending_key = (int(token_idx), int(layer))
        p = self.pending.setdefault(pending_key, {})
        if condition in p:
            raise ValueError("duplicate pending condition")
        p[condition] = dense_metrics["scale"].tolist()
        if set(p) == set(CONDITIONS):
            for numerator, denominator in PAIRS:
                ratio = (np.asarray(p[numerator], np.float64) + EPS) / (np.asarray(p[denominator], np.float64) + EPS)
                rec = self.paired[(int(layer), numerator, denominator)]
                rec["total_count"] += int(ratio.size)
                for threshold in THRESHOLDS:
                    rec[f"count_gt_{str(threshold).replace('.', '_')}"] += int(np.count_nonzero(ratio > threshold))
                logs = np.log(ratio)
                rec["sum_log_ratio"] = np.float64(rec["sum_log_ratio"] + logs.sum(dtype=np.float64))
                rec["sum_sq_log_ratio"] = np.float64(rec["sum_sq_log_ratio"] + np.square(logs).sum(dtype=np.float64))
                rec["max_ratio"] = max(rec["max_ratio"], float(ratio.max()))
            del self.pending[pending_key]
        self.keys_seen.add((int(layer), condition))
        self.batch_count += 1

    def _records(self, mapping, serializer):
        return [{"key": list(key), "state": serializer(value)} for key, value in sorted(mapping.items())]

    def to_state(self):
        return {
            "REDUCER_SCHEMA_VERSION": SCHEMA_VERSION,
            "identity": self.identity,
            "batch_count": self.batch_count,
            "moments": self._records(self.moments, lambda x: x.state()),
            "layer_histograms": self._records(self.layer_hist, lambda x: x.state()),
            "global_histograms": self._records(self.global_hist, lambda x: x.state()),
            "topk": self._records(self.topk, lambda x: x.state()),
            "paired": self._records(self.paired, lambda x: {k: float(v) if isinstance(v, np.floating) else v for k, v in x.items()}),
            "pending": [{"key": list(k), "conditions": v} for k, v in sorted(self.pending.items())],
            "last_token": [{"key": list(k), "token_idx": v} for k, v in sorted(self.last_token.items())],
            "keys_seen": [list(k) for k in sorted(self.keys_seen)],
        }

    @classmethod
    def from_state(cls, state, expected_identity):
        if state.get("REDUCER_SCHEMA_VERSION") != SCHEMA_VERSION:
            raise ValueError("reducer schema version mismatch")
        if state.get("identity") != expected_identity:
            raise ValueError("reducer identity mismatch")
        obj = cls(expected_identity)
        obj.batch_count = int(state["batch_count"])
        for row in state["moments"]:
            obj.moments[tuple(row["key"])] = Moments.from_state(row["state"])
        for section, target in (("layer_histograms", obj.layer_hist), ("global_histograms", obj.global_hist)):
            for row in state[section]:
                target[tuple(row["key"])] = MultiplicativeHistogram.from_state(row["state"])
        for row in state["topk"]:
            obj.topk[tuple(row["key"])] = DeterministicTopK.from_state(row["state"])
        for row in state["paired"]:
            obj.paired[tuple(row["key"])] = dict(row["state"])
        obj.pending = {tuple(r["key"]): dict(r["conditions"]) for r in state["pending"]}
        obj.last_token = {tuple(r["key"]): int(r["token_idx"]) for r in state["last_token"]}
        obj.keys_seen = {tuple(k) for k in state["keys_seen"]}
        return obj

    def finalize(self):
        if self.pending:
            raise ValueError("unpaired condition buffers remain")
        return self.to_state()

    def discrete_signature(self):
        state = self.finalize()
        return {
            "counts_minmax": [{"key": r["key"], "count": r["state"]["count"], "min": r["state"]["min"], "max": r["state"]["max"]} for r in state["moments"]],
            "layer_histograms": state["layer_histograms"],
            "global_histograms": state["global_histograms"],
            "topk": state["topk"],
            "paired_integer_counts": [{"key": r["key"], "total_count": r["state"]["total_count"],
                "threshold_counts": {k: v for k, v in r["state"].items() if k.startswith("count_gt_")}} for r in state["paired"]],
            "keys_seen": state["keys_seen"],
            "batch_count": state["batch_count"],
        }
