#!/usr/bin/env python3
"""Prospective P0 accelerator for the frozen Stage-B V3 reducer.

The oracle module is imported unchanged.  P0 replaces only multiplicative
histogram batch insertion; all reducer ordering and all other statistics remain
the frozen V3 implementation.
"""
from __future__ import annotations

import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path("/data/zypan/worktrees/qwen-gdn-formal-equivalent-mechanism-v2")
ORACLE_PATH = REPO / "experiments/qwen_gdn/QWEN_GDN_STAGE_B_STREAMING_REDUCER_V3_CLOSURE/scripts/reducer_v3.py"


def _load_oracle():
    spec = importlib.util.spec_from_file_location("stage_b_v3_oracle_for_accel", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORACLE = _load_oracle()
SCHEMA_VERSION = ORACLE.SCHEMA_VERSION
LAYERS = ORACLE.LAYERS
CONDITIONS = ORACLE.CONDITIONS
PRIMARY = ORACLE.PRIMARY
SECONDARY = ORACLE.SECONDARY
METRICS = ORACLE.METRICS
PAIRS = ORACLE.PAIRS
THRESHOLDS = ORACLE.THRESHOLDS
TOP_M = ORACLE.TOP_M
N_GROUPS = ORACLE.N_GROUPS
EPS = ORACLE.EPS
Moments = ORACLE.Moments
DeterministicTopK = ORACLE.DeterministicTopK
canonical_values = ORACLE.canonical_values
candidate_top42_tail = ORACLE.candidate_top42_tail
moment_summary = ORACLE.moment_summary


class AcceleratedMultiplicativeHistogram(ORACLE.MultiplicativeHistogram):
    """Vectorized exact-count insertion with scalar oracle fallback at edges."""

    def update(self, values):
        x = canonical_values(values)
        self.zero_count += int(np.count_nonzero(x == 0.0))
        self.count += int(x.size)
        positive = x[x > 0.0]
        if not positive.size:
            return

        indices = np.floor(np.log(positive) / self.log_gamma).astype(np.int64)
        # Vectorized boundary correction mirrors the oracle's while loops.
        for _ in range(8):
            lower = np.power(self.gamma, indices)
            low = positive < lower
            upper = np.power(self.gamma, indices + 1)
            high = positive >= upper
            if not bool(np.any(low) or np.any(high)):
                break
            indices[low] -= 1
            indices[high] += 1
        else:
            raise AssertionError("accelerated histogram boundary correction did not converge")

        # Values extremely close to a computed boundary use the frozen scalar
        # index routine, avoiding platform-dependent vector-libm edge choices.
        lower = np.power(self.gamma, indices)
        upper = np.power(self.gamma, indices + 1)
        scale = np.maximum(np.abs(positive), 1.0)
        near = (np.abs(positive - lower) <= 16 * np.finfo(np.float64).eps * scale)
        near |= (np.abs(positive - upper) <= 16 * np.finfo(np.float64).eps * scale)
        if bool(np.any(near)):
            positions = np.flatnonzero(near)
            indices[positions] = np.fromiter(
                (self.index(float(positive[i])) for i in positions),
                dtype=np.int64,
                count=len(positions),
            )

        unique, counts = np.unique(indices, return_counts=True)
        for index, count in zip(unique.tolist(), counts.tolist()):
            self.bins[int(index)] += int(count)


class AcceleratedSampleReducerP0(ORACLE.SampleReducerV3):
    """Frozen V3 reducer with only histogram storage replaced by P0."""

    def __init__(self, identity):
        super().__init__(identity)
        self.layer_hist = defaultdict(AcceleratedMultiplicativeHistogram)
        self.global_hist = defaultdict(AcceleratedMultiplicativeHistogram)


class AcceleratedDeterministicTopK(ORACLE.DeterministicTopK):
    """NumPy local candidate selection plus frozen deterministic CPU merge."""

    def update(self, values, token_idx, group_ids):
        x = canonical_values(values)
        gids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if x.size != gids.size:
            raise ValueError("value/id cardinality mismatch")
        if len(set(map(int, gids.tolist()))) != gids.size:
            raise ValueError("duplicate EVENT_ID within batch")
        existing_ids = {(t, g) for _, t, g in self.events}
        if any((int(token_idx), int(g)) in existing_ids for g in gids.tolist()):
            raise ValueError("duplicate EVENT_ID across batches")

        # Primary key is descending metric value; token and group identity are
        # the frozen secondary/tertiary total-order keys.  Token is constant for
        # a local batch, so local extraction needs value then group identity.
        order = np.lexsort((gids, -x))[: self.k]
        incoming = [(float(x[i]), int(token_idx), int(gids[i])) for i in order.tolist()]
        self.events = sorted(
            self.events + incoming,
            key=ORACLE.V2.event_sort_key,
        )[: self.k]

    def merge(self, other):
        if self.k != other.k:
            raise ValueError("TopK K mismatch")
        result = AcceleratedDeterministicTopK(self.k)
        ids = [(t, g) for _, t, g in self.events + other.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate EVENT_ID during TopK merge")
        result.events = sorted(
            self.events + other.events,
            key=ORACLE.V2.event_sort_key,
        )[: self.k]
        return result


class AcceleratedSampleReducerP1(AcceleratedSampleReducerP0):
    """P0 histogram acceleration plus exact deterministic local TopK."""

    def __init__(self, identity):
        super().__init__(identity)
        self.topk = defaultdict(AcceleratedDeterministicTopK)


SampleReducerV3 = AcceleratedSampleReducerP1
ACCELERATION_PHASE = "P1_HISTOGRAM_AND_TOPK"
