"""Frozen original CPU pair rule only. No GPU or long-outcome access."""
import itertools
import math

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

def sign(value):
    return (value > 0) - (value < 0)

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
