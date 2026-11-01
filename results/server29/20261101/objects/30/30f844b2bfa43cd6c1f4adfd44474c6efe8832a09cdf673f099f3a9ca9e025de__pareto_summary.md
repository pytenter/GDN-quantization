# Phase A — offline Pareto summary

## Integrity gate

PASS: 40 unique finite checkpoint rows match the preserved original C5/C6 validation summaries exactly. Every row records 16 documents and 128 captures on the same frozen panel. Known prompt values agree within 1e-12.

## Frontier counts

- C5-only frontier: 4 checkpoints — C5 u63, C5 u69, C5 u82, C5 u113.
- C6-only frontier: 2 checkpoints — C6 u119, C6 u125.
- Combined frontier: 4 checkpoints — C5 u63, C5 u69, C5 u82, C5 u113.
- Global frontier family occupancy: C5=4, C6=0.

## Mandatory checkpoints

| Checkpoint | Family Pareto | Global Pareto | Dominated by | State Δ vs family canonical | Functional Δ vs family canonical |
|---|---|---|---|---:|---:|
| C5 u63 | True | True | none | +0.000% | +0.000% |
| C5 u113 | True | True | none | +2.329% | -5.091% |
| C6 u119 | True | False | C5 u25, C5 u32, C5 u38, C5 u44, C5 u50, C5 u57, C5 u63, C5 u69, C5 u75, C5 u82, C5 u88, C5 u94, C5 u100, C5 u107, C5 u113, C5 u119, C5 u125 | -0.241% | +6.790% |
| C6 u125 | True | False | C5 u75, C5 u82, C5 u88, C5 u113 | +0.000% | +0.000% |

## Explicit questions

- C5 checkpoints dominating canonical C6 u125: C5 u75, C5 u82, C5 u88, C5 u113.
- C6 checkpoints dominating canonical C5 u63: none.
- Does one family occupy most of the combined frontier? YES — C5 occupies all combined-frontier points.
- Pareto position is not interpreted as AIME accuracy.
