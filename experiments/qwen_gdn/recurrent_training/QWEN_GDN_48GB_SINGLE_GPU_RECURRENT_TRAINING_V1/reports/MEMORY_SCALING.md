# Fixed-sample recurrent horizon diagnostic

Each horizon was one fresh-process C5 Adam update on the frozen first 32 tokens of `TRAIN:first-row`, with canonical student recurrent INT8-C128 QDQ/writeback and theta=0 initialization. The script detached the cache at the specified BPTT boundary. This deliberately bounded diagnostic does not prove full 1024-token, eight-target document training.

| Horizon | Peak allocated GiB | Peak reserved GiB | Conservative free fraction | Status |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 17.318 | 17.570 | 62.92% | PASS |
| 4 | 17.785 | 18.029 | 61.95% | PASS |
| 8 | 18.422 | 18.668 | 60.58% | PASS |
| 16 | 19.731 | 19.992 | 57.81% | PASS |
| 32 | 22.634 | 22.660 | 52.18% | PASS |

Every one-update run had 24/24 present, finite, nonzero rotation gradients, 768 C128 QDQ calls, exact checked recurrent writeback across all 24 GDN layers, unchanged frozen-base fingerprints, rotation-only optimizer ownership, completed backward/Adam step, and a passing orthogonality gate. Four memory snapshots per run are in `analysis/H*.json`; exact byte values are in `analysis/memory_scaling.csv`.

The preregistered H32 continuation completed five consecutive updates, then three further diagnostic updates before saving a rotation-only checkpoint. The post-GC end-of-update allocated baseline was 16.699552 GiB for every update, a span of 0 bytes against the preregistered 256 MiB stop limit. All eight updates were finite and orthogonal with no observed memory creep. Thus `H32_ONE_UPDATE_GATE=PASS`, `H32_x5_SUSTAINED_GATE=PASS`, and the largest stable *diagnostic* horizon is 32.
