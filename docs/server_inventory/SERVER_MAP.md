# Three-server map

This map records the physical hosts behind the historical audit labels. Labels are retained only for provenance and must not be interpreted as hardware specifications.

| Audit label | Hostname | Scientific role | Physical GPUs |
|---|---|---|---|
| `ling4090` | `lthpc1` | Ling-3.0-tiny / KDA; historical 81,920-token formal archive and 256K precursor work | 2 x RTX 4090 |
| `qwen3090` | hostname was not reliably recoverable from the shell prompt | Qwen3.5-9B / GDN; canonical 81,920-token formal results | 4 x RTX 3090 |
| `new4090` | `nlpg-SYS-4029GP-TRT` | Later Ling/KDA post-fix evidence and 256K FP/INT8 follow-up | 8 x RTX 3090 |

`new4090` is a historical audit alias only. It is **not** a 2 x RTX 4090 machine.

Protocol-designation note (2026-09-28): the table records roles at the historical audit snapshot. Current Ling/KDA canonical evaluation uses the 256K dynamic-budget protocol in `../LING_CANONICAL_256K_PROTOCOL.md`; 81,920-token Ling outputs are historical only.

The live-job snapshots in this repository were gathered read-only. No GPU process, tmux session, growing output, or scientific environment was changed.
