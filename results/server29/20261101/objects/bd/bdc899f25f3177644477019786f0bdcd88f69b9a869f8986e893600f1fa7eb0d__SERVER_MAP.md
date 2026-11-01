# Three-server map

This map records the physical hosts behind the historical audit labels. Labels are retained only for provenance and must not be interpreted as hardware specifications.

| Audit label | Hostname | Scientific role | Physical GPUs |
|---|---|---|---|
| `ling4090` | `lthpc1` | Ling-3.0-tiny / KDA; canonical 81,920-token formal results and 256K Value-Hadamard follow-up | 2 x RTX 4090 |
| `qwen3090` | hostname was not reliably recoverable from the shell prompt | Qwen3.5-9B / GDN; canonical 81,920-token formal results | 4 x RTX 3090 |
| `new4090` | `nlpg-SYS-4029GP-TRT` | Later Ling/KDA post-fix evidence and 256K FP/INT8 follow-up | 8 x RTX 3090 |

`new4090` is a historical audit alias only. It is **not** a 2 x RTX 4090 machine.

The live-job snapshots in this repository were gathered read-only. No GPU process, tmux session, growing output, or scientific environment was changed.
