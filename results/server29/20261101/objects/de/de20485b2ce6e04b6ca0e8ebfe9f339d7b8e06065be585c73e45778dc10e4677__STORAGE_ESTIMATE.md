# Storage estimate

## Basis

- Frozen paired tokens: 1,854,451
- GDN layers: 24
- INT8 conditions: 4
- Token-layer-condition rows: 178,027,296

## Raw typed-column payload

| Component | Formula | Bytes | GiB |
|---|---:|---:|---:|
| controlled errors | rows × (token i32 + layer u8 + 4×f32) | 3,738,573,216 | 3.481 |
| recurrent drift | rows × (token i32 + layer u8 + 3×f32) | 3,026,464,032 | 2.818 |
| function space | 7,417,804 rows × 17 bytes | 126,102,668 | 0.117 |
| unchanged reducer artifacts | measured Full60 projection | 1,494,503,040 | 1.392 |
| subtotal |  | 8,385,642,956 | 7.808 |

Partition columns are encoded in paths and not repeated in every row. Parquet
metadata, finite-status ledgers, sample metadata, reports, and hashes are
budgeted above the raw subtotal.

## Planning envelope

- Expected final: 6.5–9.5 GiB depending on Zstandard compressibility.
- Approved conservative final budget: 9.5 GiB.
- Largest in-progress shard, checkpoints, and atomic duplicates: up to 3.5 GiB.
- Conservative temporary peak: 13 GiB.
- Required untouched reserve: 15 GiB.
- Required free space before launch: 28 GiB.
- Free at audit: 30.68 GiB — PASS, with only 2.68 GiB headroom.

Storing float16 metrics is rejected: it would change numerical evidence.
Storing complete recurrent states or logits is also rejected because it would
increase storage by orders of magnitude and is unnecessary for the questions.
