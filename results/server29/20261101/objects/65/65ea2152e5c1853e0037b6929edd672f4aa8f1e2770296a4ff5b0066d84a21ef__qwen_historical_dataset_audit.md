# Qwen historical dataset audit

- Status: **PASS**
- `QWEN_HISTORICAL_DATASET = HuggingFaceH4/MATH-500 test split, seed-20260827 fixed 30-item subset`
- `MATCH_MATH500 = true` (`exact=30/30`, `normalized=30/30`)
- `MATCH_AIME24 = false` (`exact=0/30`, `normalized=0/30`)
- `N_EXACT_MATCH = 30`
- Ordered problem-text list SHA256: `1530f65df60fa54318b2f1ea527ca99d3fdb67b506281ac28eebe8d0daddbe4d`
- Normalization: Unicode NFKC, split on arbitrary whitespace, rejoin with one ASCII space, trim implied

## Source proof

- Recorded MATH-500 Arrow: `/data/khzhang/hf_cache/HuggingFaceH4___math-500/default/0.0.0/ff5b20257d8185524591543f8ff5993951537bb8/math-500-test.arrow`
- MATH-500 Arrow SHA256: `ff2663846092b986df3026f53904030cbaf8e061c9f978d319a7bd86b3a04ea4`
- AIME-2024 Arrow: `/data/khzhang/hf_cache/HuggingFaceH4___aime_2024/default/0.0.0/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/aime_2024-train.arrow`
- AIME-2024 Arrow SHA256: `c4dc97e685f2992dcc2313e595bcde057e14b9f2681890e766f93afcd5e9886b`
- Index-level MATH text equality: `30/30`
- Index-level gold-answer equality: `30/30`

## Historical result proof

- Raw records: `/data/zypan/results/gdn_end2end_bit_axis_screening_v1_records.jsonl` (`7ccdd360dc31f0c77cd095e0bcd1d225533936573751116af8ecffc61854f7dc`)
- Raw summary: `/data/zypan/results/gdn_end2end_bit_axis_screening_v1.json` (`8d8d243cc3c98000a504808e533e113e8b909c119fcdc6512ca5881b7acf915a`)
- Repository summary copy: `/data/zypan/GDN-quantization/results/orientation/gdn_end2end_bit_axis_screening_v1.json` (`8d8d243cc3c98000a504808e533e113e8b909c119fcdc6512ca5881b7acf915a`)
- Raw record count: `150`, all benchmark label `MATH-500`; five configs × 30.
- `AIME24_RESULTS` is empty.
- Recomputed counts equal summary: `true`

Conclusion: **the historical 30-question Qwen result is MATH-500, not AIME-2024.**
