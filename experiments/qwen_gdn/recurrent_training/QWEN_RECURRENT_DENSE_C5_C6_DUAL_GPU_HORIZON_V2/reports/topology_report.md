# V2 model topology inventory

The actual model is `Qwen3_5ForCausalLM` with 32 complete decoder blocks and 24 GDN blocks. Embedding and LM head have distinct storage (`WEIGHT_TYING_GATE=PASS`). All candidates split only between complete decoder blocks. The rotations will be colocated with their GDN block.

| Split after block | GPU0 GDN | GPU1 GDN | GPU0 static model GiB | GPU1 static model GiB |
|---:|---:|---:|---:|---:|
| 12 | 10 | 14 | 7.135 | 9.543 |
| 13 | 11 | 13 | 7.541 | 9.136 |
| 15 | 12 | 12 | 8.339 | 8.339 |
| 16 | 13 | 11 | 8.746 | 7.932 |
| 17 | 14 | 10 | 9.153 | 7.525 |
