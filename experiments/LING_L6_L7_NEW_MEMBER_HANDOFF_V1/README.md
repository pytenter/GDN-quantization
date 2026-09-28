# Ling L6/L7 new-member handoff

This directory is the reproducible entry point for the frozen Ling-3.0-tiny/KDA L6 and L7 rotations.

- GitHub: <https://github.com/pytenter/GDN-quantization>
- Branch: `handoff/ling-l6-l7-reproduction-v1`
- Hugging Face dataset: <https://huggingface.co/datasets/pytenter/ling-kda-artifacts>
- Verified Hugging Face commit: `bbc37a200721ee6e5b02f3793d7731c4c53464e2`
- Official model: `inclusionAI/Ling-3.0-tiny` at revision `e3a47d5b986e7141b6efd62597d598ebb392060d`

The four canonical binaries are distributed through Hugging Face, not Git. The 16-file `runtime_snapshot` is content-addressed and preserves the project-owned source chain used by the canonical trainer. The canonical trainer itself remains at `experiments/LING_RECURRENT_DENSE_L6_L7_V1/scripts/recurrent_dense_train.py`.

Public upload of the 80 recurrent traces is blocked pending a license determination. See `DATA_PROVENANCE_AND_LICENSE.md`; do not interpret their absence from Hugging Face as a failed local-integrity audit.

Use `scripts/download_hf_artifacts.sh`, then `scripts/smoke_l6_l7.py`. Full 100-update retraining is available through `scripts/train_l6_l7_from_frozen_traces.sh` only when the frozen traces have been obtained through an approved distribution route.

Clean-room verification passed for both L6 and L7 gradient and one-update paths. See `analysis/clean_room_smoke_audit.json` and `reports/FINAL_REPORT.md`.

```ini
NEW_MEMBER_DIRECT_EVAL_READY = YES
NEW_MEMBER_RETRAIN_READY_EXCEPT_TRACES = YES
```
